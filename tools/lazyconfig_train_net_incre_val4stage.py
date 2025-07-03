#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_COCOdataset, print_csv_format
from detectron2.utils import comm

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_COCOdataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)

    if args.freeze:
        for n, p in model.named_parameters():
            if n.endswith(tuple(args.freeze)) or n.startswith(tuple(args.freeze)):
                print("learnable part: ", n)
                continue
            elif "adapter_" in n:
                print("learnable part: ", n)
                continue
            else:
                print("freeze_part: ", n)
                p.requires_grad = False
    
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def load_cfg(path, opts, args):
    cfg = LazyConfig.load(path)
    LazyConfig.apply_overrides(cfg, opts)
    default_setup(cfg, args)
    return cfg

def instantiate_models(specs, args):
    """
    Build DDP models from cfg paths and checkpoints.
    specs: list of (config_path, checkpoint_path)
    """
    models = []
    for cfg_path, ckpt in specs:
        cfg = load_cfg(cfg_path, args.opts, args)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(ckpt)
        models.append(model)
    return models

def main(args):
    # Test configuration (no checkpoint needed)
    test_cfg = load_cfg(
        "../../detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s1_070.py",
        args.opts,
        args
    )

    # Model specifications: (config, checkpoint)
    model_specs = [
        (
            "../../detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s1_040.py",
            "../output/coco_s1_040/model_final.pth"
        ),
        (
            "../../detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s2_4050.py",
            "../output/coco_s2_4050_c16/model_final.pth"
        ),
        (
            "../../detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s3_5060.py",
            "../output/coco_s3_5060_c16/model_final.pth"
        ),
        (
            "../../detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s4_6070.py",
            "../output/coco_s4_6070_c16/model_final.pth"
        ),
    ]

    if args.eval_only:
        from detectron2.checkpoint import DetectionCheckpointer
        models = instantiate_models(model_specs, args)
        print(do_test(test_cfg, models))
    else:
        do_train(args, test_cfg)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
