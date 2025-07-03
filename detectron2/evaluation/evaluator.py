# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
# import clip
from .clip import *
from .clip_text import class_names_voc, BACKGROUND_CATEGORY_VOC, class_names_coco, BACKGROUND_CATEGORY_COCO, class_names_coco_stuff182_dict, coco_stuff_182_to_27
import numpy as np
from PIL import Image
from .utils import scoremap2bbox, parse_xml_to_dict, _convert_image_to_rgb, compute_AP, compute_F1, _transform_resize
from detectron2.structures import Boxes, Instances

from .visualize_heatmap import draw_feature_map
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
import os
import time, json

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
        
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            # if idx > 30:
            #     break
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def inference_on_COCOdataset(
    models, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run models on data_loader, evaluate with evaluator, and benchmark inference speed.
    """
    logger = logging.getLogger(__name__)
    total_batches = len(data_loader)
    logger.info(f"Start inference on {total_batches} batches")

    # Prepare evaluator
    if evaluator is None:
        evaluator = DatasetEvaluators([])
    elif isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    # Warm-up settings
    num_devices = get_world_size()
    num_warmup = min(5, total_batches - 1)
    start_time = time.perf_counter()
    total_data_time = total_compute_time = total_eval_time = 0.0

    # Load task mapping once
    task_json_path = "ETM/output/coco_04080.json"
    with open(task_json_path) as f:
        task_id_all = json.load(f)

    # Predefined class offsets per task count
    OFFSETS = {
        2: [0, 40],
        3: [0, 40, 60],
        4: [0, 40, 50, 60],
        5: [0, 40, 50, 60, 70],
    }

    # Context managers for eval mode and no_grad
    with ExitStack() as stack:
        for m in models:
            if isinstance(m, nn.Module):
                stack.enter_context(inference_context(m))
        stack.enter_context(torch.no_grad())

        # Inference loop
        data_start = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - data_start

            # Reset timers after warm-up
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = total_compute_time = total_eval_time = 0.0

            # Compute
            comp_start = time.perf_counter()
            file_name = os.path.basename(inputs[0]['file_name'])
            task_ids = task_id_all[file_name]
            len_task = len(task_ids)
            use_all = sum(task_ids) == 0
            selected = [i for i, t in enumerate(task_ids) if use_all or t]

            # Run models and collect outputs
            instances_list = [models[i](inputs)[0]['instances'] for i in range(len_task)]
            height, width = inputs[0]['height'], inputs[0]['width']
            out_inst = Instances((height, width))

            # Concatenate predictions
            out_inst.pred_boxes = Boxes.cat([instances_list[i].pred_boxes for i in selected])
            out_inst.scores = torch.cat([instances_list[i].scores for i in selected], dim=0)
            offsets = OFFSETS.get(len_task, [0] * len_task)
            classes = [instances_list[i].pred_classes + offsets[i] for i in selected]
            out_inst.pred_classes = torch.cat(classes, dim=0)
            out_inst.pred_masks = torch.cat([instances_list[i].pred_masks for i in selected], dim=0)
            outputs = [{'instances': out_inst}]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - comp_start

            # Evaluation
            eval_start = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - eval_start

            # Logging
            iters = idx + 1 - num_warmup * int(idx >= num_warmup)
            d_sec = total_data_time / iters
            c_sec = total_compute_time / iters
            e_sec = total_eval_time / iters
            t_sec = (time.perf_counter() - start_time) / iters

            if idx >= num_warmup * 2 or c_sec > 5:
                eta = datetime.timedelta(seconds=int(t_sec * (total_batches - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx+1}/{total_batches}. "
                        f"Data: {d_sec:.4f}s/iter. "
                        f"Comp: {c_sec:.4f}s/iter. "
                        f"Eval: {e_sec:.4f}s/iter. "
                        f"Total: {t_sec:.4f}s/iter. ETA={eta}"
                    ),
                    n=5,
                )
            data_start = time.perf_counter()

    # Final timing logs
    total_time = time.perf_counter() - start_time
    logger.info(
        f"Total inference time: {str(datetime.timedelta(seconds=total_time))} "
        f"({total_time/(total_batches-num_warmup):.6f}s/iter per device on {num_devices} devices)"
    )
    logger.info(
        f"Total compute time: {str(datetime.timedelta(seconds=total_compute_time))} "
        f"({total_compute_time/(total_batches-num_warmup):.6f}s/iter per device)"
    )

    results = evaluator.evaluate() or {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
