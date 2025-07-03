from detectron2.config import LazyCall as L
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator, CityscapesInstanceEvaluator, CityscapesSemSegEvaluator
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.data.detection_utils import get_fed_loss_cls_weights

from ..COCO.mask_rcnn_vitdet_b_100ep_adapter import (
    dataloader,
    model,
    train,
    lr_multiplier,
    optimizer,
)

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import os

# categories
DATASET_ROOT = '../../../../datasets/coco/subcat6070'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
VAL_PATH = os.path.join(DATASET_ROOT, "val")
TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'val.json')


DATASET_CATEGORIES= [{"color": [142, 108, 45], "isthing": 1, "id": 60, "name": "dining table"},
    {"color": [196, 172, 0], "isthing": 1, "id": 61, "name": "toilet"},
    {"color": [95, 54, 80], "isthing": 1, "id": 62, "name": "tv"},
    {"color": [128, 76, 255], "isthing": 1, "id": 63, "name": "laptop"},
    {"color": [191, 162, 208], "isthing": 1, "id": 64, "name": "mouse"},
    {"color": [0, 130, 88], "isthing": 1, "id": 65, "name": "remote"},
    {"color": [0, 32, 0], "isthing": 1, "id": 66, "name": "keyboard"},
    {"color": [0, 114, 135], "isthing": 1, "id": 67, "name": "cell phone"},
    {"color": [0, 129, 133], "isthing": 1, "id": 68, "name": "microwave"},
    {"color": [0, 74, 118], "isthing": 1, "id": 69, "name": "oven"},
    ]


def get_dataset_instances_meta_train():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

DatasetCatalog.register("coco_s4_6070_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "coco_s4_6070_train"))
MetadataCatalog.get("coco_s4_6070_train").set(json_file=TRAIN_JSON, image_root=TRAIN_PATH, evaluator_type="coco", **get_dataset_instances_meta_train())


def get_dataset_instances_meta_val():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

DatasetCatalog.register("coco_s4_6070_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_s4_6070_val"))
MetadataCatalog.get("coco_s4_6070_val").set(json_file=VAL_JSON, image_root=VAL_PATH, evaluator_type="coco", **get_dataset_instances_meta_val())

dataloader.train.dataset.names = "coco_s4_6070_train"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)
dataloader.test.dataset.names = "coco_s4_6070_val" 
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir = "output/coco_s2"
)

model.roi_heads.num_classes = 10

# # Schedule
train.max_iter = 28617*8
train.eval_period = 10000
train.checkpointer = dict(period=10000, max_to_keep=100)

lr_multiplier.scheduler.milestones = [27617*7, 27617*8]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

optimizer.lr = 2e-4
trainale = ["e_attn4", "box_predictor4"] 
