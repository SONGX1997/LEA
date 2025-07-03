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
DATASET_ROOT = '../../../../datasets/coco/subcat5060'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
VAL_PATH = os.path.join(DATASET_ROOT, "val")
TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'val.json')

# categories
DATASET_CATEGORIES= [{"color": [183, 130, 88], "isthing": 1, "id": 50, "name": "broccoli"},
    {"color": [95, 32, 0], "isthing": 1, "id": 51, "name": "carrot"},
    {"color": [130, 114, 135], "isthing": 1, "id": 52, "name": "hot dog"},
    {"color": [110, 129, 133], "isthing": 1, "id": 53, "name": "pizza"},
    {"color": [166, 74, 118], "isthing": 1, "id": 54, "name": "donut"},
    {"color": [79, 210, 114], "isthing": 1, "id": 55, "name": "cake"},
    {"color": [178, 90, 62], "isthing": 1, "id": 56, "name": "chair"},
    {"color": [65, 70, 15], "isthing": 1, "id": 57, "name": "couch"},
    {"color": [127, 167, 115], "isthing": 1, "id": 58, "name": "potted plant"},
    {"color": [59, 105, 106], "isthing": 1, "id": 59, "name": "bed"},
    ]

def get_dataset_instances_meta_train():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

DatasetCatalog.register("coco_s3_5060_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "coco_s3_5060_train"))
MetadataCatalog.get("coco_s3_5060_train").set(json_file=TRAIN_JSON, image_root=TRAIN_PATH, evaluator_type="coco", **get_dataset_instances_meta_train())


def get_dataset_instances_meta_val():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

DatasetCatalog.register("coco_s3_5060_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_s3_5060_val"))
MetadataCatalog.get("coco_s3_5060_val").set(json_file=VAL_JSON, image_root=VAL_PATH, evaluator_type="coco", **get_dataset_instances_meta_val())

dataloader.train.dataset.names = "coco_s3_5060_train"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)
dataloader.test.dataset.names = "coco_s3_5060_val" 
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir = "output/coco_s2"
)

model.roi_heads.num_classes = 10

# # Schedule
train.max_iter = 29195*8
train.eval_period = 10000
train.checkpointer = dict(period=10000, max_to_keep=100)

lr_multiplier.scheduler.milestones = [29195*7, 29000*8]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

optimizer.lr = 2e-4
trainale = ["e_attn3", "box_predictor3"] 
