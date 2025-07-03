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

DATASET_ROOT = '../../../../datasets/coco/subcat4080'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
VAL_PATH = os.path.join(DATASET_ROOT, "val")
TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'val.json')

DATASET_CATEGORIES= [
    {"color": [255, 109, 65], "isthing": 1, "id": 40, "name": "wine glass"},
    {"color": [0, 143, 149], "isthing": 1, "id": 41, "name": "cup"},
    {"color": [179, 0, 194], "isthing": 1, "id": 42, "name": "fork"},
    {"color": [5, 121, 0], "isthing": 1, "id": 43, "name": "knife"},
    {"color": [227, 255, 205], "isthing": 1, "id": 44, "name": "spoon"},
    {"color": [147, 186, 208], "isthing": 1, "id": 45, "name": "bowl"},
    {"color": [153, 69, 1], "isthing": 1, "id": 46, "name": "banana"},
    {"color": [3, 95, 161], "isthing": 1, "id": 47, "name": "apple"},
    {"color": [119, 0, 170], "isthing": 1, "id": 48, "name": "sandwich"},
    {"color": [0, 165, 120], "isthing": 1, "id": 49, "name": "orange"},
    {"color": [183, 130, 88], "isthing": 1, "id": 50, "name": "broccoli"},
    {"color": [95, 32, 0], "isthing": 1, "id": 51, "name": "carrot"},
    {"color": [130, 114, 135], "isthing": 1, "id": 52, "name": "hot dog"},
    {"color": [110, 129, 133], "isthing": 1, "id": 53, "name": "pizza"},
    {"color": [166, 74, 118], "isthing": 1, "id": 54, "name": "donut"},
    {"color": [79, 210, 114], "isthing": 1, "id": 55, "name": "cake"},
    {"color": [178, 90, 62], "isthing": 1, "id": 56, "name": "chair"},
    {"color": [65, 70, 15], "isthing": 1, "id": 57, "name": "couch"},
    {"color": [127, 167, 115], "isthing": 1, "id": 58, "name": "potted plant"},
    {"color": [59, 105, 106], "isthing": 1, "id": 59, "name": "bed"},
    {"color": [142, 108, 45], "isthing": 1, "id": 60, "name": "dining table"},
    {"color": [196, 172, 0], "isthing": 1, "id": 61, "name": "toilet"},
    {"color": [95, 54, 80], "isthing": 1, "id": 62, "name": "tv"},
    {"color": [128, 76, 255], "isthing": 1, "id": 63, "name": "laptop"},
    {"color": [191, 162, 208], "isthing": 1, "id": 64, "name": "mouse"},
    {"color": [0, 130, 88], "isthing": 1, "id": 65, "name": "remote"},
    {"color": [0, 32, 0], "isthing": 1, "id": 66, "name": "keyboard"},
    {"color": [0, 114, 135], "isthing": 1, "id": 67, "name": "cell phone"},
    {"color": [0, 129, 133], "isthing": 1, "id": 68, "name": "microwave"},
    {"color": [0, 74, 118], "isthing": 1, "id": 69, "name": "oven"},
    {"color": [0, 210, 114], "isthing": 1, "id": 70, "name": "toaster"},
    {"color": [0, 90, 62], "isthing": 1, "id": 71, "name": "sink"},
    {"color": [0, 70, 15], "isthing": 1, "id": 72, "name": "refrigerator"},
    {"color": [0, 167, 115], "isthing": 1, "id": 73, "name": "book"},
    {"color": [0, 105, 106], "isthing": 1, "id": 74, "name": "clock"},
    {"color": [0, 108, 45], "isthing": 1, "id": 75, "name": "vase"},
    {"color": [0, 172, 0], "isthing": 1, "id": 76, "name": "scissors"},
    {"color": [0, 54, 80], "isthing": 1, "id": 77, "name": "teddy bear"},
    {"color": [0, 76, 255], "isthing": 1, "id": 78, "name": "hair drier"},
    {"color": [0, 162, 208], "isthing": 1, "id": 79, "name": "toothbrush"},
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
        # "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

DatasetCatalog.register("coco_s2_4080_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "coco_s2_4080_train"))
MetadataCatalog.get("coco_s2_4080_train").set(json_file=TRAIN_JSON, image_root=TRAIN_PATH, evaluator_type="coco", **get_dataset_instances_meta_train())


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
        # "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

DatasetCatalog.register("coco_s2_4080_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_s2_4080_val"))
MetadataCatalog.get("coco_s2_4080_val").set(json_file=VAL_JSON, image_root=VAL_PATH, evaluator_type="coco", **get_dataset_instances_meta_val())

dataloader.train.dataset.names = "coco_s2_4080_train"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)
dataloader.test.dataset.names = "coco_s2_4080_val"
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir = "output/coco_s2"
)


model.roi_heads.num_classes = 40 

# # Schedule
train.max_iter = 54593*10
train.eval_period = 10000
train.checkpointer = dict(period=10000, max_to_keep=100)

lr_multiplier.scheduler.milestones = [54593*9, 53593*10]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

optimizer.lr = 2e-4

trainale = ["e_attn2", "box_predictor2"] 