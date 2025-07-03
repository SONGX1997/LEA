# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

from . import detection_utils as utils
from . import transforms as T

from PIL import Image, ImageFilter
from tools.extract_memory import Mem
import math
import os
import random

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper", "DatasetMapper_ABR"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        # print("dataset_dict: ", dataset_dict)
        return dataset_dict



class DatasetMapper_ABR:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes

        self._load_old_mem()
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper_ABR] Augmentations used in {mode}: {augmentations}")

    def _load_old_mem(self):
        ###### 2. loading box rehearsal memory images of task task t-1 ######
        STEP = 0
        MEM_TYPE = "random"  # "mean"
        MEM_BUFF = 10000 #5000 2000 10000
        current_mem_file = f"{MEM_TYPE}_{MEM_BUFF}"
        OUTPUT_DIR = "../../detectron2/buff4080"
        class_names = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop" "sign","parking meter","bench","bird",
                    "cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
                    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
                    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
                    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
        old_classes = class_names[0:40] # [0:40]
        new_classes = class_names[40:60] # [40:80]
        # OUTPUT_DIR = "../../detectron2/buffvoc/1520s4"
        # class_names = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike",\
        #           "person""pottedplant","sheep","sofa", "train", "tvmonitor"]
        # old_classes = class_names[0:15]   #[0:10] [0:15]
        # new_classes = class_names[15:20]   #[10:20] [15:20]
        
        current_mem_path = os.path.join(OUTPUT_DIR, current_mem_file)
        if not os.path.exists(current_mem_path):
            os.mkdir(current_mem_path)

        self.PrototypeBoxSelection = Mem(new_classes, old_classes, MEM_TYPE, MEM_BUFF, STEP, current_mem_path)
        self.BoxRehearsal_path = self.PrototypeBoxSelection.exemplar
        # print(self.BoxRehearsal_path)
        random.shuffle(self.BoxRehearsal_path)
        self.batch_size = 8 # !!!
        self.bg_size = 0
        # print("self.BoxRehearsal_path", self.BoxRehearsal_path)
        
        self.boxes_index = list(range(len(self.BoxRehearsal_path)))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        
        ### Start transform_current_data_with_ABR ###
        # current_image = dataset_dict['image']
        # current_targets = dataset_dict['instances']
        # print("dataset_dict: ", dataset_dict)
        # print("dataset_dict00000:", dataset_dict)
        _dataset_dict = self.transform_current_data_with_ABR(dataset_dict)
        # print("_dataset_dict:", _dataset_dict)

        if "annotations" in _dataset_dict:
            self._transform_annotations(_dataset_dict, transforms, image_shape)
        # print("dataset_dict1: ", dataset_dict)
        return _dataset_dict


    def transform_current_data_with_ABR(self, dataset_dict=None):
        """ begin to mosaic or mixup the box images into current image """
        current_image = dataset_dict['image']
        # current_targets = dataset_dict['instances']
        current_bbox = []
        current_cls = []
        for i in dataset_dict['annotations']:
            current_bbox.append(i['bbox'])
            current_cls.append(i['category_id'])
        bbox_mode = dataset_dict['annotations'][0]['bbox_mode']

        # set the ratio for replay
        # MIX,MOS,NEW=1:1:2
        is_mosaic = False
        is_mixup = False
        if random.randint(0, 1)==0:
            if random.randint(0, 1)==0:
                is_mixup = True
            else:
                is_mosaic=True
        
        # (current_image, current_targets) = (img, target)

        if is_mosaic:
            current_image, new_bbox, new_cls, (w, h) = self._start_boxes_mosaic(current_image.permute(1,2,0), [], dataset_dict['width'], dataset_dict['height'], num_boxes=4)
            dataset_dict['height'] = h
            dataset_dict['width'] = w
            dataset_dict['file_name'] = 'mosaic'
            dataset_dict['image_id'] = 200000
            new_annotation = []
            for i in range(0, len(new_cls)):
                mosaic_dict = {}
                mosaic_dict['iscrowd'] = 0
                mosaic_dict['bbox'] = new_bbox[i].tolist()
                mosaic_dict['category_id'] = int(new_cls[i])
                mosaic_dict['segmentation'] = self.bbox2mask(new_bbox[i])
                mosaic_dict['bbox_mode'] = bbox_mode
                new_annotation.append(mosaic_dict)
            dataset_dict['annotations'] = new_annotation
            dataset_dict['image'] = current_image.permute(2,0,1)
        elif is_mixup:
            current_image, new_bbox, new_cls = self._start_mixup(current_image.permute(1,2,0), current_bbox, current_cls)
            new_annotation = []
            for i in range(0, len(new_cls)):
                mosaic_dict = {}
                mosaic_dict['iscrowd'] = 0
                mosaic_dict['bbox'] = new_bbox[i].tolist()
                mosaic_dict['category_id'] = int(new_cls[i])
                mosaic_dict['segmentation'] = self.bbox2mask(new_bbox[i])
                mosaic_dict['bbox_mode'] = bbox_mode
                new_annotation.append(mosaic_dict)
            dataset_dict['annotations'] = new_annotation
            dataset_dict['image'] = current_image.permute(2,0,1)

        return dataset_dict
    
    def bbox2mask(self, bbox):
        seg = []
        # bbox[] is x,y,w,h
        # left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        # left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        # right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        # right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])

        return [seg]


    def compute_overlap(self, a, b):
        """ compute the overlap of input a and b;
            input: a and b are the box
            output(bool): the overlap > 0.2 or not
        """
        area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

        iw = np.minimum(a[2], b[2]) - np.maximum(a[0], b[0]) + 1
        ih = np.minimum(a[3], b[3]) - np.maximum(a[1], b[1]) + 1

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        aa = (a[2] - a[0] + 1)*(a[3] - a[1]+1)
        ba = area

        intersection = iw*ih

        # this parameter can be changes for different datasets
        if intersection/aa > 0.3 or intersection/ba > 0.3:
            return intersection/ba, True
        else:
            return intersection/ba, False

    def _sample_per_bbox_from_boxrehearsal(self, i, im_shape):
        """Sample a box from the BoxRehearsal.
        Args:
            i (int): Index of the BoxRehearsal.
            im_shape (tuple): Shape of the current image.
        Returns:
            Image: The box image.
            np.ndarray: Cropped ground truth boxes.
            int: Index of the sampled box.
        """
        # Get the path to the box image
        box_im_path = self.PrototypeBoxSelection.current_mem_path
        if self.PrototypeBoxSelection.current_mem_path==None:
            box_im_path = self.PrototypeBoxSelection.first_mem_path

        box_im_path = os.path.join(box_im_path, self.BoxRehearsal_path[self.boxes_index[i]])

        # Open the box image and extract class name and index
        box_im = Image.open(box_im_path).convert("RGB")
        cls_name, index = os.path.splitext(self.BoxRehearsal_path[self.boxes_index[i]])[0].split('_')

        bboxes = [0, 0, box_im.size[0], box_im.size[1]]
        box_o_h, box_o_w = box_im.size[1], box_im.size[0]
        gt_classes = int(cls_name)

        # Calculate mean size of current input image and box
        im_mean_size = np.mean(im_shape)
        box_mean_size = np.mean(np.array([int(bboxes[2]), int(bboxes[3])]))
        
         # Modify the box size based on mean sizes
        if float(box_mean_size) >= float(im_mean_size*0.2) and float(box_mean_size) <= float(im_mean_size*0.7):
            box_scale = 1.0
        else:
            box_scale = random.uniform(float(im_mean_size*0.4), float(im_mean_size*0.6)) / float(box_mean_size)
        
        # Resize the box image
        bw = int(box_scale * box_o_w)
        bh = int(box_scale * box_o_h)
        if bw > 0 and bh > 0:
            box_im = box_im.resize((bw, bh))
        else:
            print("size error!!", bw, bh)
            bw = abs(bw)
            bh = abs(bh)
            box_im = box_im.resize((bw, bh))
        
        # Define ground truth boxes
        gt_boxes = [0, 0, box_im.size[0], box_im.size[1], gt_classes]
        
        return box_im, np.array([gt_boxes]), self.boxes_index[i]


    def _start_mixup(self, image, current_bbox, current_cls, alpha=2.0, beta=5.0):
        """ Mixup the input image

        Args:
            image : the original image
            targets : the original image's targets
        Returns:
            mixupped images and targets
        """
        # print("iiiidevice: ", image.device)
        image = np.array(image.cpu())
        # image.flags.writeable = True
        img_shape = image.shape
        # print("img_shape", img_shape)
        
        gts = []
        bbox_list = current_bbox
        label_list = current_cls
        for i in range(len(bbox_list)):
            gts.append(bbox_list[i] + [label_list[i]])
        gts = np.array(gts)
        # if not isinstance(targets, np.ndarray):
        #     gts = []
        #     # bbox_list = targets.bbox.tolist()
        #     # label_list = targets.extra_fields["labels"].tolist()
        #     # for i in range(len(bbox_list)):
        #     #     gts.append(bbox_list[i] + [label_list[i]])
        #     bbox_list = targets.gt_boxes.tensor.cpu().numpy().tolist()
        #     label_list = targets.gt_classes.cpu().numpy().tolist()
        #     for i in range(len(bbox_list)):
        #         gts.append(bbox_list[i] + [label_list[i]])
        #     gts = np.array(gts)
        # else:
        #     gts = targets

            
        # make sure the image has more than one targets
        # If the only target occupies 75% of the image, we abandon mixupping.
        _MIXUP=True
        if gts.shape[0] == 1:
            img_w = gts[0][2]-gts[0][0]
            img_h = gts[0][3]-gts[0][1]
            if (img_shape[1]-img_w)<(img_shape[1]*0.25) and (img_shape[0]-img_h)<(img_shape[0]*0.25):
                _MIXUP=False
        
        ##### For normal mixup ######
        if _MIXUP: # 
            # lambda: Sampling from a beta distribution 
            Lambda = torch.distributions.beta.Beta(alpha, beta).sample().item()
            num_mixup = 3 # more mixup boxes but not all used
            
            # makesure the self.boxes_index has enough boxes
            if len(self.boxes_index) < self.batch_size:
                # print("A repeat for boxes memory!")
                self.boxes_index = list(range(len(self.BoxRehearsal_path)))
                
            mixup_count = 0
            for i in range(num_mixup):
                c_img, c_gt, b_id = self._sample_per_bbox_from_boxrehearsal(i, img_shape)
            
                c_img = np.asarray(c_img)
                _c_gt = c_gt.copy()

                # assign a random location
                pos_x = random.randint(0, int(img_shape[1] * 0.6))
                pos_y = random.randint(0, int(img_shape[0] * 0.4))
                new_gt = [c_gt[0][0] + pos_x, c_gt[0][1] + pos_y, c_gt[0][2] + pos_x, c_gt[0][3] + pos_y]

                restart = True
                overlap = False
                max_iter = 0
                # compute the overlap with each gts in image
                while restart:
                    for g in gts:      
                        _, overlap = self.compute_overlap(g, new_gt)
                        if max_iter >= 20:
                            # if iteration > 20, delete current choosed sample
                            restart = False
                        elif max_iter < 10 and overlap:
                            pos_x = random.randint(0, int(img_shape[1] * 0.6))
                            pos_y = random.randint(0, int(img_shape[0] * 0.4))
                            new_gt = [c_gt[0][0] + pos_x, c_gt[0][1] + pos_y, c_gt[0][2] + pos_x, c_gt[0][3] + pos_y]
                            max_iter += 1
                            restart = True
                            break
                        elif 20 > max_iter >= 10 and overlap:
                            # if overlap is True, then change the position at right bottom
                            pos_x = random.randint(int(img_shape[1] * 0.4), img_shape[1])
                            pos_y = random.randint(int(img_shape[0] * 0.6), img_shape[0])
                            new_gt = [pos_x-(c_gt[0][2]-c_gt[0][0]), pos_y-(c_gt[0][3]-c_gt[0][1]), pos_x, pos_y]
                            max_iter += 1
                            restart = True
                            break
                        else:
                            restart = False
                            # print("!!!!{2} the g {0} and new_gt is: {1}".format(g, new_gt, overlap))

                if max_iter < 20:
                    a, b, c, d = 0, 0, 0, 0
                    if new_gt[3] >= img_shape[0]:
                        # at bottom right new gt_y is or not bigger
                        a = new_gt[3] - img_shape[0]
                        new_gt[3] = img_shape[0]
                    if new_gt[2] >= img_shape[1]:
                        # at bottom right new gt_x is or not bigger
                        b = new_gt[2] - img_shape[1]
                        new_gt[2] = img_shape[1]
                    if new_gt[0] < 0:
                        # at top left new gt_x is or not bigger
                        c = -new_gt[0]
                        new_gt[0] = 0
                    if new_gt[1] < 0:
                        # at top left new gt_y is or not bigger
                        d = -new_gt[1]
                        new_gt[1] = 0

                    # Use the formula by the paper to weight each image
                    img1 = Lambda*image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]]
                    c_img = (1-Lambda)*c_img
                    # print("img1: ", img1.shape, c_img.shape)
                    
                    # Combine the images
                    if a == 0 and b == 0:
                        if c == 0 and d == 0:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[:, :]
                        elif c != 0 and d == 0:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[:, c:]
                        elif c == 0 and d != 0:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[d:, :]
                        else:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[d:, c:]

                    elif a == 0 and b != 0:
                        image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[:, :-b]
                    elif a != 0 and b == 0:
                        image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[:-a, :]
                    else:
                        image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[:-a, :-b]

                    _c_gt[0][:-1] = new_gt
                    if gts.shape[0] == 0:
                        gts = _c_gt
                    else:
                        gts = np.insert(gts, 0, values=_c_gt, axis=0)
                    
                    # delete the mixed boxes
                    if b_id in self.boxes_index:
                        self.boxes_index.remove(b_id)

                mixup_count += 1
                if mixup_count>=2:
                    break
        
        Current_image = torch.tensor(np.uint8(image)) # Image.fromarray()
        gt_boxes = gts[:, :4]
        gt_classes = gts[:, 4]

        # Current_target = BoxList(gts[:, :4], (img_shape[1], img_shape[0]))
        # Current_target.add_field("labels", torch.tensor(gts[:, 4]))
        
        return Current_image, gt_boxes, gt_classes

    def _start_boxes_mosaic(self, s_imgs=[], targets=[], w_img=0, h_img=0, num_boxes=4):
        """ Start mosaic boxes. 
            A composite image is formed by combining four box images into a single mosaic image
        """
        
        gt4 = [] # the final groundtruth space
        if len(targets)>=1:
            # print(len(s_imgs))
            id = [-1 for i in range(len(s_imgs))] # for the new image
        else:
            id = []
        
        s_imgs_size = (w_img, h_img)

        scale = int(np.mean(s_imgs_size)) # keep the same size with current image
        s_w = scale
        s_h = scale
        
        ### FOR without NEW IMAGE
        # The scaling factor mu randomly sampled from the range of [0.4, 0.6]
        yc = int(random.uniform(s_h*0.4, s_h*0.6)) # set the mosaic center position
        xc = int(random.uniform(s_w*0.4, s_w*0.6))
        
        ### FOR ONE NEW IMAGE
        # yc = int(random.uniform(s_h*0.3, s_h*0.4)) # set the mosaic center position
        # xc = int(random.uniform(s_w*0.3, s_w*0.4))
        
        ### preparing the enough box memory for mosaic ###
        if len(self.boxes_index) < self.batch_size:
            # print("A repeat for boxes memory!")
            self.boxes_index = list(range(len(self.BoxRehearsal_path)))
            
        imgs = [] 
        for i in range(num_boxes):
            # put the new images and box memory together
            img, target, b_id = self._sample_per_bbox_from_boxrehearsal(i, s_imgs_size)
            imgs.append(img)
            targets.append(target)
            id.append(b_id)
        
        #### Begin to mosaic ####
        for i, (img, target, b_id) in enumerate(zip(imgs, targets, id)):
            (w, h) = img.size
            if i%4==0: # top right
                xc_ = xc+self.bg_size
                yc_ = yc-self.bg_size
                img4 = np.full((s_h, s_w, 3), 114., dtype=np.float32)
                x1a, y1a, x2a, y2a = xc_, max(yc_-h, 0), min(xc_+w, s_w), yc_
                x1b, y1b, x2b, y2b = 0, h-(y2a - y1a), min(w, x2a - x1a), h # should corresponding to top left
            elif i%4==1: # bottom left
                xc_ = xc-self.bg_size
                yc_ = yc+self.bg_size
                x1a, y1a, x2a, y2a = max(xc_ - w, 0), yc_, xc_, min(s_h, yc_ + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc_, w), min(y2a - y1a, h)
            elif i%4==2: # bottom right
                xc_ = xc+self.bg_size
                yc_ = yc+self.bg_size
                x1a, y1a, x2a, y2a = xc_, yc_, min(xc_ + w, s_w), min(s_h, yc_+h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            elif i%4==3: # top left
                xc_ = xc-self.bg_size
                yc_ = yc-self.bg_size
                x1a, y1a, x2a, y2a = max(xc_- w, 0), max(yc_ - h, 0), xc_, yc_
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h-(y2a - y1a), w, h

            img4[y1a:y2a, x1a:x2a] = np.asarray(img)[y1b:y2b,x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            gts = []
            if not isinstance(target, np.ndarray):
                bbox_list = target.bbox.tolist()
                label_list = target.extra_fields["labels"].tolist()
                for i in range(len(bbox_list)):
                    gts.append(bbox_list[i] + [label_list[i]])
            else:
                gts = target

            gts = np.array(gts)
            if len(gts) > 0:
                gts[:, 0] = gts[:, 0] + padw
                gts[:, 1] = gts[:, 1] + padh
                gts[:, 2] = gts[:, 2] + padw
                gts[:, 3] = gts[:, 3] + padh
            gt4.append(gts)

            # delete the mosaiced boxes
            if b_id in self.boxes_index:
                self.boxes_index.remove(b_id)
                
        # Concat/clip gts
        if len(gt4):
            gt4 = np.concatenate(gt4, 0)
            np.clip(gt4[:, 0], 0, s_w, out=gt4[:, 0])
            np.clip(gt4[:, 2], 0, s_w, out=gt4[:, 2])
            np.clip(gt4[:, 1], 0, s_h, out=gt4[:, 1])
            np.clip(gt4[:, 3], 0, s_h, out=gt4[:, 3])

        # Delete too small objects (check again)
        del_index = []
        for col in range(gt4.shape[0]):
            if (gt4[col][2]-gt4[col][0]) <= 2.0 or (gt4[col][3]-gt4[col][1]) <= 2.0:
                del_index.append(col)
        gt4 = np.delete(gt4, del_index, axis=0)

        # == transfer for input == #
        Current_image = torch.tensor(np.uint8(img4))  # Image.fromarray()
        gt_boxes = gts[:, :4]
        gt_classes = gts[:, 4]
        # Current_target = BoxList(gt4[:, :4], (s_w, s_h))
        # Current_target.add_field("labels", torch.tensor(gt4[:, 4]))

        # Visualize
        # from PIL import ImageDraw
        # a = ImageDraw.ImageDraw(Current_image)
        # for g in range(gt4.shape[0]):
        #     gt_ = gt4[g, :4]
        #     a.rectangle(((gt_[0], gt_[1]), (gt_[2], gt_[3])), fill=None, outline='blue', width=2)
        # Current_image.save('output/mosaic_mixup_boxes/{}.jpg'.format(b_id))

        return Current_image, gt_boxes, gt_classes, (s_w, s_h)