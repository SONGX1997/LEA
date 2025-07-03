#########LEA, two task setting#######
# COCO40+40, task 1
CUDA_VISIBLE_DEVICES="3" python tools/lazyconfig_train_net.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s1_040.py \
     train.output_dir="output/coco_s1_040"
# COCO40+40, task 2
CUDA_VISIBLE_DEVICES="3" python tools/lazyconfig_train_net_incre.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s2_4080.py \
    train.init_checkpoint=output/coco_s1_040/model_final.pth train.output_dir="output/coco_s2_4080"
# val COCO40+40
CUDA_VISIBLE_DEVICES="3" python tools/lazyconfig_train_net_incre_val.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_coco.py \
    --eval-only train.init_checkpoint=output/coco_s2_4080/model_final.pth 

#########LEA, two task setting#######
# COCO70+10, task 1
CUDA_VISIBLE_DEVICES="3" python tools/lazyconfig_train_net.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s1_070.py \
    train.output_dir="output/coco_s1_070"
# COCO70+10, task 2
CUDA_VISIBLE_DEVICES="3" python tools/lazyconfig_train_net_incre.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s2_7080.py \
    train.init_checkpoint=output/coco_s1_070/model_final.pth train.output_dir="output/coco_s2_7080"
# val COCO70+10
CUDA_VISIBLE_DEVICES="3" python tools/lazyconfig_train_net_incre_val.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_coco.py \
    --eval-only train.init_checkpoint=output/coco_s2_7080/model_final.pth 

#########LEA, multi task setting#######
# COCO40+10+10+10+10, task 2
CUDA_VISIBLE_DEVICES="0" python tools/lazyconfig_train_net_incre.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s2_4050.py \
    train.init_checkpoint=output/coco_s1_040/model_final.pth train.output_dir="output/coco_s2_4050"
# COCO40+10+10+10+10, task 3
CUDA_VISIBLE_DEVICES="0" python tools/lazyconfig_train_net_incre.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s3_5060.py \
    train.init_checkpoint=output/coco_s2_4050/model_final.pth train.output_dir="output/coco_s3_5060"
# COCO40+10+10+10+10, task 4
CUDA_VISIBLE_DEVICES="0" python tools/lazyconfig_train_net_incre.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s4_6070.py \
    train.init_checkpoint=output/coco_s3_5060/model_final.pth train.output_dir="output/coco_s4_6070"
# COCO40+10+10+10+10, task 5
CUDA_VISIBLE_DEVICES="0" python tools/lazyconfig_train_net_incre.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s5_7080.py \
    train.init_checkpoint=output/coco_s4_6070/model_final.pth train.output_dir="output/coco_s5_7080"
# val COCO40+10+10+10+10
CUDA_VISIBLE_DEVICES="0" python tools/lazyconfig_train_net_incre_val5stage.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_coco.py \
    --eval-only train.init_checkpoint=output/coco_s5_7080/model_final.pth 

#########LEA, multi task setting#######
# COCO40+20+20, task 2
CUDA_VISIBLE_DEVICES="1" python tools/lazyconfig_train_net_incre.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s2_4060.py \
    train.init_checkpoint=output/coco_s1_040/model_final.pth train.output_dir="output/coco_s2_4060"
# COCO40+20+20, task 3
CUDA_VISIBLE_DEVICES="1" python tools/lazyconfig_train_net_incre.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_s3_6080.py \
    train.init_checkpoint=output/coco_s2_4060/model_final.pth train.output_dir="output/coco_s3_6080"
# val COCO40+20+20
CUDA_VISIBLE_DEVICES="1" python tools/lazyconfig_train_net_incre_val3stage.py --config-file ./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_coco.py \
    train.init_checkpoint=output/coco_s3_6080/model_final.pth


