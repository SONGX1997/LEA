<h1 align="center">Learning Endogenous Attention for Incremental Object Detection</h1>


Official codebase for the CVPR 2025 paper: Learning Endogenous Attention for Incremental Object Detection
[(📄 Paper)](https://openaccess.thecvf.com/content/CVPR2025/papers/Song_Learning_Endogenous_Attention_for_Incremental_Object_Detection_CVPR_2025_paper.pdf). 
This repository is implemented based on Detectron2.

## Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Running Experiments](#running-experiments)
- [Citation](#citation)

## Introduction
Learning Endogenous Attention (LEA) provides an effective framework for incremental object detection, supporting continual learning on large-scale benchmarks such as COCO and VOC.
If you find this repository helpful, please consider citing our work!

## Installation
Follow these steps to set up the LEA environment:


```bash
# Create conda environment
conda create -n lea python=3.9
conda activate lea

# Install PyTorch and dependencies (CUDA 11.6)
conda install pytorch==1.12.1 torchvision==0.13.1 \
    torchaudio==0.12.1 cudatoolkit=11.6 \
    -c pytorch -c conda-forge

# Clone this repository
git clone https://github.com/SONGX1997/LEA.git
cd LEA

# Install LEA (and Detectron2 dependencies)
python -m pip install -e LEA
```

For more details, refer to the [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) Installation Guide.

## Dataset Preparation
Prepare datasets for training and evaluation:
```python
# Download COCO 2017 and split the data
python datasets/coco_deal.py
```

## Running Experiments
Train and evaluate LEA on COCO datasets. You can adjust settings in the provided script:
```bash
bash run_coco.sh
```

## Citation
If you use LEA, please cite:
```bibtex
@inproceedings{LEA_CVPR2025,
  title     = {Learning Endogenous Attention for Incremental Object Detection},
  author    = {X. Song and Y. He and J. Li and Q. Wang and Y. Gong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```
