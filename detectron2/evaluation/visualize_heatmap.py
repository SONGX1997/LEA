import argparse
import cv2
import numpy as np
import os
import torch
import tqdm
from detectron2.data.detection_utils import read_image
import time

import detectron2.data.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    
    # 1*256*200*256 # feat的维度要求，四维
    feature_map = feature_map.detach()#.permute(0, 3, 1, 2)
 
    # 1*256*200*256->1*200*256
    # print("feature_map: ", feature_map.shape)
    heatmap = feature_map[:,0,:,:]*0
    for c in range(feature_map.shape[1]):
    # for c in range(100):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    # print("heatmap: ", heatmap.shape)
    heatmap = np.mean(heatmap, axis=0)
    # print("heatmap: ", heatmap.shape)
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
    return heatmap
 
def draw_feature_map(img_array, save_dir, predictions):
    img_path = img_array['file_name']
    # img = img_array['image']
    # height = img.shape[1]
    # width = img.shape[2]
    # img = img.permute(2, 1, 0).detach().cpu().numpy()

    # print(height, width, img.shape)
    # img = read_image(os.path.join(img_path, imgs), format="BGR")
    img = read_image(img_path, format="BGR")
    resize = T.ResizeShortestEdge(short_edge_length=1024, max_size=1024)
    trans = resize.get_transform(img)
    img = trans.apply_image(img)
    # print(img.shape)
    i=0
    for idx, featuremap in enumerate(predictions):
        # print(idx)
        if idx == 2 or idx == 5 or idx == 8 or idx == 11:
            # print("featuremap: ", featuremap.shape, img.shape[1], img.shape[0])
            # featuremap = featuremap[0][0].view(B, self.num_heads, H, W, -1)
            heatmap = featuremap_2_heatmap(featuremap[0].unsqueeze(0).view(1, 4096, 64, 64))
            # print("heatmap: ", heatmap.shape)
           
            # 200*256->512*640
            # print("heatmap0: ", heatmap.shape, img.shape)
            if img.shape[1] > img.shape[0]:
                d = (img.shape[0] / img.shape[1])*64
                # print("d1: ", d)
                heatmap = heatmap[:np.uint8(d),:]
            elif img.shape[0] > img.shape[1]:
                d = (img.shape[1] / img.shape[0])*64
                # print("d2: ", d)
                heatmap = heatmap[:,:np.uint8(d)]
            else: 
                heatmap = heatmap
            # print("heatmap: ", heatmap.shape, img.shape)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同   
            # heatmap = cv2.resize(heatmap, (height, width), interpolation=cv2.INTER_LINEAR)      
            heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
            # 512*640*3
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像       
            superimposed_img = heatmap * 0.7 + 0.3*img  # 热力图强度因子，修改参数，得到合适的热力图
            # superimposed_img = heatmap  # 热力图强度因子，修改参数，得到合适的热力图
            # print(img_path)
            cv2.imwrite(os.path.join(save_dir, img_path.split("/")[-1].split(".")[0]+'_'+str(i)+'.jpg'), superimposed_img)  # 将图像保存                    
        i=i+1