from pycocotools.coco import COCO
import requests
import os
from os.path import join
from tqdm import tqdm
import json
import cv2
import numpy as np 
# import random

class coco_category_filter:
    """
    Downloads images of one category & filters jsons 
    to only keep annotations of this category
    """
    def __init__(self, json_path, imgs_dir, categ='person'):
        self.coco = COCO(json_path) # instanciate coco class
        self.json_path = json_path
        self.imgs_dir = imgs_dir
        self.categ = categ
        # self.catIds = categ
        self.dict_cat = {}
        self.images = self.get_imgs_from_json()
        
    def get_imgs_from_json(self):
        """returns image names of the desired category"""
        # instantiate COCO specifying the annotations json path
        # Specify a list of category names of interest
        imgIds = []
        catIds = []
        img_num = 0
        num = 0
        for i in self.categ:
            catId = self.coco.getCatIds(catNms=i)
            if i == "teddy bear":
                catId = [88]
            if i == "carrot":
                catId = [57]
            if i == "hot dog":
                catId = [58]
            if i == "dog":
                catId = [18]

            # Get the corresponding image ids and images using loadImgs
            imgId = self.coco.getImgIds(catIds=catId)
            imgIds = imgIds + imgId
            catIds = catIds + catId
            img_num = img_num + len(imgId)
            self.dict_cat[catId[0]] = num
            num = num + 1
        imgIds = list(set(imgIds))
        images = self.coco.loadImgs(imgIds)
        #print(f"{len(images)} images in '{self.json_path}' with '{self.categ}' instances")
        self.catIds = catIds # list
        #print(images)
        return images
    
    def save_imgs(self, root_dir):
        """saves the images of this category"""
        print("Saving the images with required categories ...")
        os.makedirs(os.path.join(self.imgs_dir, subset+year), exist_ok=True)
        # Save the images into a local folder
        for im in tqdm(self.images):
            img_data = cv2.imread(os.path.join(root_dir, subset+year, im['file_name']))
            copy_img_data = np.zeros(img_data.shape, np.uint8) 
            copy_img_data = img_data.copy()
            cv2.imwrite(os.path.join(self.imgs_dir, subset+year, im['file_name']), copy_img_data)
    
    def filter_json_by_category(self, new_json_path):
        """creates a new json with the desired category"""
        ### Filter images:
        print("Filtering the annotations ... ")
        json_parent = os.path.split(new_json_path)[0]
        os.makedirs(json_parent, exist_ok=True)
        imgs_ids = [x['id'] for x in self.images] # get img_ids of imgs with the category
        new_imgs = [x for x in self.coco.dataset['images'] if x['id'] in imgs_ids]
        catIds = self.catIds
        ### Filter annotations
        new_annots = [x for x in self.coco.dataset['annotations'] if x['category_id'] in catIds]
        ### Reorganize the ids
        new_imgs, annotations = self.modify_ids(new_imgs, new_annots)
        ### Filter categories
        new_categories = [x for x in self.coco.dataset['categories'] if x['id'] in catIds]
        for n, cat in enumerate(new_categories):
            new_categories[n]['id'] = self.dict_cat[new_categories[n]['id']]

        data = {
            "info": self.coco.dataset['info'],
            "licenses": self.coco.dataset['licenses'],
            "images": new_imgs, 
            "annotations": new_annots,
            "categories": new_categories 
            }
        
        with open(new_json_path, 'w') as f:
            json.dump(data, f)

    def modify_ids(self, images, annotations):
        """
        creates new ids for the images. I.e., reorganizes the ids and returns the dictionaries back
        images: list of images dictionaries
        imId_counter: image id starting from one (each dicto will start with id of last json +1)
        """
        print("Reinitialicing images and annotation IDs ...")
        ### Images
        old_new_imgs_ids = {}  # necessary for the annotations!
        for n,im in enumerate(images):
            old_new_imgs_ids[images[n]['id']] = n+1  # dicto with old im_ids and new im_ids
            images[n]['id'] = n+1 # reorganize the ids
        ### Annotations
        for n,ann in enumerate(annotations):
            annotations[n]['id'] = n+1
            old_image_id = annotations[n]['image_id']
            annotations[n]['image_id'] = old_new_imgs_ids[old_image_id]  # replace im_ids in the annotations as well
            annotations[n]['category_id'] = self.dict_cat[annotations[n]['category_id']]
        return images, annotations


def main(subset, year, root_dir, put_dir, category='cell phone'):
    json_file = join(root_dir,'annotations', 'instances_'+subset+year+'.json')   # local path
    # imgs_dir = join(put_dir, category + '_' + subset)
    imgs_dir = join(put_dir, 'subcat' + '_' + subset)
    new_json_file = join(put_dir, 'annotations', subset+".json")
    coco_filter = coco_category_filter(json_file, imgs_dir, categ=category) # instanciate class
    coco_filter.save_imgs(root_dir)
    coco_filter.filter_json_by_category(new_json_file)


if __name__ == '__main__':
    subset, year='train', '2017'  # val - train
    root_dir = '/data1/xxx/LEA/datasets/coco/'
    put_dir = '/data1/xxx/LEA/datasets/coco/subcat040/'
    subcat = [ "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird",
                "cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
                "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
                "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
                "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
    print('creating the training class order...')
    print('current class order: ' + str(subcat))
    subcat = subcat[0:40]
    main(subset, year, root_dir, put_dir, category=subcat)
