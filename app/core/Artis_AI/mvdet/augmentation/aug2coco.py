from glob import glob
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

def default_dict():
    ann = {
            "info": {
                    "description": "Kisane Dataset",
                    "version": "1.0",
                    "contributor": "GIST AI Lab",
                    "date_created": "2023/02/21"
                    },
            "licenses": [{
                            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                            "id": 1,
                            "name": "Attribution-NonCommercial-ShareAlike License"
                        }],
            "categories": [{"supercategory": "object", "id": 1, "name": "object"}]
            }
    return ann
    
    
def extract_image_list(image_dir):
    # img_list = glob(os.path.join(image_dir, '*/*/*/*/*/*_Color.png'))
    img_list = glob(os.path.join(image_dir, '*.png'))
    return img_list

def load_bbox(ann_path):
    # bboxes = pd.read_csv(ann_path).to_numpy()[:,1:5]
    bboxes = np.load(ann_path)
    bboxes = np.concatenate([bboxes[:, [0]], bboxes[:, [1]], bboxes[:, [2]] - bboxes[:, [0]], bboxes[:, [3]] - bboxes[:, [1]]], axis=1)
    return bboxes

def wrapup(img_list, save_name, img_dir):
    ann = default_dict()
    ann['images'] = []
    ann['annotations'] = []
    
    ann_id = 0
    for img_id, img_path in enumerate(tqdm(img_list)):
        ann['images'].append({
                                "license": 1,
                                "file_name": os.path.relpath(img_path, start=img_dir),
                                "height": 720,
                                "width": 1280,
                                "id": img_id
                            })
            
        ann_path = img_path.replace('.png', '.npy')
        bboxes = load_bbox(ann_path)
        
        for index in range(bboxes.shape[0]):
            bbox = bboxes[index]
            ann['annotations'].append({
                                        "area": int(bbox[2] * bbox[3]),
                                        "iscrowd": 0,
                                        "image_id": img_id,
                                        "bbox": list(map(int, bbox.tolist())),
                                        "category_id": 1,
                                        "id": ann_id
            })
            ann_id += 1
        
    
    with open(save_name, 'w') as f:
        json.dump(ann, f)
    
    
if __name__=='__main__':
    train_img_list = extract_image_list(image_dir='/SSDe/kisane_DB/kisane_DB_v0_3/multi_augmentation')
    wrapup(train_img_list, save_name='train_aug.json', img_dir='/SSDe/kisane_DB/kisane_DB_v0_3/multi_augmentation')