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
    
    
def load_bbox(ann_path, label_type='csv'):
    if label_type == 'csv':
        bboxes = pd.read_csv(ann_path).to_numpy()[:, 1:5]
    else:
        bboxes = np.load(ann_path)

    out = []
    for bbox in bboxes:
        bbox = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])]
        out.append([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
    out = np.array(out)
    return out

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
        
        if 'multi_augmentation' in img_path:
            ann_path = img_path.replace('.png', '.npy')
            bboxes = load_bbox(ann_path, label_type='npy')
        else:
            ann_path = img_path.replace('_Color.png', '_GT.csv')
            bboxes = load_bbox(ann_path, label_type='csv')
            
        for index in range(bboxes.shape[0]):
            bbox = bboxes[index].tolist()
            ann['annotations'].append({
                                        "area": bbox[2] * bbox[3],
                                        "iscrowd": 0,
                                        "image_id": img_id,
                                        "bbox": bbox,
                                        "category_id": 1,
                                        "id": ann_id
            })
            ann_id += 1
    
    with open(save_name, 'w') as f:
        json.dump(ann, f)
    
    
if __name__=='__main__':
    # Select Class for Multi-Augmentation
    base_dir = '/SSDe/kisane_DB/kisane_DB_v0_3'
    
    seed = 0
    np.random.seed(seed)
    class_list = os.listdir(os.path.join(base_dir, 'multi_data'))
    train_multi_class = np.random.choice(class_list, int(len(class_list) * 0.7), replace=False).tolist()
    val_multi_class = list(set(class_list) - set(train_multi_class))
    
    # Image List
    train_list = glob(os.path.join(base_dir, 'multi_augmentation/*.png')) + glob(os.path.join(base_dir, 'single_data/*/*/*/*/*/*_Color.png'))
    for cls in train_multi_class:
        train_list += glob(os.path.join(base_dir, 'multi_data/%s/*/*/*/*/*_Color.png' %cls))
    
    val_list = []
    for cls in val_multi_class:
        val_list += glob(os.path.join(base_dir, 'multi_data/%s/*/*/*/*/*_Color.png' %cls))
        
    wrapup(train_list, save_name='train_all.json', img_dir=base_dir)
    wrapup(val_list, save_name='val_multi.json', img_dir=base_dir)