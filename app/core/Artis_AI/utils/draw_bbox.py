from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm


def draw_bbox(image_path, bbox, save_path):
    img = cv2.imread(image_path)
    cv2.rectangle(img,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,255,0),2)
    cv2.imwrite(save_path, img)
    pass

if __name__=='__main__':
    single_image_list = glob('/SSDe/kisane_DB/kisane_DB_v0_3/single_data/*/*/*/*/*/*_Color.png')
    for image_path in tqdm(single_image_list):
        category_type = image_path.split('/')[-4]
        if category_type != 'DR':
            continue
            
        save_path = os.path.join('./samples/single_data', os.path.basename(image_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        bbox = pd.read_csv(image_path.replace('_Color.png', '_GT.csv')).to_numpy()[0][1:5]
        draw_bbox(image_path, bbox, save_path)
        
        
    multi_image_list = glob('/SSDe/kisane_DB/kisane_DB_v0_3/multi_data/*/*/*/*/*/*_Color.png')
    for image_path in tqdm(multi_image_list):
        save_path = os.path.join('./samples/multi_data', os.path.basename(image_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        bbox = pd.read_csv(image_path.replace('_Color.png', '_GT.csv')).to_numpy()[0][1:5]
        draw_bbox(image_path, bbox, save_path)