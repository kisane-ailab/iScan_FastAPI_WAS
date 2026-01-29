import cv2, os
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
from copy import copy
import random

def get_random_augmentation(base_dir, view, overlap_cri, num_target_samples, num_max_objects, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    if view == 'left':
        base_image_list = glob(os.path.join(base_dir, 'single_data', '*/*/*/*/*/*_L2_*_Color.png'))
    elif view == 'right':
        base_image_list = glob(os.path.join(base_dir, 'single_data', '*/*/*/*/*/*_R2_*_Color.png'))
    elif view == 'top':
        base_image_list = glob(os.path.join(base_dir, 'single_data', '*/*/*/*/*/*_T2_*_Color.png'))
    else:
        raise('Error!')

    created_num = 0
    patch_dir = os.path.join(base_dir, 'single_patch')
    category_list = os.listdir(patch_dir)
    while created_num < num_target_samples:
        # Load Base Image
        base_image_path = str(np.random.choice(base_image_list, 1, replace=False)[0])
        tray_type = base_image_path.split('/')[-5]

        bbox = [pd.read_csv(base_image_path.replace('_Color.png', '_GT.csv')).to_numpy()[0][1:5].tolist()]
        image = cv2.imread(base_image_path)
        
        # Add Objects
        num_objects = int(np.random.choice(list(range(num_max_objects)), 1, replace=False)[0])
        current_object = 1
        max_overlap = 0
        
        count = 0
        while current_object < num_objects:
            ori_image, ori_bbox = copy(image), copy(bbox)
            
            new_category = str(np.random.choice(category_list, 1, replace=False)[0])
            patch_list = glob(os.path.join(patch_dir, new_category, tray_type, view, '*.png'))
            if len(patch_list) == 0:
                continue
            
            patch = cv2.imread(str(np.random.choice(patch_list, 1, replace=False)[0]), cv2.IMREAD_UNCHANGED)
            image, new_bbox = add_object(image, patch)
            max_overlap = check_overlap(bbox, new_bbox)
            
            if count > 50:
                break
                        
            if max_overlap < overlap_cri:
                bbox.append(new_bbox)
                current_object += 1
            else:
                count += 1
                image = ori_image
                bbox = ori_bbox
        
        if count > 50:
            continue

        # for bbox_ix in bbox:
        #     cv2.rectangle(image,(bbox_ix[0], bbox_ix[1]),(bbox_ix[2], bbox_ix[3]),(0,255,0),2)
        # cv2.imwrite('imp.png', image), print(max_overlap)
        
        save_result(image, bbox, os.path.join(save_dir, '%s_%d.png' %(view, created_num)))
        created_num += 1
        
        if created_num % 100 == 0:
            print('Pic %d Created !!' %created_num)


def add_object(image, patch):
    image_height, image_width, _ = image.shape
    patch_height, patch_width, _ = patch.shape
    
    patch_center_x = random.randint(300, 900)
    patch_center_x = min(patch_center_x, image_width - patch_width//2)
    patch_center_y = random.randint(200, 500)
    patch_center_y = min(patch_center_y, image_height - patch_height//2)
    
    left_x = max(patch_center_x - ((patch_width // 2) + 5), 0)
    left_y = max(patch_center_y - ((patch_height // 2) + 5), 0)
    
    # Add Transparent Objects via Alpha channel
    alpha_s = patch[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    
    for c in range(0, 3):
        image[left_y:(left_y+patch_height), left_x:(left_x+patch_width), c] = (alpha_s * patch[:, :, c] +
                                        alpha_l * image[left_y:(left_y+patch_height), left_x:(left_x+patch_width), c])

    # BBOX
    new_bbox = [left_x, left_y, left_x+patch_width, left_y+patch_height]
    return image, new_bbox


def check_overlap(bboxes, new_bbox):
    overlap = []
    for bbox in bboxes:
        old_c_x, old_c_y = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        new_c_x, new_c_y = (new_bbox[0]+new_bbox[2])/2, (new_bbox[1]+new_bbox[3])/2
        
        cond1 = abs(new_c_x - old_c_x) * 2 < (abs(bbox[2] - bbox[0]) + abs(new_bbox[2] - new_bbox[0]))
        cond2 = abs(new_c_y - old_c_y) * 2 < (abs(bbox[3] - bbox[1]) + abs(new_bbox[3] - new_bbox[1]))
        if cond1 and cond2:
            overlap_region = abs(max(bbox[0], new_bbox[0]) - min(bbox[2], new_bbox[2])) * abs(max(bbox[1], new_bbox[1]) - min(bbox[3], new_bbox[3]))
            overlap.append(overlap_region / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
            overlap.append(overlap_region / ((new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])))
        else:
            overlap.append(0.)
    return max(overlap)


def save_result(image, bbox, save_path):
    cv2.imwrite(save_path, image)
    np.save(save_path.replace('.png', '.npy'), bbox)



if __name__=='__main__':
    save_dir = '/SSDe/kisane_DB/kisane_DB_v0_3/multi_augmentation'
    get_random_augmentation(base_dir='/SSDe/kisane_DB/kisane_DB_v0_3', view='right', overlap_cri=0.2, num_target_samples=50000, num_max_objects=6, save_dir=save_dir)
    get_random_augmentation(base_dir='/SSDe/kisane_DB/kisane_DB_v0_3', view='top', overlap_cri=0.2, num_target_samples=50000, num_max_objects=6, save_dir=save_dir)
    get_random_augmentation(base_dir='/SSDe/kisane_DB/kisane_DB_v0_3', view='left', overlap_cri=0.2, num_target_samples=50000, num_max_objects=6, save_dir=save_dir)