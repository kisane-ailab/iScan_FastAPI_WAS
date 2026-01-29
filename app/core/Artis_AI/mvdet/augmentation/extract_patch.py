import cv2, os
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm

def pad_box(bbox, margin, width, height):
    # x1,y1,x2,y2
    x1, y1, x2, y2 = bbox
    bbox = [max(x1-margin, 0), max(y1-margin, 0), min(x2+margin, width), min(y2+margin, height)]
    return bbox

def get_contour_img(image_path, bbox, save_path):
    '''
    :param data:
    :param created_data_dir:
    :param num_to_create:
    [description]
    get a single datum and return a cropped image
    with transparent background and bounding box information
    :return: cropped transparent image, bbox
    '''
    # get a single image
    image = cv2.imread(image_path)

    bbox = pad_box(bbox, margin=10, width=image.shape[1], height=image.shape[0])
    
    # Get the bounding boxes
    image_bbox = np.array(bbox, dtype=np.int32)

    # Crop the bounding box region from each image
    cropped_image = image[image_bbox[1]:image_bbox[3], image_bbox[0]:image_bbox[2]]
    
    # rect: (x, y, w, h)
    rect = (2, 2, cropped_image.shape[1] - 5, cropped_image.shape[0] - 5)
    mask = np.zeros(cropped_image.shape[:2], np.uint8)

    cv2.grabCut(cropped_image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

    # 0: cv2.GC_BGD, 2: cv2.GC_PR_BGD
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    grabcut = cropped_image * mask2[:, :, np.newaxis]

    img0_hsv = cv2.cvtColor(grabcut, cv2.COLOR_BGR2HSV)
    lower = np.array([1, 1, 1])
    upper = np.array([255, 255, 255])
    range_mask = cv2.inRange(img0_hsv, lower, upper)
    contours, _ = cv2.findContours(range_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    grabcut_copy = np.copy(grabcut)
    initial_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = []
    boundRect = [None] * len(initial_contours)

    for idx, contour in enumerate(initial_contours):
        area = cv2.contourArea(contour)
        if area > 4800:
            boundRect[idx] = cv2.boundingRect(contour)
            contours.append(contour)

    boundRect = [x for x in boundRect if x is not None]
    x1 = y1 = x2 = y2 = None

    # get entire bbox
    for idx, rect in enumerate(boundRect):
        x, y, w, h = rect
        if idx == 0:
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            x1 = min(x1, x)
            y1 = min(y1, y)
            x2 = max(x2, x + w)
            y2 = max(y2, y + h)

    if len(boundRect) == 0:
        return None
    else:
        rect_width = x2 - x1
        rect_height = y2 - y1
        bounding_box = np.array([x1, y1, rect_width, rect_height], dtype=np.int32)

        contour_mask = cv2.fillPoly(mask, pts=contours, color=(255, 255, 255))
        contour_mask = cv2.merge((contour_mask, contour_mask, contour_mask))

        result = cv2.bitwise_and(grabcut_copy, contour_mask)

        # set background transparent
        tmp = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(result)
        rgba = [b, g, r, alpha]
        transparent_result = cv2.merge(rgba, 4)
        product_image = transparent_result[y1:y1 + rect_height, x1:x1 + rect_width]
        cv2.imwrite(save_path, product_image)
        return product_image, bounding_box



if __name__=='__main__':
    save_dir = '/SSDe/kisane_DB/kisane_DB_v0_3/single_patch'
    
    right_image_list = glob('/SSDe/kisane_DB/kisane_DB_v0_3/single_data/*/*/*/TP5/*/*_R2_*_Color.png')
    top_image_list = glob('/SSDe/kisane_DB/kisane_DB_v0_3/single_data/*/*/*/TP5/*/*_T2_*_Color.png')
    left_image_list = glob('/SSDe/kisane_DB/kisane_DB_v0_3/single_data/*/*/*/TP5/*/*_L2_*_Color.png')
    
    for right_image in tqdm(right_image_list):                                                    
        class_name, tray_num, file_name = right_image.split('/')[-6], right_image.split('/')[-5], right_image.split('/')[-1]
        save_path = os.path.join(save_dir, class_name, tray_num, 'right', file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        bbox = pd.read_csv(right_image.replace('_Color.png', '_GT.csv')).to_numpy()[0][1:5]
        get_contour_img(right_image, bbox, save_path)
        
        
    for top_image in tqdm(top_image_list):                                                    
        class_name, tray_num, file_name = top_image.split('/')[-6], top_image.split('/')[-5], top_image.split('/')[-1]
        save_path = os.path.join(save_dir, class_name, tray_num, 'top', file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        bbox = pd.read_csv(top_image.replace('_Color.png', '_GT.csv')).to_numpy()[0][1:5]
        get_contour_img(top_image, bbox, save_path)
        

    for left_image in tqdm(left_image_list):                                                    
        class_name, tray_num, file_name = left_image.split('/')[-6], left_image.split('/')[-5], left_image.split('/')[-1]
        save_path = os.path.join(save_dir, class_name, tray_num, 'left', file_name)       
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        bbox = pd.read_csv(left_image.replace('_Color.png', '_GT.csv')).to_numpy()[0][1:5]
        get_contour_img(left_image, bbox, save_path)    