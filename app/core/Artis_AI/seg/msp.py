import os

import cv2
import numpy as np

import json

color_list_20 = [
    [192, 57, 43],
    [231, 76, 60],
    [243, 156, 18],
    [155, 89, 182],
    [230, 126, 34],
    [142, 68, 173],
    [211, 84, 0],
    [41, 128, 185],
    [236, 240, 241],
    [52, 152, 219],
    [189, 195, 199],
    [26, 188, 156],
    [149, 165, 166],
    [22, 160, 133],
    [127, 140, 141],
    [39, 174, 96],
    [52, 73, 94],
    [46, 204, 113],
    [44, 62, 80]
]

color_list_15 = [
    [128, 190, 82],
    [141, 97, 31],
    [152, 60, 125],
    [43, 57, 192],
    [34, 126, 230],
    [159, 231, 249],
    [226, 189, 215],
    [177, 183, 245],
    [210, 207, 202],
    [247, 236, 244],
    [235, 242, 209],
    [51, 40, 28],
    [192, 57, 43],
    [231, 76, 60],
    [243, 156, 18]
]


'''
color_list_20 = [
    [192, 57, 43],
    [241, 196, 15],
    [231, 76, 60],
    [243, 156, 18],
    [155, 89, 182],
    [230, 126, 34],
    [142, 68, 173],
    [211, 84, 0],
    [41, 128, 185],
    [236, 240, 241],
    [52, 152, 219],
    [189, 195, 199],
    [26, 188, 156],
    [149, 165, 166],
    [22, 160, 133],
    [127, 140, 141],
    [39, 174, 96],
    [52, 73, 94],
    [46, 204, 113],
    [44, 62, 80]
]


'''


'''
    contours = get_contours_from_img(seg)
    boxes = get_boxes_coord_from_contours(contours)
    box = get_real_box_from_boxes(boxes)
'''

def get_contours_from_img(img):
    '''
    get countours from image.

    Param:
        img : (numpy)
    Returns:
        contours : (list) [num_obj, num_contours, x, y]
    '''
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(img, 127, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def get_contours_from_mask(mask):
    mask_re = mask * 255.0
    mask_re = np.clip(mask_re, 0, 255)
    mask_re = mask_re.astype(np.uint8)
    ret, thr = cv2.threshold(mask_re, 5, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

'''def get_boxes_coord_from_contours(contours):
    boxes = list()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, w, h])

    return boxes'''

def get_boxes_coord_from_contours(contours, is_numpy=False, data_format='xywh'):
    boxes = list()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if data_format=='xywh':
            boxes.append([x, y, w, h])
        else:
            boxes.append([x, y, x+w, y+h])

    if is_numpy:
        boxes = np.array(boxes)

    return boxes


def get_real_contour_from_contours(contours):
    idx = 0
    max_area = 0.
    #print(f"=====> [get_real_contour_from_contours] len(contours) = {len(contours)}")
    if len(contours) == 0:
        return None
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            idx = i
            max_area = area
    #print(f"=====> [get_real_contour_from_contours] idx = {idx}")
    return contours[idx]


def get_real_box_from_boxes(boxes):
    '''
    get real box coord from bboxes using max area
    '''
    idx = 0
    max_area = 0.
    for i, box in enumerate(boxes):
        _x, _y, w, h = box
        area = w * h
        if area > max_area:
            idx = i
            max_area = area

    return boxes[idx]


def rotate_bound(image, angle, bbox):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 회전 행렬 계산 (scale = 1.0)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # 회전 후 새 bounding box 크기 계산
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 이동 보정 (translation)
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # 최종 회전
    rotated = cv2.warpAffine(image, M, (nW, nH))

    new_bbox = None
    if bbox is not None:
        # bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)

        # (N,2) -> (N,1,2) 로 reshape 후 affine 변환
        corners = corners.reshape(-1, 1, 2)
        transformed = cv2.transform(corners, M).reshape(-1, 2)

        # 새로운 bbox 계산
        x_coords, y_coords = transformed[:, 0], transformed[:, 1]
        new_bbox = [
            int(np.min(x_coords)), int(np.min(y_coords)),
            int(np.max(x_coords)), int(np.max(y_coords))
        ]

    return rotated, new_bbox
    
def fast_rotate(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 회전 행렬 계산 (scale = 1.0)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # 회전 후 새 bounding box 크기 계산
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 이동 보정 (translation)
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # 최종 회전
    rotated = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_NEAREST)
    
    return rotated

def delete_bg(patch):
    patch_height, patch_width, _ = patch.shape

    min_x, min_y = patch_width, patch_height
    max_x, max_y = 0, 0
    for y in range(patch_height):
        for x in range(patch_width):
            if patch[y, x, 0] | patch[y, x, 1] | patch[y, x, 2]:
                if min_x > x:
                    min_x = x
                if min_y > y:
                    min_y = y
                if max_x < x:
                    max_x = x
                if max_y < y:
                    max_y = y

    #return [min_x, min_y, max_x, max_y]
    return patch[min_y:max_y, min_x:max_x], [min_x, min_y, max_x, max_y]


def get_cls_dict(json_path, data_type='annotation_json'):
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    cls_dict = dict()

    if data_type == 'annotation_json':
        for category_item in json_data['categories']:
            name = category_item['name']
            real_cls = int(name.split('_')[-1])
            cls_dict[category_item['id']] = real_cls

    elif data_type == 'db_json':
        class_list = json_data['Class']
        for class_one_dict in class_list:
            for key, value in class_one_dict.items():
                cls_dict[int(key)] = int(value)


    return cls_dict


def get_depth_image_from_bin(bin_path, img_size=(1280, 960)):
    if not os.path.exists(bin_path):
        print(f"=====> not exists bin path : {bin_path}")
        return None
   
    with open(bin_path, 'rb') as bin_depth_file:
        len_depth_header_category = np.fromfile(bin_depth_file, dtype=np.uint16, count=1)[0]
        category_depth_header = bin_depth_file.read(len_depth_header_category).decode('utf-8')
        if category_depth_header == 'Version':
            depth_header_version = np.fromfile(bin_depth_file, dtype=np.uint16, count=3)
            depth = np.fromfile(bin_depth_file, dtype=np.uint16).reshape(480, 640)

    resize_depth = cv2.resize(depth, dsize=img_size, interpolation=cv2.INTER_LINEAR)
    resize_depth = np.expand_dims(resize_depth, axis=-1)
    #print(f"=====> resize_depth = {resize_depth.shape}")

    #img = np.concatenate((img, resize_depth), axis=2)
    return resize_depth


def get_depth_image_from_bin_original_size(bin_path):
    if not os.path.exists(bin_path):
        print(f"=====> not exists bin path : {bin_path}")
        return None
   
    with open(bin_path, 'rb') as bin_depth_file:
        len_depth_header_category = np.fromfile(bin_depth_file, dtype=np.uint16, count=1)[0]
        category_depth_header = bin_depth_file.read(len_depth_header_category).decode('utf-8')
        if category_depth_header == 'Version':
            depth_header_version = np.fromfile(bin_depth_file, dtype=np.uint16, count=3)
            depth = np.fromfile(bin_depth_file, dtype=np.uint16).reshape(480, 640)
            return depth
    
    return None
    

def concat_rgb_and_depth(img_rgb, img_depth):
    return np.concatenate((img_rgb, img_depth), axis=2)


def get_contours_from_masks(masks, is_numpy=False):
    contour_list = list()
    exist_indices_list = list()

    for i, mask in enumerate(masks):
        contours = get_contours_from_mask(mask)
        contour = get_real_contour_from_contours(contours)
        if contour is None:
            continue
    
        if is_numpy:
            contour_list.append(np.array(contour))
        else:
            contour_list.append(contour)

        #contour_array = contour.squeeze(axis=1).flatten()
        #contour_list.append(contour_array)
        exist_indices_list.append(i)

    #print(f"=====> contour_list = {len(contour_list)}")
    #if is_numpy:
    #    contour_list_npy = list()
    #    for contour in contour_list:
    #        contour_list_npy.append(np.array(contour_list))
    #    #contour_list = np.array(contour_list)
    #    return contour_list_npy, exist_indices_list

    return contour_list, exist_indices_list
