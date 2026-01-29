import torch
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from random import shuffle
from common_config import image_resolution_depth, image_resolution_rgb

import cv2

@PIPELINES.register_module()
class LoadImage:
    def __call__(self, img=''):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(img, torch.Tensor):
            results = {}
            file_name = ''
            results['filename'] = file_name
            results['ori_filename'] = file_name
            results['img'] = img
            results['img_fields'] = ['img']
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            
        else:
            results = {}
            results['filename'] = img
            results['ori_filename'] = img
            img = mmcv.imread(img)
            results['img'] = img
            results['img_fields'] = ['img']
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape

        return results

@PIPELINES.register_module()
class LoadDepth:
    def __call__(self, file_name=''):
        results = {}
        if isinstance(file_name, torch.Tensor):
            depth_file_name = ''
            results['filename'] = depth_file_name
            results['ori_filename'] = depth_file_name
            results['img'] = file_name
            results['img_fields'] = ['img']
            results['img_shape'] = file_name.shape
            results['ori_shape'] = file_name.shape

        else:
            valid_depth_info_line = 18
            results['filename'] = file_name
            results['ori_filename'] = file_name
            #img = mmcv.imread(file_name)
            ###file_name_depth = file_name.replace('jpg', 'txt')
            file_name_depth = file_name.replace('_Color.jpg', '_Depth.txt')
            depth_file = open(file_name_depth, 'r')
            lines = depth_file.readlines()

            img_list = list()
            for current_read_line_index in range(valid_depth_info_line, len(lines)):
                current_line_depth_info = lines[current_read_line_index].strip()
                img_list += current_line_depth_info.split('\t')

            img = np.array(img_list).astype(np.float32)
            img = np.reshape(img, (image_resolution_depth[0], image_resolution_depth[1], 1))

            results['img'] = img
            results['img_fields'] = ['img']
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
        return results

@PIPELINES.register_module()
class LoadImageAndDepth:
    def __call__(self, file_name=''):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        results = {}
        if isinstance(file_name, torch.Tensor):
            img_file_name = ''
            results['filename'] = img_file_name
            results['ori_filename'] = img_file_name
            ###img = mmcv.imread(file_name)
            results['img'] = file_name
            results['img_fields'] = ['img']
            results['img_shape'] = file_name.shape
            results['ori_shape'] = file_name.shape
        elif isinstance(file_name, np.ndarray):
            img_file_name = ''
            results['filename'] = img_file_name
            results['ori_filename'] = img_file_name
            results['img'] = torch.Tensor(file_name).permute(2, 0, 1)
            results['img_fields'] = ['img']
            results['img_shape'] = file_name.shape
            results['ori_shape'] = file_name.shape
        else:
            img = mmcv.imread(file_name)
            depth_filename = file_name.replace('_Color.jpg', '_Depth.bin')
            with open(depth_filename, 'rb') as bin_depth_file:
                len_depth_header_category = np.fromfile(bin_depth_file, dtype=np.uint16, count=1)[0]
                # 안전한 디코딩 시도
                try:
                    category_depth_header = bin_depth_file.read(len_depth_header_category).decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        category_depth_header = bin_depth_file.read(len_depth_header_category).decode('utf-8-sig')
                    except UnicodeDecodeError:
                        try:
                            category_depth_header = bin_depth_file.read(len_depth_header_category).decode('latin-1')
                        except UnicodeDecodeError:
                            category_depth_header = bin_depth_file.read(len_depth_header_category).decode('utf-8', errors='ignore')
                if category_depth_header == 'Version':
                    depth_header_version = np.fromfile(bin_depth_file, dtype=np.uint16, count=3)
                    depth = np.fromfile(bin_depth_file, dtype=np.uint16).reshape(image_resolution_depth[0], image_resolution_depth[1])
                    resize_depth = cv2.resize(depth, dsize=(image_resolution_rgb[1], image_resolution_rgb[0]), interpolation=cv2.INTER_LINEAR)
                    resize_depth = np.expand_dims(resize_depth, axis=-1)
                    concat_img = np.concatenate((img, resize_depth), axis=2)
                else:
                    print('[Artis_AI] ***** Wrong Depth Header *****')
                    print('[Artis_AI] Please Check Depth Bin File !!!')
                    # Depth 헤더가 잘못된 경우 RGB 이미지에 빈 depth 채널 추가
                    empty_depth = np.zeros((img.shape[0], img.shape[1], 1), dtype=img.dtype)
                    concat_img = np.concatenate((img, empty_depth), axis=2)

            results['filename'] = file_name
            results['ori_filename'] = file_name  # img_info 대신 file_name 사용
            results['img'] = concat_img
            results['img_shape'] = concat_img.shape
            results['ori_shape'] = concat_img.shape
            results['img_fields'] = ['img']

        return results

