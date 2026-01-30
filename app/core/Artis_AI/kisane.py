# -*- coding:utf-8 -*-
import copy
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import mmcv
import torch
import time
from mmcv import Config
from mmdet.models import build_detector
from mmdet.utils import rfnext_init_model, build_dp, compat_cfg, replace_cfg_vals, update_data_root
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
import mvdet
import numpy as np
from utils.nms import nms
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.transforms import ToPILImage

#import matplotlib.pyplot as plt

from classification.model.ClsNet import ClsNet1

import json
import csv
from datetime import datetime

import mmdet.models.detectors.single_stage as SingleStage

from collections.abc import Mapping, Sequence
from mmcv.parallel import collate, scatter, DataContainer
from torch.utils.data.dataloader import default_collate
import importlib.util
if importlib.util.find_spec("tensorrt") is not None:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    is_using_tensorrt = True
else:
    is_using_tensorrt = False

import common_config as cc
from common_config import make_artis_ai_log, CurrentDateTime
from image_fusion import fuse_reflection_free_image

# SEG #
from seg.data import cfg as sg_cfg
from seg.data import set_cfg as sg_set_cfg
from seg.utils.re_build_model import get_class_information, re_build_all_config
from seg.yolact import Yolact
from seg.msp import *
from seg.utils.augmentations import FastBaseTransform
from seg.utils import timer
from seg.layers.output_utils import postprocess
from ultralytics import YOLO
from collections import defaultdict
from matplotlib import font_manager
from ultralytics.aug import wrapper_detect_spike, has_internal_loop, wrapping_hull_contour

class PreloadedImagesOnGPU:
    def __init__(self, ai_seg_model=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.original_image = None
        self.image = []
        self.depth = []
        self.sg_frame = None
        self.sg_batch = None
        self.flag_preload_rgbd = False
        self.ai_seg_model = ai_seg_model
        self.model_input = {}

    def preload_image(self, img_path, flag, depth_flag, model_cfg=None):
        if flag == 'RGB':
            self.original_image = mmcv.imread(img_path)
            if model_cfg.model.backbone.type == 'ResNet':
                self.image = torch.tensor(self.original_image.transpose(2, 0, 1), device=self.device)
                self.flag_preload_rgbd = False
            elif model_cfg.model.backbone.type == 'ResNet_RGBD':
                depth_path = img_path.replace('_Color.jpg', '_Depth.bin')
                with open(depth_path, 'rb') as depth_file:
                    len_depth_header_string = np.fromfile(depth_file, dtype=np.uint16, count=1)[0]
                    # 안전한 디코딩 시도
                    try:
                        current_depth_header = depth_file.read(len_depth_header_string).decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            current_depth_header = depth_file.read(len_depth_header_string).decode('utf-8-sig')
                        except UnicodeDecodeError:
                            try:
                                current_depth_header = depth_file.read(len_depth_header_string).decode('latin-1')
                            except UnicodeDecodeError:
                                current_depth_header = depth_file.read(len_depth_header_string).decode('utf-8', errors='ignore')
                    if current_depth_header == 'Version':
                        depth_header_version = np.fromfile(depth_file, dtype=np.uint16, count=3)
                        depth = np.fromfile(depth_file, dtype=np.uint16).reshape(cc.image_resolution_depth[0], cc.image_resolution_depth[1])
                        resize_depth = cv2.resize(depth, dsize=(cc.image_resolution_rgb[1], cc.image_resolution_rgb[0]), interpolation=cv2.INTER_LINEAR)
                        resize_depth = np.expand_dims(resize_depth, axis=-1)
                        concat_img = np.concatenate((self.original_image, resize_depth), axis=2)
                        if concat_img.dtype == np.uint16:
                            concat_img = concat_img.astype(np.float32)
                        self.image = torch.tensor(concat_img.transpose(2, 0, 1), device=self.device)
                        self.flag_preload_rgbd = True
                    else:
                        print(f'{CurrentDateTime(0)} [Artis_AI] ***** Wrong Depth Header *****')
                        self.flag_preload_rgbd = False
                        # Depth 헤더가 잘못된 경우 RGB 이미지에 빈 depth 채널 추가하여 4채널로 만들기
                        empty_depth = np.zeros((self.original_image.shape[0], self.original_image.shape[1], 1), dtype=self.original_image.dtype)
                        concat_img = np.concatenate((self.original_image, empty_depth), axis=2)
                        self.image = torch.tensor(concat_img.transpose(2, 0, 1), device=self.device)
        elif flag == 'SG':
            self.original_image = cv2.imread(img_path)
            depth_path = img_path.replace('_Color.jpg', '_Depth.bin')

            if model_cfg is None:
                self.sg_frame = None
                self.sg_batch = None
            elif self.ai_seg_model == 'yolac':
                img_depth = get_depth_image_from_bin(img_path.replace('_Color.jpg', '_Depth.bin'), img_size=(1280, 960))
                img_depth = img_depth.astype(np.float32)
                # if depth_flag == 'depth':  # only depth
                if model_cfg["TYPE"] == 'depth':  # only depth
                    img_input = img_depth.copy()
                    # img_input = img_depth.astype(np.float32)
                # elif depth_flag == 'rgbd':  # rgbd
                elif model_cfg["TYPE"] == 'rgbd':  # rgbd
                    img_input = concat_rgb_and_depth(self.original_image.copy().astype(np.float32), img_depth)
                else:
                    img_input = self.original_image.copy()
                MEANS = model_cfg["MEANS"]
                STD = model_cfg["STD"]

                self.sg_frame = torch.from_numpy(img_input).cuda().float()
                self.sg_batch = FastBaseTransform(MEANS, STD)(self.sg_frame.unsqueeze(0))
            # self.sg_depth_img = Image.open(img_path.replace("_Color", "_Depth")).convert("RGBA")
            else:
                img_depth = get_depth_image_from_bin(img_path.replace('_Color.jpg', '_Depth.bin'), img_size=(1280, 960))
                img_depth = img_depth.astype(np.float32)
                for mode in ["OD", "OC"]:
                    if model_cfg[mode]["TYPE"] == 'rgbd':
                        img_input = concat_rgb_and_depth(self.original_image.copy().astype(np.float32), img_depth)
                    elif model_cfg[mode]["TYPE"] == 'depth':
                        img_input = img_depth.copy()
                    else:
                        img_input = self.original_image.copy()
                    self.model_input[mode] = img_input.copy()
            self.sg_depth_img = cv2.imread(img_path.replace("_Color", "_Depth"))
        else:
            if self.flag_preload_rgbd:
                self.depth = self.image
            else:
                self.original_image = mmcv.imread(img_path) if self.original_image is None else self.original_image
                depth_path = img_path.replace('_Color.jpg', '_Depth.bin')
                with open(depth_path, 'rb') as depth_file:
                    len_depth_header_string = np.fromfile(depth_file, dtype=np.uint16, count=1)[0]
                    # 안전한 디코딩 시도
                    try:
                        current_depth_header = depth_file.read(len_depth_header_string).decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            current_depth_header = depth_file.read(len_depth_header_string).decode('utf-8-sig')
                        except UnicodeDecodeError:
                            try:
                                current_depth_header = depth_file.read(len_depth_header_string).decode('latin-1')
                            except UnicodeDecodeError:
                                current_depth_header = depth_file.read(len_depth_header_string).decode('utf-8', errors='ignore')
                    if current_depth_header == 'Version':
                        depth_header_version = np.fromfile(depth_file, dtype=np.uint16, count=3)
                        depth = np.fromfile(depth_file, dtype=np.uint16).reshape(cc.image_resolution_depth[0], cc.image_resolution_depth[1])
                        if depth_flag == 'rgbd':
                            resize_depth = cv2.resize(depth,dsize=(cc.image_resolution_rgb[1], cc.image_resolution_rgb[0]), interpolation=cv2.INTER_LINEAR)
                            resize_depth = np.expand_dims(resize_depth, axis=-1)
                            concat_img = np.concatenate((self.original_image, resize_depth), axis=2)
                            if concat_img.dtype == np.uint16:
                                concat_img = concat_img.astype(np.float32)
                            self.depth = torch.tensor(concat_img.transpose(2, 0, 1), device=self.device)
                        elif depth_flag == 'depth':
                            depth = np.expand_dims(depth, axis=-1)
                            depth = np.array(depth).astype(np.float32)
                            self.depth = torch.tensor(depth.transpose(2, 0, 1), device=self.device)
                    else:
                        print(f'{CurrentDateTime(0)} [Artis_AI] ***** Wrong Depth Header *****')
                        # Depth 헤더가 잘못된 경우 RGB 이미지만 사용
                        if depth_flag == 'rgbd':
                            # RGB 이미지에 빈 depth 채널 추가하여 4채널로 만들기
                            empty_depth = np.zeros((self.original_image.shape[0], self.original_image.shape[1], 1), dtype=self.original_image.dtype)
                            concat_img = np.concatenate((self.original_image, empty_depth), axis=2)
                            self.depth = torch.tensor(concat_img.transpose(2, 0, 1), device=self.device)
                        elif depth_flag == 'depth':
                            # Depth 모드인데 헤더가 잘못된 경우 빈 depth 생성
                            empty_depth = np.zeros((cc.image_resolution_depth[0], cc.image_resolution_depth[1], 1), dtype=np.float32)
                            self.depth = torch.tensor(empty_depth.transpose(2, 0, 1), device=self.device)

class Kisane:
    def __init__(self, config_file_path):
        self.class_summary = {} # Depth 클래스 정보
        self.timer_starter = torch.cuda.Event(enable_timing=True)
        self.timer_ender = torch.cuda.Event(enable_timing=True)
        self.timer_starter_det_0 = torch.cuda.Event(enable_timing=True)
        self.timer_ender_det_0 = torch.cuda.Event(enable_timing=True)
        self.timer_starter_det_1 = torch.cuda.Event(enable_timing=True)
        self.timer_ender_det_1 = torch.cuda.Event(enable_timing=True)
        self.depth_instance = None  # Depth 인스턴스 참조

        print(f"{CurrentDateTime(0)} [Artis_AI][init] {datetime.now()}")
        self.init_params(config_file_path)
        self.load_models(config_file_path)

    def set_depth_instance(self, depth_instance):
        self.depth_instance = depth_instance
    
    def init_params(self, config_file_path):
        self.gpu_id = 0
        self.device = "cuda:%d" % self.gpu_id
        self.set_seed(0)
        
        # RGB-D Model
        self.flag_use_tensorrt_det_0 = False
        # RGB Model
        self.flag_use_tensorrt_det_1 = False
        self.flag_use_tensorrt_seg = False
        self.config_file_path = config_file_path
        self.send_result_format = 0
        self.ai_combine_mode = cc.artis_ai_model_mode["mode_rgb_with_depth"]
        self.depth_type = 'rgb'
        self.det_cfg0 = None        ##### Model Config For Auto GT Detection #####
        self.det_cfg1 = None        ##### Model Config For Obj Detection & Classification #####

        self.flag_ignore_depth = False

        cc.artis_ai_json_config = cc.get_config(self.config_file_path, cc.artis_ai_json_config)

        self.area_del_cnt = 0

        if "ai_model_mode" in cc.artis_ai_json_config:
            self.ai_combine_mode = int(cc.artis_ai_json_config["ai_model_mode"])
            if (cc.artis_ai_model_mode["mode_rgb_only"] > self.ai_combine_mode
                or cc.artis_ai_model_mode["mode_seg_with_feature_matching"] < self.ai_combine_mode):
                self.ai_combine_mode = cc.artis_ai_model_mode["mode_rgb_with_depth"]
        else:
            self.ai_combine_mode = cc.artis_ai_model_mode["mode_rgb_with_depth"]
        print(f"{CurrentDateTime(0)} [Artis_AI] AI Model Mode : {cc.artis_ai_model_mode_detail[self.ai_combine_mode]}")

        if "ai_valid_area" in cc.artis_ai_json_config:
            self.l_thr = int(cc.artis_ai_json_config["ai_valid_area"]["left_x"])
            self.t_thr = int(cc.artis_ai_json_config["ai_valid_area"]["left_y"])
            self.r_thr = int(cc.artis_ai_json_config["ai_valid_area"]["right_x"])
            self.b_thr = int(cc.artis_ai_json_config["ai_valid_area"]["right_y"])
        else:
            self.l_thr = 150
            self.t_thr = 50
            self.r_thr = 1180
            self.b_thr = 910

        self.ai_sg_model = cc.artis_ai_json_config.get("ai_sg_model", "yolov8")
        self.input_channels = {
            "OD": int(cc.artis_ai_json_config.get("ai_od_channel", 4)),
            "OC": int(cc.artis_ai_json_config.get("ai_oc_channel", 4))
        }
        print(f"{CurrentDateTime(0)} [Artis_AI] AI SG Input Channels : {self.input_channels}")

        self.overlap_thr = float(cc.artis_ai_json_config.get("overlap_thr", 0.75))

        self.flag_ignore_depth = bool(cc.artis_ai_json_config.get("flag_ignore_depth", False))
        
        #If Use Feature Matching, It is not necessary to set tensor rt for rgb model.
        #Cause, To classify the objects, It is not uses rgb model. It will be process by feature matching model.
        if self.ai_combine_mode < cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
            if "ai_tensorrt_rgb" in cc.artis_ai_json_config:
                self.flag_use_tensorrt_det_1 = bool(cc.artis_ai_json_config["ai_tensorrt_rgb"])
                if not os.path.exists(cc.artis_ai_det_1_trt_config_path) or not is_using_tensorrt:
                    self.flag_use_tensorrt_det_1 = False

        if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_depth"]:
            if "ai_tensorrt_depth" in cc.artis_ai_json_config:
                self.flag_use_tensorrt_det_0 = bool(cc.artis_ai_json_config["ai_tensorrt_depth"])
                if not os.path.exists(cc.artis_ai_det_0_trt_config_path) or not is_using_tensorrt:
                    self.flag_use_tensorrt_det_0 = False
        elif (self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_seg"]
              or self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_feature_matching"]
              or self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]):
            if "ai_tensorrt_sg" in cc.artis_ai_json_config:
                #self.flag_use_tensorrt_sg = bool(artis_ai_json_config["ai_tensorrt_sg"])
                #if not os.path.exists(cc.artis_ai_seg_trt_config_path):
                #    self.flag_use_tensorrt_sg = False
                self.flag_use_tensorrt_sg = False
            print(f"{CurrentDateTime(0)} [Artis_AI] AI SG Model Name : {self.ai_sg_model}")
        elif self.ai_combine_mode == cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
            if "ai_tensorrt_depth" in cc.artis_ai_json_config:
                self.flag_use_tensorrt_det_0 = bool(cc.artis_ai_json_config["ai_tensorrt_depth"])
                if not os.path.exists(cc.artis_ai_det_0_trt_config_path) or not is_using_tensorrt:
                    self.flag_use_tensorrt_det_0 = False

        self.ai_thresh_mode_oc = cc.artis_ai_json_config.get("ai_thresh_mode_rgb", False) or cc.artis_ai_json_config.get("ai_thresh_mode_oc", False)
        self.ai_thresh_sub_value = 0.2
        if "ai_thresh_sub_value" in cc.artis_ai_json_config and 0 <= cc.artis_ai_json_config["ai_thresh_sub_value"] < 1:
            self.ai_thresh_sub_value = cc.artis_ai_json_config["ai_thresh_sub_value"]
        self.ai_thresh_del_cnt = 0

        self.sg_except_class = []
        self.sg_error_class = []

    def load_models(self, config_file_path):
        print(f'{CurrentDateTime(0)} [Artis_AI][load_models] kisan_config.json path')
        print(f'{CurrentDateTime(0)} {config_file_path}')
        
        self.send_result_format = bool(cc.artis_ai_json_config["ai_send_result_format"])

        print(f"{CurrentDateTime(0)} [Artis_AI][load_models] Check the model and config files")
        if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_depth"]:
            print(f'{CurrentDateTime(0)} [Artis_AI][load_models] AI Combine Mode : RGB/D OD + RGB/D OC')

            if self.flag_use_tensorrt_det_0:
                det0_config_path = cc.artis_ai_det_0_trt_config_path
            else:
                det0_config_path = cc.artis_ai_det_0_config_path
            if self.flag_use_tensorrt_det_1:
                det1_config_path = cc.artis_ai_det_1_trt_config_path
            else:
                det1_config_path = cc.artis_ai_det_1_config_path
            det0_model_path = cc.artis_ai_det_0_model_path
            det1_model_path = cc.artis_ai_det_1_model_path
            #cls0_config_path = cc.artis_ai_cls_0_config_path
            #cls0_model_path = cc.artis_ai_cls_0_model_path
            det1_lookup_path = cc.artis_ai_det_1_lookup_path
            det0_lookup_path = cc.artis_ai_det_0_lookup_path

            print(f"{CurrentDateTime(0)} {det0_config_path} exists : {os.path.exists(det0_config_path)}")
            print(f"{CurrentDateTime(0)} {det0_model_path} exists : {os.path.exists(det0_model_path)}")
            print(f"{CurrentDateTime(0)} {det0_lookup_path} exists : {os.path.exists(det0_lookup_path)}")
            #print(f"{CurrentDateTime(0)} {cls0_config_path} exists : {os.path.exists(cls0_config_path)}")
            #print(f"{CurrentDateTime(0)} {cls0_model_path} exists : {os.path.exists(cls0_model_path)}")
            print(f"{CurrentDateTime(0)} {det1_config_path} exists : {os.path.exists(det1_config_path)}")
            print(f"{CurrentDateTime(0)} {det1_model_path} exists : {os.path.exists(det1_model_path)}")
            print(f"{CurrentDateTime(0)} {det1_lookup_path} exists : {os.path.exists(det1_lookup_path)}")
            ### ***** For Resnet Classification ***** ###
            '''
            if not os.path.exists(det0_config_path) or not os.path.exists(det0_model_path) \
                or not os.path.exists(cls0_config_path) or not os.path.exists(cls0_model_path):
                return False
            '''
            if (not os.path.exists(det0_config_path) or not os.path.exists(det0_model_path)
                    or not os.path.exists(det1_model_path) or not os.path.exists(det1_config_path)
                    or not os.path.exists(det0_lookup_path) or not os.path.exists(det1_lookup_path)):
                print(f'{CurrentDateTime(0)} [Artis_AI][load_models] AI Combine Mode Should Have 2 Detection Models [RGB / Depth]')
                return False
        elif (self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_seg"]
              or self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]):
            print(f'{CurrentDateTime(0)} [Artis_AI][load_models] AI Combine Mode : SG + RGB/D')
            # TODO sg tensorrt 관련 추가
            det0_lookup_path = cc.artis_sg_lookup_path
            print(f"{CurrentDateTime(0)} {det0_lookup_path} exists : {os.path.exists(det0_lookup_path)}")

            if self.ai_sg_model == "yolac":
                det0_config_path = cc.artis_ai_sg_config_path
                print(f"{CurrentDateTime(0)} {det0_config_path} exists : {os.path.exists(det0_config_path)}")

                if not os.path.exists(det0_lookup_path) or not os.path.exists(det0_config_path):
                    print(f'{CurrentDateTime(0)} [Artis_AI][load_models] This AI Mode Should Have Det Model & Feature DB/Model [SEG / Feat DB & Model]')
                    return False

            det0_model_path = cc.artis_ai_sg_model_path[self.ai_sg_model]
            print(f"{CurrentDateTime(0)} {det0_model_path} exists : {os.path.exists(det0_model_path)}")

            if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_seg"]:
                if self.flag_use_tensorrt_det_1:
                    det1_config_path = cc.artis_ai_det_1_trt_config_path
                else:
                    det1_config_path = cc.artis_ai_det_1_config_path
                det1_model_path = cc.artis_ai_det_1_model_path
                det1_lookup_path = cc.artis_ai_det_1_lookup_path

                print(f"{CurrentDateTime(0)} {det1_config_path} exists : {os.path.exists(det1_config_path)}")
                print(f"{CurrentDateTime(0)} {det1_lookup_path} exists : {os.path.exists(det1_lookup_path)}")
                if not os.path.exists(det1_lookup_path) or not os.path.exists(det1_config_path):
                    print(
                        f'{CurrentDateTime(0)} [Artis_AI][load_models] AI Combine Mode Should Have 2 Detection Models [RGB / SG]')
                    return False
            else:
                det1_model_path = cc.artis_ai_sg_oc_model_path
                det1_lookup_path = cc.artis_sg_oc_lookup_path

                print(f"{CurrentDateTime(0)} {det1_lookup_path} exists : {os.path.exists(det1_lookup_path)}")
            print(f"{CurrentDateTime(0)} {det1_model_path} exists : {os.path.exists(det1_model_path)}")
            if not os.path.exists(det0_model_path) or not os.path.exists(det1_model_path):
                print(f'{CurrentDateTime(0)} [Artis_AI][load_models] AI Combine Mode Should Have 2 Detection Models [RGB / SG]')
                return False
        elif self.ai_combine_mode == cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
            print(f"{CurrentDateTime(0)} [Artis_AI][load_models] AI Combine Mode : RGBD + Feature Matching")
            if self.flag_use_tensorrt_det_0:
                det0_config_path = cc.artis_ai_det_0_trt_config_path
            else:
                det0_config_path = cc.artis_ai_det_0_config_path
            det0_model_path = cc.artis_ai_det_0_model_path
            feature_matching_db_path = cc.artis_ai_feature_db_path
            feature_matching_model_path = cc.artis_ai_feature_extract_model
            feature_matching_lookup_path = cc.artis_ai_feature_lookup_path
            det0_lookup_path = cc.artis_ai_det_0_lookup_path

            print(f"{CurrentDateTime(0)} {det0_config_path} exists : {os.path.exists(det0_config_path)}")
            print(f"{CurrentDateTime(0)} {det0_model_path} exists : {os.path.exists(det0_model_path)}")
            print(f"{CurrentDateTime(0)} {det0_lookup_path} exists : {os.path.exists(det0_lookup_path)}")
            print(f"{CurrentDateTime(0)} {feature_matching_db_path} exists : {os.path.exists(feature_matching_db_path)}")
            print(f"{CurrentDateTime(0)} {feature_matching_model_path} exists : {os.path.exists(feature_matching_model_path)}")
            print(f"{CurrentDateTime(0)} {feature_matching_lookup_path} exists : {os.path.exists(feature_matching_lookup_path)}")

            if (not os.path.exists(det0_config_path) or not os.path.exists(det0_model_path) or not os.path.exists(det0_lookup_path)
                    or not os.path.exists(feature_matching_db_path) or not os.path.exists(feature_matching_model_path) or not os.path.exists(feature_matching_lookup_path)):
                print(f'{CurrentDateTime(0)} [Artis_AI][load_models] This AI Mode Should Have Det Model & Feature DB/Model [RGBD / Feat DB & Model]')
                return False
        elif self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_feature_matching"]:
            print(f"{CurrentDateTime(0)} [Artis_AI][load_models] AI Combine Mode : SEG + Feature Matching")
            # TODO sg tensorrt 관련 추가

            det0_lookup_path = cc.artis_sg_lookup_path
            print(f"{CurrentDateTime(0)} {det0_lookup_path} exists : {os.path.exists(det0_lookup_path)}")

            if self.ai_sg_model == "yolac":
                det0_config_path = cc.artis_ai_sg_config_path
                print(f"{CurrentDateTime(0)} {det0_config_path} exists : {os.path.exists(det0_config_path)}")
                if not os.path.exists(det0_lookup_path) or not os.path.exists(det0_config_path):
                    print(f'{CurrentDateTime(0)} [Artis_AI][load_models] This AI Mode Should Have Det Model & Feature DB/Model [SEG / Feat DB & Model]')
                    return False

            det0_model_path = cc.artis_ai_sg_model_path[self.ai_sg_model]
            feature_matching_db_path = cc.artis_ai_feature_db_path
            feature_matching_model_path = cc.artis_ai_feature_extract_model
            feature_matching_lookup_path = cc.artis_ai_feature_lookup_path
            
            print(f"{CurrentDateTime(0)} {det0_model_path} exists : {os.path.exists(det0_model_path)}")
            print(f"{CurrentDateTime(0)} {feature_matching_db_path} exists : {os.path.exists(feature_matching_db_path)}")
            print(f"{CurrentDateTime(0)} {feature_matching_model_path} exists : {os.path.exists(feature_matching_model_path)}")
            print(f"{CurrentDateTime(0)} {feature_matching_lookup_path} exists : {os.path.exists(feature_matching_lookup_path)}")

            if (not os.path.exists(det0_model_path)
                or not os.path.exists(feature_matching_db_path) or not os.path.exists(feature_matching_model_path) or not os.path.exists(feature_matching_lookup_path)):
                print(f'{CurrentDateTime(0)} [Artis_AI][load_models] This AI Mode Should Have Det Model & Feature DB/Model [SEG / Feat DB & Model]')
                return False
        else:
            det0_config_path = None
            det0_model_path = None
            cls0_config_path = None
            cls0_model_path = None

            if self.flag_use_tensorrt_det_1:
                det1_config_path = cc.artis_ai_det_1_trt_config_path
            else:
                det1_config_path = cc.artis_ai_det_1_config_path
            det1_model_path = cc.artis_ai_det_1_model_path
            det1_lookup_path = cc.artis_ai_det_1_lookup_path

            print(f"{CurrentDateTime(0)} {det1_config_path} exists : {os.path.exists(det1_config_path)}")
            print(f"{CurrentDateTime(0)} {det1_model_path} exists : {os.path.exists(det1_model_path)}")
            print(f"{CurrentDateTime(0)} {det1_lookup_path} exists : {os.path.exists(det1_lookup_path)}")
            if not os.path.exists(det1_config_path) or not os.path.exists(det1_model_path) \
                or not os.path.exists(det1_lookup_path):
                return False

        if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_depth"]:
            ##### Load Depth Detection Model #####
            self.det_cfg0 = Config.fromfile(det0_config_path)
            self.det_cfg0 = replace_cfg_vals(self.det_cfg0)
            update_data_root(self.det_cfg0)
            self.det_cfg0 = compat_cfg(self.det_cfg0)

            if self.flag_use_tensorrt_det_0:
                if self.det_cfg0.model.type == 'YOLOF_Depth_TRT' or self.det_cfg0.model.type == 'YOLOF_RGBD_TRT':
                    cfg_path = os.path.dirname(det0_config_path)
                    for each_file in os.listdir(cfg_path):
                        name, ext = os.path.splitext(each_file)
                        if ext == '.trt' and 'yolof' in name and 'depth' in name:
                            if 'fp16' in name:
                                SingleStage.TRT_PATH_DEPTH = os.path.join(cfg_path, each_file)
                                print(f'{CurrentDateTime(0)} [Artis_AI] ***** 16 bit Tensor RT Depth Model Exist *****')
                                break
                            elif 'int8' in name:
                                SingleStage.TRT_PATH_DEPTH = os.path.join(cfg_path, each_file)
                                print(f'{CurrentDateTime(0)} [Artis_AI] ***** 8 bit Tensor RT Depth Model Exist *****')
                                break
                            else:
                                self.flag_use_tensorrt_det_0 = False

            ###self.pipeline0 = self.load_pipeline(det_cfg0)
            self.pipeline0 = self.load_pipeline(self.det_cfg0)
            self.det0_model = self.load_det_model(self.det_cfg0, device_id=self.gpu_id, checkpoint_path=det0_model_path)
            if self.flag_use_tensorrt_det_0:
                self.det0_model.module.half()

            ### ***** For Resnet Classification ***** ###
            '''
            self.cls_lookup = {}
            with open(cls0_config_path) as json_file:
                cls_config = json.load(json_file)
                cls_list = cls_config["Class"]
                for i in range(len(cls_list)):
                    if str(i) in cls_list[i]:
                        self.cls_lookup[i] = cls_list[i][str(i)]
            print(f"[Artis_AI] ResNet {len(self.cls_lookup)} 클래스 모델")
            print(self.cls_lookup)

            self.cls0_model = ClsNet1(net="resnet34", num_classes=len(self.cls_lookup))
            self.cls0_model.load_state_dict((torch.load(cls0_model_path, map_location="cpu")["state_dict"]))
            self.cls0_model.to(self.device)
            self.cls0_model.eval()
            self.cls0_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
            '''
        elif (self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_seg"]
              or self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_feature_matching"]
              or self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]):
            # TODO Tensor RT 모델 추가 : self.flag_use_tensorrt_sg
            self.sg_info = None
            self.sg_except_class = [9999990, 9999991]
            if self.ai_sg_model == "yolac":
                self.det0_lookup = get_cls_dict(det0_lookup_path, data_type="db_json")

                ##### Load SG Model #####
                sg_set_cfg(cc.artis_ai_sg_model_config)
                # MSP : re-load label_map
                new_label_map, new_classes = get_class_information(det0_lookup_path)
                re_build_all_config(sg_cfg, new_label_map, new_classes)
                
                torch.backends.cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                #torch.set_default_tensor_type('torch.FloatTensor')

                import ast
                self.sg_info = {
                    "MEANS" : (103.94, 116.78, 123.68, 450.0),
                    "STD": (57.38, 57.12, 58.40, 1.0),
                    "TYPE": "rgbd",
                    "CHANNEL": 4
                }
                with open(det0_config_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=det0_config_path)
                    for node in tree.body:
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if target.id == "MEANS":
                                    self.sg_info["MEANS"] = ast.literal_eval(node.value)
                                if target.id == "STD":
                                    self.sg_info["STD"] = ast.literal_eval(node.value)
                            self.sg_info["CHANNEL"] = len(self.sg_info["MEANS"])
                            if self.sg_info["CHANNEL"] == 1:
                                self.sg_info["TYPE"] = "depth"
                            elif self.sg_info["CHANNEL"] == 3:
                                self.sg_info["TYPE"] = "rgb"
                            else:
                                self.sg_info["TYPE"] = "rgbd"                        

                sg_cfg.backbone.in_channels = self.sg_info["CHANNEL"]
                print(f'{CurrentDateTime(0)} [Artis_AI][load_models] SG CFG={self.sg_info}')
            
                self.det0_model = Yolact()
                self.det0_model.load_weights(det0_model_path)
                self.det0_model.eval()

                self.det0_model = self.det0_model.cuda()

                self.det0_model.detect.use_fast_nms = True
                self.det0_model.detect.use_cross_class_nms = False
                sg_cfg.mask_proto_debug = False
            else:
                self.sg_info = {
                    "OD": {
                        "MEANS": (103.94, 116.78, 123.68, 450.0),
                        "STD": (57.38, 57.12, 58.40, 1.0),
                        "TYPE": "rgbd",
                        "CHANNEL": 4
                    },
                    "OC": {
                        "MEANS": (103.94, 116.78, 123.68, 450.0),
                        "STD": (57.38, 57.12, 58.40, 1.0),
                        "TYPE": "rgbd",
                        "CHANNEL": 4
                    }
                }
                print(f"{CurrentDateTime(0)}[Artis_AI][segmentation][debug] Before Call Class YOLO self.input_channels['OD'] : {self.input_channels['OD']}")
                self.det0_model = YOLO(det0_model_path, channels=self.input_channels["OD"])
                #self.det0_model.names.update({0: 0, 1: 9999993, 2: 9999994, 3: 9999995, 4: 9999996, 5: 9999997, 6: 9999998})
                #self.det0_model.model.names.update({0: 0, 1: 9999993, 2: 9999994, 3: 9999995, 4: 9999996, 5: 9999997, 6: 9999998})
                #self.det0_model.save(det0_model_path.replace(".pt", "_update.pt"))

                ##### Check Input Channel by Model's Structure #####
                self.sg_info["OD"]["CHANNEL"] = self.det0_model.model.model[0].conv.in_channels
                print(f"{CurrentDateTime(0)}[Artis_AI][segmentation][debug] After Call Class YOLO self.input_channels['OD'] : { self.sg_info['OD']['CHANNEL']}")
                if self.sg_info["OD"]["CHANNEL"] != self.input_channels["OD"]:
                    self.det0_model = YOLO(det0_model_path, channels=self.sg_info["OD"]["CHANNEL"])
                    print(f"{CurrentDateTime(0)}[Artis_AI][segmentation][debug] Config Channels ({self.input_channels['OD']})"
                          f" and Model Channels ({self.sg_info['OD']['CHANNEL']}) are different."
                          f" Model will operate {self.sg_info['OD']['CHANNEL']} Ch Network.")
                    self.input_channels["OD"] = self.sg_info['OD']['CHANNEL']

                if self.sg_info["OD"]["CHANNEL"] == 1:
                    self.sg_info["OD"]["TYPE"] = "depth"
                elif self.sg_info["OD"]["CHANNEL"] == 3:
                    self.sg_info["OD"]["TYPE"] = "rgb"
                else:
                    self.sg_info["OD"]["TYPE"] = "rgbd"

                self.det0_lookup = {}
                if os.path.exists(det0_lookup_path):
                    with open(det0_lookup_path) as json_file:
                        det_config = json.load(json_file)
                        cls_list = det_config["Class"]
                        for cls_item in cls_list:
                            for idx, cls in cls_item.items():
                                self.det0_lookup[int(idx)] = int(cls)
                else:
                    classes_ids = list(self.det0_model.names.keys())
                    print(classes_ids)
                    for i in classes_ids:
                        self.det0_lookup[int(i)] = int(self.det0_model.names[i])
            print(f"{CurrentDateTime(0)} [Artis_AI] SG OD {len(self.det0_lookup)} 클래스 모델")
            print(f"{CurrentDateTime(0)} {self.det0_lookup}")

            if self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]:
                self.det1_model = YOLO(det1_model_path, channels=self.input_channels["OC"])
                ##### Check OC Input Channel by Model's Structure #####
                self.sg_info["OC"]["CHANNEL"] = self.det1_model.model.model[0].conv.in_channels
                print(
                    f"{CurrentDateTime(0)}[Artis_AI][segmentation][debug] After Call Class YOLO self.input_channels['OC'] : {self.sg_info['OC']['CHANNEL']}")
                if self.sg_info["OC"]["CHANNEL"] != self.input_channels["OC"]:
                    self.det1_model = YOLO(det1_model_path, channels=self.sg_info["OC"]["CHANNEL"])
                    print(
                        f"{CurrentDateTime(0)}[Artis_AI][segmentation][debug] Config Channels ({self.input_channels['OC']})"
                        f" and Model Channels ({self.sg_info['OC']['CHANNEL']}) are different."
                        f" Model will operate {self.sg_info['OC']['CHANNEL']} Ch Network.")
                    self.input_channels["OC"] = self.sg_info['OC']['CHANNEL']

                if self.sg_info["OC"]["CHANNEL"] == 1:
                    self.sg_info["OC"]["TYPE"] = "depth"
                elif self.sg_info["OC"]["CHANNEL"] == 3:
                    self.sg_info["OC"]["TYPE"] = "rgb"
                else:
                    self.sg_info["OC"]["TYPE"] = "rgbd"

                self.det1_lookup_path = det1_lookup_path
                if os.path.exists(self.det1_lookup_path):
                    self.det1_lookup = {}
                    self.det1_threshold = {}
                    with open(self.det1_lookup_path) as json_file:
                        det_config = json.load(json_file)
                        cls_list = det_config["Class"]
                        th_list = det_config["stats"]["class"] if "stats" in det_config and "class" in det_config[
                            "stats"] else {}
                        for cls_item in cls_list:
                            for idx, cls in cls_item.items():
                                self.det1_lookup[int(idx)] = int(cls)
                                if self.ai_thresh_mode_oc:
                                    self.det1_threshold[int(idx)] = max(th_list[str(idx)]["smin"] - self.ai_thresh_sub_value,
                                                                 0.4) if str(idx) in th_list else 0.4
                                else:
                                    self.det1_threshold[int(idx)] = 0.4
                        for err_cls in det_config.get("error", []):
                            err_cls = int(err_cls)
                            self.sg_error_class.append(err_cls)
                else:
                    classes_ids = list(self.det1_model.names.keys())
                    print(classes_ids)
                    self.det1_lookup = {}
                    self.det1_threshold = {}
                    for idx in classes_ids:
                        self.det1_lookup[int(idx)] = int(self.det1_model.names[idx])
                        self.det1_threshold[int(idx)] = 0.4
                print(f"{CurrentDateTime(0)} [Artis_AI] SG OC {len(self.det1_lookup)} 클래스 모델")
                print(f"{CurrentDateTime(0)} {self.det1_lookup}")
                print(f"{CurrentDateTime(0)} {self.det1_threshold}")
                print(f"{CurrentDateTime(0)} [Artis_AI] SG OC Error Class : {self.sg_error_class}")

        elif self.ai_combine_mode == cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
            ##### Load Depth Detection Model #####
            self.det_cfg0 = Config.fromfile(det0_config_path)
            self.det_cfg0 = replace_cfg_vals(self.det_cfg0)
            update_data_root(self.det_cfg0)
            self.det_cfg0 = compat_cfg(self.det_cfg0)

            if self.flag_use_tensorrt_det_0:
                if self.det_cfg0.model.type == 'YOLOF_Depth_TRT' or self.det_cfg0.model.type == 'YOLOF_RGBD_TRT':
                    cfg_path = os.path.dirname(det0_config_path)
                    for each_file in os.listdir(cfg_path):
                        name, ext = os.path.splitext(each_file)
                        if ext == '.trt' and 'yolof' in name and 'depth' in name:
                            if 'fp16' in name:
                                SingleStage.TRT_PATH_DEPTH = os.path.join(cfg_path, each_file)
                                print(f'{CurrentDateTime(0)} [Artis_AI] ***** 16 bit Tensor RT Depth Model Exist *****')
                                break
                            elif 'int8' in name:
                                SingleStage.TRT_PATH_DEPTH = os.path.join(cfg_path, each_file)
                                print(f'{CurrentDateTime(0)} [Artis_AI] ***** 8 bit Tensor RT Depth Model Exist *****')
                                break
                            else:
                                self.flag_use_tensorrt_det_0 = False
            ###self.pipeline0 = self.load_pipeline(det_cfg0)
            self.pipeline0 = self.load_pipeline(self.det_cfg0)
            self.det0_model = self.load_det_model(self.det_cfg0, device_id=self.gpu_id, checkpoint_path=det0_model_path)
            if self.flag_use_tensorrt_det_0:
                self.det0_model.module.half()

        else:
            self.pipeline0 = None
            self.det0_model = None

        self.cls0_model = None
        self.cls0_transform = None
        self.cls_lookup = {}

        if self.ai_combine_mode < cc.artis_ai_model_mode["mode_seg_with_seg"]:
            ##### Load RGB Detection Model #####
            self.det_cfg1 = Config.fromfile(det1_config_path)
            self.det_cfg1  = replace_cfg_vals(self.det_cfg1)
            update_data_root(self.det_cfg1)
            self.det_cfg1  = compat_cfg(self.det_cfg1)

            if self.flag_use_tensorrt_det_1:
                if self.det_cfg1.model.type == 'YOLOF_TRT' or self.det_cfg1.model.type == 'YOLOF_RGBD_TRT':
                    cfg_path = os.path.dirname(det1_config_path)
                    for each_file in os.listdir(cfg_path):
                        name, ext = os.path.splitext(each_file)
                        if ext == '.trt' and 'yolof' in name and 'rgb' in name:
                            if 'fp16' in name:
                                SingleStage.TRT_PATH = os.path.join(cfg_path, each_file)
                                print(f'{CurrentDateTime(0)} [Artis_AI] ***** 16 bit Tensor RT RGB Model Exist *****')
                                break
                            elif 'int8' in name:
                                SingleStage.TRT_PATH = os.path.join(cfg_path, each_file)
                                print(f'{CurrentDateTime(0)} [Artis_AI] ***** 8 bit Tensor RT RGB Model Exist *****')
                                break
                            else:
                                self.flag_use_tensorrt_det_1 = False

            # Load Detection Model
            self.pipeline1 = self.load_pipeline(self.det_cfg1)
            self.det1_model = self.load_det_model(self.det_cfg1, device_id=self.gpu_id, checkpoint_path=det1_model_path)
            if self.flag_use_tensorrt_det_1:
                self.det1_model.module.half()

            # Load Classification Configure
            self.det1_lookup = {}
            self.det1_threshold = {}
            self.det1_lookup_path = det1_lookup_path
            with open(self.det1_lookup_path) as json_file:
                det_config = json.load(json_file)
                cls_list = det_config["Class"]
                th_list = det_config["stats"]["class"] if "stats" in det_config and "class" in det_config["stats"] else {}
                for cls_item in cls_list:
                    for idx, cls in cls_item.items():
                        self.det1_lookup[int(idx)] = int(cls)
                        if self.ai_thresh_mode_oc:
                            self.det1_threshold[int(idx)] = max(th_list[str(idx)]["smin"] - self.ai_thresh_sub_value, 0.4) if str(idx) in th_list else 0.4
                        else:
                            self.det1_threshold[int(idx)] = 0.4
            print(f"{CurrentDateTime(0)} [Artis_AI] RGB/D OC {len(self.det1_lookup)} 클래스 모델")
            print(f"{CurrentDateTime(0)} {self.det1_lookup}")
            print(f"{CurrentDateTime(0)} [Artis_AI] RGB/D OC threshold mode : {self.ai_thresh_mode_oc}")
            print(f"{CurrentDateTime(0)} [Artis_AI] RGB/D OC substract value : {self.ai_thresh_sub_value}")
            print(f"{CurrentDateTime(0)} [Artis_AI] RGB/D OC threshold {self.det1_threshold}")
        
        if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_depth"] \
                or self.ai_combine_mode == cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
            self.det0_lookup = {}
            self.det0_lookup_path = det0_lookup_path
            with open(self.det0_lookup_path) as json_file:
                det_config = json.load(json_file)
                cls_list = det_config["Class"]
                for cls_item in cls_list:
                    for idx, cls in cls_item.items():
                        self.det0_lookup[int(idx)] = int(cls)
            print(f"{CurrentDateTime(0)} [Artis_AI] Depth OD {len(self.det0_lookup)} 클래스 모델")
            print(f"{CurrentDateTime(0)} {self.det0_lookup}")

        return True
    
    def load_engine(self, engine_file_path):
        engine = None
        if is_using_tensorrt:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            with open(engine_file_path, "rb") as f:
                engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
        return engine
        
    def prepare_trt_engine(self):
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        if is_using_tensorrt:
            for binding in self.cls_engine:
                size = trt.volume(self.cls_engine.get_binding_shape(binding))
                dtype = trt.nptype(self.cls_engine.get_binding_dtype(binding))
                host_mem = cuda.pagelocked_empty(size, dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)

                bindings.append(int(cuda_mem))
                if self.cls_engine.binding_is_input(binding):
                    host_inputs.append(host_mem)
                    cuda_inputs.append(cuda_mem)
                else:
                    host_outputs.append(host_mem)
                    cuda_outputs.append(cuda_mem)
            return host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, self.cls_engine.create_execution_context()
        else:
            return host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, None
        
    def infer(self, input_data):
        host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, context = self.cls_engine_params
        
        np.copyto(host_inputs[0], input_data.ravel())
        cuda.memcpy_htod(cuda_inputs[0], host_inputs[0])
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(host_outputs[0], cuda_outputs[0])

        return host_outputs
        
    def warm_up(self, loop_max, det_max, cls_max):
        print(f"{CurrentDateTime(0)} [Artis_AI][WARM_UP] {loop_max} loops")
        loop_cnt = 0
        det_time = 0
        cls_time = 0
        warmup_image_height, warmup_image_width = cc.image_resolution_rgb[0], cc.image_resolution_rgb[1]

        if self.ai_combine_mode:
            while loop_cnt < loop_max:
                self.timer_starter.record()
                if (self.ai_combine_mode == cc.artis_ai_model_mode["mode_depth_with_feature_matching"] or
                    self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_depth"]):
                    ch0 = 3 if self.det_cfg0.model.backbone.type == "ResNet" else 4
                    dataset = self.load_dataset([np.zeros((warmup_image_height, warmup_image_width, ch0))], self.pipeline0)
                    self.det_inference(dataset, self.ai_combine_mode, False)

                if (self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_feature_matching"] or
                    self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_seg"] or
                    self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]):
                    if self.ai_sg_model == "yolac":
                        ch = self.sg_info["CHANNEL"]
                        means = self.sg_info["MEANS"]
                        std = self.sg_info["STD"]
                        frame = torch.from_numpy(np.zeros((warmup_image_height, warmup_image_width, ch))).cuda().float()
                        batch = FastBaseTransform(means, std)(frame.unsqueeze(0))
                        self.det0_model(batch)
                    else:
                        img_npy = np.zeros((warmup_image_height, warmup_image_width, self.input_channels["OD"]))
                        self.det0_model.predict(img_npy, conf=0.4, verbose=False)

                    if self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]:
                        img_npy = np.zeros((warmup_image_height, warmup_image_width, self.input_channels["OC"]))
                        self.det1_model.predict(img_npy, conf=0.4, verbose=False)

                if (self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_depth"] or
                    self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_seg"]):
                    ch1 = 3 if self.det_cfg1.model.backbone.type == "ResNet" else 4
                    dataset = self.load_dataset([np.zeros((warmup_image_height, warmup_image_width, ch1))],
                                                self.pipeline1)
                    self.det_inference(dataset, self.ai_combine_mode, True)
                self.timer_ender.record()
                torch.cuda.synchronize()
                det_time = self.timer_starter.elapsed_time(self.timer_ender) / 1000
                print(f"{CurrentDateTime(0)} [{loop_cnt + 1}] RGBD/SEG + Yolof1: {det_time:.2f}")

                if det_time < det_max:
                    break

                loop_cnt += 1

            if loop_cnt == loop_max:
                #return False, "0x02" + "|" + "0x01" + "|" + "NG" + "|" + str(det_max) + "|" + "0" + "|" + "0" + "|" + "0x03"
                return False, [str(det_max), "0", "0"]
            ### ***** For Resnet Classification ***** ###
            '''
            img_crop = np.zeros((224, 224, 3))
            img_crop = Image.fromarray(np.uint8(img_crop)).convert("RGB")
            img_crop = self.cls0_transform(img_crop)
            img_crop = torch.unsqueeze(img_crop, 0)
            img_crop = img_crop.to(self.device)
            if self.flag_use_tensorrt:
                img_crop = img_crop.half()

            loop_cnt = 0
            while loop_cnt < loop_max:
                self.timer_starter.record()
                self.cls0_model(img_crop)
                self.timer_ender.record()
                torch.cuda.synchronize()
                cls_time = self.timer_starter.elapsed_time(self.timer_ender) / 1000
                print(f"{CurrentDateTime(0)} [{loop_cnt + 1}] Resnet : {cls_time:.2f}")

                if cls_time < cls_max:
                    break

                loop_cnt += 1

            if loop_cnt == loop_max:
                #return False, "0x02" + "|" + "0x01" + "|" + "NG" + "|" + str(det_time) + "|" + str(cls_max) + "|" + "0" + "|" + "0x03"
                return False, [str(det_max), str(cls_max), "0"]
            '''
        else:
            loop_cnt = 0
            det_time = 0
            while loop_cnt < loop_max:
                self.timer_starter.record()
                ch1 = 3 if self.det_cfg1.model.backbone.type == "ResNet" else 4
                dataset = self.load_dataset([np.zeros((warmup_image_height, warmup_image_width, ch1))], self.pipeline1)
                self.det_inference(dataset, self.ai_combine_mode, True)
                self.timer_ender.record()
                torch.cuda.synchronize()
                det_time = self.timer_starter.elapsed_time(self.timer_ender) / 1000
                print(f"{CurrentDateTime(0)} [{loop_cnt + 1}] Yolof1 : {det_time:.2f}")

                if det_time < det_max:
                    break

                loop_cnt += 1

            if loop_cnt == loop_max:
                #return False, "0x02" + "|" + "0x01" + "|" + "NG" + "|" + str(det_max) + "|" + "0" + "|" + "0" + "|" +  "0x03"
                return False, [str(det_max), "0", "0"]

        #return True, "0x02" + "|" + "0x01" + "|" + "OK" + "|" + str(round(det_time, 2)) + "|" + str(round(cls_time, 2)) + "|" + "0" + "|" + "0x03"
        return True, [str(round(det_time, 2)), str(round(cls_time, 2)), "0"]


    def reinit(self, config_file_path):
        print(f"{CurrentDateTime(0)} [Artis_AI][reinit] Config & Model Re-initialize")
        self.init_params(config_file_path)
        self.load_models(config_file_path)

    def set_seed(self,seed):
        torch.backends.cudnn.benchmark = True

    def load_pipeline(self, cfg):
        if cfg.model.backbone.type == 'ResNet':
            cfg.data.test.pipeline[0].type = "LoadImage"
            print(f'{CurrentDateTime(0)} [Artis_AI] RGB Model PipeLine : {cfg.data.test.pipeline[0].type}')
        elif cfg.model.backbone.type == 'ResNet_Depth':
            self.depth_type = 'depth'
            cfg.data.test.pipeline[0].type = "LoadDepth"
            print(f'{CurrentDateTime(0)} [Artis_AI] Depth Model PipeLine : {cfg.data.test.pipeline[0].type} / Depth Type : {self.depth_type}')
        elif cfg.model.backbone.type == 'ResNet_RGBD':
            cfg.data.test.pipeline[0].type = 'LoadImageAndDepth'
            ##### Auto GT Detection Model #####
            if cfg.model.bbox_head.num_classes == 1:
                self.depth_type = 'rgbd'
                print(f'{CurrentDateTime(0)} [Artis_AI] RGBD Model (For Auto GT) PipeLine : {cfg.data.test.pipeline[0].type} / Depth Type : {self.depth_type}')
            ##### Object Det & Cls Model By using RGB-D #####
            else:
                print(f'{CurrentDateTime(0)} [Artis_AI] RGBD Model (For Det & Cls) PipeLine : {cfg.data.test.pipeline[0].type}')
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        test_pipeline = Compose(cfg.data.test.pipeline)
        return test_pipeline

    def load_det_model(self, cfg, device_id=0, checkpoint_path=""):
        # Load pre-trained models
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        rfnext_init_model(model, cfg=cfg)
        checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
        model.CLASSES = checkpoint["meta"]["CLASSES"]

        # CUDA set
        model = build_dp(model, "cuda", device_ids=[device_id])

        # Freeze
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model
        
    def simple_collate(self, batch: Sequence, samples_per_gpu: int = 1):
        """
        This function simplifies the original `collate` function by
        reducing some of its complexity and handling fewer special cases.

        Args:
            batch (Sequence): A batch of data to be collated.
            samples_per_gpu (int): The number of samples per GPU. Default is 1.
        """

        if not isinstance(batch, Sequence):
            raise TypeError(f'{batch.dtype} is not supported.')

        if isinstance(batch[0], DataContainer):
            stacked = []
            stacked.append([sample.data for sample in batch[:samples_per_gpu]])
            return DataContainer(stacked, batch[0].stack, batch[0].padding_value)

        elif isinstance(batch[0], Sequence):
            transposed = zip(*batch)
            return [self.simple_collate(samples, samples_per_gpu) for samples in transposed]

        elif isinstance(batch[0], Mapping):
            return {key: self.simple_collate([d[key] for d in batch], samples_per_gpu) for key in batch[0]}

        else:
            return default_collate(batch)

    def load_dataset(self, imgs, test_pipeline):
        datas = []
        for img in imgs:
            data = test_pipeline(img)
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(imgs))
        # just get the actual data from DataContainer
        data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
        data["img"] = [img.data[0] for img in data["img"]]

        # scatter to specified GPU
        data = scatter(data, [self.device])[0]
        return data

    def det_inference(self, data, ai_combine_mode, is_det1):
        if is_det1:
            if self.flag_use_tensorrt_det_1:
                data['img'] = [img.half() for img in data['img']]
            with torch.no_grad():
                results = self.det1_model(return_loss=False, rescale=True, **data)[0]
        else:
            if self.flag_use_tensorrt_det_0:
                data['img'] = [img.half() for img in data['img']]
            with torch.no_grad():
                results = self.det0_model(return_loss=False, rescale=True, **data)[0]

        return results

    def detection(self, img_list):
        bbox_output = []
        bbox_len = []
        dataset = self.load_dataset([img_list[1]], self.pipeline0)

        # Inference
        #results = self.det_inference(dataset)  # x1,y1,x2,y2,score
        with torch.no_grad():
            results = self.det0_model(return_loss=False, rescale=True, **dataset)[0]

        threshold = 0.4
        bbox_result = results[0][results[0][:, 4] > threshold]

        nms_threshold = 0.5
        selected_indices = nms(bbox_result[:, :4], bbox_result[:, 4], overlap_threshold=nms_threshold)
        bbox_result = bbox_result[selected_indices]
        bbox_tmp = []
        for r in bbox_result:
            if (r[0] + r[2]) / 2 >= self.l_thr and (r[0] + r[2])/2 <= self.r_thr:
                #bbox_tmp.append(r)
                bbox_tmp.append([int(r[0]), int(r[1]), int(r[2]), int(r[3]), 0, round(r[4], 2)])
        bbox_result = np.array(bbox_tmp)
        #bbox_output.append(bbox_result)
        for r in bbox_result:
            bbox_output.append(r)
        bbox_len.append(len(bbox_result))

        del results, dataset

        if len(bbox_output) <= 0:
            return False, bbox_output
        #bbox_output = bbox_output[0]

        return True, bbox_output
    
    def det_cls(self, img_list, is_det1, run_mode):
        bbox_output = []

        cc.artis_ai_json_config = cc.get_config(self.config_file_path, cc.artis_ai_json_config)
        if "ai_valid_area" in cc.artis_ai_json_config:
            self.l_thr = int(cc.artis_ai_json_config["ai_valid_area"]["left_x"])
            self.t_thr = int(cc.artis_ai_json_config["ai_valid_area"]["left_y"])
            self.r_thr = int(cc.artis_ai_json_config["ai_valid_area"]["right_x"])
            self.b_thr = int(cc.artis_ai_json_config["ai_valid_area"]["right_y"])

        print(f'{CurrentDateTime(0)} [Artis_AI] Detection Skip Area : [ {self.l_thr}, {self.t_thr}, {self.r_thr}, {self.b_thr} ]')

        if is_det1:
            self.ai_thresh_mode_oc = cc.artis_ai_json_config.get("ai_thresh_mode_rgb", False) or cc.artis_ai_json_config.get("ai_thresh_mode_oc", False)
            self.ai_thresh_sub_value = 0.2
            if "ai_thresh_sub_value" in cc.artis_ai_json_config and 0 <= cc.artis_ai_json_config["ai_thresh_sub_value"] < 1:
                self.ai_thresh_sub_value = cc.artis_ai_json_config["ai_thresh_sub_value"]
            print(f"{CurrentDateTime(0)} [Artis_AI] RGB/D threshold mode : {self.ai_thresh_mode_oc}")
            print(f"{CurrentDateTime(0)} [Artis_AI] RGB/D substract value : {self.ai_thresh_sub_value}")

            with open(self.det1_lookup_path) as json_file:
                det_config = json.load(json_file)
                cls_list = det_config["Class"]
                th_list = det_config["stats"]["class"] if "stats" in det_config and "class" in det_config["stats"] else {}
                for cls_item in cls_list:
                    for idx, cls in cls_item.items():
                        if self.ai_thresh_mode_oc:
                            self.det1_threshold[int(idx)] = max(th_list[str(idx)]["smin"] - self.ai_thresh_sub_value, 0.4) if str(idx) in th_list else 0.4
                        else:
                            self.det1_threshold[int(idx)] = 0.4

            dataset = self.load_dataset([img_list], self.pipeline1)
            if self.flag_use_tensorrt_det_1:
                dataset['img'] = [img.half() for img in dataset['img']]
            with torch.no_grad():
                ###dataset['img'][0] = dataset['img'][0].unsqueeze(0)
                results = self.det1_model(return_loss=False, rescale=True, **dataset)[0]
        else:
            dataset = self.load_dataset([img_list], self.pipeline0)
            if self.flag_use_tensorrt_det_0:
                dataset['img'] = [img.half() for img in dataset['img']]
            with torch.no_grad():
                results = self.det0_model(return_loss=False, rescale=True, **dataset)[0]

        threshold = 0.4
        nms_threshold = 0.5
        for i in range(len(results)):
            bbox_result = results[i][results[i][:, 4] > threshold]
            selected_indices = nms(bbox_result[:, :4], bbox_result[:, 4], overlap_threshold=nms_threshold)
            bbox_result = bbox_result[selected_indices]
            bbox_tmp = []

            for r in bbox_result:
                if is_det1:
                    is_inside_area = self.l_thr <= (r[0] + r[2]) // 2 <= self.r_thr and self.t_thr <= (r[1] + r[3]) // 2 <= self.b_thr
                    cls_thresh = self.det1_threshold[i]
                    if not is_inside_area:
                        self.area_del_cnt += 1
                        print(f"{CurrentDateTime(0)} [Artis_AI] {self.area_del_cnt}번째 {self.det1_lookup[i]} 클래스 : "
                              f"영역 제외 {not is_inside_area} ")
                        cls_thresh = 2.0
                    bbox = [int(r[0]), int(r[1]), int(r[2]), int(r[3]), self.det1_lookup[i], r[4]]
                    bbox = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3]), bbox[4], bbox[5], cls_thresh]
                    bbox_tmp.append(bbox)
                else:
                    depth_l_thr = self.l_thr // 2
                    depth_r_thr = self.r_thr // 2
                    depth_t_thr = self.t_thr // 2
                    depth_b_thr = self.b_thr // 2

                    if self.depth_type == 'rgbd':
                        r[0] = r[0] // 2
                        r[1] = r[1] // 2
                        r[2] = r[2] // 2
                        r[3] = r[3] // 2

                    is_pass = None
                    if depth_l_thr <= (r[0] + r[2]) // 2 <= depth_r_thr and depth_t_thr <= (r[1] + r[3]) // 2 <= depth_b_thr:
                        is_pass = True

                    bbox = [int(r[0]), int(r[1]), int(r[2]), int(r[3]), self.det0_lookup[i], r[4]]
                    bbox = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3]), bbox[4], bbox[5], is_pass]
                    bbox_tmp.append(bbox)

            bbox_result = np.array(bbox_tmp)
            for r in bbox_result:
                bbox_output.append(r)

        del results, dataset

        del_cnt = 0
        if is_det1:
            remove_output = self.remove_overlap(bbox_output)
            bbox_output = []
            for result in remove_output:
                if result[5] >= result[6]:
                    result = [result[0], result[1], result[2], result[3], result[4], result[5], True]
                elif result[6] == 2.0:
                    result = [result[0], result[1], result[2], result[3], result[4], result[5], None]
                else:
                    print(f"{CurrentDateTime(0)} [Artis_AI] {del_cnt}번째 {int(result[4])} 클래스 : "
                                f"임계값 미만 False {result[5]} >= {result[6]}")
                    del_cnt += 1
                    result = [result[0], result[1], result[2], result[3], result[4], result[5], False]
                bbox_output.append(result)
            if run_mode == "NewItem" and self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_only"]:
                bbox_output = self.merge_detected_area(bbox_output)
            self.ai_thresh_del_cnt += del_cnt
        else:
            for each_bbox_output_depth in bbox_output:
                each_bbox_output_depth[0] = str(int(each_bbox_output_depth[0]) * 2)
                each_bbox_output_depth[1] = str(int(each_bbox_output_depth[1]) * 2)
                each_bbox_output_depth[2] = str(int(each_bbox_output_depth[2]) * 2)
                each_bbox_output_depth[3] = str(int(each_bbox_output_depth[3]) * 2)
            bbox_output = self.remove_overlap(bbox_output)
            if run_mode == "NewItem":
                bbox_output = self.merge_detected_area(bbox_output)

        if len(bbox_output) <= 0:
            return False, bbox_output, del_cnt

        return True, bbox_output, del_cnt

    
    def convert_mask2contours(self, masks, classes, image):
        '''
        This function is used to points of contour from mask.

        Params:
            masks : (numpy) (n_obj, img_h, img_w)
        Returns:
            contour_list : (list) (n_obj, n_contour x 2).   n_contour is different for object.
        '''

        contour_list = list()
        bbox_list = list()
        class_list = list()
        patch_list = list()
        for idx, mask in enumerate(masks):
            #start_time = time.time()
            contours = get_contours_from_mask(mask)
            #print(f"get_contours_from_mask : {(time.time() - start_time) * 1000:0.0f}")
            #start_time = time.time()
            if len(contours) == 0:
                continue
            contour, bbox = get_real_contour_from_contours(contours)
            #print(f"get_real_contour_from_contours : {(time.time() - start_time) * 1000:0.0f}")
            #start_time = time.time()
            contour_array = contour.squeeze(axis=1).flatten()
            #print(f"contour.squeeze : {(time.time() - start_time) * 1000:0.0f}")
            contour_list.append(contour_array)
            bbox_list.append(bbox)
            class_list.append(classes[idx])

            # 컨투어 내부만 남긴 마스크 생성
            #start_time = time.time()
            if image is not None:
                #x, y, w, h = cv2.boundingRect(contour)
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]

                # ROI 크기만큼 마스크 생성
                mask_small = np.zeros((h, w), dtype=np.uint8)

                # contour 좌표를 ROI 기준으로 shift
                contour_shifted = contour - [x, y]

                # 작은 마스크에만 그리기
                cv2.fillPoly(mask_small, [contour_shifted], 255)

                # 이미지 ROI에 bitwise_and 적용
                crop = image[y:y+h, x:x+w]
                mask_img = cv2.bitwise_and(crop, crop, mask=mask_small)

                if len(contour) < 5:
                    angle = 0
                else:
                    (_, _), (_, _), angle = cv2.fitEllipse(contour)
                patch = fast_rotate(mask_img, -angle)
                patch_list.append(patch)
            #print(f"make patch : {(time.time() - start_time) * 1000:0.0f}")

        return contour_list, bbox_list, class_list, patch_list


    def inference_seg_yolac(self, run_mode, sg_frame, sg_batch, sg_image=None):
        bbox_output = []
        del_cnt = 0

        start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_time.record()
        cc.artis_ai_json_config = cc.get_config(self.config_file_path, cc.artis_ai_json_config)
        self.l_thr = int(cc.artis_ai_json_config["ai_valid_area"].get("left_x", 150))
        self.t_thr = int(cc.artis_ai_json_config["ai_valid_area"].get("left_y", 50))
        self.r_thr = int(cc.artis_ai_json_config["ai_valid_area"].get("right_x", 1180))
        self.b_thr = int(cc.artis_ai_json_config["ai_valid_area"].get("right_y", 910))

        ai_score_threshold = float(cc.artis_ai_json_config.get("ai_score_threshold", 0.4))
        ai_iou_threshold = float(cc.artis_ai_json_config.get("ai_iou_threshold", 0.3))
        ai_pixel_area_threshold = int(cc.artis_ai_json_config.get("ai_pixel_area_threshold", 7000))
        ai_top_k = int(cc.artis_ai_json_config.get("ai_top_k", 15))

        print(f'{CurrentDateTime(0)} [Artis_AI] SG Skip Area : [ {self.l_thr}, {self.t_thr}, {self.r_thr}, {self.b_thr} ]')
        print(f'{CurrentDateTime(0)} [Artis_AI] SG top_k = {ai_top_k}, score_threshold : {ai_score_threshold}, ',
              f'iou_threshold : {ai_iou_threshold}, pixel_area_threshold : {ai_pixel_area_threshold}')
        
        end_time.record()
        torch.cuda.synchronize()
        print(f"{CurrentDateTime(0)} [Artis_AI][SG] 1. Read Parameter : {start_time.elapsed_time(end_time):0.0f}")
        start_time.record()

        with torch.no_grad():
            sg_pred = self.det0_model(sg_batch)
            end_time.record()
            torch.cuda.synchronize()
            print(f"{CurrentDateTime(0)} [Artis_AI][SG] 2. self.det0_model(sg_batch) : {start_time.elapsed_time(end_time):0.0f}")
            start_time.record()
            h, w, _ = sg_frame.shape
            # inference()
            save = sg_cfg.rescore_bbox
            sg_cfg.rescore_bbox = True
            t = postprocess(sg_pred, w, h, visualize_lincomb=False,
                            crop_masks=True,
                            score_threshold=ai_score_threshold,
                            iou_threshold=ai_iou_threshold,
                            pixel_area_threshold=ai_pixel_area_threshold)
            sg_cfg.rescore_bbox = save
            end_time.record()
            torch.cuda.synchronize()
            print(f"{CurrentDateTime(0)} [Artis_AI][SG] 3. Postprocess : {start_time.elapsed_time(end_time):0.0f}")
            start_time.record()

            # sort object from scores
            idx = t[1].argsort(0, descending=True)[:ai_top_k]      # t[1] = scores
            #masks = t[3][idx]
            classes, scores, boxes, masks = [x[idx].cpu().numpy() for x in t]
            end_time.record()
            torch.cuda.synchronize()
            print(f"{CurrentDateTime(0)} [Artis_AI][SG] 4. Sort object from scores : {start_time.elapsed_time(end_time):0.0f}")
            start_time.record()

            # remove object under score_threshold
            num_dets_to_consider = min(ai_top_k, classes.shape[0])
            for j in range(num_dets_to_consider):
                if scores[j] < ai_score_threshold:
                    num_dets_to_consider = j
                    break
            end_time.record()
            torch.cuda.synchronize()
            print(f"{CurrentDateTime(0)} [Artis_AI][SG] 5. Remove object under score_threshold : {start_time.elapsed_time(end_time):0.0f}")
            start_time.record()

            classes = classes[:num_dets_to_consider]
            scores = scores[:num_dets_to_consider]
            boxes = boxes[:num_dets_to_consider]
            masks = masks[:num_dets_to_consider]

            # Get Contours and re-boxes
            contours, exist_indices = get_contours_from_masks(masks, is_numpy=True)
            boxes = get_boxes_coord_from_contours(contours, is_numpy=True, data_format='xyxy')
            scores = scores[exist_indices]
            masks = masks[exist_indices]
            classes = classes[exist_indices]

            end_time.record()
            torch.cuda.synchronize()
            print(f"{CurrentDateTime(0)} [Artis_AI][SG] 6. Get Contours and re-boxes : {start_time.elapsed_time(end_time):0.0f}")
            start_time.record()

            if len(classes) > 0:
                # Skip Area 적용
                for j in range(len(classes)):
                    cls_id = self.det0_lookup[classes[j] + 1]
                    score = scores[j]
                    #x1, y1, x2, y2 = boxes[j, :]
                    x1, y1, x2, y2 = boxes[j]
                    #print(x1, y1, x2, y2)
                    contour = contours[j]
                    contour_list = contour.tolist()

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if cx < self.l_thr or self.r_thr < cx or cy < self.t_thr or self.b_thr < cy:
                        print(f"{CurrentDateTime(0)} [Artis_AI] SG 영역제외 : {cls_id} 클래스 {score} in {(x1, y1)}, {(x2, y2)}")
                        bbox_output.append([x1, y1, x2, y2, cls_id, score, None, contour_list, None])
                        continue

                    if self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_feature_matching"]:
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1

                        # ROI 크기만큼 마스크 생성
                        mask_small = np.zeros((h, w), dtype=np.uint8)

                        # contour 좌표를 ROI 기준으로 shift
                        contour_shifted = contour - [x, y]

                        # 작은 마스크에만 그리기
                        cv2.fillPoly(mask_small, [contour_shifted], 255)
                        #cv2.imwrite(f"1.fillPoly_{j}.png", mask_small)

                        # 이미지 ROI에 bitwise_and 적용
                        crop = sg_image[y:y+h, x:x+w]
                        mask_img = cv2.bitwise_and(crop, crop, mask=mask_small)
                        #cv2.imwrite(f"2.mask_img_{j}.png", mask_img)

                        '''if len(contour) < 5:
                            angle = 0
                        else:
                            (_, _), (_, _), angle = cv2.fitEllipse(contour)
                        patch = fast_rotate(mask_img, -angle)'''
                        patch = mask_img
                        #cv2.imwrite(f"3.patch_{j}.png", patch)
                        
                        #bbox_output.append([x1, y1, x2, y2, cls_id, score, contour, patches[j]])
                        bbox_output.append([x1, y1, x2, y2, cls_id, score, True, contour_list, patch])
                    else:
                        bbox_output.append([x1, y1, x2, y2, cls_id, score, True, contour_list])
                end_time.record()
                torch.cuda.synchronize()
                print(f"{CurrentDateTime(0)} [Artis_AI][SG] 8. make bbox_output : {start_time.elapsed_time(end_time):0.0f}")

        return len(bbox_output) <= 0, bbox_output, del_cnt

    def inference_seg_yolo(self, run_mode, is_det1, img_dict):
        del_cnt = 0

        cc.artis_ai_json_config = cc.get_config(self.config_file_path, cc.artis_ai_json_config)
        self.l_thr = int(cc.artis_ai_json_config["ai_valid_area"].get("left_x", 150))
        self.t_thr = int(cc.artis_ai_json_config["ai_valid_area"].get("left_y", 50))
        self.r_thr = int(cc.artis_ai_json_config["ai_valid_area"].get("right_x", 1180))
        self.b_thr = int(cc.artis_ai_json_config["ai_valid_area"].get("right_y", 910))

        print(
            f'{CurrentDateTime(0)} [Artis_AI] SG Skip Area : [ {self.l_thr}, {self.t_thr}, {self.r_thr}, {self.b_thr} ]')

        if is_det1:
            self.ai_thresh_mode_oc = (cc.artis_ai_json_config.get("ai_thresh_mode_rgb", False)
                                      or cc.artis_ai_json_config.get("ai_thresh_mode_oc",False))
            self.ai_thresh_sub_value = 0.2
            if "ai_thresh_sub_value" in cc.artis_ai_json_config and 0 <= cc.artis_ai_json_config["ai_thresh_sub_value"] < 1:
                self.ai_thresh_sub_value = cc.artis_ai_json_config["ai_thresh_sub_value"]
            print(f"{CurrentDateTime(0)} [Artis_AI] SG OC threshold mode : {self.ai_thresh_mode_oc}")
            print(f"{CurrentDateTime(0)} [Artis_AI] SG OC substract value : {self.ai_thresh_sub_value}")

            if os.path.exists(self.det1_lookup_path):
                with open(self.det1_lookup_path) as json_file:
                    det_config = json.load(json_file)
                    cls_list = det_config["Class"]
                    th_list = det_config["stats"]["class"] if "stats" in det_config and "class" in det_config["stats"] else {}
                    for cls_item in cls_list:
                        for idx, cls in cls_item.items():
                            if self.ai_thresh_mode_oc:
                                self.det1_threshold[int(idx)] = max(th_list[str(idx)]["smin"] - self.ai_thresh_sub_value, 0.4) if str(idx) in th_list else 0.4
                            else:
                                self.det1_threshold[int(idx)] = 0.4
            sg_model = self.det1_model
            cls_lookup = self.det1_lookup
            img_npy = img_dict["OC"]
        else:
            sg_model = self.det0_model
            cls_lookup = self.det0_lookup
            img_npy = img_dict["OD"]

        img_h, img_w = img_npy.shape[:2]
        inf_output = []
        with torch.no_grad():
            results = sg_model.predict(img_npy, conf=0.4, verbose=False)
            for result in results:
                if result.masks is None or result.boxes is None:
                    continue

                for mask, box in zip(result.masks.xy, result.boxes):
                    contour = np.int32([mask])
                    x1, y1, w, h = cv2.boundingRect(contour)  # bounding box 계산
                    x2, y2 = min(x1 + w, img_w), min(y1 + h, img_h) # 이미지 경게 클리핑
                    
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cls_idx = int(box.cls[0].cpu().numpy())
                    cls_id = cls_lookup[cls_idx]
                    score = box.conf[0].cpu().numpy()
                    if cx < self.l_thr or self.r_thr < cx or cy < self.t_thr or self.b_thr < cy:
                        print(f"{CurrentDateTime(0)} [Artis_AI] SG 영역제외 : {cls_id} 클래스 {score} in {(x1, y1)}, {(x2, y2)}")
                        inf_output.append([x1, y1, x2, y2, cls_id, score, None, mask])
                        continue
                    
                    score_pass = True
                    if is_det1:
                        score_thr = self.det1_threshold[cls_idx]
                        score_pass = score >= score_thr
                        if not score_pass:
                            del_cnt += 1
                            print(f"{CurrentDateTime(0)} [Artis_AI] {del_cnt}번째 {cls_id} 클래스 : "
                                        f"임계값 미만 False {score} >= {score_thr}")

                    inf_output.append([x1, y1, x2, y2, cls_id, score, score_pass, mask])

        tmp_bbox_output = self.remove_overlap(inf_output)
        if run_mode == "NewItem" and not is_det1:
            tmp_bbox_output = self.merge_detected_area(tmp_bbox_output, add_mask=True, img_h=img_h, img_w=img_w)

        spike_output = []
        for idx, bbox in enumerate(tmp_bbox_output):
            mask = bbox[-1]
            contour = np.int32([mask])
            if contour is None:
                continue

            x1, y1, w, h = cv2.boundingRect(contour)
            x2, y2 = min(x1 + w, img_w), min(y1 + h, img_h)

            cls_id = int(bbox[4])

            cvt_cnt = contour[0].reshape(-1, 1, 2).tolist()

            if cls_id not in [9999990, 9999991]:
                is_spike, info, unspike_cnt = wrapper_detect_spike(cvt_cnt)
                if is_spike:
                    print(f"{CurrentDateTime(0)} [Artis_AI] {idx}번째 {bbox[4]} 클래스 : spike 제거")
                    cvt_cnt = unspike_cnt

            spike_output.append([x1, y1, x2, y2, bbox[4], bbox[5], bbox[6], cvt_cnt])

        bbox_output = []
        for idx, out in enumerate(spike_output):
            contour = out[7]
            x1, y1, x2, y2 = out[0], out[1], out[2], out[3]

            if int(out[4]) in [9999990, 9999991]:
                has_loop = has_internal_loop(contour)
                if has_loop:
                    raw_hull, x1, y1, x2, y2, used = wrapping_hull_contour(spike_output, seed_idx=idx, img_w=img_w, img_h=img_h)
                    spike_output[idx][7] = raw_hull
                    contour = raw_hull
                    print(f"{CurrentDateTime(0)} [Artis_AI] {idx}번째 {out[4]} 클래스 loop 제거 : candidate [{used}]")

            if self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_feature_matching"]:
                # 원본 이미지에서 ROI 부분만 crop
                if img_npy.shape[-1] == 4:
                    img_rgb = img_npy[:, :, :3].astype(np.uint8)
                else:
                    img_rgb = img_npy.astype(np.uint8)
                # crop = img_npy[y1:y2, x1:x2]
                crop = img_rgb[y1:y2, x1:x2]

                # ROI 크기만큼 마스크 생성
                mask_small = np.zeros(crop.shape[:2], dtype=np.uint8)

                # contour 좌표를 ROI 기준으로 shift
                #contour_shifted = contour[0] - [x1, y1]
                contour_arr = np.asarray(contour)
                contour_arr = contour_arr.reshape(1, -1, 2)[0]
                contour_shifted = contour_arr - [x1, y1]

                cv2.fillPoly(mask_small, [contour_shifted], 255)
                # cv2.imwrite(f"1.fillPoly_{len(bbox_output)}.png", mask_small)

                # 이미지 ROI에 bitwise_and 적용
                mask_img = cv2.bitwise_and(crop, crop, mask=mask_small)
                # cv2.imwrite(f"2.mask_img_{len(bbox_output)}.png", mask_img)

                '''if len(contour[0]) < 5:
                    angle = 0
                else:
                    (_, _), (_, _), angle = cv2.fitEllipse(contour[0])
                patch = fast_rotate(mask_img, -angle)'''
                patch = mask_img
                ##cv2.imwrite(f"3.patch_{len(bbox_output)}.png", patch)

                bbox_output.append([x1, y1, x2, y2, out[4], out[5], out[6], contour, patch])
            else:
                bbox_output.append([x1, y1, x2, y2, out[4], out[5], out[6], contour])

        return len(bbox_output) <= 0, bbox_output, del_cnt

    def classification1(self, img_list, bbox_output):
        img_list = img_list[[2, 1, 0], :, :]
        imgT = ToPILImage()(img_list)
        cls_result = []

        #image_T = img_list[0]
        #imgT = Image.open(image_T)

        for bbox in bbox_output:
            imgT_crop = imgT.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            imgT_crop= self.cls0_transform(imgT_crop)
            imgT_crop= torch.unsqueeze(imgT_crop, 0)
            imgT_crop= imgT_crop.to(self.device)

            with torch.no_grad():
                outputs = self.cls0_model(imgT_crop)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confs, predicted = torch.max(probs, 1)
                # _, predicted = outputs.max(1)

                tmp_result = self.cls_lookup[predicted.item()]

                cls_result.append([bbox[0], bbox[1], bbox[2], bbox[3], tmp_result, str(round(confs.item(), 2))])

            del imgT_crop, outputs

        return cls_result
        
    def trt_classification(self, img_list, bbox_output):
        img_list = img_list[[2, 1, 0], :, :]
        imgT = ToPILImage()(img_list)
        #cls_result = {}
        idx = 0
        cls_result = []

        #image_T = img_list[0]
        #imgT = Image.open(image_T)
        ###timer_starter, timer_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        n_bbox = 1
        sum_time = 0
        for bbox in bbox_output:
            imgT_crop = imgT.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            imgT_crop = self.cls0_transform(imgT_crop)

            imgT_crop = torch.unsqueeze(imgT_crop, 0)
            rt_img = np.array(imgT_crop, dtype=np.float32)

            with torch.no_grad():
                ###timer_starter.record()
                outputs = self.infer(rt_img)
                ###timer_ender.record()
                ###torch.cuda.synchronize()
                ###cls_time = timer_starter.elapsed_time(timer_ender)
                ###sum_time += cls_time
                # print(f"{n_bbox}th bbox:", cls_time)

                probs = torch.nn.functional.softmax(torch.from_numpy(outputs[0]).to(torch.float32), dim=0)
                confs, predicted = torch.max(probs, 0)
                tmp_result = self.cls_lookup[predicted.item()]
                #top_confs = confs.item()
                #tmp_confs = top_confs
                #cls_result[idx] = [top_result, str(round(tmp_confs, 2))]
                cls_result.append([bbox[0], bbox[1], bbox[2], bbox[3], tmp_result, str(round(confs.item(), 2))])
                idx += 1
                n_bbox += 1

            del imgT_crop, outputs

        ###print(f"TRT Classification Model Time: {sum_time / n_bbox}ms")

        return cls_result

    def adjust(self, point, minPoint, maxPoint):
        if point <= minPoint:
            return 0

        if point >= maxPoint:
            return maxPoint

        return int(float(point))

    def save_imgs(self, img_path, save_path, bbox_output, flag):
        img_t = cv2.imread(img_path)

        color_plt = [(50, 50, 150), (0, 0, 150), (50, 150, 50), (0, 150, 0), (150, 50, 50), (150, 0, 0), (50, 150, 150), (0, 150, 150), (150, 50, 150), (150, 0, 150), (150, 150, 50), (150, 150, 0), (150, 150, 150)]
        idx = 0
        img_h, img_w, _ = img_t.shape
        '''
        img_t[:, 0:self.l_thr, 0] = cv2.subtract(img_t[:, 0:self.l_thr, 0], 100)
        img_t[:, 0:self.l_thr, 1] = cv2.subtract(img_t[:, 0:self.l_thr, 1], 100)
        img_t[:, 0:self.l_thr, 2] = cv2.subtract(img_t[:, 0:self.l_thr, 2], 100)
        img_t[:, self.r_thr:img_w, 0] = cv2.subtract(img_t[:, self.r_thr:img_w, 0], 100)
        img_t[:, self.r_thr:img_w, 1] = cv2.subtract(img_t[:, self.r_thr:img_w, 1], 100)
        img_t[:, self.r_thr:img_w, 2] = cv2.subtract(img_t[:, self.r_thr:img_w, 2], 100)
        '''
        for i in range(len(bbox_output)):
            result = bbox_output[i]
            if result[6] != True:
                continue
            score = round(float(result[5]), 3)
            bbox = result[0:4]
            lbl = str(i + 1) + "(" + str(int(result[4])) + ")"
            color_idx = color_plt[idx % len(color_plt)]
            
            img_t = cv2.rectangle(img_t, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_idx, 3)
            img_t = cv2.putText(img_t, lbl,
                                    (self.adjust(int(bbox[0]) + 10, 10, img_w - 100),
                                     (self.adjust((int(bbox[1]) + int(bbox[3])) / 2, 10, img_h - 100))),
                                    cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,0), thickness=3, fontScale=1.2)
            img_t = cv2.putText(img_t, "(" + str(score) + ")",
                                    (self.adjust(int(bbox[0]) + 10, 10, img_w - 100),
                                     (self.adjust((int(bbox[1]) + int(bbox[3])) / 2 + 25, 10, img_h - 100))),
                                    cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,0), thickness=3, fontScale=1)

            idx += 1

        if flag == 'Depth':
            result_file_name = os.path.basename(img_path).replace('_Color', '_Depth')
        else:
            result_file_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(save_path, result_file_name), img_t)

        return img_t

    def save_seg(self, img_path, save_path, image, seg_output, visualize_bbox=False, visualize_score=False):
        result_file_name = os.path.basename(img_path).replace('_Color', '_Depth')
        result_path = os.path.join(save_path, result_file_name)

        imageHeight, imageWidth = image.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        fontThickness = 2

        bgColors = [
            [80, 60, 160],     # 0 - 은은한 바이올렛 브라운
            [178, 0, 0],       # 1 - 톤다운 파랑
            [0, 153, 153],     # 2 - 톤다운 노랑 (청록 계열)
            [255, 120, 150],   # 3 - 소프트 코랄 핑크
            [0, 178, 0],       # 4 - 톤다운 녹색
            [120, 60, 80],     # 5 - 부드러운 네이비 레드톤 블루
            [0, 128, 255],     # 6 - 주황
            [255, 170, 0],     # 7 - 살구색
            [130, 200, 180],   # 8 - 진한 민트블루
            [0, 200, 200],     # 9 - 청록빛 민트
            [200, 180, 255],   # 10 - 라일락 퍼플
            [220, 190, 160],   # 11 - 웜 베이지
            [200, 100, 20],    # 12 - 딥 오렌지
            [0, 128, 128],     # 13 - 올리브
            [128, 0, 128],     # 14 - 자주
            [70, 180, 150]     # 15 - 청녹
        ]

        bboxes = []

        # 빵과 오류 케이스 구분 / 좌표 가공하기
        real_flag_list = np.zeros(len(seg_output))
        total_real_obj = 0
        for out in seg_output:
            if out[6] == True:
                x1, y1, x2, y2 = int(out[0]) // 2, int(out[1]) // 2, int(out[2]) // 2, int(out[3]) // 2
                # --- 좌표 클램핑 ---
                x1 = max(0, min(x1, imageWidth - 1))
                y1 = max(0, min(y1, imageHeight - 1))
                x2 = max(0, min(x2, imageWidth - 1))
                y2 = max(0, min(y2, imageHeight - 1))

                bboxes.append([x1, y1, x2, y2, out[4], round(float(out[5]), 3), out[7]])
                idx_obj = len(bboxes) - 1
                if int(out[4]) < 9000000:
                    total_real_obj += 1
                    real_flag_list[idx_obj] = True
                else:
                    real_flag_list[idx_obj] = False

        # contour 그리기
        real_obj, err_obj = 0, 0
        for idx_obj, out in enumerate(bboxes):
            contour = out[6]
            if real_flag_list[idx_obj]:
                color = bgColors[real_obj % len(bgColors)]
                real_obj += 1
            else:
                color = bgColors[(total_real_obj + err_obj) % len(bgColors)]
                err_obj += 1

            for i, data in enumerate(contour):
                pts_x, pts_y = data[0]
                pts_x, pts_y = int(pts_x/2), int(pts_y/2)
                pts_x_prev, pts_y_prev = contour[(i-1) % len(contour)][0]
                pts_x_prev, pts_y_prev = int(pts_x_prev/2), int(pts_y_prev/2)
                cv2.line(image, (pts_x, pts_y), (pts_x_prev, pts_y_prev), color, 3)

        # BBOX 그리기
        real_obj, err_obj = 0, 0
        if visualize_bbox:
            for idx_obj, out in enumerate(bboxes):
                x1, y1, x2, y2 = int(out[0]), int(out[1]), int(out[2]), int(out[3])
                if real_flag_list[idx_obj]:
                    color = bgColors[real_obj % len(bgColors)]
                    real_obj += 1
                else:
                    color = bgColors[(total_real_obj + err_obj) % len(bgColors)]
                    err_obj += 1
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        real_obj, err_obj = 0, 0
        for idx_obj, out in enumerate(bboxes):
            x1, y1, x2, y2 = int(out[0]), int(out[1]), int(out[2]), int(out[3])
            if real_flag_list[idx_obj]:
                color = bgColors[real_obj % len(bgColors)]
                real_obj += 1
                num_obj_str = str(real_obj)
            else:
                color = bgColors[(total_real_obj + err_obj) % len(bgColors)]
                err_obj += 1
                num_obj_str = str(total_real_obj + err_obj)

            # --- 번호 표시 ---
            #num_obj_str = str(idx_obj + 1)
            (tw, th), _ = cv2.getTextSize(num_obj_str, font, fontScale, fontThickness)
            margin = 5
            num_bg_w = tw + 2 * margin
            num_bg_h = th + 2 * margin
            num_x = max(0, min(x1 + margin, imageWidth - num_bg_w))
            num_y = max(0, min(y1 + margin, imageHeight - num_bg_h))
           
            # 반투명 사각형
            overlay = image.copy()
            cv2.rectangle(overlay,
                            (num_x - margin, num_y - margin),
                            (num_x + num_bg_w, num_y + num_bg_h),
                            color, -1)
            alpha = 0.3
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # 번호 텍스트
            cv2.putText(image, num_obj_str, (num_x, num_y + th),
                            font, fontScale, (0, 0, 0), fontThickness, cv2.LINE_AA)

            # --- 텍스트 표시 ---
            if visualize_score:
                cls_id = int(out[4])

                (text_w, text_h), baseline = cv2.getTextSize(str(cls_id), font, fontScale, fontThickness)
                totalTextHeight = text_h
                
                centerX = x1 + text_w if cls_id in self.sg_except_class else (x1 + x2) // 2
                centerY = y1 + text_h if cls_id in self.sg_except_class else (y1 + y2) // 2

                score = f"({round(float(out[5]), 3)})"
                (score_w, score_h), _ = cv2.getTextSize(score, font, fontScale, fontThickness)
                totalTextHeight = totalTextHeight + 5 if cls_id in self.sg_except_class else totalTextHeight + score_h + 5

                textX = max(0, min(centerX - text_w // 2, imageWidth - text_w))
                textY = max(0, min(centerY - totalTextHeight // 2 + text_h, imageHeight - totalTextHeight))

                # --- 반투명 배경 ---
                margin = 10
                bg_w = max(text_w, score_w) + 2 * margin
                bg_h = totalTextHeight + 2 * margin
                bg_x = max(0, min(centerX - bg_w // 2, imageWidth - bg_w))
                bg_y = max(0, min(centerY - totalTextHeight // 2 - margin, imageHeight - bg_h))

                overlay = image.copy()
                cv2.rectangle(overlay, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h),
                            (255, 255, 255), -1)
                image = cv2.addWeighted(overlay, 0.2, image, 0.8, 0)

                cv2.putText(image, str(cls_id), (textX, textY),
                    font, fontScale, (10, 10, 10), fontThickness, cv2.LINE_AA)

                scoreX = textX + text_w + 1 if cls_id in self.sg_except_class else centerX - score_w // 2
                scoreY = textY if cls_id in self.sg_except_class else textY + score_h + 5
                cv2.putText(image, score, (scoreX, scoreY),
                            font, fontScale, (10, 10, 10), fontThickness, cv2.LINE_AA)

        # ====== 최종 저장 ======
        cv2.imwrite(result_path, image)

    def intersect_single(self, box_a, box_b):
        max_xy = np.minimum(box_a[2:], box_b[2:])
        min_xy = np.maximum(box_a[:2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[0] * inter[1]

    def calc_overlap(self, box_a, box_b):
        inter = self.intersect_single(box_a, box_b)
        area_a = ((box_a[2] - box_a[0]) *
                  (box_a[3] - box_a[1]))  # [A,B]
        area_b = ((box_b[2] - box_b[0]) *
                  (box_b[3] - box_b[1]))  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]
    
    def remove_overlap(self, bbox_output):
        bbox_temp = bbox_output.copy()
        del_index_list = []
        if len(bbox_temp) > 0:
            bbox_temp.sort(reverse=True, key=lambda x: x[5])
            for key_index, key_det_result in enumerate(bbox_temp):
                key_score = key_det_result[5]
                key_box = [int(key_det_result[0]), int(key_det_result[1]), int(key_det_result[2]), int(key_det_result[3])]
                for index, current_det_result in enumerate(bbox_temp):
                    if key_index == index:
                        continue
                    #del_index = len(bbox_temp) - (index + 1)
                    current_score = current_det_result[5]
                    current_box = [int(current_det_result[0]), int(current_det_result[1]), int(current_det_result[2]), int(current_det_result[3])]
                    overlap = self.calc_overlap(current_box, key_box)
                    if key_score >= current_score and overlap > self.overlap_thr:
                        if key_score == current_score:
                            print(f'{CurrentDateTime(0)} [Artis_AI][remove_overlap] key_score={key_score}, current_score={current_score}, overlap={overlap}')
                        del_index_list.append(index)

        bbox_output = []
        del_index_list.sort()
        for index, det_result in enumerate(bbox_temp):
            if not index in del_index_list:
                bbox_output.append(det_result)
        print(f'{CurrentDateTime(0)} [Artis_AI][remove_overlap] {len(del_index_list)}개 중첩 물체 제거')

        return bbox_output

    def merge_detected_area(self, bbox, add_mask=False, img_h=960, img_w=1280):
        cc.artis_ai_current_log = f'Start Automatic GT In New Item Mode'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

        merge_diff_thr_w = 150
        merge_diff_thr_h = 100
        merge_overlap_thr = 0.8

        bbox_backup = copy.deepcopy(bbox)

        if len(bbox) <= 1:
            cc.artis_ai_current_log = f'[merge_detected_area] No BBox to Merge.'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

            cc.artis_ai_current_log = f'Finish Automatic GT In New Item Mode'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

            return bbox_backup

        total_merge_bbox = []
        # bbox center 미리 계산
        centers = [((int(b[0]) + int(b[2])) // 2, (int(b[1]) + int(b[3])) // 2) for b in bbox]
        merged_flags = [False] * len(bbox)
        for key_index, key_det_result in enumerate(bbox):
            cls_id = key_det_result[4]
            if int(cls_id) >= 9000000:
                total_merge_bbox.append(copy.deepcopy(key_det_result))
                merged_flags[key_index] = True
                cc.artis_ai_current_log = f'[merge_detected_area] Area No.{key_index} class id = {cls_id} 제외'
                make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)
            elif key_det_result[6] == None:
                total_merge_bbox.append(copy.deepcopy(key_det_result))
                merged_flags[key_index] = True
                cc.artis_ai_current_log = f'[merge_detected_area] Area No.{key_index} 외곽 영역 제외'
                make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)

            if merged_flags[key_index]:
                continue

            # merge_bbox는 shallow copy로 시작, 내부 리스트는 deepcopy로 처리
            merge_bbox = []
            merged_mask = []
            for val in key_det_result:
                if isinstance(val, list):
                    merge_bbox.append(val.copy())
                else:
                    merge_bbox.append(val)
            if add_mask:
                merged_mask = np.zeros((img_h, img_w), dtype=np.uint8) if add_mask else []
                cv2.fillPoly(merged_mask, [np.int32(key_det_result[-1])], 255)

            merge_indexes = {key_index}
            for current_index, current_det_result in enumerate(bbox):
                if key_index == current_index or merged_flags[current_index]:
                    continue

                cx, cy = centers[current_index]
                kx, ky = centers[key_index]
                if abs(cx - kx) < merge_diff_thr_w and abs(cy - ky) < merge_diff_thr_h:
                    merge_indexes.add(current_index)

                    key_box_before_merge = merge_bbox.copy()  # 로그용 이전 상태

                    # merge bbox 계산
                    merge_bbox[0] = min(int(merge_bbox[0]), int(current_det_result[0]))
                    merge_bbox[1] = min(int(merge_bbox[1]), int(current_det_result[1]))
                    merge_bbox[2] = max(int(merge_bbox[2]), int(current_det_result[2]))
                    merge_bbox[3] = max(int(merge_bbox[3]), int(current_det_result[3]))
                    merge_bbox[5] = min(float(merge_bbox[5]), float(current_det_result[5]))

                    # mask merge
                    if add_mask:
                        cur_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                        cv2.fillPoly(cur_mask, [np.int32(current_det_result[-1])], 255)
                        merged_mask = cv2.bitwise_or(merged_mask, cur_mask)

                    cc.artis_ai_current_log = (
                        f'[merge_detected_area] Area No.{key_index} [{key_box_before_merge[:4]}] '
                        f'and Area No.{current_index} [{current_det_result[:4]}] are merged. '
                        f'Merged Area: [{merge_bbox[:4]}]'
                    )
                    make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)

            # 병합한 mask 업데이트
            if add_mask:
                contours, hierarchy = cv2.findContours(
                    merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if len(contours) == 0:
                    merge_bbox[-1] = None
                else:
                    contour = max(contours, key=cv2.contourArea)
                    polygon = contour.reshape(-1, 2)
                    merge_bbox[-1] = polygon.astype(np.float32)

            current_coords = list(map(int, merge_bbox[:4]))
            flag_merge = all(self.calc_overlap(list(map(int, tb[:4])), current_coords) < merge_overlap_thr
                                     for tb in total_merge_bbox)
            # 중복확인 + 추가
            if flag_merge:
                total_merge_bbox.append(copy.deepcopy(merge_bbox))
                for idx in merge_indexes:
                    bbox[idx] = copy.deepcopy(merge_bbox)
                    merged_flags[idx] = True

        cc.artis_ai_current_log = f'[merge_detected_area] Area Merge Finished. Before : {len(bbox_backup)} -> After : {len(total_merge_bbox)}'
        make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)
        cc.artis_ai_current_log = f'Finish Automatic GT In New Item Mode'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

        return total_merge_bbox

    def _apply_image_fusion(self, imgs, bbox_output, handler_func):
        """이미지 합성 공통 로직을 처리하는 헬퍼 메서드"""
        try:
            fused_image = fuse_reflection_free_image(imgs, bbox_output, depth_instance=self.depth_instance)
            if fused_image is not None:
                handler_func(fused_image)
                return True
            return False
        except Exception as e:
            cc.artis_ai_current_log = f'Image Fusion 실패: {e}'
            make_artis_ai_log(cc.artis_ai_current_log, 'warning', True)
            return False

    def _update_preloaded_image_for_rgb_seg(self, preloadedImage, fused_image):
        """mode_rgb_with_seg용 preloadedImage 업데이트"""
        # Mode 2: 새 tensor 먼저 생성
        if preloadedImage.flag_preload_rgbd and preloadedImage.image.shape[0] == 4:
            depth_channel = preloadedImage.image[3:4, :, :]
            rgb_tensor = torch.tensor(fused_image.transpose(2, 0, 1), device=preloadedImage.device)
            new_image = torch.cat([rgb_tensor, depth_channel], dim=0)
        else:
            new_image = torch.tensor(fused_image.transpose(2, 0, 1), device=preloadedImage.device)
        # 성공 시에만 교체
        preloadedImage.original_image = fused_image.copy()
        preloadedImage.image = new_image

    def _update_preloaded_image_for_seg_seg(self, preloadedImage, fused_image):
        """mode_seg_with_seg용 preloadedImage 업데이트"""
        # Mode 3: 새 model_input 먼저 생성
        if preloadedImage.model_input is not None and preloadedImage.model_input.shape[-1] == 4:
            depth_channel = preloadedImage.model_input[:, :, 3:4]
            new_model_input = np.concatenate([fused_image.astype(np.float32), depth_channel], axis=-1)
        else:
            new_model_input = fused_image.copy()
        # 성공 시에만 교체
        preloadedImage.original_image = fused_image.copy()
        preloadedImage.model_input = new_model_input

    def _update_bbox_crops_for_feature_matching(self, bbox_output, fused_image):
        """mode_seg_with_feature_matching용 bbox crop 이미지 업데이트"""
        # Feature Matching용 crop 이미지 업데이트 (bbox[8] = patch)
        for bbox in bbox_output:
            if bbox[6]:  # valid bbox
                crop_img = fused_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                if len(bbox) > 8:
                    bbox[8] = crop_img

    def inference(self, imgs, is_debug, save_path, run_mode):
        ###print(f"[Artis_AI][inference] {datetime.now()}")

        current_mode_name = cc.artis_ai_model_mode_detail.get(self.ai_combine_mode, f"Unknown({self.ai_combine_mode})")
        cc.artis_ai_current_log = f'Inference Mode: {current_mode_name} ({self.ai_combine_mode})'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

        # Preload RGB Images
        preloadedImage = PreloadedImagesOnGPU(ai_seg_model=self.ai_sg_model)
        if self.ai_combine_mode == cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
            preloadedImage.preload_image(imgs[1], 'RGB', self.depth_type, self.det_cfg0)
        elif (self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_feature_matching"]
              or self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]):
            preloadedImage.preload_image(imgs[1], 'SG', self.depth_type, self.sg_info)
        else:
            preloadedImage.preload_image(imgs[1], 'RGB', self.depth_type, self.det_cfg1)

        cc.artis_ai_current_log = f'Load RGB Image'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', False)

        timer_total_starter, timer_total_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timer_total_starter.record()

        invalid_bbox_det_0 = []
        invalid_bbox_rgb = []
        invalid_bbox_integrate = []

        #with open(self.config_file_path) as json_file:
        #    current_json_config = json.load(json_file)
        cc.artis_ai_json_config = cc.get_config(self.config_file_path, cc.artis_ai_json_config)
        self.send_result_format = bool(cc.artis_ai_json_config["ai_send_result_format"])

        if "overlap_thr" in cc.artis_ai_json_config:
            self.overlap_thr = float(cc.artis_ai_json_config["overlap_thr"])
        print(f'{CurrentDateTime(0)} [Artis_AI] Remove Overlap Thr : [ {self.overlap_thr} ]')

        if "flag_ignore_depth" in cc.artis_ai_json_config:
            self.flag_ignore_depth = bool(cc.artis_ai_json_config["flag_ignore_depth"])
        print(f'{CurrentDateTime(0)} [Artis_AI] Flag For Ignore Depth Result : [ { self.flag_ignore_depth} ]')

        cc.artis_ai_current_log = f'Load Config Json File'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', False)

        if cc.artis_ai_model_mode["mode_rgb_only"] < self.ai_combine_mode < cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
            artis_det_0_name = ''
            if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_depth"]:
                preloadedImage.preload_image(imgs[1], 'Depth', self.depth_type, self.det_cfg0)

                if self.depth_type == 'rgbd':
                    artis_det_0_name = 'RGB-D'
                else:
                    artis_det_0_name = 'Depth'
                cc.artis_ai_current_log = f'Load {artis_det_0_name} Information'
            else:
                artis_det_0_name = 'SG'
                preloadedImage.preload_image(imgs[1], 'SG', self.depth_type, self.sg_info)
                cc.artis_ai_current_log = f'Load {artis_det_0_name} Image'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', False)

            # ========== OD (Object Detection) 실행 ==========
            self.timer_starter_det_0.record()
            if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_depth"]:
                is_det0, bbox_output_det_0, _ = self.det_cls(preloadedImage.depth, False, run_mode)
                bbox_output_det_0.sort(key=lambda x:((int(x[3]) + int(x[1])) // 2, (int(x[2]) + int(x[0])) // 2))
            else:
                # [2]mode_rgb_with_seg, [3]mode_seg_with_seg
                if self.ai_sg_model == "yolac":
                    is_det0, bbox_output_det_0, _ = self.inference_seg_yolac(run_mode, preloadedImage.sg_frame, preloadedImage.sg_batch)
                else:
                    is_det0, bbox_output_det_0, _ = self.inference_seg_yolo(run_mode, False, preloadedImage.model_input)
                bbox_output_det_0.sort(key=lambda x:((int(x[3]) + int(x[1])) // 2, (int(x[2]) + int(x[0])) // 2))
                self.save_seg(imgs[1], os.path.dirname(imgs[1]), preloadedImage.sg_depth_img, bbox_output_det_0, True, True)
            self.timer_ender_det_0.record()

            # ========== Image Fusion ([2]mode_rgb_with_seg, [3]mode_seg_with_seg) ==========
            image_fusion_enable = cc.artis_ai_json_config.get("image_fusion", False)
            if image_fusion_enable:
                def handler_func(fused_image):
                    if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_seg"]:
                        self._update_preloaded_image_for_rgb_seg(preloadedImage, fused_image)
                    elif self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]:
                        self._update_preloaded_image_for_seg_seg(preloadedImage, fused_image)
                    cc.artis_ai_current_log = f'Image Fusion 완료: OC 입력 이미지 교체됨'
                    make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
                
                self._apply_image_fusion(imgs, bbox_output_det_0, handler_func)
            else:
                cc.artis_ai_current_log = f'Image Fusion 스킵: image_fusion=false'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

            # ========== OC (Object Classification) 실행 ==========
            self.timer_starter_det_1.record()
            if self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]:
                artis_det_1_name = 'SG'
                is_det1, bbox_output, del_cnt = self.inference_seg_yolo(run_mode, True, preloadedImage.model_input)
            else:
                # [1]mode_rgb_with_depth, [2]mode_rgb_with_seg
                artis_det_1_name = 'RGB/D'
                is_det1, bbox_output, del_cnt = self.det_cls(preloadedImage.image, True, run_mode)

            bbox_output.sort(key=lambda x:((int(x[3]) + int(x[1])) // 2, (int(x[2]) + int(x[0])) // 2))
            self.timer_ender_det_1.record()

            ##### Sort By Center Point Ascending Order #####
            ##bbox_output.sort(key=lambda x:((int(x[3]) + int(x[1])) // 2, (int(x[2]) + int(x[0])) // 2))
            ##bbox_output_det_0.sort(key=lambda x:((int(x[3]) + int(x[1])) // 2, (int(x[2]) + int(x[0])) // 2))
            ##### Sort By Center Point Ascending Order #####

            torch.cuda.synchronize()
            det_time_det_0 = self.timer_starter_det_0.elapsed_time(self.timer_ender_det_0)
            det_time_det_1 = self.timer_starter_det_1.elapsed_time(self.timer_ender_det_1)
            det_time = det_time_det_0 + det_time_det_1

            cc.artis_ai_current_log = f'Finish {artis_det_0_name} OD & {artis_det_1_name} OC Inference'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', False)

            self.timer_starter.record()
            ### ***** For Resnet Classification ***** ###
            #match_result, cls_result = self.classification1(imgs, bbox_output0)
            '''
            if self.flag_use_tensorrt:
                cls_result = self.trt_classification(preloadedImage.image, bbox_output)
            else:
                cls_result = self.classification1(preloadedImage.image, bbox_output)
            '''
            self.timer_ender.record()
            torch.cuda.synchronize()
            cls_time = self.timer_starter.elapsed_time(self.timer_ender)

            if is_debug:
                #save_path = '/home/nvidia/kisan/JetsonMan/workspace/output/'
                print(f'{CurrentDateTime(0)} [Artis_AI] Debug Image Save Path : {save_path}')
                img_rgb = self.save_imgs(imgs[1], save_path, bbox_output, 'RGB')
                img_depth = self.save_imgs(imgs[1], save_path, bbox_output_det_0, 'Depth')
                #if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_seg"]:
                #    self.save_seg(imgs[1], os.path.dirname(imgs[1]), preloadedImage.sg_depth_img, bbox_output_det_0, True, True)

            '''
            print('[Artis_AI] ***** Resnet Classfication Result *****')
            for each_cls_result in cls_result:
                print(each_cls_result)
            
            print(f"[Artis_AI] Yolof0 {len(bbox_output)}개 검출, {det_time} ms")
            print(f"[Artis_AI] ResNet {len(bbox_output)}개 인식, {cls_time} ms")
            '''

            cc.artis_ai_current_log = f'{artis_det_0_name} OD {len(bbox_output_det_0)}개 검출, {det_time_det_0} ms'
            make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)

            cc.artis_ai_current_log = f'{artis_det_1_name} OC {len(bbox_output)}개 검출, {det_time_det_1} ms'
            make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)


            valid_thr_diff_coordinate = 75
            total_bbox_cnt = len(bbox_output_det_0)
            # match_bbox = [depth_bbox_index, rgb_bbox_index] if both bboxes are decided to same area
            match_bbox = [[0, 0] for i in range(total_bbox_cnt)]
            match_bbox_index = 0
            for bbox_det_0_index, each_bbox_output_det_0 in enumerate(bbox_output_det_0):
                if int(each_bbox_output_det_0[4]) not in self.sg_except_class and each_bbox_output_det_0[6]:
                    det_0_bbox = [int(each_bbox_output_det_0[0]), int(each_bbox_output_det_0[1]),
                                int(each_bbox_output_det_0[2]), int(each_bbox_output_det_0[3])]
                    det_0_bbox_center = [(det_0_bbox[0] + det_0_bbox[2]) // 2, (det_0_bbox[1] + det_0_bbox[3]) // 2]
                    for bbox_rgb_index, each_bbox_output in enumerate(bbox_output):
                        if each_bbox_output[6] and int(each_bbox_output[4]) not in self.sg_except_class and int(each_bbox_output[4]) not in self.sg_error_class:
                            rgb_bbox = [int(each_bbox_output[0]), int(each_bbox_output[1]), int(each_bbox_output[2]), int(each_bbox_output[3])]
                            rgb_bbox_center = [(rgb_bbox[0] + rgb_bbox[2]) // 2, (rgb_bbox[1] + rgb_bbox[3]) // 2]
                            if (abs(det_0_bbox_center[0] - rgb_bbox_center[0]) < valid_thr_diff_coordinate
                                    and abs(det_0_bbox_center[1] - rgb_bbox_center[1]) < valid_thr_diff_coordinate):
                                match_bbox[match_bbox_index] = [bbox_det_0_index + 1, bbox_rgb_index + 1]
                                match_bbox_index += 1
                                break
                else:
                    match_bbox[match_bbox_index] = [bbox_det_0_index + 1, len(bbox_output) + 1]
                    match_bbox_index += 1

            flag_bbox_valid = True
            for each_match_bbox in match_bbox:
                if each_match_bbox[0] == 0 or each_match_bbox[1] == 0:
                    flag_bbox_valid = False
                    break
            if len(bbox_output) == 0 and len(bbox_output_det_0) == 0:
                flag_bbox_valid = True

            match_bbox = np.array(match_bbox)
            if len(match_bbox) > 0:
                for current_bbox_index, each_bbox_output_det_0 in enumerate(bbox_output_det_0):
                    if not current_bbox_index + 1 in match_bbox[:, 0] and each_bbox_output_det_0[6]:
                        invalid_bbox_det_0.append(each_bbox_output_det_0)
                        invalid_bbox_integrate.append(each_bbox_output_det_0)
                for current_bbox_index, each_bbox_output in enumerate(bbox_output):
                    if not current_bbox_index + 1 in match_bbox[:, 1] and each_bbox_output[6]:
                        invalid_bbox_rgb.append(each_bbox_output)
                    ###invalid_bbox_integrate.append(each_bbox_output)

            cc.artis_ai_current_log = f'Finish Compare {artis_det_0_name} OD BBox & {artis_det_1_name} OC BBox'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', False)

            # ========= 미인식 결과 처리 ========= #
            if self.ai_thresh_mode_oc and del_cnt > 0:
                flag_bbox_valid = False
                for each_del_output in bbox_output:
                    if each_del_output[6] == False:
                        invalid_bbox_integrate.append(each_del_output)
                        invalid_bbox_rgb.append(each_del_output)
            # ========= 미인식 결과 처리 ========= #

            # Ignore Depth Detection Result to Prevent Errors due to Depth ROI #
            if self.flag_ignore_depth:
                flag_bbox_valid = True

            # 에러코드 설정
            if flag_bbox_valid:
                cc.artis_ai_current_error_code = '0x00000000'
                cc.artis_ai_current_error_reason = ''
                cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
            else:
                #error_key = 'detect_untrained_view' if len(del_output) else 'detect_untrained_item'
                error_key = 'detect_untrained_item'
                cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['inference'][error_key]))
                cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['inference'][cc.artis_ai_error_code['inference'][error_key]]
                cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)

            cc.artis_ai_result_json['artis_error_info']['object_total_cnt'] = len(invalid_bbox_integrate)
            for index, each_invalid_bbox in enumerate(invalid_bbox_integrate):
                current_json_data = {str(index): [int(each_invalid_bbox[0]), int(each_invalid_bbox[1]),
                                                  int(each_invalid_bbox[2]), int(each_invalid_bbox[3])]}
                cc.artis_ai_result_json['artis_error_info']['object_bbox'].update(current_json_data)

                if self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]:
                    current_json_data = {
                        str(index): {'cnt': len(each_invalid_bbox[7]), 'point': each_invalid_bbox[7]}}
                    cc.artis_ai_result_json['artis_error_info']['object_contour'].update(current_json_data)

            # 전체 결과 저장  if str(each_bbox_output[4]) not in self.sg_special_class:
            real_idx = 0
            flag_class_valid = True
            for index, each_bbox_output in enumerate(bbox_output):
                current_class_info = int(each_bbox_output[4])
                if current_class_info not in self.sg_except_class and current_class_info not in self.sg_error_class:
                    current_json_data = {str(real_idx): [int(each_bbox_output[0]), int(each_bbox_output[1]),
                                                      int(each_bbox_output[2]), int(each_bbox_output[3])]}
                    cc.artis_ai_result_json['artis_object_bbox'].update(current_json_data)

                    if current_class_info >= cc.artis_ai_class_invalid_item:
                        current_json_data = {str(real_idx): cc.artis_ai_class_invalid_item}
                        flag_class_valid = False
                    elif each_bbox_output[6] == False:
                        current_json_data = {str(real_idx): cc.artis_ai_class_untrained_view}
                    else:
                        current_json_data = {str(real_idx): current_class_info}
                    cc.artis_ai_result_json['artis_object_detail'].update(current_json_data)

                    current_json_data = {str(real_idx): str(each_bbox_output[5])}
                    cc.artis_ai_result_json['artis_object_score'].update(current_json_data)

                    if self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]:
                        current_json_data = {
                            str(real_idx): {'cnt': len(each_bbox_output[7]), 'point': each_bbox_output[7]}}
                        cc.artis_ai_result_json['artis_object_contour'].update(current_json_data)

                    real_idx += 1

            if not flag_class_valid:
                cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['inference']['detect_invalid_item']))
                cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['inference'][cc.artis_ai_error_code['inference']['detect_invalid_item']]
                cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)

            if flag_bbox_valid:
                cc.artis_ai_result_json['artis_result_info'] = {'object_total_cnt': real_idx}
            else:
                cc.artis_ai_result_json['artis_result_info'] = {'object_total_cnt': real_idx + len(invalid_bbox_det_0)}
                ### Save Invalid BBox To Json ###
                for index, each_invalid_det_0_bbox in enumerate(invalid_bbox_det_0):
                    integrate_index = index + real_idx
                    current_json_data = {str(integrate_index): [int(each_invalid_det_0_bbox[0]), int(each_invalid_det_0_bbox[1]),
                                                      int(each_invalid_det_0_bbox[2]), int(each_invalid_det_0_bbox[3])]}
                    cc.artis_ai_result_json['artis_object_bbox'].update(current_json_data)

                    if int(each_invalid_det_0_bbox[4]) >= cc.artis_ai_class_invalid_item:
                        current_json_data = {str(integrate_index): int(cc.artis_ai_class_invalid_item)}
                    else:
                        current_json_data = {str(integrate_index): int(cc.artis_ai_class_untrained_item)}
                    cc.artis_ai_result_json['artis_object_detail'].update(current_json_data)

                    current_json_data = {str(integrate_index): str(each_invalid_det_0_bbox[5])}
                    cc.artis_ai_result_json['artis_object_score'].update(current_json_data)

                    if self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_seg"]:
                        current_json_data = {
                            str(integrate_index): {'cnt': len(each_invalid_det_0_bbox[7]), 'point': each_invalid_det_0_bbox[7]}}
                        cc.artis_ai_result_json['artis_object_contour'].update(current_json_data)
                ### Save Invalid BBox To Json ###

            # Det0 : Depth / Seg 결과
            artis_det_0_key = "depth" if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_depth"] else "seg"
            for index_det_0, each_bbox_output_det_0 in enumerate(bbox_output_det_0):
                current_json_data = {str(index_det_0):[int(each_bbox_output_det_0[0]), int(each_bbox_output_det_0[1]),
                                                       int(each_bbox_output_det_0[2]), int(each_bbox_output_det_0[3])]}
                cc.artis_ai_result_json[artis_det_0_key]['object_bbox'].update(current_json_data)

                current_json_data = {str(index_det_0):int(each_bbox_output_det_0[4])}
                cc.artis_ai_result_json[artis_det_0_key]['object_detail'].update(current_json_data)

                current_json_data = {str(index_det_0): str(each_bbox_output_det_0[5])}
                cc.artis_ai_result_json[artis_det_0_key]['object_score'].update(current_json_data)

                # fill contours
                if artis_det_0_key == 'seg':
                    current_json_data = {
                        str(index_det_0): {'cnt': len(each_bbox_output_det_0[7]), 'point': each_bbox_output_det_0[7]}}
                    cc.artis_ai_result_json[artis_det_0_key]['object_contour'].update(current_json_data)
            cc.artis_ai_result_json[artis_det_0_key]['result_info'] = {'object_total_cnt': len(bbox_output_det_0)}
            
            # Det1 : RGB 결과
            artis_det_1_key = "seg_oc" if self.ai_combine_mode == cc.artis_ai_model_mode[
                "mode_seg_with_seg"] else "rgb"
            for index_det_1, each_bbox_output in enumerate(bbox_output):
                current_json_data = {str(index_det_1):[int(each_bbox_output[0]), int(each_bbox_output[1]),
                                                     int(each_bbox_output[2]), int(each_bbox_output[3])]}
                cc.artis_ai_result_json[artis_det_1_key]['object_bbox'].update(current_json_data)

                current_json_data = {str(index_det_1):int(each_bbox_output[4])}
                cc.artis_ai_result_json[artis_det_1_key]['object_detail'].update(current_json_data)

                current_json_data = {str(index_det_1): str(each_bbox_output[5])}
                cc.artis_ai_result_json[artis_det_1_key]['object_score'].update(current_json_data)

                # contour
                if artis_det_1_key == 'seg_oc':
                    current_json_data = {
                        str(index_det_1): {'cnt': len(each_bbox_output[7]), 'point': each_bbox_output[7]}}
                    cc.artis_ai_result_json[artis_det_1_key]['object_contour'].update(current_json_data)

            cc.artis_ai_result_json[artis_det_1_key]['result_info'] = {'object_total_cnt': len(bbox_output)}

            if flag_bbox_valid:
                cc.artis_ai_current_log = f' ***** {artis_det_0_name} OD Det and {artis_det_1_name} OC Det Results are Valid *****'
                make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)
            else:
                cc.artis_ai_current_log = f'***** Invalid Results between {artis_det_0_name} OD Det and {artis_det_1_name} OC Det *****'
                make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)                
                for each_invalid_bbox in invalid_bbox_integrate:
                    cc.artis_ai_current_log = f'Invalid BBox : {int(each_invalid_bbox[0])}, {int(each_invalid_bbox[1])}, {int(each_invalid_bbox[2])}, {int(each_invalid_bbox[3])}'
                    make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)
                    ###print('[Artis_AI] Invalid BBox : %d, %d, %d, %d' % (int(each_invalid_bbox[0]), int(each_invalid_bbox[1]),
                    ###                                                        int(each_invalid_bbox[2]), int(each_invalid_bbox[3])))
        else:
            det0_time = 0
            cls_time = 0

            self.timer_starter.record()
            if self.ai_combine_mode == cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:   #feature matching mode
                det_key_name = "RGB/D OD"
                is_det, bbox_output, del_cnt = self.det_cls(preloadedImage.image, False, run_mode)
                bbox_output.sort(key=lambda x:((int(x[3]) + int(x[1])) // 2, (int(x[2]) + int(x[0])) // 2))

                tmp_output = []
                for idx, bbox in enumerate(bbox_output):
                    crop_img = preloadedImage.original_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    if isinstance(bbox, np.ndarray):
                        bbox = bbox.tolist()
                    bbox.append(crop_img)
                    tmp_output.append(bbox)
                    #cv2.imwrite(f"crop_{idx + 1}.png", crop_img)
                bbox_output = tmp_output
            elif self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_feature_matching"]:
                det_key_name = "SG OD"
                if self.ai_sg_model == "yolac":
                    is_det0, bbox_output, del_cnt = self.inference_seg_yolac(run_mode, preloadedImage.sg_frame, preloadedImage.sg_batch, preloadedImage.original_image)
                else:
                    is_det0, bbox_output, del_cnt = self.inference_seg_yolo(run_mode, False, preloadedImage.model_input)
                start_time = time.time()
                bbox_output.sort(key=lambda x:((int(x[3]) + int(x[1])) // 2, (int(x[2]) + int(x[0])) // 2))
                self.save_seg(imgs[1], os.path.dirname(imgs[1]), preloadedImage.sg_depth_img, bbox_output, True, True)
                print(f"{CurrentDateTime(0)} [Artis_AI][SG] 9. save seg : {(time.time() - start_time) * 1000:0.0f}")
                
                # ========== Image Fusion ([5]mode_seg_with_feature_matching) ==========
                image_fusion_enable = cc.artis_ai_json_config.get("image_fusion", False)
                if image_fusion_enable:
                    def handler_func(fused_image):
                        self._update_bbox_crops_for_feature_matching(bbox_output, fused_image)
                        cc.artis_ai_current_log = f'Image Fusion 완료 - FM 모드'
                        make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
                    
                    self._apply_image_fusion(imgs, bbox_output, handler_func)
                else:
                    cc.artis_ai_current_log = f'Image Fusion 스킵: image_fusion=false'
                    make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
                # ========== Image Fusion 끝 ==========
            else:
                det_key_name = "RGB/D OC"
                is_det, bbox_output, del_cnt = self.det_cls(preloadedImage.image, True, run_mode)
                bbox_output.sort(key=lambda x:((int(x[3]) + int(x[1])) // 2, (int(x[2]) + int(x[0])) // 2))

            ##### Sort By Center Point Ascending Order #####
            ##bbox_output.sort(key=lambda x:((int(x[3]) + int(x[1])) // 2, (int(x[2]) + int(x[0])) // 2))
            ##### Sort By Center Point Ascending Order #####

            self.timer_ender.record()
            torch.cuda.synchronize()
            det_time = self.timer_starter.elapsed_time(self.timer_ender)

            cc.artis_ai_current_log = f'Finish {det_key_name} Inference'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', False)

            if is_debug:
                # self.save_imgs(imgs[1], save_path, bbox_output)
                img1 = self.save_imgs(imgs[1], save_path, bbox_output, 'RGB')
            
            flag_bbox_valid = True
            if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_only"]:
                # ========= 미인식 결과 처리 ========= #
                if self.ai_thresh_mode_oc and del_cnt > 0:
                    flag_bbox_valid = False
                    for each_del_output in bbox_output:
                        if each_del_output[6] == False:
                            invalid_bbox_integrate.append(each_del_output)
                            invalid_bbox_rgb.append(each_del_output)
                # ========= 미인식 결과 처리 ========= #

                # Ignore Depth Detection Result to Prevent Errors due to Depth ROI #
                if self.flag_ignore_depth:
                    flag_bbox_valid = True

                #current_time = datetime.now()
                #inference_result_time = current_time.strftime('%Y%m%d%H%M%S%f')
                if flag_bbox_valid:
                    cc.artis_ai_current_error_code = '0x00000000'
                    cc.artis_ai_current_error_reason = ''
                    cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
                else:
                    #error_key = 'detect_untrained_view' if len(del_output) else 'detect_untrained_item'
                    error_key = 'detect_untrained_item'
                    cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['inference'][error_key]))
                    cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['inference'][cc.artis_ai_error_code['inference'][error_key]]
                    cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
            else:
                cc.artis_ai_current_error_code = '0x00000000'
                cc.artis_ai_current_error_reason = ''
                cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)

            cc.artis_ai_result_json['artis_error_info']['object_total_cnt'] = len(invalid_bbox_integrate)
            for index, each_invalid_bbox in enumerate(invalid_bbox_integrate):
                current_json_data = {str(index): [int(each_invalid_bbox[0]), int(each_invalid_bbox[1]),
                                                  int(each_invalid_bbox[2]), int(each_invalid_bbox[3])]}
                cc.artis_ai_result_json['artis_error_info']['object_bbox'].update(current_json_data)

            #cc.artis_ai_result_json['artis_result_info'] = {'object_total_cnt': len(bbox_output)}
            real_idx = 0
            for index, each_bbox_output in enumerate(bbox_output):
                if int(each_bbox_output[4]) not in self.sg_except_class and each_bbox_output[6] == True:
                    current_json_data = {str(real_idx): [int(each_bbox_output[0]), int(each_bbox_output[1]),
                                                    int(each_bbox_output[2]), int(each_bbox_output[3])]}
                    cc.artis_ai_result_json['artis_object_bbox'].update(current_json_data)

                    current_json_data = {str(real_idx): str(each_bbox_output[5])}
                    cc.artis_ai_result_json['artis_object_score'].update(current_json_data)
                    real_idx += 1
            ##print(cc.artis_ai_result_json['artis_object_bbox'])
            ##print(cc.artis_ai_result_json['artis_object_score'])
            cc.artis_ai_result_json['artis_result_info'] = {'object_total_cnt': real_idx}

            real_idx = 0
            if self.ai_combine_mode >= cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
                flag_class_valid = True
                for index, each_bbox_output in enumerate(bbox_output):
                    current_class_info = int(each_bbox_output[4])
                    if each_bbox_output[4] not in self.sg_except_class and each_bbox_output[6] == True:
                        if current_class_info >= cc.artis_ai_class_invalid_item:
                            current_json_data = {str(real_idx): cc.artis_ai_class_invalid_item}
                            flag_class_valid = False
                        else:
                            current_json_data = {str(real_idx):current_class_info}
                        cc.artis_ai_result_json['artis_object_detail'].update(current_json_data)
                        real_idx += 1

                if not flag_class_valid:
                    cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['inference']['detect_invalid_item']))
                    cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['inference'][cc.artis_ai_error_code['inference']['detect_invalid_item']]
                    cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
            else:
                flag_class_valid = True
                for index, each_bbox_output in enumerate(bbox_output):
                    current_class_info = int(each_bbox_output[4])
                    if int(each_bbox_output[4]) not in self.sg_except_class and each_bbox_output[6] == True:
                        if current_class_info >= cc.artis_ai_class_invalid_item:
                            current_json_data = {str(real_idx):cc.artis_ai_class_invalid_item}
                            flag_class_valid = False
                        elif each_bbox_output[6] == False:
                            current_json_data = {str(real_idx):cc.artis_ai_class_untrained_view}
                        else:
                            current_json_data = {str(real_idx):current_class_info}
                        cc.artis_ai_result_json['artis_object_detail'].update(current_json_data)
                        real_idx += 1

                if not flag_class_valid:
                    cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['inference']['detect_invalid_item']))
                    cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['inference'][cc.artis_ai_error_code['inference']['detect_invalid_item']]
                    cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)

            artis_det_key = "depth" if self.ai_combine_mode == cc.artis_ai_model_mode["mode_depth_with_feature_matching"] else "seg"
            for index_rgb, each_bbox_output in enumerate(bbox_output):
                current_json_data = {str(index_rgb): [int(each_bbox_output[0]), int(each_bbox_output[1]),
                                                      int(each_bbox_output[2]), int(each_bbox_output[3])]}
                cc.artis_ai_result_json['rgb']['object_bbox'].update(current_json_data)
                if self.ai_combine_mode >= cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
                    cc.artis_ai_result_json[artis_det_key]['object_bbox'].update(current_json_data)

                current_json_data = {str(index_rgb): int(each_bbox_output[4])}
                cc.artis_ai_result_json['rgb']['object_detail'].update(current_json_data)
                if self.ai_combine_mode >= cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
                    cc.artis_ai_result_json[artis_det_key]['object_detail'].update(current_json_data)

                current_json_data = {str(index_rgb): str(each_bbox_output[5])}
                cc.artis_ai_result_json['rgb']['object_score'].update(current_json_data)
                if self.ai_combine_mode >= cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
                    cc.artis_ai_result_json[artis_det_key]['object_score'].update(current_json_data)

                # fill contours
                if self.ai_combine_mode == cc.artis_ai_model_mode["mode_seg_with_feature_matching"]:
                    current_json_data = {
                        str(index_rgb): {'cnt': len(each_bbox_output[7]), 'point': each_bbox_output[7]}}
                    cc.artis_ai_result_json[artis_det_key]['object_contour'].update(current_json_data)

            cc.artis_ai_result_json['rgb']['result_info'] = {'object_total_cnt': len(bbox_output)}
            if self.ai_combine_mode >= cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
                cc.artis_ai_result_json[artis_det_key]['result_info'] = {'object_total_cnt': len(bbox_output)}

            if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_only"]:
                if not flag_bbox_valid:
                    cc.artis_ai_current_log = f'***** Invalid Results of RGB Det *****'
                    make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)
                    for each_invalid_bbox in invalid_bbox_integrate:
                        cc.artis_ai_current_log = f'Invalid BBox : {int(each_invalid_bbox[0])}, {int(each_invalid_bbox[1])}, {int(each_invalid_bbox[2])}, {int(each_invalid_bbox[3])}'
                        make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)
                cc.artis_ai_current_log = f'RGB {len(bbox_output)} 개 인식, {det_time} ms'
            else:
                cc.artis_ai_current_log = f'{det_key_name} {len(bbox_output)} 개 인식, {det_time} ms'
            make_artis_ai_log(cc.artis_ai_current_log, 'debug', True)


        timer_total_ender.record()
        torch.cuda.synchronize()
        total_time = timer_total_starter.elapsed_time(timer_total_ender)

        max_detect_object_cnt = 50
        obj_idx = 0
        my_string = \
            "0x02" + "|" + \
            "0x02" + "|" + \
            "OK" + "|" + \
            str(len(bbox_output) + len(invalid_bbox_det_0)) + "|"
        for i in range(len(bbox_output)):
            my_string += str(int(bbox_output[i][4])) + "|" + \
                    str(round(float(bbox_output[i][5]), 2)) + "|" + \
                    str(int(bbox_output[i][0])) + "|" + \
                    str(int(bbox_output[i][1])) + "|" + \
                    str(int(bbox_output[i][2])) + "|" + \
                    str(int(bbox_output[i][3])) + "|"
            obj_idx += 1

        for i in range(len(invalid_bbox_det_0)):
            my_string += str(cc.artis_ai_class_untrained_item) + "|" + \
                         str(round(float(invalid_bbox_det_0[i][5]), 2)) + "|" + \
                         str(int(invalid_bbox_det_0[i][0])) + "|" + \
                         str(int(invalid_bbox_det_0[i][1])) + "|" + \
                         str(int(invalid_bbox_det_0[i][2])) + "|" + \
                         str(int(invalid_bbox_det_0[i][3])) + "|"
            obj_idx += 1

        while obj_idx < max_detect_object_cnt:
            my_string += "0|" + \
                        "0|" + \
                        "0|" + \
                        "0|" + \
                        "0|" + \
                        "0|"
            obj_idx += 1

        #if cc.artis_ai_model_mode["mode_rgb_only"] < self.ai_combine_mode < cc.artis_ai_model_mode["mode_rgb_with_seg"]:
        if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_depth"]:
            obj_idx_det_0 = 0
            my_string += str(len(bbox_output_det_0)) + "|"
            for i in range(len(bbox_output_det_0)):
                my_string += (str(int(bbox_output_det_0[i][4])) + "|" +
                            str(round(float(bbox_output_det_0[i][5]), 2)) + "|" +
                            str(int(bbox_output_det_0[i][0])) + "|" +
                            str(int(bbox_output_det_0[i][1])) + "|" +
                            str(int(bbox_output_det_0[i][2])) + "|" +
                            str(int(bbox_output_det_0[i][3])) + "|"
                            )
                obj_idx_det_0 += 1

            while obj_idx_det_0 < max_detect_object_cnt:
                my_string += "0|" + \
                            "0|" + \
                            "0|" + \
                            "0|" + \
                            "0|" + \
                            "0|"
                obj_idx_det_0 += 1
        
        my_string += str(round(det_time/1000, 2)) + "|"
        my_string += str(round(cls_time/1000, 2)) + "|"
        #my_string += str(round(total_time, 2)) + "|"
        #my_string += str(0) + "|"
        #my_string += "0x03"
        
        #my_string = "0x02|0x02|" + json.dumps(result_json, indent=2) + "|0x03"
        #my_string = "0x02|0x02|" + json.dumps(result_json) + "|0x03"

        #with open('/home/nvidia/kisan/JetsonMan/workspace/Debug.json', 'w', encoding='utf-8') as debug_json:
        #    json.dump(compare_result_json, debug_json, ensure_ascii=False, indent=4)

        del timer_total_starter, timer_total_ender

        # mode_rgb_with_depth에서만 bbox_output + invalid_bbox_det_0를 합쳐서 반환
        if self.ai_combine_mode == cc.artis_ai_model_mode["mode_rgb_with_depth"]:
            bbox_all = bbox_output + invalid_bbox_det_0
        else:
            bbox_all = invalid_bbox_integrate

        ###return my_string, total_time / 1000, bbox_output, send_result_json, self.send_result_format
        return my_string, total_time, bbox_output, self.send_result_format, self.ai_combine_mode, bbox_all

if __name__=="__main__":
    import common_config as cc
    from inference import make_default_result_json
    from feature_matching import Feature_Matching

    config_file_path = "./kisan_config.json"
    k1 = Kisane(config_file_path)
    f1 = Feature_Matching(config_file_path)
    k1.warm_up(5, 0.1, 0.1)

    cc.artis_ai_result_json = make_default_result_json(cc.artis_ai_result_json)
    cam1_path = "./sample/Cam_1_Color.jpg"
    cam2_path = "./sample/Cam_2_Color.jpg"
    imgs = [cam2_path, cam2_path, cam1_path]
    for i in range(1):
        send_str, total_time, bbox_output, send_result_format, ai_combine_mode, box_all = k1.inference(imgs, True, "./", "")
        if int(k1.ai_combine_mode) >= cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
            f1.inference_by_feature_matching("./", bbox_output)
        with open("./result.json", 'w') as file:
            json.dump(cc.artis_ai_result_json, file, indent=4)
