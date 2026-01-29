import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import torch
from mmcv import Config
from mmdet.models import build_detector
from mmdet.utils import rfnext_init_model, build_dp, compat_cfg, replace_cfg_vals, update_data_root
from mmcv.runner import load_checkpoint

from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
import mvdet
from glob import glob
import numpy as np
import yaml
from camera.projection import get_bbox, project_homography_points, hungarian_matching
from utils.nms import nms
import cv2
from PIL import Image
from torchvision import transforms
import re

import matplotlib.pyplot as plt
import time

from classification.model.ClsNet import ClsNet1

class Kisane:
    def __init__(self):
        # Load Arguments
        self.homography_path = 'camera/r_output/matrix/homography.yaml'

        # num_classes = 200
        num_classes = 10
        self.id2class = {1:'0001', 2:'0002', 3:'0003', 4:'0018', 5:'0031', 6:'1005', 7:'1016', 8:'1026', 9:'1039', 10:'1047'}
        det_config_path = 'checkpoints/rm_config_100.py'
        det_checkpoint_path = "checkpoints/rm_epoch_100.pth"
        cls1_checkpoint_path = 'checkpoints/rm_model_best_one.pth.tar'

        self.set_seed(0)

        # Load Configure
        det_cfg = Config.fromfile(det_config_path)
        det_cfg = replace_cfg_vals(det_cfg)
        update_data_root(det_cfg)
        det_cfg = compat_cfg(det_cfg)

        # Load Model
        self.gpu_id = 0
        self.pipeline = self.load_pipeline(det_cfg)
        self.det_model = self.load_model(det_cfg, device_id=self.gpu_id,
                                         checkpoint_path=det_checkpoint_path)

        self.cls_model1 = ClsNet1(net='resnet34', num_classes=num_classes)
        self.cls_model1.load_state_dict((torch.load(cls1_checkpoint_path, map_location='cpu')['state_dict']))
        #self.cls_model1.to('cuda:%d' % self.gpu_id)
        self.cls_model1.to('cpu')
        self.cls_model1.eval()


    def warm_up(self, loop_max, det_max, cls_max):
        print("WARM_UP :")

        start_time = time.time()

        #timer_starter, timer_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        loop_cnt = 0
        det_time = 0
        while loop_cnt < loop_max:
            #timer_starter.record()
            start_time = time.time()
            dataset = self.load_dataset([np.zeros((800, 1200, 3))], self.pipeline, device_id=self.gpu_id)
            self.det_inference(dataset)
            #timer_ender.record()
            #torch.cuda.synchronize()
            #det_time = timer_starter.elapsed_time(timer_ender) / 1000
            det_time = time.time() - start_time
            print(det_time)

            if det_time < det_max:
                break

            loop_cnt += 1

        if loop_cnt == loop_max:
            return "0x02" + "|" + "0x01" + "|" + "NG" + "|" + str(det_max) + "|" + "0" + "|" + "0x03"

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ##inference_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        inference_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])

        #device = 'cuda:{}'.format(self.gpu_id)
        device = 'cpu'
        img_crop = np.zeros((244, 244, 3))
        img_crop = Image.fromarray(np.uint8(img_crop)).convert('RGB')
        img_crop = inference_transform(img_crop)
        img_crop = torch.unsqueeze(img_crop, 0)
        img_crop = img_crop.to(device)
        loop_cnt = 0
        cls_time = 0
        while loop_cnt < loop_max:
            #timer_starter.record()
            start_time = time.time()
            self.cls_model1(img_crop)
            #timer_ender.record()
            #torch.cuda.synchronize()
            #cls_time = timer_starter.elapsed_time(timer_ender) / 1000
            cls_time = time.time() - start_time
            print(cls_time)

            if cls_time < cls_max:
                break

            loop_cnt += 1

        if loop_cnt == loop_max:
            return "0x02" + "|" + "0x01" + "|" + "NG" + "|" + str(det_time) + "|" + str(cls_max) + "|" + "0x03"

        return "0x02" + "|" + "0x01" + "|" + "OK" + "|" + str(round(det_time, 2)) + "|" + str(round(cls_time, 2)) + "|" + "0x03"

    def set_seed(self,seed):
        torch.backends.cudnn.benchmark = True

    def load_pipeline(self,cfg):
        cfg.data.test.pipeline[0].type = 'LoadImage'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        test_pipeline = Compose(cfg.data.test.pipeline)
        return test_pipeline

    def load_model(self, cfg, device_id=0, checkpoint_path=''):
        # Load pre-trained models
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        rfnext_init_model(model, cfg=cfg)
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']

        # CUDA set
        #model = build_dp(model, 'cuda', device_ids=[device_id])
        model = build_dp(model, 'cpu', device_ids=[device_id])

        # Freeze
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def load_dataset(self,imgs, test_pipeline, device_id=0):
        datas = []
        for img in imgs:
            data = test_pipeline(img)
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(imgs))
        # just get the actual data from DataContainer
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]

        # scatter to specified GPU
        #data = scatter(data, ['cuda:%d' % device_id])[0]
        return data

    def det_inference(self, data):
        with torch.no_grad():
            results = self.det_model(return_loss=False, rescale=True, **data)[0]
        return results


    def detection(self, img_list):
        bbox_output = []
        bbox_len = []
        for img in img_list:
            dataset = self.load_dataset([img], self.pipeline, device_id=self.gpu_id)

            # Inference
            results = self.det_inference(dataset)  # x1,y1,x2,y2,score

            threshold = 0.2
            bbox_result = results[0][results[0][:, 4] > threshold]
            print(bbox_result)

            nms_threshold = 0.5
            selected_indices = nms(bbox_result[:, :4], bbox_result[:, 4], overlap_threshold=nms_threshold)
            bbox_result = bbox_result[selected_indices]

            bbox_output.append(bbox_result)
            bbox_len.append(len(bbox_result))
            os.makedirs(os.path.join('tmp/', 'bbox'), exist_ok=True)
            np.save(os.path.join('tmp/', 'bbox', os.path.basename(img).replace('.png', '.npy')), bbox_result)

        if len(bbox_len) == len(img_list):
            for bl in bbox_len:
                if bl <= 0:
                    print("Detection Fail")
                    return False, bbox_output
            return True, bbox_output
        else:
            print("Detection Fail")
            return False, bbox_output
        

    def classification1(self, img_list, bbox_output):

        match_result = {}

        if len(bbox_output[0]):
            for ix in range(len(bbox_output[0])):
                match_result['obj%d' % ix] = {'left': [], 'top': [], 'right': []}

                match_result['obj%d' % ix]['top'] = [bbox_output[0][ix][0], bbox_output[0][ix][1],
                                                     bbox_output[0][ix][2],
                                                     bbox_output[0][ix][3]]

        inference_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
        #device = 'cuda:{}'.format(self.gpu_id)
        device = 'cpu'

        image_T = img_list[0]
        imgT = Image.open(image_T)

        cls_result = {}
        for ob in match_result:
            if len(match_result[ob]['top']) == 4:
                imgT_crop = imgT.crop(match_result[ob]['top'])
            else:
                imgT_crop = np.zeros((244, 244, 3))

            imgT_crop= inference_transform(imgT_crop)
            imgT_crop= torch.unsqueeze(imgT_crop, 0)
            imgT_crop= imgT_crop.to(device)

            outputs = self.cls_model1(imgT_crop)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confs, predicted = torch.max(probs, 1)
            # _, predicted = outputs.max(1)

            tmp_result = self.id2class[predicted.item() + 1]

            cls_result[ob] = [tmp_result, str(round(confs.item(), 2))]

        return match_result, cls_result


    def adjust(self, point, minPoint, maxPoint):
        if point <= minPoint:
            return 0

        if point >= maxPoint:
            return maxPoint

        return point

    def save_imgs(self, img_list, save_path, cls_result, match_result):
        if len(img_list) == 1:
            img_t = cv2.imread(img_list[0])
        else:
            img_l, img_t, img_r = cv2.imread(img_list[0]), cv2.imread(img_list[1]), cv2.imread(img_list[2])
        color_plt = [(0, 0, 150), (0, 150, 0), (150, 0, 0), (0, 150, 150), (150, 0, 150), (150, 150, 0),
                     (150, 150, 150)]

        idx = 0
        for ob in match_result:
            lbl = cls_result[ob][0]
            score = cls_result[ob][1]
            color_idx = color_plt[idx % len(color_plt)]
            if len(match_result[ob]['left']):
                img_l = cv2.rectangle(img_l, (int(match_result[ob]['left'][0]), int(match_result[ob]['left'][1])),
                                  (int(match_result[ob]['left'][2]), int(match_result[ob]['left'][3])),
                                  color_idx, 2)
                img_l = cv2.putText(img_l, lbl,
                                    (self.adjust(int(match_result[ob]['left'][0]), 0, 1180),
                                     (self.adjust(int(match_result[ob]['left'][1]) - 10, 0, 680))),
                                    cv2.FONT_HERSHEY_SIMPLEX, color=color_idx, thickness=2, fontScale=1)
                img_l = cv2.putText(img_l, '(' + score + ')',
                                    (self.adjust(int(match_result[ob]['left'][0]), 0, 1180),
                                     (self.adjust(int(match_result[ob]['left'][1]) + 20, 0, 680))),
                                    cv2.FONT_HERSHEY_SIMPLEX, color=color_idx, thickness=2, fontScale=0.7)

            if len(match_result[ob]['top']):
                img_t = cv2.rectangle(img_t, (int(match_result[ob]['top'][0]), int(match_result[ob]['top'][1])),
                                  (int(match_result[ob]['top'][2]), int(match_result[ob]['top'][3])), color_idx,
                                  2)
                img_t = cv2.putText(img_t, lbl,
                                    (self.adjust(int(match_result[ob]['top'][0]), 0, 1180),
                                     (self.adjust(int(match_result[ob]['top'][1]) - 10, 0, 680))),
                                    cv2.FONT_HERSHEY_SIMPLEX, color=color_idx, thickness=2, fontScale=1)
                img_t = cv2.putText(img_t, '(' + score + ')',
                                    (self.adjust(int(match_result[ob]['top'][0]), 0, 1180),
                                     (self.adjust(int(match_result[ob]['top'][1]) + 20, 0, 680))),
                                    cv2.FONT_HERSHEY_SIMPLEX, color=color_idx, thickness=2, fontScale=0.7)

            if len(match_result[ob]['right']):
                img_r = cv2.rectangle(img_r, (int(match_result[ob]['right'][0]), int(match_result[ob]['right'][1])),
                                  (int(match_result[ob]['right'][2]), int(match_result[ob]['right'][3])),
                                  color_idx, 2)

                img_r = cv2.putText(img_r, lbl,
                                (self.adjust(int(match_result[ob]['right'][0] - 10), 0, 1180),
                                (self.adjust(int(match_result[ob]['right'][1]) - 10, 0, 680))),
                                cv2.FONT_HERSHEY_SIMPLEX, color=color_idx, thickness=2, fontScale=1)
                img_r = cv2.putText(img_r, '(' + score + ')',
                                (self.adjust(int(match_result[ob]['right'][0]), 0, 1180),
                                (self.adjust(int(match_result[ob]['right'][1]) + 20, 0, 680))),
                                cv2.FONT_HERSHEY_SIMPLEX, color=color_idx, thickness=2, fontScale=0.7)

            if len(img_list) == 1:
                cv2.imwrite(os.path.join(save_path, os.path.basename(img_list[0])), img_t)
            else:
                cv2.imwrite(os.path.join(save_path, os.path.basename(img_list[0])), img_l)
                cv2.imwrite(os.path.join(save_path, os.path.basename(img_list[1])), img_t)
                cv2.imwrite(os.path.join(save_path, os.path.basename(img_list[2])), img_r)
            idx += 1

    def inference(self, imgs, is_debug, save_path):
        out_str = {'result': []}

        #timer_total_starter, timer_total_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #timer_total_starter.record()
        start_time = time.time()

        #self.timer_starter.record()

        is_det, bbox_output = self.detection(imgs)
        #self.timer_ender.record()
        #torch.cuda.synchronize()
        # timer_starter.elapsed_time(timer_ender) / 1000

        if is_det:
            match_result, cls_result = self.classification1(imgs, bbox_output)
        else:
            match_result = []
            cls_result = []

        #timer_total_ender.record()
        #torch.cuda.synchronize()
        #total_time = timer_total_starter.elapsed_time(timer_total_ender) / 1000
        total_time = time.time() - start_time
        print(total_time)

        obj_idx = 0
        my_string = \
            "0x02" + "|" + \
            "0x02" + "|" + \
            str(len(match_result)) + "|"
        for ob in match_result:
            my_string += cls_result[ob][0] + "|" + \
                         cls_result[ob][1] + "|" + \
                        str(int(match_result[ob]['top'][0])) + "|" + \
                        str(int(match_result[ob]['top'][1])) + "|" + \
                        str(int(match_result[ob]['top'][2])) + "|" + \
                        str(int(match_result[ob]['top'][3])) + "|"
            obj_idx += 1

        while obj_idx < 50:
            my_string += "0|" + \
                        "0|" + \
                        "0|" + \
                        "0|" + \
                        "0|"
            obj_idx += 1
        my_string += str(round(total_time, 2)) + "|"
        my_string += "0x03"

        if is_debug:
            print("SAVE_PATH : ", save_path, "\n")
            self.save_imgs(imgs, save_path, cls_result, match_result)

        return my_string
