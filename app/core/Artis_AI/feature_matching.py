import os
import time
import cv2
import json
import torch
import numpy as np
import warnings

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import mmcv
import common_config as cc
from common_config import make_artis_ai_log

import torch.nn.functional as F

from dino.vision_transformer import vit_small

import importlib.util
if importlib.util.find_spec("tensorrt") is not None:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    is_using_tensorrt_in_fm = True
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
else:
    is_using_tensorrt_in_fm = False

class Feature_Matching:
    def __init__(self, config_file_path):
        self.init(config_file_path)

    def init(self, config_file_path):
        print(f'[Artis_AI][feature_matching][Initialize]')

        self.gpu_id = 0
        self.device = "cuda:%d" % self.gpu_id
        self.db_feature = None
        self.filtered_db_feature = None
        self.feature_extractor = None
        self.context_feature_extractor = None
        self.feature_db_lookup = None
        self.flag_use_feature_matching = False
        self.flag_use_tensorrt_fm = False
        self.feature_threshold = 0.5
        self.timer_starter = torch.cuda.Event(enable_timing=True)
        self.timer_ender = torch.cuda.Event(enable_timing=True)

        if os.path.exists(config_file_path):
            print(f"Config File Path : {config_file_path}")
            self.ai_config_file_path = config_file_path
            #with open(self.ai_config_file_path, "r", encoding="utf-8") as json_file:
            #    self.artis_ai_json_config = json.load(json_file)
            with open(config_file_path, encoding="utf-8") as json_file:
                json_raw = json_file.read()
                self.artis_ai_json_config = json.loads(cc.remove_json_comments(json_raw))
            if "ai_tensorrt_feature" in self.artis_ai_json_config:
                self.flag_use_tensorrt_fm = bool(self.artis_ai_json_config["ai_tensorrt_feature"])
            if "ai_feature_threshold" in self.artis_ai_json_config:
                self.feature_threshold = float(self.artis_ai_json_config["ai_feature_threshold"])
        else:
            self.artis_ai_json_config = {}

        if self.flag_use_tensorrt_fm:
            if not os.path.exists(cc.artis_ai_feature_extract_model.replace(".pth", ".trt")):
                self.flag_use_tensorrt_fm = False

        feature_db_load_path = cc.artis_ai_feature_db_path
        if not os.path.exists(feature_db_load_path):
            print(f'[Artis_AI][feature_matching][load_feature_db] Not Exist Feature Database : {feature_db_load_path}')
            flag_load_feature_db = False
        else:
            self.db_feature = torch.load(feature_db_load_path, map_location='cpu').detach()
            self.db_feature = self.db_feature / self.db_feature.norm(dim=1, keepdim=True)
            flag_load_feature_db = True

        flag_load_feature_extractor = self.load_feature_extract_model(self.ai_config_file_path,
                                                                      self.flag_use_tensorrt_fm)
        flag_load_feature_db_lookup = False
        if os.path.exists(cc.artis_ai_feature_lookup_path):
            with open(cc.artis_ai_feature_lookup_path) as json_feature_lookup:
                data = json.load(json_feature_lookup)
            self.feature_db_lookup = data["mapper"]
            dict_db_size = data['size']
            sizes = [dict_db_size[str(i)] for i in range(len(dict_db_size))]
            self.feature_db_size = torch.tensor(sizes, dtype=torch.long)
            flag_load_feature_db_lookup = True

        if not flag_load_feature_db or not flag_load_feature_extractor or not flag_load_feature_db_lookup:
            self.flag_use_feature_matching = False
        else:
            self.flag_use_feature_matching = True


    def load_trt_engine_for_feature_model(self, trt_engine_path):
        with open(trt_engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine


    def load_feature_extract_model(self, ai_config_file_path, flag_use_trt):
        if flag_use_trt:
            feature_extract_model_path = cc.artis_ai_feature_extract_model.replace(".pth", ".trt")
            print(f"[Artis_AI][feature_matching][load_feature_extract_model] Use Tensor RT Model For Feature Extractor.")
        else:
            feature_extract_model_path = cc.artis_ai_feature_extract_model
            print(f"[Artis_AI][feature_matching][load_feature_extract_model] Use Normal Model For Feature Extractor.")
        if not os.path.exists(feature_extract_model_path):
            print(f'[Artis_AI][feature_matching][load_feature_extact_model] Not Exist Feature Extract Model : {feature_extract_model_path}')
            return False
        else:
            if flag_use_trt:
                self.feature_extractor = self.load_trt_engine_for_feature_model(feature_extract_model_path)
                self.context_feature_extractor = self.feature_extractor.create_execution_context()
            else:
                self.feature_extractor = vit_small(patch_size=8, num_classes=0)
                state = torch.load(feature_extract_model_path)
                self.feature_extractor.load_state_dict(state, strict=False)
                self.feature_extractor.eval().to(self.device)

            print(f"[Artis_AI][feature_matching][load_feature_extract_model] Load Feature Extract Model (DINO) Finished : {feature_extract_model_path}")
            print(f"[Artis_AI][feature_matching][load_feature_extract_model] Flag Use Tensor RT : {flag_use_trt}")
            return True

    def allocate_buffers_for_feature_extract_model(self, context, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(engine.num_bindings):
            binding_name = engine.get_binding_name(i)
            binding_shape = context.get_binding_shape(i)
            size = trt.volume(binding_shape)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding_name):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream


    def _convert_image_to_rgb(self, image: Image.Image):
        return image.convert("RGB")


    preprocess = Compose([
        ###_convert_image_to_rgb,
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    def resize_with_padding(self, img: Image.Image,
                            target_size: int = 224,
                            padding_color=(0, 0, 0)) -> Image.Image:

        orig_w, orig_h = img.size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized_img = img.resize((new_w, new_h), Image.LANCZOS)
        new_img = Image.new(img.mode, (target_size, target_size), padding_color)

        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        new_img.paste(resized_img, (paste_x, paste_y))

        return new_img

    def do_extract_feature_by_trt(self, context, bindings, inputs, outputs, stream):
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()
        return outputs[0]['host']

    def warm_up(self, loop_max, time_max):
        loop_cnt = 0
        cls_img_size, cls_img_ch = 448, 3
        warm_up_batch_size = 1
        total_processing_time = 0
        while loop_cnt < loop_max:
            self.timer_starter.record()
            total_warm_up_imgs = []
            total_feature_tensors = []
            warm_up_img = np.zeros((cls_img_size, cls_img_size, cls_img_ch), dtype=np.uint8)
            warm_up_img = warm_up_img[:, :, ::-1]  # BGR TO RGB
            warm_up_img = Image.fromarray(warm_up_img)
            warm_up_img = self.resize_with_padding(warm_up_img, target_size=224, padding_color=(0, 0, 0))
            warm_up_img = self._convert_image_to_rgb(warm_up_img)
            warm_up_img = self.preprocess(warm_up_img).unsqueeze(0)

            total_warm_up_imgs.append(warm_up_img)
            tensor_input = torch.cat(total_warm_up_imgs, dim=0)

            if self.flag_use_tensorrt_fm:
                self.context_feature_extractor.set_binding_shape(0, tuple(tensor_input.shape))
                inputs, outputs, bindings, stream = self.allocate_buffers_for_feature_extract_model(self.context_feature_extractor, self.feature_extractor)
                input_data = tensor_input.cpu().numpy().astype(np.float32).ravel()
                np.copyto(inputs[0]['host'], input_data)
                img_feature = self.do_extract_feature_by_trt(self.context_feature_extractor, bindings, inputs, outputs, stream)
                total_feature_tensors.append(img_feature)
                query = np.concatenate(total_feature_tensors, axis=0)
                query = torch.from_numpy(query).to('cpu')
            else:
                with torch.no_grad():
                    tensor_input = tensor_input.to(self.device, non_blocking=True).float()
                    img_feature = self.feature_extractor(tensor_input)
                    total_feature_tensors.append(img_feature.detach().cpu())
                query = torch.cat(total_feature_tensors, dim=0)
                query = query.detach().cpu()

            query_norm = query / query.norm(dim=1, keepdim=True)
            sim = query_norm @ self.db_feature.T
            values, indices = torch.max(sim, dim=1)

            loop_cnt += 1

            self.timer_ender.record()
            torch.cuda.synchronize()
            feature_processing_time = self.timer_starter.elapsed_time(self.timer_ender)
            total_processing_time += feature_processing_time
            feature_processing_time /= 1000
            if feature_processing_time < time_max:
                print(f"[Artis_AI][feature_matching][warm_up] GPUs are already standby. It is not need to warm up process")
                break

        print(f"[Artis_AI][feature_matching][warm_up] Warm up Total Processing Count & Time : {loop_cnt}, {total_processing_time} ms")


    def inference_by_feature_matching(self, save_path, bboxes):
        if not self.flag_use_feature_matching:
            print(f"[Artis_AI][feature_matching][inference_by_feature_matching] Can not Process Feature Matching. Please Check Feature DB & Feature Model File")
            return [], [], 0
        if not len(bboxes):
            print(f"[Artis_AI][feature_matching][inference_by_feature_matching] No Object To Classify. Please Check Images.")
            return [], [], 0

        #debug_image_save_path = "/home/nvidia/JetsonMan/debug_image/"
        #img_path = arg_img_path[1]
        #img = mmcv.imread(img_path)
        self.timer_starter.record()
        crop_img_list = []
        crop_img_size = []
        img_size_margin = 50
        idx_list = []

        with open(self.ai_config_file_path, encoding="utf-8") as cur_json_file:
            cur_json_data = cur_json_file.read()
            self.artis_ai_json_config = json.loads(cc.remove_json_comments(cur_json_data))
            if "ai_feature_threshold" in self.artis_ai_json_config:
                self.feature_threshold = float(self.artis_ai_json_config["ai_feature_threshold"])

        for index, bbox in enumerate(bboxes):
            #crop_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            '''
            ### For Test ###
            pre_angle = 0
            debug_image_save_name = f"FM_origin_crop_img_{index}.jpg"
            cv2.imwrite(os.path.join(debug_image_save_path, debug_image_save_name), crop_img)

            gray_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            bin_crop_img = cv2.adaptiveThreshold(gray_crop_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 10)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed_bin_crop_img = cv2.morphologyEx(bin_crop_img, cv2.MORPH_CLOSE, kernel, iterations=5)

            debug_image_save_name = f"FM_binary_origin_crop_img_{index}.jpg"
            cv2.imwrite(os.path.join(debug_image_save_path, debug_image_save_name), bin_crop_img)
            contours, _ = cv2.findContours(closed_bin_crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                contour_filtered = [max_contour]
            else:
                contour_filtered = []

            cnt = contour_filtered[0]

            ###cnt = contours[0]
            if len(cnt) < 5:
                print(f"Crop Image No.{index} - Not Enough to Detect Ellipse / total cnt of contours : {len(cnt)}")
                angle = pre_angle
            else:
                print(f"Crop Image No.{index} - Success to Detect Ellipse / total cnt of contours : {len(cnt)}")
                (_, _), (_, _), angle = cv2.fitEllipse(cnt)
                pre_angle = angle

            if abs(angle) > rotate_angle_thr:
                rotate_img = rotate_bound(crop_img, -angle)
            else:
                rotate_img = crop_img.copy()

            print(f"Crop Image No.{index} - Detected Angle : {angle}")
            rotate_patch_height, rotate_patch_width, _ = rotate_img.shape

            
            #min_x, min_y = rotate_patch_width, rotate_patch_height
            #max_x, max_y = 0, 0

            #for y in range(rotate_patch_height):
            #    for x in range(rotate_patch_width):
            #        if rotate_img[y, x, 0] | rotate_img[y, x, 1] | rotate_img[y, x, 2]:
            #            if min_x > x:
            #                min_x = x
            #            if min_y > y:
            #                min_y = y
            #            if max_x < x:
            #                max_x = x
            #            if max_y < y:
            #                max_y = y
            
            #rotate_img = rotate_img[min_y:max_y, min_x:max_x]
            
            #debug_image_save_name = f"FM_binary_deleteBG_crop_img_{index}.jpg"
            #cv2.imwrite(os.path.join(debug_image_save_path, debug_image_save_name), bin_crop_img)
            
            crop_img = rotate_img.copy()
            debug_image_save_name = f"FM_Rotated_crop_img_{index}.jpg"
            cv2.imwrite(os.path.join(debug_image_save_path, debug_image_save_name), crop_img)
            ### For Test ###
            '''
            if int(bbox[4]) >= 9000000 or bbox[6] != True:
                continue
            idx_list.append(index)
            patch = bbox[-1]
            #cv2.imwrite(f"{save_path}/patch_{index+1}.png", patch)
            crop_img = patch[:, :, ::-1]  # BGR TO RGB
            crop_img = Image.fromarray(crop_img)
            ### For Test ###
            crop_img_width, crop_img_height = crop_img.size
            crop_img_size.append([crop_img_height, crop_img_width])
            #crop_img = self.resize_with_padding(crop_img, target_size=224, padding_color=(255, 255, 255))
            crop_img = self.resize_with_padding(crop_img, target_size=224, padding_color=(0, 0, 0))
            crop_img = self._convert_image_to_rgb(crop_img)
            crop_img = self.preprocess(crop_img).unsqueeze(0)
            crop_img_list.append(crop_img)

        mapping_class_ids = []
        mapping_class_scores = []
        if len(crop_img_list) > 0:
            total_crop_img = torch.cat(crop_img_list, dim=0)
            batch_size = total_crop_img.shape[0]
            cnt_of_each_batch = 10
            feature_outputs = []

            #Extract Feature using Crop Images
            if self.flag_use_tensorrt_fm:
                for process_start_idx in range(0, batch_size, cnt_of_each_batch):
                    process_end_idx = min(process_start_idx + 10, batch_size)
                    sub_batch = total_crop_img[process_start_idx:process_end_idx]

                    self.context_feature_extractor.set_binding_shape(0, tuple(sub_batch.shape))
                    inputs, outputs, bindings, stream = self.allocate_buffers_for_feature_extract_model(
                        self.context_feature_extractor, self.feature_extractor)
                    input_data = sub_batch.cpu().numpy().astype(np.float32).ravel()
                    np.copyto(inputs[0]['host'], input_data)

                    out = self.do_extract_feature_by_trt(self.context_feature_extractor, bindings, inputs, outputs, stream)
                    feature_outputs.append(out)

                total_feature_outputs = np.concatenate(feature_outputs, axis=0)
                embedding_dim = int(len(total_feature_outputs) // batch_size)
                queries_np = np.array(total_feature_outputs).reshape(batch_size, embedding_dim)
                queries = torch.from_numpy(queries_np).to('cpu')
                queries = queries / queries.norm(dim=1, keepdim=True)
            else:
                with torch.no_grad():
                    for process_start_idx in range(0, batch_size, cnt_of_each_batch):
                        process_end_idx = min(process_start_idx + 10, batch_size)
                        sub_batch = total_crop_img[process_start_idx:process_end_idx].to(self.device, non_blocking=True).float()

                        out = self.feature_extractor(sub_batch)
                        feature_outputs.append(out.detach().cpu())

                queries = torch.cat(feature_outputs, dim=0)
                queries = queries.detach().cpu()
                queries = queries / queries.norm(dim=1, keepdim=True)

            scores, class_ids = [], []
            total_crop_img_size = torch.tensor(crop_img_size)

            for query_idx, current_query in enumerate(queries):
                diff = torch.abs(self.feature_db_size - total_crop_img_size[query_idx])
                mask = (diff[:, 0] <= img_size_margin) & (diff[:, 1] <= img_size_margin)
                valid_idx = torch.where(mask)[0]
                valid_idx = valid_idx.to(self.db_feature.device)

                if len(valid_idx) == 0:
                    self.filtered_db_feature = self.db_feature
                    valid_idx = torch.arange(0, len(self.db_feature))
                else:
                    self.filtered_db_feature = self.db_feature[valid_idx]

                sim = current_query @ self.filtered_db_feature.T
                current_score, current_id = torch.max(sim, dim=0)
                global_current_id = valid_idx[current_id]
                scores.append(current_score)
                class_ids.append(global_current_id)

                print(f"[feature_matching] Input Idx : {query_idx}"
                      f" || Input Size : {crop_img_size[query_idx]}"
                      f" || Filtered DB Feature Cnt :  {len(self.filtered_db_feature)} / {len(self.db_feature)}")
                '''
                ##### Code For Debug #####
                print(f"[feature_matching] shape sim : {sim.shape}")
                print(f"[feature_matching] sim : {sim}")
                '''
            scores = torch.tensor(scores)
            class_ids = torch.tensor(class_ids)

            #for current_idx, (each_score, each_class_id) in enumerate(zip(scores, class_ids)):
            for (current_idx, each_score, each_class_id) in zip(idx_list, scores, class_ids):
                each_class_id = str(each_class_id.item())
                if float(each_score.item()) < self.feature_threshold:
                    mapping_class_id = cc.artis_ai_class_untrained_item
                    error_key = 'detect_untrained_item'
                    cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['inference'][error_key]))
                    cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['inference'][cc.artis_ai_error_code['inference'][error_key]]
                    cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
                else:
                    mapping_class_id = self.feature_db_lookup[each_class_id]

                each_score = str(each_score.item())

                real_idx = 0
                real_bbox = cc.artis_ai_result_json['rgb']['object_bbox'][str(current_idx)]
                for box_id, bbox in enumerate(cc.artis_ai_result_json['artis_object_bbox'].values()):
                    if bbox[0] == real_bbox[0] and bbox[1] == real_bbox[1] and bbox[2] == real_bbox[2] and bbox[3] == real_bbox[3]:
                        real_idx = box_id
                        break

                cc.artis_ai_result_json['artis_object_detail'].update({str(real_idx):mapping_class_id})
                cc.artis_ai_result_json['rgb']['object_detail'].update({str(current_idx):mapping_class_id})

                cc.artis_ai_result_json['artis_object_score'].update({str(real_idx):each_score})
                cc.artis_ai_result_json['rgb']['object_score'].update({str(current_idx):each_score})

                #mapping_class_ids.append(mapping_class_id)
                mapping_class_ids.append(each_class_id)
                mapping_class_scores.append(each_score)

        mapping_class_ids = np.array(mapping_class_ids)
        mapping_class_scores = np.array(mapping_class_scores)

        self.timer_ender.record()
        torch.cuda.synchronize()
        feature_matching_time = self.timer_starter.elapsed_time(self.timer_ender)
        print(f"[Artis_AI][inference_by_feature_matching] Feature Matching Finished.")
        return mapping_class_scores, mapping_class_ids, feature_matching_time













