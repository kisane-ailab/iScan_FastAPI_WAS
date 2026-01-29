import os
from dino.vision_transformer import vit_small

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import csv
import torch
import time
from PIL import Image
import numpy as np
import random

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from torchvision import transforms
import cv2
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import json
import glob

import torch.nn.functional as F

import pyzipper
from utils.crypto import load_and_decrypt

from datetime import datetime
import shutil

import importlib.util
if importlib.util.find_spec("tensorrt") is not None:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    is_using_tensorrt = True
else:
    is_using_tensorrt = False


train_key = load_and_decrypt("utils/a31_train.bin")
rotate_idx = 0
extracted_feature_dim = 384
invalid_class_no = 9999998
# 시드 고정 함수
# ------------------------

def CurrentDateTime(format_type=0):
    if format_type == 0:
        t = time.time()
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}" + "]"
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    return timestamp

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(777)


# 동적 배치 처리를 위한 버퍼 할당 함수
def allocate_buffers(engine, max_batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        if -1 in binding_shape:
            binding_shape = list(binding_shape)
            binding_shape[0] = max_batch_size
        else:
            binding_shape = tuple(binding_shape)
        size = trt.volume(binding_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem, "name": binding, "shape": binding_shape})
        else:
            outputs.append({"host": host_mem, "device": device_mem, "name": binding, "shape": binding_shape})
    return inputs, outputs, bindings, stream

# 이미지 전처리 함수
# ------------------------
class RandomCutout(object):
    def __init__(self, n_holes=1, length=70, p=0.3, color=(0, 0, 0)):
        self.n_holes = n_holes
        self.length = length
        self.p = p
        self.color = color

    def __call__(self, img):
        if random.random() > self.p:
            return img
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(self.n_holes):
            x = random.randint(0, w)
            y = random.randint(0, h)
            x1 = max(0, x - self.length // 2)
            y1 = max(0, y - self.length // 2)
            x2 = min(w, x + self.length // 2)
            y2 = min(h, y + self.length // 2)
            draw.rectangle([x1, y1, x2, y2], fill=self.color)
        return img

class CyclicRotation:
    #def __init__(self, angles=[0, 45, 90, 135, 180, 225, 270, 315, 360], fill=(255, 255, 255)):
    def __init__(self, angles=[45, 90, 135, 180, 225, 270, 315], fill=(0, 0, 0)):
        self.angles = angles
        self.fill = fill
        self.index = 0

    def __call__(self, img):
        #angle = self.angles[self.index]
        #self.index = (self.index + 1) % len(self.angles)
        angle = self.angles[rotate_idx]
        return TF.rotate(img, angle=angle, fill=self.fill)

# Augmentation pipeline
def _convert_image_to_rgb(image):
    return image.convert("RGB")
augmentation_transforms = transforms.Compose([
    #CyclicRotation(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 0.3)),
    # 주석 처리
    # transforms.RandomHorizontalFlip(p=0.01),
    # transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0.1),
    # RandomCutout(n_holes=1, length=10, p=0.01, color=(255, 255, 255)), 
    _convert_image_to_rgb,
    #ToTensor(),
    #Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Original pipeline
def _convert_image_to_rgb(image):
    return image.convert("RGB")
original_transforms = transforms.Compose([
    _convert_image_to_rgb,
    #ToTensor(),
    #Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

final_transforms = transforms.Compose([
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Input preprocessing
def resize_with_padding(img: Image.Image,
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



###################################################################
def delete(args):

    ''' Tensor DB load'''
    combined_tensor = torch.load(args.feature_db_path)
    print(f"Loaded tensor with shape: {combined_tensor.shape}")
    try:
        indices_to_del = [int(i)-1 for i in args.modify_num.split(',')]
        print(f"Indices to delete: {indices_to_del}")
    except ValueError:
        print("Error: modify_num should contain integers separated by commas.")
        return

    '''   # Check if indices are valid  '''
    max_idx = combined_tensor.shape[0] - 1
    invalid_indices = [i for i in indices_to_del if i < 0 or i > max_idx]
    if invalid_indices:
        print(f"Invalid indices found: {invalid_indices}. Please provide valid indices between 0 and {max_idx}.")
        return

    # Replace
    for idx in indices_to_del:
        combined_tensor[idx] = torch.full_like(combined_tensor[idx], 100)
        print(f"Deleting index {idx} information from the tensor.")

    # # Delete
    # keep = [i for i in range(combined_tensor.size(0)) if i not in indices_to_del]
    # combined_tensor = combined_tensor[keep]
    # print(f"Deleted rows at indices {indices_to_del}; new shape is {combined_tensor.shape}")

    torch.save(combined_tensor, args.feature_db_path)
    print(f"Tensor DB saved to {args.feature_db_path}, {combined_tensor.shape}")


###################################################################
import argparse

'''
now = datetime.now()
        start_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{start_time}][make_feature][mode_all] Mode All - Making Feature DB Start")
        now = datetime.now()
        end_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{end_time}][make_feature][mode_all] Mode All - Making Feature DB Finish")
        print(f"Start : {start_time} End : {end_time}")
'''


import common_config as cc
import asyncio
from tqdm import tqdm

# 학습 로그
log_json = {
    "item_current": [],
    "item_new": [],
    "item_change": [],

    "ai_model_stage": "TRAIN_START",
    "ai_model_stage_eta": ["0", "0", "0"],
    "ai_model_error_code": "OK",

    "ai_model_version": "",
}

class MakeFeatureDB:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        cc.artis_ai_json_config = cc.get_config(self.config_file_path, cc.artis_ai_json_config)
        if "ai_tensorrt_feature" in cc.artis_ai_json_config and is_using_tensorrt:
            self.flag_use_tensorrt = bool(cc.artis_ai_json_config["ai_tensorrt_feature"])
        else:
            self.flag_use_tensorrt = False

        if self.flag_use_tensorrt:
            self.feature_extract_model_path = cc.artis_ai_feature_extract_model.replace(".pth", ".trt")
        else:
            self.feature_extract_model_path = cc.artis_ai_feature_extract_model
        self.ai_feature_mode = cc.artis_ai_json_config.get("ai_feature_mode", 1)

        print(
            f"{CurrentDateTime(0)}[MakeFeatureDB] Flag For Use Tensor RT Feature Extractor Model : {self.flag_use_tensorrt}")
        print(f"{CurrentDateTime(0)}[MakeFeatureDB] Load Feature Extractor Model Path : {self.feature_extract_model_path}")
        if not os.path.exists(self.feature_extract_model_path):
            print(
                f"{CurrentDateTime(0)}[MakeFeatureDB] Can not Load Feature Extractor Model. Please Check Path : {self.feature_extract_model_path}")
            #return False, f"Feature Extractor Model 로드 실패 : {self.feature_extract_model_path}"

        self.max_batch_size = 10
        self.aug_per_image = 10 # Number of augmentations per image
        self.max_images_for_apply = 10 # Flag for use tensor rt based feature extractor : default 10
        self.angle_per_rotate = 30 # Flag for use tensor rt based feature extractor : default 45
        self.img_per_product = 3 # Number of augmented images per product
        self.save_debug_image = False # Flag for save debug images when making feature db
        self.modify_num = '' # List of # of products to modify (comma-separated)

        self.C, self.H, self.W = 3, 224, 224

        self.db_root_path = None
        self.db_sync_report = {}

        self.final_save_path = cc.artis_ai_feature_db_path

        self.eta = 3600
        self.eta_pre = 3600
        self.start_time = 0

        self.load_model()

    def load_model(self):
        # ------------------------
        # TensorRT 엔진 로드 및 버퍼 할당
        # ------------------------
        if self.flag_use_tensorrt:
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)

            with open(self.feature_extract_model_path, "rb") as f:
                engine_data = f.read()

            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = engine.create_execution_context()

            # 버퍼 배치는 최대 10개로 설정
            # 버퍼 최대치는 모델 trt 변환시 사용하는 min~max 범위에서 자유롭게 설정 가능!
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(engine, self.max_batch_size)

            # 바인딩 인덱스 안전화
            in_name = self.inputs[0]["name"]
            out_name = self.outputs[0]["name"]
            self.in_idx = engine.get_binding_index(in_name)
            self.out_idx = engine.get_binding_index(out_name)
            print(f"{CurrentDateTime(0)}[MakeFeatureDB] Load Tensor RT based Feature Extractor Model")
        # ------------------------
        # 일반 모델 로드
        # ------------------------
        else:
            self.context = vit_small(patch_size=8, num_classes=0)
            print(f"context = vit_small(patch_size=8, num_classes=0)")
            weights = torch.load(self.feature_extract_model_path)
            print(f"weights = torch.load(engine_file_path)")
            self.context.load_state_dict(weights, strict=False)
            print(f"context.load_state_dict(weights, strict=False)")
            self.context.eval().to("cuda")
            print(f"{CurrentDateTime(0)}[MakeFeatureDB][main] Load Normal Feature Extractor Model")

    def set_db(self, files, db_root_path):
        self.db_root_path = os.path.join(db_root_path, "db").replace("\\", "/")
        if not os.path.exists(self.db_root_path):
            return False, f"DB Path가 존재하지 않습니다 : {self.db_root_path}"

        self.db_sync_report = {}
        self.train_log = log_json.copy()
        for file in files:
            file_name = file["original_name"]
            raw_data = file["data"]
            if not ".json" in file_name and not "db_sync_report" in file_name:
                continue
            processed_data = None
            try:
                # UTF-8 디코딩 시도, 실패하면 다른 인코딩 시도
                try:
                    text_data = raw_data.decode('utf-8')
                except UnicodeDecodeError:
                    # UTF-8 디코딩 실패 시 다른 인코딩 시도
                    try:
                        text_data = raw_data.decode('utf-8-sig')  # BOM 제거
                    except UnicodeDecodeError:
                        try:
                            text_data = raw_data.decode('latin-1')  # fallback
                        except UnicodeDecodeError:
                            # 마지막 수단으로 바이너리 데이터를 그대로 사용
                            text_data = raw_data.decode('utf-8', errors='ignore')
                processed_data = json.loads(cc.remove_json_comments(text_data))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                print(f"{cc.CurrentDateTime(0)} [MakeFeatureDB] 파일을 JSON으로 파싱할 수 없습니다: {e}")
                return False, f"파일을 JSON으로 파싱할 수 없습니다: {e}"
            except Exception as e:
                print(f"{cc.CurrentDateTime(0)} [MakeFeatureDB] 설정 파일 오류: {e}")
                return False, f"설정 파일 오류: {e}"

            if processed_data is not None:
                self.db_sync_report = processed_data
                if self.ai_feature_mode == 1:
                    self.db_sync_report["file_new"] = []
                    if (self.db_sync_report.get("V2_single", {}) is None
                        or self.db_sync_report.get("V2_single", {}).get("file_new", None) is None
                        or len(self.db_sync_report["V2_single"]["file_new"]) == 0):
                        print(f"{cc.CurrentDateTime(0)} [MakeFeatureDB] 새로운 DB가 없습니다")
                        return False, f"새로운 DB가 없습니다"
                    else:
                        for filename in self.db_sync_report["V2_single"]["file_new"].keys():
                            if "Cam_2_Color.jpg" in filename or ".zip" in filename:
                                filename_new = os.path.join(db_root_path, filename).replace("\\", "/")
                                self.db_sync_report["file_new"].append(filename_new)
                print(f"file_new : {self.db_sync_report['file_new']}")

                self.train_log["item_current"] = self.db_sync_report.get("item_current", [])
                self.train_log["item_new"] = self.db_sync_report.get("item_new", [])
                self.train_log["item_change"] = self.db_sync_report.get("item_change", [])
                if "V2_single" in self.db_sync_report and self.db_sync_report["V2_single"] is not None:
                    self.train_log["item_current"] += self.db_sync_report["V2_single"].get("item_current", [])
                    self.train_log["item_new"] += self.db_sync_report["V2_single"].get("item_new", [])
                    self.train_log["item_change"] += self.db_sync_report["V2_single"].get("item_change", [])
                self.train_log["ai_model_version"] = "0.0.0"
                self.train_log["ai_model_stage_eta"] = ["1", "0", "0"]
                self.eta = 3600
                self.eta_pre = 3600

        print(f"train_log : {self.train_log}")
        return True, ""


    async def all(self):
        # ------------------------
        # 이미지 불러오기 및 배치별 추론
        # ------------------------

        self.start_time = time.time()

        flag_exist_object_patch = False
        flag_exist_object_img = False
        flag_exist_zip_db = False
        total_classes = {}
        total_classes_size = {}
        unique_classes = []

        class_folder = []
        total_class_list = []

        db_folder = self.db_root_path + "/V2_single/"
        tmp_save_path = self.final_save_path.replace(".pt", "_tmp.pt")

        if "single_patch" in db_folder:
            if os.path.exists(db_folder):
                if len(os.listdir(db_folder)):
                    total_class_list = os.listdir(db_folder)
                    class_folder = db_folder
                    flag_exist_object_patch = True
                else:
                    print(
                        f"{CurrentDateTime(0)}[make_feature][mode_all] Must Exist Class Folder in {db_folder}, Please Check the Path : {db_folder}")
                    return
        elif "V2_single" in db_folder:
            if os.path.exists(db_folder):
                if len(os.listdir(db_folder)):
                    total_class_list = os.listdir(db_folder)
                    class_folder = db_folder
                    flag_exist_object_img = True
                else:
                    print(
                        f"{CurrentDateTime(0)}[make_feature][mode_all] Must Exist Class Folder in {db_folder}, Please Check the Path : {db_folder}")
                    return
        else:
            if len(os.listdir(db_folder)):
                class_folder = db_folder
                flag_exist_zip_db = True
            else:
                print(
                    f"{CurrentDateTime(0)}[make_feature][mode_all] Must Exist Zip File in {db_folder}, Please Check the Path : {db_folder}")
                return

        if flag_exist_object_patch or flag_exist_object_img:
            class_names = [x for x in total_class_list if x != 'Base' and int(x) < 9000000]
            class_names.sort()
        else:
            class_names = []

        # 각 이미지당 생성할 augmentation 개수
        aug_per_image = self.aug_per_image
        num_augmented = aug_per_image - 1
        # print(f"Augmentations per image: {aug_per_image} (including original)")

        rotate_per_image = 361
        num_rotated = rotate_per_image - 1

        # 배치에 쌓일 이미지 텐서들을 저장할 리스트
        images_batch = []
        # 배치 번호 카운터
        batch_counter = 100000
        max_images_for_apply = self.max_images_for_apply
        angle_per_rotate = self.angle_per_rotate
        list_rotation_angle = list(range(0, 360, angle_per_rotate))
        print(f"{CurrentDateTime(0)}[make_feature][mode_all] Total Rotation Angle : {list_rotation_angle}")

        object_package_type = ['TYPE1_FRONT', 'TYPE1_SIDE', 'TYPE2_FRONT', 'TYPE2_SIDE', 'TYPE3_FRONT',
                               'TYPE3_SIDE']  # Non-Vinyl, Vinyl
        valid_class_id = 0

        if flag_exist_object_patch:
            with tqdm(object_package_type) as pbar:
                for current_object_package_type in object_package_type:
                    for cls_idx, cls_name in enumerate(class_names):
                        img_paths = glob.glob(os.path.join(class_folder, cls_name, current_object_package_type, '*', '*', '*', '*_patch1.png'))
                        if len(img_paths) > max_images_for_apply:
                            selected_img_paths = random.sample(img_paths, max_images_for_apply)
                        else:
                            selected_img_paths = img_paths.copy()

                        if len(selected_img_paths) == 0:
                            continue

                        if 'TYPE1' in current_object_package_type:
                            print(f"{CurrentDateTime(0)}[make_feature][mode_all] Make Feature Map for Class {cls_name} [Package Type : Non-Vinyl]")
                        elif 'TYPE2' in current_object_package_type:
                            print(f"{CurrentDateTime(0)}[make_feature][mode_all] Make Feature Map for Class {cls_name} [Package Type : Vinyl]")
                        else:
                            print(f"{CurrentDateTime(0)}[make_feature][mode_all] Make Feature Map for Class {cls_name} [Package Type : Unknown]")

                        print(f"{CurrentDateTime(0)}[make_feature][mode_all] Current DB Total Cnt : {len(img_paths)}")

                        for current_rotate_angle in list_rotation_angle:
                            view_list = []
                            total_current_class_feature = []
                            selected_imgs_height, selected_imgs_width = [], []
                            process_img_count = 0
                            for img_idx, img_path in enumerate(selected_img_paths):
                                try:
                                    img = cv2.imread(img_path)
                                except Exception as e:
                                    print(f"{CurrentDateTime(0)}[make_feature][mode_all] Error loading image {img_path}: {e}")
                                    continue

                                img = img[:, :, ::-1]
                                img = Image.fromarray(img)
                                size_check_img = img.copy()
                                pattern_img = resize_with_padding(img, target_size=224, padding_color=(0, 0, 0))

                                if current_rotate_angle == 0:
                                    transform_img = original_transforms(pattern_img)
                                else:
                                    transform_img = TF.rotate(pattern_img, angle=current_rotate_angle, fill=(0, 0, 0))
                                    transform_img = augmentation_transforms(transform_img)
                                    size_check_img = TF.rotate(size_check_img, angle=current_rotate_angle, fill=(0, 0, 0), expand=True)

                                transform_img = final_transforms(transform_img).unsqueeze(0)

                                np_size_check_img = np.array(size_check_img)
                                if np_size_check_img.shape[2] == 4:
                                    mask = (np_size_check_img[..., :3].sum(axis=2) != 0) & (np_size_check_img[..., 3] != 0)
                                else:
                                    mask = (np_size_check_img.sum(axis=2) != 0)

                                coords = np.argwhere(mask)
                                y0, x0 = coords.min(axis=0)
                                y1, x1 = coords.max(axis=0) + 1
                                fit_crop_size_check_img = size_check_img.crop((x0, y0, x1, y1))

                                current_transform_img_width, current_transform_img_height = fit_crop_size_check_img.size
                                selected_imgs_height.append(current_transform_img_height)
                                selected_imgs_width.append(current_transform_img_width)

                                view_list.append(transform_img)

                                if self.save_debug_image:
                                    debug_image_save_file_path = f"./debug_image/cls_{cls_name}_type_{current_object_package_type}_angle_{current_rotate_angle}_idx_{img_idx}.jpg"
                                    fit_crop_size_check_img.save(debug_image_save_file_path)

                                process_img_count += 1

                            current_img_mean_height = int(sum(selected_imgs_height) / len(selected_imgs_height))
                            current_img_mean_width = int(sum(selected_imgs_width) / len(selected_imgs_width))
                            print(f"{CurrentDateTime(0)}[make_feature][mode_all] [Class : {cls_name}] || [Total Applied Images : {process_img_count}]"
                                  f" || [Type : {current_object_package_type}] || [Rotate : {current_rotate_angle}]"
                                  f" || [Image Height : {current_img_mean_height} Image Width : {current_img_mean_width}]")

                            ### For Mean Feature Logic ###
                            for view in view_list:
                                images_batch.append(view)

                            if self.flag_use_tensorrt:
                                feature_tensors = []
                                batch_tensor_t = torch.cat(images_batch, dim=0).contiguous()

                                for process_start_idx in range(0, process_img_count, self.max_batch_size):
                                    process_end_idx = min(process_start_idx + self.max_batch_size, process_img_count)
                                    sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                    sub_batch_cnt = sub_batch.shape[0]

                                    batch_tensor = sub_batch.numpy()
                                    np.copyto(self.inputs[0]["host"][:batch_tensor.size], batch_tensor.ravel())
                                    self.context.set_binding_shape(self.in_idx, (sub_batch_cnt, self.C, self.H, self.W))  # input에 따른 동적
                                    cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
                                    self.context.execute_v2(bindings=self.bindings)
                                    cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
                                    self.stream.synchronize()

                                    out_shape = tuple(self.context.get_binding_shape(self.out_idx))
                                    output_np = self.outputs[0]["host"][:np.prod(out_shape)].reshape(out_shape)
                                    sub_batch_output_tensor = torch.from_numpy(output_np)
                                    feature_tensors.append(sub_batch_output_tensor)

                                feature_tensors = np.concatenate(feature_tensors, axis=0)
                            else:
                                feature_tensors = []
                                batch_tensor_t = torch.cat(images_batch, dim=0).contiguous().to("cuda")
                                with torch.no_grad():
                                    for process_start_idx in range(0, process_img_count, self.max_batch_size):
                                        process_end_idx = min(process_start_idx + self.max_batch_size, process_img_count)
                                        sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                        sub_batch_output_tensor = self.context(sub_batch)
                                        feature_tensors.append(sub_batch_output_tensor.detach().cpu())

                                feature_tensors = np.concatenate(feature_tensors, axis=0)

                            output_tensor = torch.from_numpy(np.array(feature_tensors).reshape(process_img_count, extracted_feature_dim))
                            images_batch = []
                            total_current_class_feature.append(output_tensor)

                            if len(total_current_class_feature):
                                combined_tensor = torch.cat(total_current_class_feature, dim=0)
                                combined_tensor = combined_tensor.view(-1, len(total_current_class_feature) * process_img_count, extracted_feature_dim)
                                combined_tensor = combined_tensor.mean(dim=1, keepdim=True)
                                if os.path.exists(tmp_save_path):
                                    if valid_class_id > 0:
                                        total_class_tensor_db = torch.load(tmp_save_path)
                                        if total_class_tensor_db.dim() == 3:
                                            total_class_tensor_db = total_class_tensor_db.squeeze(1)
                                        total_class_tensor_db = total_class_tensor_db.unsqueeze(1)
                                        total_class_tensor_db = torch.cat((total_class_tensor_db, combined_tensor), dim=0).squeeze(1)
                                        torch.save(total_class_tensor_db, tmp_save_path)
                                        #print(f"{CurrentDateTime(0)}[make_feature][mode_all] Making Feature DB Finished. shape info : {combined_tensor.shape} / {total_class_tensor_db.shape} class info : {cls_name} || rotation info : {current_rotate_angle}")
                                    else:
                                        os.remove(tmp_save_path)
                                        torch.save(combined_tensor, tmp_save_path)
                                        print(f"{CurrentDateTime(0)}[make_feature][mode_all] Remove Existing Feature DB & Make New Feature DB")
                                else:
                                    print(f"{CurrentDateTime(0)}[make_feature][mode_all] Making First Feature Finished. {combined_tensor.shape}")
                                    torch.save(combined_tensor, tmp_save_path)

                                total_classes[valid_class_id] = int(cls_name)
                                total_classes_size[valid_class_id] = [int(current_img_mean_height), int(current_img_mean_width)]
                                valid_class_id += 1
                    pbar.update(1)
                    progress = pbar.n / pbar.total
                    elapsed = pbar.format_dict['elapsed']
                    eta = (elapsed / progress) - elapsed if progress > 0 else None
                    self.eta = eta if eta is not None else self.eta
            shutil.copy(tmp_save_path, self.final_save_path)
            json_save_path = cc.artis_ai_feature_lookup_path
            print(f"{CurrentDateTime(0)}[make_feature][mode_all] Json file saved to {json_save_path}")
            if len(total_classes):
                if os.path.exists(json_save_path):
                    os.remove(json_save_path)
                with open(json_save_path, 'w') as f:
                    json.dump({'mapper': total_classes, 'size': total_classes_size}, f, indent=4)
                print(f"{CurrentDateTime(0)}[make_feature][mode_all] Total Classes For Feature DB : {len(total_classes)}")
                total_class_tensor = torch.load(self.final_save_path)
                print(f"{CurrentDateTime(0)}[make_feature][mode_all] Combined tensor saved to {self.final_save_path}")
                print(f"{CurrentDateTime(0)}[make_feature][mode_all] Total Class Tensor Shape : {total_class_tensor.shape}")

            print(f"{CurrentDateTime(0)}[make_feature][mode_all] ********** Making Feature DB By Using Patch Images is Finished. **********")

        elif flag_exist_object_img:
            with tqdm(object_package_type) as pbar:
                for current_object_package_type in object_package_type:
                    for cls_idx, cls_name in enumerate(class_names):
                        img_paths = (glob.glob(os.path.join(class_folder, cls_name, current_object_package_type, '*', '*', '*2_Color.jpg'))
                                     + glob.glob(os.path.join(class_folder, cls_name, current_object_package_type, '*', '*', '*.zip')))

                        if len(img_paths) > max_images_for_apply:
                            selected_img_paths = random.sample(img_paths, max_images_for_apply)
                        else:
                            selected_img_paths = img_paths.copy()

                        if len(selected_img_paths) == 0:
                            continue

                        if 'TYPE1' in current_object_package_type:
                            print(f"{CurrentDateTime(0)}[make_feature][mode_all] Make Feature Map for Class {cls_name} [Package Type : Non-Vinyl]")
                        elif 'TYPE2' in current_object_package_type:
                            print(f"{CurrentDateTime(0)}[make_feature][mode_all] Make Feature Map for Class {cls_name} [Package Type : Vinyl]")
                        else:
                            print(f"{CurrentDateTime(0)}[make_feature][mode_all] Make Feature Map for Class {cls_name} [Package Type : Unknown]")

                        print(f"{CurrentDateTime(0)}[make_feature][mode_all] Current Class DB Total Cnt : {len(img_paths)} || Apply DB Cnt: {len(selected_img_paths)}")
                        for img_idx, img_path in enumerate(selected_img_paths):
                            total_gt_info = []
                            flag_exist_contour_info = False
                            if ".zip" in img_path:
                                zip_db_img_filename = "Cam_2_Color.jpg"
                                zip_db_gt_filename = "artis_result_debug.json"
                                with pyzipper.AESZipFile(img_path, 'r') as zf:
                                    zf.pwd = train_key
                                    filelist = zf.namelist()
                                    if zip_db_img_filename in filelist:
                                        jpg_data = zf.read(zip_db_img_filename)
                                        image_array = np.frombuffer(jpg_data, dtype=np.uint8)
                                        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                                        if img is None:
                                            print(f"{CurrentDateTime(0)}[make_feature][mode_all] Image is None {img_path}")
                                            continue
                                    if zip_db_gt_filename in filelist:
                                        json_data = zf.read(zip_db_gt_filename)
                                        gt_data = json.loads(json_data.decode('utf-8-sig'))
                                        if "seg_oc" in gt_data and gt_data["seg_oc"]["result_info"]["object_total_cnt"]:
                                            field = "seg_oc"
                                            flag_exist_contour_info = True
                                        elif "seg" in gt_data and gt_data["seg"]["result_info"]["object_total_cnt"]:
                                            field = "seg"
                                            flag_exist_contour_info = True
                                        elif "rgb" in gt_data and gt_data["rgb"]["result_info"]["object_total_cnt"]:
                                            field = "rgb"
                                        elif "depth" in gt_data and gt_data["depth"]["result_info"]["object_total_cnt"]:
                                            field = "depth"
                                        else:
                                            break

                                        for gt_idx in range(gt_data[field]["result_info"]["object_total_cnt"]):
                                            bbox = gt_data[field]["object_bbox"][str(gt_idx)]
                                            bbox_info = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(cls_name)]
                                            if flag_exist_contour_info:
                                                contour = gt_data[field]["object_contour"][str(gt_idx)]["point"]
                                                gt_info = [bbox_info, contour]
                                            else:
                                                gt_info = bbox_info
                                            total_gt_info.append(gt_info)
                            else:
                                try:
                                    img = cv2.imread(img_path)
                                except Exception as e:
                                    print(f"{CurrentDateTime(0)}[make_feature][mode_all] Error loading image {img_path}: {e}")
                                    continue
                                gt_path = img_path.replace('Cam_2_Color.jpg', 'artis_result_debug.json')
                                if os.path.exists(gt_path):
                                    with open(gt_path, "r", encoding='utf-8-sig') as json_file:
                                        gt_data = json.load(json_file)
                                        if "seg_oc" in gt_data and gt_data["seg_oc"]["result_info"]["object_total_cnt"]:
                                            field = "seg_oc"
                                            flag_exist_contour_info = True
                                        elif "seg" in gt_data and gt_data["seg"]["result_info"]["object_total_cnt"]:
                                            field = "seg"
                                            flag_exist_contour_info = True
                                        elif "rgb" in gt_data and gt_data["rgb"]["result_info"]["object_total_cnt"]:
                                            field = "rgb"
                                        elif "depth" in gt_data and gt_data["depth"]["result_info"]["object_total_cnt"]:
                                            field = "depth"
                                        else:
                                            break

                                        for gt_idx in range(gt_data[field]["result_info"]["object_total_cnt"]):
                                            bbox = gt_data[field]["object_bbox"][str(gt_idx)]
                                            bbox_info = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(cls_name)]
                                            if flag_exist_contour_info:
                                                contour = gt_data[field]["object_contour"][str(gt_idx)]["point"]
                                                gt_info = [bbox_info, contour]
                                            else:
                                                gt_info = bbox_info
                                            total_gt_info.append(gt_info)
                                else:
                                    gt_path = img_path.replace('_Color.jpg', '_GT.csv')
                                    if os.path.exists(gt_path):
                                        with open(gt_path, newline='', encoding="utf-8") as csv_file:
                                            reader = csv.reader(csv_file)
                                            next(reader)
                                            for each_row in reader:
                                                bbox = [int(each_row[1]), int(each_row[2]), int(each_row[3]), int(each_row[4]), int(cls_name)]
                                                gt_info = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4])]
                                                total_gt_info.append(gt_info)

                            if not len(total_gt_info):
                                print(f"{CurrentDateTime(0)}[make_feature][mode_all] Please Check GT Information : {img_path}")
                                continue

                            origin_img = img.copy()
                            for current_rotate_angle in list_rotation_angle:
                                for current_gt_info in total_gt_info:
                                    if flag_exist_contour_info:
                                        y, x = current_gt_info[0][1], current_gt_info[0][0]
                                        h, w = current_gt_info[0][3] - current_gt_info[0][1], current_gt_info[0][2] - current_gt_info[0][0]
                                        contour_shifted = np.array(current_gt_info[1]) - np.array([x, y])
                                        current_mask = np.zeros((h, w), dtype=np.uint8)
                                        cv2.fillPoly(current_mask, [contour_shifted], 255)
                                        temp_crop_img = origin_img[y:y + h, x:x + w]
                                        current_img = cv2.bitwise_and(temp_crop_img, temp_crop_img, mask=current_mask)
                                    else:
                                        current_img = origin_img[current_gt_info[1]:current_gt_info[3], current_gt_info[0]:current_gt_info[2]]
                                    current_img = current_img[:, :, ::-1]
                                    current_img = Image.fromarray(current_img)
                                    size_check_img = current_img.copy()
                                    pattern_img = resize_with_padding(current_img, target_size=224, padding_color=(0, 0, 0))

                                    total_current_class_feature = []
                                    images_batch = []

                                    if current_rotate_angle == 0:
                                        transform_img = original_transforms(pattern_img)
                                    else:
                                        transform_img = TF.rotate(pattern_img, angle=current_rotate_angle, fill=(0, 0, 0))
                                        transform_img = augmentation_transforms(transform_img)
                                        size_check_img = TF.rotate(size_check_img, angle=current_rotate_angle, fill=(0, 0, 0), expand=True)

                                    transform_img = final_transforms(transform_img).unsqueeze(0)

                                    np_size_check_img = np.array(size_check_img)
                                    if np_size_check_img.shape[2] == 4:
                                        mask = (np_size_check_img[..., :3].sum(axis=2) != 0) & (np_size_check_img[..., 3] != 0)
                                    else:
                                        mask = (np_size_check_img.sum(axis=2) != 0)

                                    coords = np.argwhere(mask)
                                    y0, x0 = coords.min(axis=0)
                                    y1, x1 = coords.max(axis=0) + 1
                                    fit_crop_size_check_img = size_check_img.crop((x0, y0, x1, y1))

                                    current_img_mean_width, current_img_mean_height = fit_crop_size_check_img.size
                                    images_batch.append(transform_img)

                                    if self.flag_use_tensorrt:
                                        feature_tensors = []
                                        batch_tensor_t = torch.cat(images_batch, dim=0).contiguous()

                                        for process_start_idx in range(0, len(images_batch), self.max_batch_size):
                                            process_end_idx = min(process_start_idx + self.max_batch_size, len(images_batch))
                                            sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                            sub_batch_cnt = sub_batch.shape[0]

                                            batch_tensor = sub_batch.numpy()
                                            np.copyto(self.inputs[0]["host"][:batch_tensor.size], batch_tensor.ravel())
                                            self.context.set_binding_shape(self.in_idx, (sub_batch_cnt, self.C, self.H, self.W))  # input에 따른 동적
                                            cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
                                            self.context.execute_v2(bindings=self.bindings)
                                            cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
                                            self.stream.synchronize()

                                            out_shape = tuple(self.context.get_binding_shape(self.out_idx))
                                            output_np = self.outputs[0]["host"][:np.prod(out_shape)].reshape(out_shape)
                                            sub_batch_output_tensor = torch.from_numpy(output_np)
                                            feature_tensors.append(sub_batch_output_tensor)

                                        feature_tensors = np.concatenate(feature_tensors, axis=0)
                                    else:
                                        feature_tensors = []
                                        batch_tensor_t = torch.cat(images_batch, dim=0).contiguous().to("cuda")
                                        with torch.no_grad():
                                            for process_start_idx in range(0, len(images_batch), self.max_batch_size):
                                                process_end_idx = min(process_start_idx + self.max_batch_size, len(images_batch))
                                                sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                                sub_batch_output_tensor = self.context(sub_batch)
                                                feature_tensors.append(sub_batch_output_tensor.detach().cpu())

                                        feature_tensors = np.concatenate(feature_tensors, axis=0)

                                    output_tensor = torch.from_numpy(np.array(feature_tensors).reshape(len(images_batch), extracted_feature_dim))
                                    total_current_class_feature.append(output_tensor)

                                    if len(total_current_class_feature):
                                        combined_tensor = torch.cat(total_current_class_feature, dim=0)
                                        combined_tensor = combined_tensor.view(-1, len(total_current_class_feature) * len(images_batch), extracted_feature_dim)
                                        combined_tensor = combined_tensor.mean(dim=1, keepdim=True)
                                        if os.path.exists(tmp_save_path):
                                            if valid_class_id > 0:
                                                total_class_tensor_db = torch.load(tmp_save_path)
                                                if total_class_tensor_db.dim() == 3:
                                                    total_class_tensor_db = total_class_tensor_db.squeeze(1)
                                                total_class_tensor_db = total_class_tensor_db.unsqueeze(1)
                                                total_class_tensor_db = torch.cat((total_class_tensor_db, combined_tensor), dim=0).squeeze(1)
                                                torch.save(total_class_tensor_db, tmp_save_path)
                                                #print(f"{CurrentDateTime(0)}[make_feature][mode_all] Making Feature DB Finished. shape info : {combined_tensor.shape} / {total_class_tensor_db.shape} class info : {cls_name} || rotation info : {current_rotate_angle}")
                                            else:
                                                os.remove(tmp_save_path)
                                                torch.save(combined_tensor, tmp_save_path)
                                                print(f"{CurrentDateTime(0)}[make_feature][mode_all] Remove Existing Feature DB & Make New Feature DB")
                                        else:
                                            print(f"{CurrentDateTime(0)}[make_feature][mode_all] Making First Feature Finished. {combined_tensor.shape}")
                                            torch.save(combined_tensor, tmp_save_path)

                                        total_classes[valid_class_id] = int(cls_name)
                                        total_classes_size[valid_class_id] = [int(current_img_mean_height), int(current_img_mean_width)]
                                        valid_class_id += 1
                    pbar.update(1)
                    progress = pbar.n / pbar.total
                    elapsed = pbar.format_dict['elapsed']
                    eta = (elapsed / progress) - elapsed if progress > 0 else None
                    self.eta = eta if eta is not None else self.eta
            shutil.copy(tmp_save_path, self.final_save_path)
            json_save_path = cc.artis_ai_feature_lookup_path
            print(f"{CurrentDateTime(0)}[make_feature][mode_all] Json file saved to {json_save_path}")
            if len(total_classes):
                if os.path.exists(json_save_path):
                    os.remove(json_save_path)
                with open(json_save_path, 'w') as f:
                    json.dump({'mapper': total_classes, 'size': total_classes_size}, f, indent=4)
                print(f"{CurrentDateTime(0)}[make_feature][mode_all] Total Classes For Feature DB : {len(total_classes)}")
                total_class_tensor = torch.load(self.final_save_path)
                print(f"{CurrentDateTime(0)}[make_feature][mode_all] Combined tensor saved to {self.final_save_path}")
                print(f"{CurrentDateTime(0)}[make_feature][mode_all] Total Class Tensor Shape : {total_class_tensor.shape}")

            print(f"[make_feature][mode_all] ********** Making Feature DB By Using Single Images is Finished. **********")

        elif flag_exist_zip_db:
            db_img_filename = "Cam_2_Color.jpg"
            db_gt_filename = "artis_result_debug.json"
            db_paths = glob.glob(os.path.join(db_folder, '*.zip'))
            with tqdm(len(db_paths)) as pbar:
                for each_db_path in db_paths:
                    total_gt_info = []
                    flag_exist_contour_info = False
                    with pyzipper.AESZipFile(each_db_path, 'r') as zf:
                        zf.pwd = train_key
                        filelist = zf.namelist()
                        if db_img_filename in filelist:
                            jpg_data = zf.read(db_img_filename)
                            image_array = np.frombuffer(jpg_data, dtype=np.uint8)
                            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                            if img is None:
                                print(
                                    f"{CurrentDateTime(0)}[make_feature][mode_all] Image is Invalid Check DB Path : {each_db_path}")
                                continue
                        if db_gt_filename in filelist:
                            json_data = zf.read(db_gt_filename)
                            gt_data = json.loads(json_data.decode('utf-8-sig'))
                            if "seg_oc" in gt_data and gt_data["seg_oc"]["result_info"]["object_total_cnt"]:
                                field = "seg_oc"
                                flag_exist_contour_info = True
                            elif gt_data["seg"]["result_info"]["object_total_cnt"]:
                                field = "seg"
                                flag_exist_contour_info = True
                            elif gt_data["artis_result_info"]["object_total_cnt"]:
                                field = "total"
                            elif "rgb" in gt_data and gt_data["rgb"]["result_info"]["object_total_cnt"]:
                                field = "rgb"
                            elif "depth" in gt_data and gt_data["depth"]["result_info"]["object_total_cnt"]:
                                field = "depth"
                            else:
                                break
                            
                            if field == "total":
                                for gt_idx in range(gt_data["artis_result_info"]["object_total_cnt"]):
                                    bbox = gt_data["artis_object_bbox"][str(gt_idx)]
                                    cls = int(gt_data["artis_object_detail"][str(gt_idx)])
                                    ### 9999998, 9999999 classes are invalid class number ###
                                    ### 9999998 : It is not satisfied for recognition score (score is under threshold.)
                                    ### 9999999 : This ROI is not matched between RGB Det result & RGBD Det result. (RGBD model is more detected this area than RGB Model)
                                    if 0 < cls < invalid_class_no:
                                        gt_info = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), cls]
                                        total_gt_info.append(gt_info)
                                    else:
                                        print(
                                            f"{CurrentDateTime(0)}[make_feature][mode_all] Invalid Class Info Detected. Path : {each_db_path} GT Idx : {gt_idx} class : {cls}")
                            else:
                                for gt_idx in range(gt_data[field]["result_info"]["object_total_cnt"]):
                                    bbox = gt_data[field]["object_bbox"][str(gt_idx)]
                                    cls = int(gt_data[field]["object_detail"][str(gt_idx)])
                                    if 0 < cls < 9000000:
                                        gt_info = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), cls]
                                        if flag_exist_contour_info:
                                            contour = gt_data[field]["object_contour"][str(gt_idx)]["point"]
                                            gt_info = [gt_info, contour]
                                        total_gt_info.append(gt_info)
                                    else:
                                        print(
                                            f"{CurrentDateTime(0)}[make_feature][mode_add] Invalid Class Info Detected. Path : {each_db_path} GT Idx : {gt_idx} class : {cls}")

                    if not len(total_gt_info):
                        print(
                            f"{CurrentDateTime(0)}[make_feature][mode_all] GT Information is not Exist. Please Check Zip DB : {each_db_path}")
                        continue

                    origin_img = img.copy()
                    for current_rotate_angle in list_rotation_angle:
                        for current_gt_info in total_gt_info:
                            if flag_exist_contour_info:
                                y, x = current_gt_info[0][1], current_gt_info[0][0]
                                h, w = current_gt_info[0][3] - current_gt_info[0][1], current_gt_info[0][2] - current_gt_info[0][0]
                                contour_shifted = np.array(current_gt_info[1]) - np.array([x, y])
                                current_mask = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillPoly(current_mask, [contour_shifted], 255)
                                temp_crop_img = origin_img[y:y + h, x:x + w]
                                current_img = cv2.bitwise_and(temp_crop_img, temp_crop_img, mask=current_mask)
                            else:
                                current_img = origin_img[current_gt_info[1]:current_gt_info[3], current_gt_info[0]:current_gt_info[2]]
                            current_img = current_img[:, :, ::-1]
                            current_img = Image.fromarray(current_img)
                            size_check_img = current_img.copy()
                            pattern_img = resize_with_padding(current_img, target_size=224, padding_color=(0, 0, 0))

                            total_current_class_feature = []
                            images_batch = []

                            if current_rotate_angle == 0:
                                transform_img = original_transforms(pattern_img)
                            else:
                                transform_img = TF.rotate(pattern_img, angle=current_rotate_angle, fill=(0, 0, 0))
                                transform_img = augmentation_transforms(transform_img)
                                size_check_img = TF.rotate(size_check_img, angle=current_rotate_angle, fill=(0, 0, 0),
                                                           expand=True)

                            transform_img = final_transforms(transform_img).unsqueeze(0)

                            np_size_check_img = np.array(size_check_img)
                            if np_size_check_img.shape[2] == 4:
                                mask = (np_size_check_img[..., :3].sum(axis=2) != 0) & (np_size_check_img[..., 3] != 0)
                            else:
                                mask = (np_size_check_img.sum(axis=2) != 0)

                            coords = np.argwhere(mask)
                            y0, x0 = coords.min(axis=0)
                            y1, x1 = coords.max(axis=0) + 1
                            fit_crop_size_check_img = size_check_img.crop((x0, y0, x1, y1))

                            current_img_mean_width, current_img_mean_height = fit_crop_size_check_img.size

                            images_batch.append(transform_img)
                            if self.flag_use_tensorrt:
                                feature_tensors = []
                                batch_tensor_t = torch.cat(images_batch, dim=0).contiguous()

                                for process_start_idx in range(0, len(images_batch), self.max_batch_size):
                                    process_end_idx = min(process_start_idx + self.max_batch_size, len(images_batch))
                                    sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                    sub_batch_cnt = sub_batch.shape[0]

                                    batch_tensor = sub_batch.numpy()
                                    np.copyto(self.inputs[0]["host"][:batch_tensor.size], batch_tensor.ravel())
                                    self.context.set_binding_shape(self.in_idx, (sub_batch_cnt, self.C, self.H, self.W))
                                    cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
                                    self.context.execute_v2(bindings=self.bindings)
                                    cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
                                    self.stream.synchronize()

                                    out_shape = tuple(self.context.get_binding_shape(self.out_idx))
                                    output_np = self.outputs[0]["host"][:np.prod(out_shape)].reshape(out_shape)
                                    sub_batch_output_tensor = torch.from_numpy(output_np)
                                    feature_tensors.append(sub_batch_output_tensor)

                                feature_tensors = np.concatenate(feature_tensors, axis=0)
                            else:
                                feature_tensors = []
                                batch_tensor_t = torch.cat(images_batch, dim=0).contiguous().to("cuda")
                                with torch.no_grad():
                                    for process_start_idx in range(0, len(images_batch), self.max_batch_size):
                                        process_end_idx = min(process_start_idx + self.max_batch_size, len(images_batch))
                                        sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                        sub_batch_output_tensor = self.context(sub_batch)
                                        feature_tensors.append(sub_batch_output_tensor.detach().cpu())

                                feature_tensors = np.concatenate(feature_tensors, axis=0)

                            output_tensor = torch.from_numpy(
                                np.array(feature_tensors).reshape(len(images_batch), extracted_feature_dim))
                            total_current_class_feature.append(output_tensor)

                            if len(total_current_class_feature):
                                combined_tensor = torch.cat(total_current_class_feature, dim=0)
                                combined_tensor = combined_tensor.view(-1,
                                                                       len(total_current_class_feature) * len(images_batch),
                                                                       extracted_feature_dim)
                                combined_tensor = combined_tensor.mean(dim=1, keepdim=True)
                                if os.path.exists(tmp_save_path):
                                    if valid_class_id > 0:
                                        total_class_tensor_db = torch.load(tmp_save_path)
                                        if total_class_tensor_db.dim() == 3:
                                            total_class_tensor_db = total_class_tensor_db.squeeze(1)
                                        total_class_tensor_db = total_class_tensor_db.unsqueeze(1)
                                        total_class_tensor_db = torch.cat((total_class_tensor_db, combined_tensor),
                                                                          dim=0).squeeze(1)
                                        torch.save(total_class_tensor_db, tmp_save_path)
                                    else:
                                        os.remove(tmp_save_path)
                                        torch.save(combined_tensor, tmp_save_path)
                                        print(
                                            f"{CurrentDateTime(0)}[make_feature][mode_all] Remove Existing Feature DB & Make New Feature DB")
                                else:
                                    print(
                                        f"{CurrentDateTime(0)}[make_feature][mode_all] Making First Feature Finished. {combined_tensor.shape}")
                                    torch.save(combined_tensor, tmp_save_path)

                                if flag_exist_contour_info:
                                    total_classes[valid_class_id] = int(current_gt_info[0][4])
                                    if not int(current_gt_info[0][4]) in unique_classes:
                                        unique_classes.append(int(current_gt_info[0][4]))
                                else:
                                    total_classes[valid_class_id] = int(current_gt_info[4])
                                    if not int(current_gt_info[4]) in unique_classes:
                                        unique_classes.append(int(current_gt_info[4]))
                                total_classes_size[valid_class_id] = [int(current_img_mean_height),
                                                                      int(current_img_mean_width)]
                                print(f"{CurrentDateTime(0)}[make_feature][mode_all] Making Feature DB Finished. shape info : {combined_tensor.shape} / {total_class_tensor_db.shape}"
                                      f" class info : {total_classes_size[valid_class_id]} || rotation info : {current_rotate_angle}")
                                valid_class_id += 1
                    pbar.update(1)
                    progress = pbar.n / pbar.total
                    elapsed = pbar.format_dict['elapsed']
                    eta = (elapsed / progress) - elapsed if progress > 0 else None
                    self.eta = eta if eta is not None else self.eta
            shutil.copy(tmp_save_path, self.final_save_path)
            json_save_path = cc.artis_ai_feature_lookup_path
            print(f"{CurrentDateTime(0)}[make_feature][mode_all] Json file saved to {json_save_path}")
            if len(total_classes):
                if os.path.exists(json_save_path):
                    os.remove(json_save_path)
                with open(json_save_path, 'w') as f:
                    json.dump({'mapper': total_classes, 'size': total_classes_size}, f, indent=4)
                print(
                    f"{CurrentDateTime(0)}[make_feature][mode_all] Total Classes For Feature DB : {len(total_classes)}")
                print(f"{CurrentDateTime(0)}[make_feature][mode_all] Total Unique Classes Cnt : {len(unique_classes)}")
                print(
                    f"{CurrentDateTime(0)}[make_feature][mode_all] Total Unique Classes For Feature DB : {unique_classes}")
                total_class_tensor = torch.load(self.final_save_path)
                print(f"{CurrentDateTime(0)}[make_feature][mode_all] Combined tensor saved to {self.final_save_path}")
                print(
                    f"{CurrentDateTime(0)}[make_feature][mode_all] Total Class Tensor Shape : {total_class_tensor.shape}")

            print(
                f"{CurrentDateTime(0)}[make_feature][mode_all] ********** Making Feature DB By Using Zip DB is Finished. **********")

        self.eta = 0

    async def add(self):
        ''' Tensor DB load'''
        img_folder_to_add = self.db_root_path + "/V2_single/"
        load_feature_db_path = ""
        json_mapper_path = ""

        self.start_time = time.time()

        if os.path.isdir(cc.artis_ai_feature_db_path):
            erase_check_path = cc.artis_ai_feature_db_path
        else:
            erase_check_path = os.path.dirname(cc.artis_ai_feature_db_path)
        for file in glob.glob(os.path.join(erase_check_path, "*backup.pt")):
            os.remove(file)
        for file in glob.glob(os.path.join(erase_check_path, "*backup.json")):
            os.remove(file)

        if os.path.isdir(cc.artis_ai_feature_db_path):
            file_list = os.listdir(cc.artis_ai_feature_db_path)
            for each_file in file_list:
                name, ext = os.path.splitext(each_file)
                if ext == ".pt":
                    load_feature_db_path = os.path.join(cc.artis_ai_feature_db_path, each_file)
                    break
        else:
            load_feature_db_path = cc.artis_ai_feature_db_path

        if os.path.exists(load_feature_db_path):
            combined_tensor = torch.load(load_feature_db_path)
        else:
            print(
                f"{CurrentDateTime(0)}[make_feature][mode_add] It is not exist feature db file (.pt). Check Path : {load_feature_db_path}")
            return

        if combined_tensor.dim() == 3:
            combined_tensor = combined_tensor.squeeze(1)
        print(f"{CurrentDateTime(0)}[make_feature][mode_add] Loaded tensor with shape: {combined_tensor.shape}")
        num_classes = int(combined_tensor.shape[0])

        '''load_path = os.path.dirname(load_feature_db_path)
        file_list = os.listdir(load_path)
        for each_file in file_list:
            name, ext = os.path.splitext(each_file)
            if ext == ".json":
                json_mapper_path = os.path.join(load_path, each_file)
                break'''
        json_mapper_path = cc.artis_ai_feature_lookup_path
        if os.path.exists(json_mapper_path):
            with open(json_mapper_path, 'r') as f:
                mapper = json.load(f)
            class_map = mapper['mapper']
            class_size = mapper['size']
        else:
            print(
                f"{CurrentDateTime(0)}[make_feature][mode_add] It is not exist feature db json file (.json). Check Path : {json_mapper_path}")
            return

        # backup original feature db file
        backup_feature_db_path = load_feature_db_path.replace(".pt", "_backup.pt")
        shutil.copy(load_feature_db_path, backup_feature_db_path)
        backup_json_mapper_path = json_mapper_path.replace(".json", "_backup.json")
        shutil.copy(json_mapper_path, backup_json_mapper_path)

        flag_exist_object_patch = False
        flag_exist_object_img = False
        flag_exist_zip_db = False

        total_modify_class = []

        if "single_patch" in img_folder_to_add:
            if os.path.exists(img_folder_to_add):
                if len(os.listdir(img_folder_to_add)):
                    total_modify_class = os.listdir(img_folder_to_add)
                    flag_exist_object_patch = True
                else:
                    print(
                        f"{CurrentDateTime(0)}[make_feature][mode_add] Must Exist Class Folder in {img_folder_to_add}, Please Check the Path : {img_folder_to_add}")
                    return
        elif "single" in img_folder_to_add:
            if os.path.exists(img_folder_to_add):
                if len(os.listdir(img_folder_to_add)):
                    #total_modify_class = os.listdir(img_folder_to_add)
                    tmp_modify_class = os.listdir(img_folder_to_add)
                    for cls in tmp_modify_class:
                        if cls in self.train_log["item_new"] or cls in self.train_log["item_change"]:
                            total_modify_class.append(cls)
                    flag_exist_object_img = True
                else:
                    print(
                        f"{CurrentDateTime(0)}[make_feature][mode_add] Must Exist Class Folder in {img_folder_to_add}, Please Check the Path : {img_folder_to_add}")
                    return
        else:
            if len(os.listdir(img_folder_to_add)):
                flag_exist_zip_db = True
            else:
                print(
                    f"{CurrentDateTime(0)}[make_feature][mode_add] Must Exist Zip DB in {img_folder_to_add}, Please Check the Path : {img_folder_to_add}")

        print(f"{CurrentDateTime(0)}[make_feature][mode_add] Current Class Map Info\n{class_map}")

        # class_names = []
        # if flag_exist_object_img or flag_exist_object_patch:
        #    class_names = total_modify_class.sort()
        #    print(f"{CurrentDateTime(0)}[make_feature][mode_add] Total Class to Add\n{total_modify_class}")

        if flag_exist_object_patch or flag_exist_object_img:
            class_names = [x for x in total_modify_class if x != 'Base']
            class_names.sort()
        else:
            class_names = []

        # 각 이미지당 생성할 augmentation 개수
        aug_per_image = self.aug_per_image
        num_augmented = aug_per_image - 1

        rotate_per_image = 361
        num_rotated = rotate_per_image - 1

        # 배치에 쌓일 이미지 텐서들을 저장할 리스트
        images_batch = []
        # 배치 번호 카운터
        batch_counter = 100000

        max_images_for_apply = self.max_images_for_apply
        angle_per_rotate = self.angle_per_rotate
        list_rotation_angle = list(range(0, 360, angle_per_rotate))
        print(f"{CurrentDateTime(0)}[make_feature][mode_add] Total Rotation Angle : {list_rotation_angle}")

        object_package_type = ['TYPE1_FRONT', 'TYPE1_SIDE', 'TYPE2_FRONT', 'TYPE2_SIDE', 'TYPE3_FRONT',
                               'TYPE3_SIDE']  # Non-Vinyl, Vinyl
        valid_class_idx = 0

        if flag_exist_object_patch:
            with tqdm(object_package_type) as pbar:
                for current_object_package_type in object_package_type:
                    for cls_idx, cls_name in enumerate(class_names):
                        img_paths = glob.glob(
                            os.path.join(img_folder_to_add, cls_name, current_object_package_type, '*', '*', '*',
                                         '*_patch1.png'))

                        if len(img_paths) > max_images_for_apply:
                            selected_img_paths = random.sample(img_paths, max_images_for_apply)
                        else:
                            selected_img_paths = img_paths.copy()

                        if len(selected_img_paths) == 0:
                            continue

                        if 'TYPE1' in current_object_package_type:
                            print(
                                f"{CurrentDateTime(0)}[make_feature][mode_add] Make Feature Map for Class {cls_name} [Package Type : Non-Vinyl]")
                        elif 'TYPE2' in current_object_package_type:
                            print(
                                f"{CurrentDateTime(0)}[make_feature][mode_add] Make Feature Map for Class {cls_name} [Package Type : Vinyl]")
                        else:
                            print(
                                f"{CurrentDateTime(0)}[make_feature][mode_add] Make Feature Map for Class {cls_name} [Package Type : Unknown]")

                        print(
                            f"{CurrentDateTime(0)}[make_feature][mode_add] Class {cls_name} Extract Path Cnt: {len(img_paths)} || Apply DB Cnt: {len(selected_img_paths)}")

                        for current_rotate_angle in list_rotation_angle:
                            view_list = []
                            total_current_class_feature = []
                            selected_imgs_height, selected_imgs_width = [], []
                            process_img_count = 0
                            for img_idx, img_path in enumerate(selected_img_paths):
                                try:
                                    img = cv2.imread(img_path)
                                except Exception as e:
                                    print(
                                        f"{CurrentDateTime(0)}[make_feature][mode_add] Error loading image {img_path}: {e}")
                                    continue

                                img = img[:, :, ::-1]
                                img = Image.fromarray(img)
                                size_check_img = img.copy()
                                pattern_img = resize_with_padding(img, target_size=224, padding_color=(0, 0, 0))

                                ### For Mean Feature Logic ###
                                if current_rotate_angle == 0:
                                    transform_img = original_transforms(pattern_img)
                                else:
                                    transform_img = TF.rotate(pattern_img, angle=current_rotate_angle, fill=(0, 0, 0))
                                    transform_img = augmentation_transforms(transform_img)
                                    size_check_img = TF.rotate(size_check_img, angle=current_rotate_angle, fill=(0, 0, 0),
                                                               expand=True)

                                transform_img = final_transforms(transform_img).unsqueeze(0)

                                np_size_check_img = np.array(size_check_img)
                                if np_size_check_img.shape[2] == 4:
                                    mask = (np_size_check_img[..., :3].sum(axis=2) != 0) & (np_size_check_img[..., 3] != 0)
                                else:
                                    mask = (np_size_check_img.sum(axis=2) != 0)

                                coords = np.argwhere(mask)
                                y0, x0 = coords.min(axis=0)
                                y1, x1 = coords.max(axis=0) + 1
                                fit_crop_size_check_img = size_check_img.crop((x0, y0, x1, y1))

                                current_transform_img_width, current_transform_img_height = fit_crop_size_check_img.size
                                selected_imgs_height.append(current_transform_img_height)
                                selected_imgs_width.append(current_transform_img_width)

                                view_list.append(transform_img)

                                if self.save_debug_image:
                                    debug_image_save_file_path = f"./debug_image/cls_{cls_name}_type_{current_object_package_type}_angle_{current_rotate_angle}_idx_{img_idx}.jpg"
                                    fit_crop_size_check_img.save(debug_image_save_file_path)

                                process_img_count += 1

                            current_img_mean_height = int(sum(selected_imgs_height) / len(selected_imgs_height))
                            current_img_mean_width = int(sum(selected_imgs_width) / len(selected_imgs_width))
                            print(
                                f"{CurrentDateTime(0)}[make_feature][mode_add] [Class : {cls_name}] || [Total Applied Images : {process_img_count}]"
                                f" || [Type : {current_object_package_type}] || [Rotate : {current_rotate_angle}]"
                                f" || [Image Height : {current_img_mean_height} Image Width : {current_img_mean_width}]")

                            for view in view_list:
                                images_batch.append(view)

                            if self.flag_use_tensorrt:
                                feature_tensors = []
                                batch_tensor_t = torch.cat(images_batch, dim=0).contiguous()

                                for process_start_idx in range(0, process_img_count, self.max_batch_size):
                                    process_end_idx = min(process_start_idx + self.max_batch_size, process_img_count)
                                    sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                    sub_batch_cnt = sub_batch.shape[0]

                                    batch_tensor = sub_batch.numpy()
                                    np.copyto(self.inputs[0]["host"][:batch_tensor.size], batch_tensor.ravel())
                                    self.context.set_binding_shape(self.in_idx, (sub_batch_cnt, self.C, self.H, self.W))
                                    cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
                                    self.context.execute_v2(bindings=self.bindings)
                                    cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
                                    self.stream.synchronize()

                                    out_shape = tuple(self.context.get_binding_shape(self.out_idx))
                                    output_np = self.outputs[0]["host"][:np.prod(out_shape)].reshape(out_shape)
                                    sub_batch_output_tensor = torch.from_numpy(output_np)
                                    feature_tensors.append(sub_batch_output_tensor)

                                feature_tensors = np.concatenate(feature_tensors, axis=0)
                            else:
                                feature_tensors = []
                                batch_tensor_t = torch.cat(images_batch, dim=0).contiguous().to("cuda")
                                with torch.no_grad():
                                    for process_start_idx in range(0, process_img_count, self.max_batch_size):
                                        process_end_idx = min(process_start_idx + self.max_batch_size, process_img_count)
                                        sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                        sub_batch_output_tensor = self.context(sub_batch)
                                        feature_tensors.append(sub_batch_output_tensor.detach().cpu())

                                feature_tensors = np.concatenate(feature_tensors, axis=0)

                            output_tensor = torch.from_numpy(
                                np.array(feature_tensors).reshape(process_img_count, extracted_feature_dim))
                            images_batch = []
                            total_current_class_feature.append(output_tensor)

                            if len(total_current_class_feature):
                                current_class_idx = str(valid_class_idx + num_classes)
                                class_map[current_class_idx] = int(cls_name)
                                class_size[current_class_idx] = [int(current_img_mean_height), int(current_img_mean_width)]
                                with open(backup_json_mapper_path, 'w') as f:
                                    json.dump({'mapper': class_map, 'size': class_size}, f, indent=4)
                                valid_class_idx += 1

                                total_class_tensor = torch.load(backup_feature_db_path)
                                if total_class_tensor.dim() == 3:
                                    total_class_tensor = total_class_tensor.squeeze(1)

                                combined_new_tensor = torch.cat(total_current_class_feature, dim=0)
                                combined_new_tensor = combined_new_tensor.view(-1,
                                                                               len(total_current_class_feature) * process_img_count,
                                                                               extracted_feature_dim)
                                combined_new_tensor = combined_new_tensor.mean(dim=1, keepdim=True)
                                total_class_tensor = total_class_tensor.unsqueeze(1)
                                # Append to the existing tensor
                                total_class_tensor = torch.cat((total_class_tensor, combined_new_tensor), dim=0).squeeze(1)
                                # Save the updated tensor
                                torch.save(total_class_tensor, backup_feature_db_path)
                                print(
                                    f"{CurrentDateTime(0)}[make_feature][mode_add] New Class Added - Class : {cls_name} || Angle : {current_rotate_angle}")
                                print(
                                    f"{CurrentDateTime(0)}[make_feature][mode_add] New class tensor shape : {combined_new_tensor.shape} || Updated tensor shape: {total_class_tensor.shape}")
                pbar.update(1)
                progress = pbar.n / pbar.total
                elapsed = pbar.format_dict['elapsed']
                eta = (elapsed / progress) - elapsed if progress > 0 else None
                self.eta = eta if eta is not None else self.eta
        elif flag_exist_object_img:
            with tqdm(object_package_type) as pbar:
                for current_object_package_type in object_package_type:
                    for cls_idx, cls_name in enumerate(class_names):
                        img_paths = (glob.glob(
                            os.path.join(img_folder_to_add, cls_name, current_object_package_type, '*', '*',
                                         '*2_Color.jpg'))
                                     + glob.glob(
                                    os.path.join(img_folder_to_add, cls_name, current_object_package_type, '*', '*',
                                                 '*.zip')))

                        if len(img_paths) > max_images_for_apply:
                            selected_img_paths = random.sample(img_paths, max_images_for_apply)
                        else:
                            selected_img_paths = img_paths.copy()

                        if len(selected_img_paths) == 0:
                            continue

                        real_db_cnt = 0
                        for img_idx, img_path in enumerate(selected_img_paths):
                            img_path = img_path.replace("\\", "/")
                            if img_path not in self.db_sync_report["file_new"]:
                                continue
                            total_gt_info = []
                            flag_exist_contour_info = False
                            if ".zip" in img_path:
                                zip_db_img_filename = "Cam_2_Color.jpg"
                                zip_db_gt_filename = "artis_result_debug.json"
                                with pyzipper.AESZipFile(img_path, 'r') as zf:
                                    zf.pwd = train_key
                                    filelist = zf.namelist()
                                    if zip_db_img_filename in filelist:
                                        jpg_data = zf.read(zip_db_img_filename)
                                        image_array = np.frombuffer(jpg_data, dtype=np.uint8)
                                        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                                        if img is None:
                                            print(f"{CurrentDateTime(0)}[make_feature][mode_add] Image is None {img_path}")
                                            continue
                                    if zip_db_gt_filename in filelist:
                                        json_data = zf.read(zip_db_gt_filename)
                                        gt_data = json.loads(json_data.decode('utf-8-sig'))
                                        if "seg_oc" in gt_data and gt_data["seg_oc"]["result_info"]["object_total_cnt"]:
                                            field = "seg_oc"
                                            flag_exist_contour_info = True
                                        elif "seg" in gt_data and gt_data["seg"]["result_info"]["object_total_cnt"]:
                                            field = "seg"
                                            flag_exist_contour_info = True
                                        elif "rgb" in gt_data and gt_data["rgb"]["result_info"]["object_total_cnt"]:
                                            field = "rgb"
                                        elif "depth" in gt_data and gt_data["depth"]["result_info"]["object_total_cnt"]:
                                            field = "depth"
                                        else:
                                            break

                                        for gt_idx in range(gt_data[field]["result_info"]["object_total_cnt"]):
                                            bbox = gt_data[field]["object_bbox"][str(gt_idx)]
                                            bbox_info = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(cls_name)]
                                            if flag_exist_contour_info:
                                                contour = gt_data[field]["object_contour"][str(gt_idx)]["point"]
                                                gt_info = [bbox_info, contour]
                                            else:
                                                gt_info = bbox_info
                                            total_gt_info.append(gt_info)
                            else:
                                try:
                                    img = cv2.imread(img_path)
                                except Exception as e:
                                    print(
                                        f"{CurrentDateTime(0)}[make_feature][mode_add] Error loading image {img_path}: {e}")
                                    continue
                                gt_path = img_path.replace('Cam_2_Color.jpg', 'artis_result_debug.json')
                                if os.path.exists(gt_path):
                                    with open(gt_path, "r", encoding='utf-8-sig') as json_file:
                                        gt_data = json.load(json_file)
                                        if "seg_oc" in gt_data and gt_data["seg_oc"]["result_info"]["object_total_cnt"]:
                                            field = "seg_oc"
                                            flag_exist_contour_info = True
                                        elif "seg" in gt_data and gt_data["seg"]["result_info"]["object_total_cnt"]:
                                            field = "seg"
                                            flag_exist_contour_info = True
                                        elif "rgb" in gt_data and gt_data["rgb"]["result_info"]["object_total_cnt"]:
                                            field = "rgb"
                                        elif "depth" in gt_data and gt_data["depth"]["result_info"]["object_total_cnt"]:
                                            field = "depth"
                                        else:
                                            break

                                        for gt_idx in range(gt_data[field]["result_info"]["object_total_cnt"]):
                                            bbox = gt_data[field]["object_bbox"][str(gt_idx)]
                                            bbox_info = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(cls_name)]
                                            if flag_exist_contour_info:
                                                contour = gt_data[field]["object_contour"][str(gt_idx)]["point"]
                                                gt_info = [bbox_info, contour]
                                            else:
                                                gt_info = bbox_info
                                            total_gt_info.append(gt_info)
                                else:
                                    gt_path = img_path.replace('_Color.jpg', '_GT.csv')
                                    if os.path.exists(gt_path):
                                        with open(gt_path, newline='', encoding="utf-8") as csv_file:
                                            reader = csv.reader(csv_file)
                                            next(reader)
                                            for each_row in reader:
                                                bbox = [int(each_row[1]), int(each_row[2]), int(each_row[3]),
                                                        int(each_row[4]), int(cls_name)]
                                                gt_info = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                                                           int(bbox[4])]
                                                total_gt_info.append(gt_info)

                            if not len(total_gt_info):
                                print(
                                    f"{CurrentDateTime(0)}[make_feature][mode_add] Please Check GT Information : {img_path}")
                                continue

                            real_db_cnt += 1

                            origin_img = img.copy()
                            for current_rotate_angle in list_rotation_angle:
                                for current_gt_info in total_gt_info:
                                    if flag_exist_contour_info:
                                        y, x = current_gt_info[0][1], current_gt_info[0][0]
                                        h, w = current_gt_info[0][3] - current_gt_info[0][1], current_gt_info[0][2] - current_gt_info[0][0]
                                        contour_shifted = np.array(current_gt_info[1]) - np.array([x, y])
                                        current_mask = np.zeros((h, w), dtype=np.uint8)
                                        cv2.fillPoly(current_mask, [contour_shifted], 255)
                                        temp_crop_img = origin_img[y:y + h, x:x + w]
                                        current_img = cv2.bitwise_and(temp_crop_img, temp_crop_img, mask=current_mask)
                                    else:
                                        current_img = origin_img[current_gt_info[1]:current_gt_info[3], current_gt_info[0]:current_gt_info[2]]

                                    current_img = current_img[:, :, ::-1]
                                    current_img = Image.fromarray(current_img)
                                    size_check_img = current_img.copy()
                                    pattern_img = resize_with_padding(current_img, target_size=224, padding_color=(0, 0, 0))

                                    total_current_class_feature = []
                                    images_batch = []

                                    if current_rotate_angle == 0:
                                        transform_img = original_transforms(pattern_img)
                                    else:
                                        transform_img = TF.rotate(pattern_img, angle=current_rotate_angle, fill=(0, 0, 0))
                                        transform_img = augmentation_transforms(transform_img)
                                        size_check_img = TF.rotate(size_check_img, angle=current_rotate_angle,
                                                                   fill=(0, 0, 0), expand=True)

                                    transform_img = final_transforms(transform_img).unsqueeze(0)

                                    np_size_check_img = np.array(size_check_img)
                                    if np_size_check_img.shape[2] == 4:
                                        mask = (np_size_check_img[..., :3].sum(axis=2) != 0) & (
                                                    np_size_check_img[..., 3] != 0)
                                    else:
                                        mask = (np_size_check_img.sum(axis=2) != 0)

                                    coords = np.argwhere(mask)
                                    y0, x0 = coords.min(axis=0)
                                    y1, x1 = coords.max(axis=0) + 1
                                    fit_crop_size_check_img = size_check_img.crop((x0, y0, x1, y1))

                                    current_img_mean_width, current_img_mean_height = fit_crop_size_check_img.size
                                    images_batch.append(transform_img)

                                    if self.flag_use_tensorrt:
                                        feature_tensors = []
                                        batch_tensor_t = torch.cat(images_batch, dim=0).contiguous()

                                        for process_start_idx in range(0, len(images_batch), self.max_batch_size):
                                            process_end_idx = min(process_start_idx + self.max_batch_size, len(images_batch))
                                            sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                            sub_batch_cnt = sub_batch.shape[0]

                                            batch_tensor = sub_batch.numpy()
                                            np.copyto(self.inputs[0]["host"][:batch_tensor.size], batch_tensor.ravel())
                                            self.context.set_binding_shape(self.in_idx, (sub_batch_cnt, self.C, self.H, self.W))
                                            cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
                                            self.context.execute_v2(bindings=self.bindings)
                                            cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
                                            self.stream.synchronize()

                                            out_shape = tuple(self.context.get_binding_shape(self.out_idx))
                                            output_np = self.outputs[0]["host"][:np.prod(out_shape)].reshape(out_shape)
                                            sub_batch_output_tensor = torch.from_numpy(output_np)
                                            feature_tensors.append(sub_batch_output_tensor)

                                        feature_tensors = np.concatenate(feature_tensors, axis=0)
                                    else:
                                        feature_tensors = []
                                        batch_tensor_t = torch.cat(images_batch, dim=0).contiguous().to("cuda")
                                        with torch.no_grad():
                                            for process_start_idx in range(0, len(images_batch), self.max_batch_size):
                                                process_end_idx = min(process_start_idx + self.max_batch_size, len(images_batch))
                                                sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                                sub_batch_output_tensor = self.context(sub_batch)
                                                feature_tensors.append(sub_batch_output_tensor.detach().cpu())

                                        feature_tensors = np.concatenate(feature_tensors, axis=0)

                                    output_tensor = torch.from_numpy(
                                        np.array(feature_tensors).reshape(len(images_batch), extracted_feature_dim))
                                    total_current_class_feature.append(output_tensor)

                                    if len(total_current_class_feature):
                                        current_class_idx = str(valid_class_idx + num_classes)
                                        class_map[current_class_idx] = int(cls_name)
                                        class_size[current_class_idx] = [int(current_img_mean_height),
                                                                         int(current_img_mean_width)]
                                        with open(backup_json_mapper_path, 'w') as f:
                                            json.dump({'mapper': class_map, 'size': class_size}, f, indent=4)
                                        valid_class_idx += 1

                                        total_class_tensor = torch.load(backup_feature_db_path)
                                        if total_class_tensor.dim() == 3:
                                            total_class_tensor = total_class_tensor.squeeze(1)

                                        # Combine all new tensors into a single tensor
                                        combined_new_tensor = torch.cat(total_current_class_feature, dim=0)
                                        combined_new_tensor = combined_new_tensor.view(-1,
                                                                                       len(total_current_class_feature) * len(
                                                                                           images_batch),
                                                                                       extracted_feature_dim)
                                        combined_new_tensor = combined_new_tensor.mean(dim=1, keepdim=True)
                                        total_class_tensor = total_class_tensor.unsqueeze(1)
                                        # Append to the existing tensor
                                        total_class_tensor = torch.cat((total_class_tensor, combined_new_tensor),
                                                                       dim=0).squeeze(1)
                                        # Save the updated tensor
                                        torch.save(total_class_tensor, backup_feature_db_path)
                                        print(
                                            f"{CurrentDateTime(0)}[make_feature][mode_add] New Class Added - Class : {cls_name} || Angle : {current_rotate_angle}")
                                        print(
                                            f"{CurrentDateTime(0)}[make_feature][mode_add] New class tensor shape : {combined_new_tensor.shape} || Updated tensor shape: {total_class_tensor.shape}")
                
                        if 'TYPE1' in current_object_package_type:
                            print(f"{CurrentDateTime(0)}[make_feature][mode_add] Make Feature Map for Class {cls_name} [Package Type : Non-Vinyl]")
                        elif 'TYPE2' in current_object_package_type:
                            print(f"{CurrentDateTime(0)}[make_feature][mode_add] Make Feature Map for Class {cls_name} [Package Type : Vinyl]")
                        else:
                            print(f"{CurrentDateTime(0)}[make_feature][mode_add] Make Feature Map for Class {cls_name} [Package Type : Unknown]")

                        print(f"{CurrentDateTime(0)}[make_feature][mode_add] Current Class DB Total Cnt : {len(img_paths)} || Apply DB Cnt: {real_db_cnt}")
                            
                pbar.update(1)
                progress = pbar.n / pbar.total
                elapsed = pbar.format_dict['elapsed']
                eta = (elapsed / progress) - elapsed if progress > 0 else None
                self.eta = eta if eta is not None else self.eta
        elif flag_exist_zip_db:
            db_img_filename = "Cam_2_Color.jpg"
            db_gt_filename = "artis_result_debug.json"
            db_paths = glob.glob(os.path.join(img_folder_to_add, '*.zip'))
            with tqdm(len(db_paths)) as pbar:
                for each_db_path in db_paths:
                    total_gt_info = []
                    flag_exist_contour_info = False
                    with pyzipper.AESZipFile(each_db_path, 'r') as zf:
                        zf.pwd = train_key
                        filelist = zf.namelist()
                        if db_img_filename in filelist:
                            jpg_data = zf.read(db_img_filename)
                            image_array = np.frombuffer(jpg_data, dtype=np.uint8)
                            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                            if img is None:
                                print(
                                    f"{CurrentDateTime(0)}[make_feature][mode_add] Image is Invalid Check DB Path : {each_db_path}")
                                continue
                        if db_gt_filename in filelist:
                            json_data = zf.read(db_gt_filename)
                            gt_data = json.loads(json_data.decode('utf-8-sig'))
                            if "seg_oc" in gt_data and gt_data["seg_oc"]["result_info"]["object_total_cnt"]:
                                field = "seg_oc"
                                flag_exist_contour_info = True
                            elif gt_data["seg"]["result_info"]["object_total_cnt"]:
                                field = "seg"
                                flag_exist_contour_info = True
                            elif gt_data["artis_result_info"]["object_total_cnt"]:
                                field = "total"
                            elif "rgb" in gt_data and gt_data["rgb"]["result_info"]["object_total_cnt"]:
                                field = "rgb"
                            elif "depth" in gt_data and gt_data["depth"]["result_info"]["object_total_cnt"]:
                                field = "depth"
                            else:
                                break

                            if field == "total":
                                for gt_idx in range(gt_data["artis_result_info"]["object_total_cnt"]):
                                    bbox = gt_data["artis_object_bbox"][str(gt_idx)]
                                    cls = int(gt_data["artis_object_detail"][str(gt_idx)])
                                    ### 9999998, 9999999 classes are invalid class number ###
                                    ### 9999998 : It is not satisfied for recognition score (score is under threshold.)
                                    ### 9999999 : This ROI is not matched between RGB Det result & RGBD Det result. (RGBD model is more detected this area than RGB Model)
                                    if 0 < cls < invalid_class_no:
                                        gt_info = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), cls]
                                        total_gt_info.append(gt_info)
                                    else:
                                        print(
                                            f"{CurrentDateTime(0)}[make_feature][mode_add] Invalid Class Info Detected. Path : {each_db_path} GT Idx : {gt_idx} class : {cls}")
                            else:
                                for gt_idx in range(gt_data[field]["result_info"]["object_total_cnt"]):
                                    bbox = gt_data[field]["object_bbox"][str(gt_idx)]
                                    cls = int(gt_data[field]["object_detail"][str(gt_idx)])
                                    if 0 < cls < 9000000:
                                        gt_info = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), cls]
                                        if flag_exist_contour_info:
                                            contour = gt_data[field]["object_contour"][str(gt_idx)]["point"]
                                            gt_info = [gt_info, contour]
                                        total_gt_info.append(gt_info)
                                    else:
                                        print(
                                            f"{CurrentDateTime(0)}[make_feature][mode_add] Invalid Class Info Detected. Path : {each_db_path} GT Idx : {gt_idx} class : {cls}")

                    if not len(total_gt_info):
                        print(
                            f"{CurrentDateTime(0)}[make_feature][mode_add] GT Information is not Exist. Please Check Zip DB : {each_db_path}")
                        continue

                    origin_img = img.copy()
                    for current_rotate_angle in list_rotation_angle:
                        for current_gt_info in total_gt_info:
                            if flag_exist_contour_info:
                                y, x = current_gt_info[0][1], current_gt_info[0][0]
                                h, w = current_gt_info[0][3] - current_gt_info[0][1], current_gt_info[0][2] - current_gt_info[0][0]
                                contour_shifted = np.array(current_gt_info[1]) - np.array([x, y])
                                current_mask = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillPoly(current_mask, [contour_shifted], 255)
                                temp_crop_img = origin_img[y:y + h, x:x + w]
                                current_img = cv2.bitwise_and(temp_crop_img, temp_crop_img, mask=current_mask)
                            else:
                                current_img = origin_img[current_gt_info[1]:current_gt_info[3], current_gt_info[0]:current_gt_info[2]]
                            current_img = current_img[:, :, ::-1]
                            current_img = Image.fromarray(current_img)
                            size_check_img = current_img.copy()
                            pattern_img = resize_with_padding(current_img, target_size=224, padding_color=(0, 0, 0))

                            total_current_class_feature = []
                            images_batch = []

                            if current_rotate_angle == 0:
                                transform_img = original_transforms(pattern_img)
                            else:
                                transform_img = TF.rotate(pattern_img, angle=current_rotate_angle, fill=(0, 0, 0))
                                transform_img = augmentation_transforms(transform_img)
                                size_check_img = TF.rotate(size_check_img, angle=current_rotate_angle, fill=(0, 0, 0),
                                                           expand=True)

                            transform_img = final_transforms(transform_img).unsqueeze(0)

                            np_size_check_img = np.array(size_check_img)
                            if np_size_check_img.shape[2] == 4:
                                mask = (np_size_check_img[..., :3].sum(axis=2) != 0) & (np_size_check_img[..., 3] != 0)
                            else:
                                mask = (np_size_check_img.sum(axis=2) != 0)

                            coords = np.argwhere(mask)
                            y0, x0 = coords.min(axis=0)
                            y1, x1 = coords.max(axis=0) + 1
                            fit_crop_size_check_img = size_check_img.crop((x0, y0, x1, y1))

                            current_img_mean_width, current_img_mean_height = fit_crop_size_check_img.size

                            images_batch.append(transform_img)
                            if self.flag_use_tensorrt:
                                feature_tensors = []
                                batch_tensor_t = torch.cat(images_batch, dim=0).contiguous()

                                for process_start_idx in range(0, len(images_batch), self.max_batch_size):
                                    process_end_idx = min(process_start_idx + self.max_batch_size, len(images_batch))
                                    sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                    sub_batch_cnt = sub_batch.shape[0]

                                    batch_tensor = sub_batch.numpy()
                                    np.copyto(self.inputs[0]["host"][:batch_tensor.size], batch_tensor.ravel())
                                    self.context.set_binding_shape(self.in_idx, (sub_batch_cnt, self.C, self.H, self.W))
                                    cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
                                    self.context.execute_v2(bindings=self.bindings)
                                    cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
                                    self.stream.synchronize()

                                    out_shape = tuple(self.context.get_binding_shape(self.out_idx))
                                    output_np = self.outputs[0]["host"][:np.prod(out_shape)].reshape(out_shape)
                                    sub_batch_output_tensor = torch.from_numpy(output_np)
                                    feature_tensors.append(sub_batch_output_tensor)

                                feature_tensors = np.concatenate(feature_tensors, axis=0)
                            else:
                                feature_tensors = []
                                batch_tensor_t = torch.cat(images_batch, dim=0).contiguous().to("cuda")
                                with torch.no_grad():
                                    for process_start_idx in range(0, len(images_batch), self.max_batch_size):
                                        process_end_idx = min(process_start_idx + self.max_batch_size, len(images_batch))
                                        sub_batch = batch_tensor_t[process_start_idx:process_end_idx]
                                        sub_batch_output_tensor = self.context(sub_batch)
                                        feature_tensors.append(sub_batch_output_tensor.detach().cpu())

                                feature_tensors = np.concatenate(feature_tensors, axis=0)

                            output_tensor = torch.from_numpy(
                                np.array(feature_tensors).reshape(len(images_batch), extracted_feature_dim))
                            total_current_class_feature.append(output_tensor)

                            if len(total_current_class_feature):
                                current_class_idx = str(valid_class_idx + num_classes)
                                if flag_exist_contour_info:
                                    class_map[current_class_idx] = int(current_gt_info[0][4])
                                else:
                                    class_map[current_class_idx] = int(current_gt_info[4])
                                class_size[current_class_idx] = [int(current_img_mean_height), int(current_img_mean_width)]
                                with open(backup_json_mapper_path, 'w') as f:
                                    json.dump({'mapper': class_map, 'size': class_size}, f, indent=4)
                                valid_class_idx += 1

                                total_class_tensor = torch.load(backup_feature_db_path)
                                if total_class_tensor.dim() == 3:
                                    total_class_tensor = total_class_tensor.squeeze(1)

                                # Combine all new tensors into a single tensor
                                combined_new_tensor = torch.cat(total_current_class_feature, dim=0)
                                # Mean pooling
                                ###img_per_product = args.img_per_product
                                combined_new_tensor = combined_new_tensor.view(-1, len(total_current_class_feature) * len(
                                    images_batch), extracted_feature_dim)
                                combined_new_tensor = combined_new_tensor.mean(dim=1, keepdim=True)
                                total_class_tensor = total_class_tensor.unsqueeze(1)
                                # Append to the existing tensor
                                total_class_tensor = torch.cat((total_class_tensor, combined_new_tensor), dim=0).squeeze(1)
                                # Save the updated tensor
                                torch.save(total_class_tensor, backup_feature_db_path)
                                print(
                                    f"{CurrentDateTime(0)}[make_feature][mode_add] New Class Added - Class : {class_map[current_class_idx]} || Angle : {current_rotate_angle}")
                                print(
                                    f"{CurrentDateTime(0)}[make_feature][mode_add] New class tensor shape : {combined_new_tensor.shape} || Updated tensor shape: {total_class_tensor.shape}")
                    pbar.update(1)
                    progress = pbar.n / pbar.total
                    elapsed = pbar.format_dict['elapsed']
                    eta = (elapsed / progress) - elapsed if progress > 0 else None
                    self.eta = eta if eta is not None else self.eta
        shutil.copy(backup_feature_db_path, load_feature_db_path)
        shutil.copy(backup_json_mapper_path, json_mapper_path)
        print(
            f"{CurrentDateTime(0)}[make_feature][mode_add] Adding Feature DB Process is Finished. Save Path : {load_feature_db_path}")
        print(f"{CurrentDateTime(0)}[make_feature][mode_add] Json file saved to {json_mapper_path}")

        self.eta = 0

    def run(self):
        cc.artis_ai_json_config = cc.get_config(self.config_file_path, cc.artis_ai_json_config)
        self.ai_feature_mode = cc.artis_ai_json_config.get("ai_feature_mode", 1)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if self.ai_feature_mode == 0:
            loop.run_until_complete(self.all())
        elif self.ai_feature_mode == 1:
            loop.run_until_complete(self.add())
        loop.close()


if __name__ == "__main__":
    inf_feature_db = MakeFeatureDB(cc.path_to_config)
    processed_files = []

    file_path = "../../../sample/EdgeMan/db/db_sync_report.json"
    with open(file_path, 'rb') as f:
        file_content = f.read()
    processed_files.append({
        "original_name": "db_sync_report.json",
        "size": os.path.getsize(file_path),
        "data": file_content
    })
    db_root_path = os.path.join("/mynas/uploads/", "TestVendor", "db_key", "test_db_key_123")

    ret, msg = inf_feature_db.set_db(processed_files, db_root_path)
    if ret:
        inf_feature_db.run()
