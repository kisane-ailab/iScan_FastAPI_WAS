import os
import time
from enum import Enum

class CamInfo(Enum):
    LEFT = "left"      # Cam_2, 왼쪽 카메라
    RIGHT = "right"    # Cam_1, 오른쪽 카메라
    SINGLE = "single"  # Cam_Single, 단일 카메라

artis_ai_version_major = 0
artis_ai_version_minor = 3
artis_ai_version_inner = 6
artis_ai_version = str(artis_ai_version_major) + "." + str(artis_ai_version_minor) + "." + str(artis_ai_version_inner)

artis_ai_db_version_major = 0
artis_ai_db_version_minor = 0
artis_ai_db_version_inner = 3
artis_ai_db_version = str(artis_ai_db_version_major) + "." + str(artis_ai_db_version_minor) + "." + str(artis_ai_db_version_inner)

artis_ai_model_version = "0.0.0"

artis_ai_depth_format_ver_major = 0
artis_ai_depth_format_ver_minor = 0
artis_ai_depth_format_ver_inner = 1

artis_debug_image_path = "\\temp\\artis_result_debug.jpg"

'''path_to_root = "/home/nvidia/JetsonMan/"
path_to_checkpoint = path_to_root + "/resource/Artis_AI/checkpoints/"
path_depth_offset = path_to_root + "/resource/camera/temp"
path_to_log = path_to_root + "/log/"

path_to_cal_file = path_to_root + "/resource/camera/calibration_results.xml"'''

path_to_root = os.path.dirname(os.path.abspath(__file__))
path_to_checkpoint = path_to_root + "/checkpoints/"
path_depth_offset = path_to_root + "/camera/temp"
path_to_log = path_to_root + "/log/"

path_to_config = path_to_root + "/kisan_config.json"
path_to_cal_file = path_to_root + "/camera/calibration/default/calibration_results.xml"

artis_transform_img_file = "artis_transform_debug.jpg"
artis_fusion_img_file = "artis_fusion_debug.jpg"
artis_debug_img_file = "artis_combined_debug.jpg"

artis_ai_config_file = "kisan_config.json"
artis_edge_config = "edge_manager_setting"

artis_ai_det_0_config_path = path_to_checkpoint + "yolof_depth_config.py"
artis_ai_det_0_model_path = path_to_checkpoint + "yolof_depth.pth"
artis_ai_det_0_trt_config_path = path_to_checkpoint + "yolof_depth_trt_config.py"

artis_ai_sg_model_path = {"yolac": path_to_checkpoint + "best_model_seg.pth",
                          "yolov8": path_to_checkpoint + "best_model_seg.pt"}
artis_ai_sg_model_config = "yolact_resnet50_config"
artis_ai_sg_config_path = path_to_checkpoint + "best_config_seg.py"

artis_ai_sg_oc_model_path = path_to_checkpoint + "best_model_seg_oc.pt"

artis_ai_det_1_config_path = path_to_checkpoint + "best_config.py"
artis_ai_det_1_model_path = path_to_checkpoint + "best_model.pth"
artis_ai_det_1_trt_config_path = path_to_checkpoint + "yolof_rgb_trt_config.py"

artis_ai_det_0_lookup_path = path_to_checkpoint + "db_class_info_depth.json"
artis_ai_det_1_lookup_path = path_to_checkpoint + "db_class_info.json"
artis_depth_lookup_path = path_to_checkpoint + "depth_class_info.json"
artis_sg_lookup_path = path_to_checkpoint + "db_class_info_seg.json"
artis_sg_oc_lookup_path = path_to_checkpoint + "db_class_info_seg_oc.json"

artis_ai_cls_0_config_path = path_to_checkpoint + "rm_cls_config.json"
artis_ai_cls_0_model_path = path_to_checkpoint + "model_best.pth.tar"

artis_ai_gen_depth_model_path = path_to_checkpoint + "pretrained_RAFT_fast.pth"

artis_ai_feature_db_path = path_to_checkpoint + 'feature_db.pt'
artis_ai_feature_extract_model = path_to_checkpoint + 'feature_extractor.pth'
artis_ai_feature_lookup_path = path_to_checkpoint + 'db_class_info_cls.json'

# Artis_AI 모델 경로
path_kisane_model = "/run/JetsonMan/Artis_AI/model/kisane_model.pt"
path_depth_model = "/run/JetsonMan/Artis_AI/model/depth_model.pt"
path_depth_lookup = "/run/JetsonMan/Artis_AI/model/depth_lookup.json"

image_resolution_rgb = [960, 1280]
image_resolution_depth = [image_resolution_rgb[0] // 2, image_resolution_rgb[1] // 2]

depth_reference_distance = 470
### Definition of AI Model Mode ###
artis_ai_model_mode = {"mode_rgb_only": 0,                      #Only RGB Det/Cls Model
                       "mode_rgb_with_depth": 1,                #RGB Det/Cls Model + RGBD Det Model
                       "mode_rgb_with_seg": 2,                  #RGB Det/Cls Model + Seg Det Model
                       "mode_seg_with_seg":3,                   #Seg Det + Cls Model
                       "mode_depth_with_feature_matching": 4,   #RGBD Det Model + Feature Matching Cls Model
                       "mode_seg_with_feature_matching": 5}     #Seg Det Model + Feature Matching Cls Model
artis_ai_model_mode_detail = {v:k for k,v in artis_ai_model_mode.items()}

### Definition of Error Code for Artis AI ###
artis_ai_error_code = dict()
artis_ai_error_reason = dict()
artis_ai_error_code["communication"] = {"invalid_packet_stx_or_etx": 0x10000000,
                                     "invalid_packet_cmd": 0x10000001}
artis_ai_error_reason["communication"] = {v:k for k,v in artis_ai_error_code["communication"].items()}

artis_ai_error_code["image"] = {"invalid_image_count_for_inference": 0x20000000,
                             "not_exist_inference_image": 0x20000001,
                             "invalid_debug_image": 0x20000002,
                             "invalid_inference_image": 0x20000003}
artis_ai_error_reason["image"] = {v:k for k,v in artis_ai_error_code["image"].items()}

artis_ai_error_code["inference"] = {"detect_overlaped_item":  0x30000000,
                                    "detect_untrained_item":  0x30000100,
                                    "detect_invalid_item":    0x30000101,
                                    "detect_reflection_item": 0x30000200,
                                    "detect_stand_item":      0x30000201,
                                    "detect_max_depth_item":  0x30000202}
'''
artis_ai_error_code["inference"] = {"detect_overlaped_item":  0x30000000,
                                    "detect_untrained_item":  0x30000100,
                                    "detect_untrained_view":  0x30000101,
                                    "detect_invalid_item":    0x30000102,
                                    "detect_reflection_item": 0x30000200,
                                    "detect_stand_item":      0x30000201,
                                    "detect_max_depth_item":  0x30000202}
'''
artis_ai_error_reason["inference"] = {v:k for k,v in artis_ai_error_code["inference"].items()}

artis_ai_current_error_code = 0
artis_ai_current_error_reason = ""
artis_ai_current_log = ""
artis_ai_current_log_time = ""
artis_ai_result_json = {}

# RGB / Depth bbox가 일치하지 않는 경우에 대한 클래스 번호
artis_ai_class_untrained_item = 9999999
# 인식 Score가 낮아 미인식 처리가 된 경우에 대한 클래스 번호
artis_ai_class_untrained_view = 9999998
# 손, 핸드폰 등의 유효하지 않은 아이템에 대한 클래스 번호
artis_ai_class_invalid_item = 9999900

def CurrentDateTime(format_type=0):
    if format_type == 0:
        t = time.time()
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}" + "]"
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    return timestamp

def make_error_code(result_json, error_code_current, error_reason_current):
    error_code_before = result_json['artis_error_info']['error_code']
    error_reason_before = result_json['artis_error_info']['error_reason']
    #error_code_detail = list(map(list, result_json['artis_error_info']['error_code_detail'].items()))
    #error_reason_detail = list(map(list, result_json['artis_error_info']['error_reason_detail'].items()))
    error_code_detail = result_json['artis_error_info']['error_code_detail']
    error_reason_detail = result_json['artis_error_info']['error_reason_detail']
    if error_code_before != 'None':
        error_code_no_before = int(error_code_before, 16)
    else:
        error_code_no_before = 0x00000000
    if error_code_current != 'None':
        error_code_no_current = int(error_code_current, 16)
    else:
        error_code_no_current = 0x00000000
    if error_code_no_before < error_code_no_current:
        if error_code_before != '0x00000000' and error_reason_before != 'None':
            if not error_code_before in error_code_detail:
                error_code_detail.append(error_code_before)
            if not error_reason_before in error_reason_detail:
                error_reason_detail.append(error_reason_before)
        result_json['artis_error_info']['error_code'] = error_code_current
        result_json['artis_error_info']['error_reason'] = error_reason_current
        result_json['artis_error_info']['error_code_detail'] = error_code_detail
        result_json['artis_error_info']['error_reason_detail'] = error_reason_detail
    else:
        if error_code_current != '0x00000000' and error_reason_current != 'None':
            if not error_code_current in error_code_detail:
                error_code_detail.append(error_code_current)
            if not error_reason_current in error_reason_detail:
                error_reason_detail.append(error_reason_current)
        result_json['artis_error_info']['error_code'] = error_code_before
        result_json['artis_error_info']['error_reason'] = error_reason_before
        result_json['artis_error_info']['error_code_detail'] = error_code_detail
        result_json['artis_error_info']['error_reason_detail'] = error_reason_detail

    return result_json

def make_artis_ai_log(log_string, log_category, flag_print):
    current_log_time = CurrentDateTime(0)
    # To prevent parsing error in edge/jetson manager
    log_string = log_string.replace("|", ", ")
    ###current_log = current_log_time + " [Artis_AI] " + "[" + log_category + "] " + str(log_string) + "\n"
    current_log = current_log_time + " [Artis_AI] " + "[" + log_category + "] " + str(log_string)
    #save_log = current_log + "\n"
    #artis_ai_result_json["log_Artis_AI"].append(save_log)

    if flag_print:
        print(f"{current_log}")

    return True

## Set Version ##
artis_ai_model_integrity = "none"
artis_ai_model_fail_list = []

from utils.checksum import validate_checksum
if os.path.exists(path_to_checkpoint + "version.json"):
    artis_ai_model_version, artis_ai_model_integrity, artis_ai_model_fail_list = validate_checksum(path_to_checkpoint, "version.json")
elif os.path.exists(path_to_checkpoint + "version.txt"):
    with open(path_to_checkpoint + "version.txt") as file:
        artis_ai_model_version = file.readline()

os.makedirs(path_to_log + "/Artis_AI_Model/", exist_ok=True)
with open(path_to_log + "/Artis_AI_Model/version.txt", "w") as version_txt:
    version_txt.write(artis_ai_model_version)

#version_history_txt_dir = '/home/nvidia/JetsonMan/log/Artis_AI'
version_history_txt_dir = path_to_log + '/Artis_AI/'
os.makedirs(version_history_txt_dir, exist_ok=True)
version_history_txt_dir = os.path.join(version_history_txt_dir, 'version.txt')
with open(version_history_txt_dir, 'w') as version_txt:
    version_txt.write(artis_ai_version)

## Set Config
import json, re
artis_ai_json_config = {"log_on_off":"On"}
def remove_json_comments(json_str):
    # // 주석 제거
    json_str = re.sub(r'//.*', '', json_str)
    # /* */ 주석 제거
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    return json_str

def get_config(config_file_path, artis_ai_json_config):
    if os.path.exists(config_file_path):
        with open(config_file_path, encoding="utf-8") as json_file:
            json_raw = json_file.read()
            json_config = json.loads(remove_json_comments(json_raw))
            if artis_edge_config in json_config:
                for key, value in json_config[artis_edge_config].items():
                    if isinstance(value, dict) and key == "crop":
                        artis_ai_json_config["crop_lx"] = value.get("left_x", 180)
                        artis_ai_json_config["crop_ly"] = value.get("left_y", 0)
                        artis_ai_json_config["crop_rx"] = value.get("right_x", 130)
                        artis_ai_json_config["crop_ry"] = value.get("right_y", 0)
                        artis_ai_json_config["crop_width"] = value.get("width", 1600)
                        artis_ai_json_config["crop_height"] = value.get("height", 1200)
                    else:
                        artis_ai_json_config[key] = value
            
            if artis_ai_config_file in json_config:
                for key, value in json_config[artis_ai_config_file].items():
                    if isinstance(value, dict):
                        if key == "multi_view_processing":
                            for subkey, subval in value.items():
                                artis_ai_json_config[subkey] = subval
                        else:
                            for subkey, subval in value.items():
                                if key not in artis_ai_json_config:
                                    artis_ai_json_config[key] = {}
                                artis_ai_json_config[key][subkey] = subval
                    else:
                        artis_ai_json_config[key] = value

            for key, value in json_config.items():
                if key == artis_edge_config or key == artis_ai_config_file:
                    continue

                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        if key not in artis_ai_json_config:
                            artis_ai_json_config[key] = {}
                        artis_ai_json_config[key][subkey] = subval
                else:
                    artis_ai_json_config[key] = value

    return artis_ai_json_config