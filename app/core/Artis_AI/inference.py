import zmq
import sys
import os.path
from kisane import Kisane
from depth import Depth
from feature_matching import Feature_Matching

import logging
import time
from datetime import datetime
import json
import numpy as np
import cv2

import common_config as cc
from common_config import make_artis_ai_log, CurrentDateTime

class StdoutRedirector(object):
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message.strip() != "":
            self.logger.info(message)

    def flush(self):
        pass

class CustomRotatingFileHandler(logging.Handler):
    def __init__(self, base_filename, maxBytes, backupCount):
        super().__init__()
        self.base_filename = base_filename
        self.maxBytes = maxBytes
        self.backupCount = backupCount
        self.log_directory = os.path.dirname(base_filename)  # Assuming base_filename includes directory path
        self.current_filename = self._get_new_filename()
        self.current_file = open(self.current_filename, 'a')
        self.current_size = os.path.getsize(self.current_filename)

    def _get_existing_files(self):
        return sorted([f for f in os.listdir(self.log_directory) if f.startswith(os.path.basename(self.base_filename)) and f.endswith('.log')])

    def _get_new_filename(self):
        existing_files = self._get_existing_files()
        if not existing_files:
            return os.path.join(self.log_directory, f"{os.path.basename(self.base_filename)}_00001.log")
        last_file = existing_files[-1]
        last_index = int(last_file.split('_')[-1].split('.')[0])
        new_index = last_index + 1
        return os.path.join(self.log_directory, f"{os.path.basename(self.base_filename)}_{new_index:05d}.log")

    def _rotate_file(self):
        self.current_file.close()
        self.current_filename = self._get_new_filename()
        self.current_file = open(self.current_filename, 'a')
        self.current_size = 0
        self._cleanup_old_files()

    def _cleanup_old_files(self):
        existing_files = self._get_existing_files()
        if len(existing_files) > self.backupCount:
            files_to_delete = existing_files[:len(existing_files) - self.backupCount]
            for file in files_to_delete:
                os.remove(os.path.join(self.log_directory, file))

    def emit(self, record):
        msg = self.format(record)
        msg_size = len(msg) + 1  # +1 for newline character
        if self.current_size + msg_size > self.maxBytes:
            self._rotate_file()
        
        self.current_file.write(msg + '\n')
        self.current_file.flush()
        self.current_size += msg_size

    def close(self):
        self.current_file.close()
        super().close()

def check_string(recv_str):
    recv_token = recv_str.split("|")
    cnt_token = len(recv_token)

    # 0x01 Protocol Error
    if recv_token[0] != "0x02" or recv_token[cnt_token - 1] != "0x03":
        ###send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x01" + "|" + "0x03"
        cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['communication']['invalid_packet_stx_or_etx']))
        cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['communication'][cc.artis_ai_error_code['communication']['invalid_packet_stx_or_etx']]
        cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
        send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x01" + "|" + "0x03"
        return False, send_str, []

    # 0x02 Undefined Command
    cmd_str = recv_token[1]
    if cmd_str not in ["0x01", "0x02", "0xEE", "0xFF", "0xF5", "0xF6"]:
        ###send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x02" + "|" + "0x03"
        cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['communication']['invalid_packet_cmd']))
        cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['communication'][cc.artis_ai_error_code['communication']['invalid_packet_cmd']]
        cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
        send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x02" + "|" + "0x03"
        return False, send_str, [cmd_str]

    if cmd_str == "0x01":
        return True, None, [0x01, int(recv_token[2]), float(recv_token[3]), float(recv_token[4]), float(recv_token[5])]
    elif cmd_str == "0xEE":
        return True, None, [0xEE]
    elif cmd_str == "0xFF":
        return True, None, [0xFF]
    elif cmd_str == "0xF5":
        config_path = recv_token[2]
        if os.path.exists(config_path) == False:
            send_str = "0x02" + "|" + "0xF5" + "|" + "NG" + "|" + "0x03"
            return False, send_str, [config_path]
        else:
            return True, None, [0xF5, config_path]
    elif cmd_str == "0xF6":
        # UpdateCropPoint: 0x02|0xF6|left_x|left_y|right_x|right_y|0x03
        if len(recv_token) < 7:
            send_str = "0x02" + "|" + "0xF6" + "|" + "NG" + "|" + "0x01" + "|" + "0x03"
            return False, send_str, [0xF6]

        left_x = int(recv_token[2])
        left_y = int(recv_token[3])
        right_x = int(recv_token[4])
        right_y = int(recv_token[5])

        return True, None, [0xF6, left_x, left_y, right_x, right_y]

    # 0x03 Unsupported Image Count
    num_img = int(recv_token[2])
    if not (recv_token[2] == "1" or recv_token[2] == "3"):
        ###send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x03" + "|" + "0x03"
        cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['image']['invalid_image_count_for_inference']))
        cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['image'][cc.artis_ai_error_code['image']['invalid_image_count_for_inference']]
        cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
        send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x03" + "|" + "0x03"
        return False, send_str, [0x02, num_img]

    # 0x04 Input Absolute Path < Image Count
    #if 2 + num_img < cnt_token and \
    #    (".png" not in recv_token[2+num_img] or ".jpg" not in recv_token[2+num_img]):
    #    send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x04" + "|" + "0x03"
    #    return False, send_str, [0x02, num_img]

    img_path = recv_token[3:3 + num_img]
    # 0x05 Image file is not exist
    for img in img_path:
        if os.path.exists(img) == False:
            ###send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x05" + "|" + "0x03"
            cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['image']['not_exist_inference_image']))
            cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['image'][cc.artis_ai_error_code['image']['not_exist_inference_image']]
            cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
            send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x05" + "|" + "0x03"
            return False, send_str, [0x02, num_img, img_path]

    # 0x06 Debug option is not exist
    debug_str = recv_token[3 + num_img]
    if debug_str == "0x03":
        send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x06" + "|" + "0x03"
        return False, send_str, [0x02, num_img, img_path]

    if debug_str == "True":
        is_debug = True
    else:
        is_debug = False

    # 0x07 Debug Save Path is not exist
    save_path = recv_token[4 + num_img]
    if is_debug == True and save_path == "0x03":
        ###send_str = "0x02" + "|" + "0x02" + "|" + + "NG" + "|" + "0x07" + "|" + "0x03"
        cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['image']['invalid_debug_image']))
        cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['image'][cc.artis_ai_error_code['image']['invalid_debug_image']]
        cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
        send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x07" + "|" + "0x03"
        return False, send_str, [0x02, num_img, img_path, is_debug]

    if not is_debug:
        save_path = ""

    # 0x08 Input images not exist
    for img in img_path:
        with open(img, "r") as file:
            file.seek(0, os.SEEK_END)
            if int(file.tell()) < 1000:
                ###send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x08" + "|" + img + "|" + "0x03"
                cc.artis_ai_current_error_code = str(hex(cc.artis_ai_error_code['image']['invalid_inference_image']))
                cc.artis_ai_current_error_reason = cc.artis_ai_error_reason['image'][cc.artis_ai_error_code['image']['invalid_inference_image']]
                cc.artis_ai_result_json = cc.make_error_code(cc.artis_ai_result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
                send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + "0x08" + "|" + img + "|" + "0x03"
                return False, send_str, [0x02, num_img, img_path, is_debug]

    run_mode = recv_token[8]

    #print("SAVE_PATH : " + save_path + "\n")
    return True, None, [0x02, img_path, is_debug, save_path, run_mode]

def make_default_result_json(result_json):
    result_json['artis_error_info'] = {'version': {}, 'error_code': '0x00000000', 'error_reason': 'OK',
                                       'error_code_detail':[], 'error_reason_detail':[],
                                       'object_total_cnt': 0, 'object_bbox': {}}
    json_artis_ai_version = {
        "Artis_AI": str(cc.artis_ai_version), 
        "Artis_AI_Model": str(cc.artis_ai_model_version),
        "Artis_AI_DB": str(cc.artis_ai_db_version)
    }
    result_json['artis_error_info']['version'].update(json_artis_ai_version)
    if cc.artis_ai_model_integrity == "fail":
        result_json['integrity_check'] = {
            "result": "fail",
            "fail_list": cc.artis_ai_model_fail_list
        }
    result_json['artis_error_info']['object_total_cnt'] = 0
    result_json['artis_error_info']['object_bbox'] = {}

    result_json['artis_result_info'] = {'object_total_cnt': 0}
    result_json['artis_object_bbox'] = {}
    result_json['artis_object_detail'] = {}
    result_json['artis_object_score'] = {}

    result_json['depth'] = {'object_bbox':{}, 'object_detail':{}, 'object_score':{}}
    result_json['depth']['result_info'] = {'object_total_cnt': 0}

    result_json['seg'] = {'object_bbox': {}, 'object_detail': {}, 'object_score': {}, 'object_contour':{}}
    result_json['seg']['result_info'] = {'object_total_cnt': 0}

    result_json['seg_oc'] = {'object_bbox': {}, 'object_detail': {}, 'object_score': {}, 'object_contour': {}}
    result_json['seg_oc']['result_info'] = {'object_total_cnt': 0}

    result_json['rgb'] = {'object_bbox': {}, 'object_detail': {}, 'object_score': {}}
    result_json['rgb']['result_info'] = {'object_total_cnt': 0}
    #result_json['log_Artis_AI'] = []

    return result_json

def calculate_depth_offset(depth_file):
    if not os.path.exists(depth_file):
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : There is no depth file ({depth_file})")
        return None

    try:
        with open(depth_file, 'rb') as f:
            len_header = np.fromfile(f, dtype=np.uint16, count=1)[0]
            # 안전한 디코딩 시도
            try:
                header_str = f.read(len_header).decode('utf-8')
            except UnicodeDecodeError:
                try:
                    header_str = f.read(len_header).decode('utf-8-sig')
                except UnicodeDecodeError:
                    try:
                        header_str = f.read(len_header).decode('latin-1')
                    except UnicodeDecodeError:
                        header_str = f.read(len_header).decode('utf-8', errors='ignore')

            if header_str != 'Version':
                print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : Invalid depth header: {header_str}")
                return None

            _ = np.fromfile(f, dtype=np.uint16, count=3)
            depth = np.fromfile(f, dtype=np.uint16).reshape(cc.image_resolution_depth)
    except Exception as e:
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : Failed to read depth file ({depth_file})")
        return None

    # 중앙 100x100 영역에서 유효 Depth 값 추출
    window_size = 100
    ref = cc.depth_reference_distance
    min_valid, max_valid = ref - 30, ref + 30 # 440 ~ 500mm

    h, w = cc.image_resolution_depth
    cy, cx = h // 2, w // 2
    region = depth[cy - window_size//2:cy + window_size//2, cx - window_size//2:cx + window_size//2]
    valid_values = region[(region > min_valid) & (region < max_valid)]

    # 유효 Depth 값이 없으면 None 반환
    if valid_values.size == 0:
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : 유효한 Depth 값이 없습니다.")
        return None
    
    # 유효 Depth 값이 너무 적으면 None 반환
    if valid_values.size < window_size*window_size*0.5:  # 50% 미만이면
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : 유효한 Depth 값이 너무 적습니다. ({valid_values.size}/{window_size*window_size})")
        return None

    # 이상치 제거
    q1 = np.percentile(valid_values, 25)
    q3 = np.percentile(valid_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_values = valid_values[(valid_values >= lower_bound) & (valid_values <= upper_bound)]

    if filtered_values.size == 0:
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : 이상치 제거 후 유효한 값이 없습니다.")
        return None

    # 중앙값 계산 및 보정값 계산
    median_depth = np.median(filtered_values)
    offset = ref - median_depth

    # 보정값 범위 제한
    max_offset = 50  # 최대 50mm까지만 보정
    if abs(offset) > max_offset:
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정값이 너무 큽니다. ({offset:.1f}mm -> {max_offset}mm로 제한)")
        offset = max_offset if offset > 0 else -max_offset

    # 상세 로깅
    print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 정보:")
    print(f"{CurrentDateTime(0)}   - 검사 영역: {window_size}x{window_size} (중앙)")
    print(f"{CurrentDateTime(0)}   - 유효 범위: {min_valid}~{max_valid}mm")
    print(f"{CurrentDateTime(0)}   - 유효값 개수: {valid_values.size}/{window_size*window_size}")
    print(f"{CurrentDateTime(0)}   - 중앙값: {median_depth:.1f}mm")
    print(f"{CurrentDateTime(0)}   - 보정값: {offset:.1f}mm")

    return offset

def load_depth_offset_from_file():
    depth_offset_file = os.path.join(cc.path_depth_offset, "depth_offset.txt")
    
    if os.path.exists(depth_offset_file):
        try:
            with open(depth_offset_file, 'r') as f:
                depth_offset_str = f.read().strip()
                if depth_offset_str and depth_offset_str != "None":
                    depth_offset = float(depth_offset_str)
                    print(f"{CurrentDateTime(0)} [Artis_AI] depth_offset loaded from file: {depth_offset}")
                    return depth_offset
                else:
                    print(f"{CurrentDateTime(0)} [Artis_AI] depth_offset file contains invalid value: {depth_offset_str}")
        except Exception as e:
            print(f"{CurrentDateTime(0)} [Artis_AI] Failed to read depth_offset file: {e}")
    
    return None

def save_depth_offset_to_file(depth_offset):
    depth_offset_file = os.path.join(cc.path_depth_offset, "depth_offset.txt")
    
    try:
        with open(depth_offset_file, 'w') as f:
            f.write(str(depth_offset) + '\n')
        print(f"{CurrentDateTime(0)} [Artis_AI] depth_offset saved to file: {depth_offset_file}")
        return True
    except Exception as e:
        print(f"{CurrentDateTime(0)} [Artis_AI] Failed to save depth_offset file: {e}")
        return False

def calculate_and_save_depth_offset(crop_lx, crop_ly, crop_rx, crop_ry, crop_width, crop_height, cmd_args):
    depth_file = os.path.join(cc.path_depth_offset, "Cam_2_Depth.bin")
    
    if not os.path.exists(depth_file):
        img_l, img_r = do_preprocess(crop_lx, crop_ly, crop_rx, crop_ry, crop_width, crop_height)
        
        if img_l is not None and img_r is not None:
            ret, dep_time = d1.inference([0, img_l, img_r], cmd_args[2], cmd_args[3], cmd_args[4], 0)
            depth_offset = calculate_depth_offset(depth_file)
            print(f"{CurrentDateTime(0)} [Artis_AI] depth_offset calculated: {depth_offset}")
        else:
            return None
    else:
        depth_offset = calculate_depth_offset(depth_file)
        print(f"{CurrentDateTime(0)} [Artis_AI] depth_offset calculated: {depth_offset}")
    
    if depth_offset is not None:
        save_depth_offset_to_file(depth_offset)
    
    return depth_offset

def do_preprocess(crop_lx, crop_ly, crop_rx, crop_ry, crop_width, crop_height):
    temp_dir = cc.path_depth_offset
    input_img_l = os.path.join(temp_dir, "Cal_left.jpg")
    input_img_r = os.path.join(temp_dir, "Cal_right.jpg")
    output_img_l = os.path.join(temp_dir, "Cam_2_Color.jpg")
    output_img_r = os.path.join(temp_dir, "Cam_1_Color.jpg")

    if not os.path.exists(input_img_l) or not os.path.exists(input_img_r):
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : Input image files not found in {temp_dir}")
        return None, None
    
    left_img_data = cv2.imread(input_img_l)
    right_img_data = cv2.imread(input_img_r)

    if left_img_data is None or right_img_data is None:
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : Failed to read image files")
        return None, None
    
    # 왼쪽 이미지 crop
    height, width = left_img_data.shape[:2]
    x1 = max(0, crop_lx)
    y1 = max(0, crop_ly)
    x2 = min(width, crop_lx + crop_width)
    y2 = min(height, crop_ly + crop_height)
    left_img_data = left_img_data[y1:y2, x1:x2]
    
    # 오른쪽 이미지 crop
    height, width = right_img_data.shape[:2]
    x1 = max(0, crop_rx)
    y1 = max(0, crop_ry)
    x2 = min(width, crop_rx + crop_width)
    y2 = min(height, crop_ry + crop_height)
    right_img_data = right_img_data[y1:y2, x1:x2]

    cv2.imwrite(output_img_l, left_img_data)
    cv2.imwrite(output_img_r, right_img_data)
    
    return output_img_l, output_img_r

if __name__=="__main__":
    args = sys.argv
    port_num = "15000"
    config_file_path = os.getcwd() + "/" + cc.artis_ai_config_file
    log_on_off = "On"
    cal_file_path = cc.path_to_cal_file
    min_depth, max_depth = 200, 500
    crop_lx, crop_ly = 180, 0
    crop_rx, crop_ry = 130, 0
    crop_width, crop_height = 1600, 1200

    if len(args) == 1:
        if not os.path.exists(config_file_path):
            print(f"{CurrentDateTime(0)} [Artis_AI] Fail to open json file\n")
            print(f"{CurrentDateTime(0)} [Artis_AI] {config_file_path} not exist!!!!")
            sys.exit(0)
            
        cc.artis_ai_json_config = cc.get_config(config_file_path, cc.artis_ai_json_config)
        port_num = str(cc.artis_ai_json_config["zmq_port_ai_inference"])
        min_depth, max_depth = cc.artis_ai_json_config["depth_min"], cc.artis_ai_json_config["depth_max"]
        crop_lx, crop_ly = cc.artis_ai_json_config["crop_lx"], cc.artis_ai_json_config["crop_ly"]
        crop_rx, crop_ry = cc.artis_ai_json_config["crop_rx"], cc.artis_ai_json_config["crop_ry"]
        crop_width, crop_height = cc.artis_ai_json_config["crop_width"], cc.artis_ai_json_config["crop_height"]
        log_on_off = cc.artis_ai_json_config["log_on_off"]
        
        print(f"{CurrentDateTime(0)} [Artis_AI] Success to parse json file\n")
        print(f"{CurrentDateTime(0)} [Artis_AI] {cc.artis_ai_json_config}\n")
    elif len(args) == 13:
        port_num = args[1]
        config_file_path = args[2]
        log_on_off = args[3]
        cal_file_path = args[4]
        min_depth = int(args[5])
        max_depth = int(args[6])
        crop_lx = int(args[7])
        crop_ly = int(args[8])
        crop_rx = int(args[9])
        crop_ry = int(args[10])
        crop_width = int(args[11])
        crop_height = int(args[12])

        cc.artis_ai_json_config["zmq_port_ai_inference"] = port_num
        cc.artis_ai_json_config["log_on_off"] = log_on_off
        cc.artis_ai_json_config["depth_min"], cc.artis_ai_json_config["depth_max"] = min_depth, max_depth
        cc.artis_ai_json_config["crop_lx"], cc.artis_ai_json_config["crop_ly"] = crop_lx, crop_ly
        cc.artis_ai_json_config["crop_rx"], cc.artis_ai_json_config["crop_ry"] = crop_rx, crop_ry
        cc.artis_ai_json_config["crop_width"], cc.artis_ai_json_config["crop_height"] = crop_width, crop_height

        print(f"{CurrentDateTime(0)} [Artis_AI] Success to parse argument\n")
        print(f"{CurrentDateTime(0)} [Artis_AI] {args}\n")
    else:
        print(f"{CurrentDateTime(0)} [Artis_AI] Fail to parse argument\n")
        print(f"{CurrentDateTime(0)} [Artis_AI] {args}\n")
        print(f"{CurrentDateTime(0)} [Artis_AI] Ex) python inference.py port_num(15000) config_file_path log(On/Off) [cal_file_path depth_min depth_max crop_args]")
        sys.exit(0)

    sys.stdout.flush()
    sys.stderr.flush()

    if log_on_off == "On":
        # 현재 프로그램 실행 경로를 가져오고 출력
        current_path = os.path.abspath(os.path.dirname(__file__))
        print(f"{CurrentDateTime(0)} [Artis_AI] Current execution path: {current_path}")

        # 로그 파일 경로 설정
        #log_directory = os.path.join(current_path, '/home/nvidia/JetsonMan/log/Artis_AI')
        log_directory = os.path.join(current_path, cc.path_to_log + '/Artis_AI')
        folder_name_date = time.strftime("%Y%m%d")
        log_directory = os.path.join(log_directory, folder_name_date)
        os.makedirs(log_directory, exist_ok=True)  # 디렉토리 없으면 생성

        # 프로그램 시작 시간
        ###start_time = time.strftime("%Y%m%d_%H%M%S")
        start_time = time.strftime("%H%M%S")

        # 로그 파일 핸들러 설정
        log_file = os.path.join(log_directory, f'Artis_AI_{start_time}')
        handler = CustomRotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=50)

        # 로거 설정
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        # 표준 출력을 로거로 리다이렉션
        sys.stdout = StdoutRedirector(logger)
        sys.stderr = StdoutRedirector(logger)

        print(f"{CurrentDateTime(0)} [Artis_AI] Number of arguments: {len(args)}")
        print(f"{CurrentDateTime(0)} [Artis_AI] inference.py execution arguments:")
        for i, arg in enumerate(sys.argv):
            print(f"{CurrentDateTime(0)} [{i}]: {arg}")

    print(f"{CurrentDateTime(0)} [Artis_AI] Version Information : {cc.artis_ai_version}\n")
    print(f"{CurrentDateTime(0)} [Artis_AI] Model Version Information : {cc.artis_ai_model_version}")
    print(f"{CurrentDateTime(0)} [Artis_AI] Model Integrity Check : {cc.artis_ai_model_integrity}\n")
    if cc.artis_ai_model_integrity == "fail":
        print(f"{CurrentDateTime(0)} [Artis_AI] Model Integrity Fail List : {cc.artis_ai_model_fail_list}\n")
        
    k1 = Kisane(config_file_path)
    d1 = Depth(cal_file_path, [min_depth, max_depth], [crop_lx, crop_ly, crop_rx, crop_ry, crop_width, crop_height], config_file_path)
    k1.set_depth_instance(d1)  # Depth 인스턴스 연결
    f1 = Feature_Matching(config_file_path)

    context = zmq.Context()
    client_socket = context.socket(zmq.REP)
    client_socket.connect("tcp://localhost:" + port_num)
    print(f"{CurrentDateTime(0)} [Artis_AI] ZeroMQ Connect")

    is_on = True
    inf_cnt = 0

    while is_on:
        cc.artis_ai_current_error_code = 0
        cc.artis_ai_current_error_reason = ''
        cc.artis_ai_current_log = ''
        cc.artis_ai_result_json = make_default_result_json(cc.artis_ai_result_json)

        recv_string = client_socket.recv_string()

        cc.artis_ai_current_log = 'recv (JetsonMan -> Artis_AI)'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
        ###print(f"[Artis_AI] recv (JetsonMan -> Artis_AI)")
        cc.artis_ai_current_log = f'recv string : {recv_string}'
        make_artis_ai_log(cc.artis_ai_current_log, 'debug', False)
        ###print(f"[Artis_AI] [recv_string] {recv_string}")

        is_pass, send_str, cmd_args = check_string(recv_string)
        send_result_json = ''
        result_format = 0
        cc.artis_ai_current_log = f'CMD : {cmd_args}'
        make_artis_ai_log(cc.artis_ai_current_log, 'debug', False)
        ###print(f"[Artis_AI] CMD : {cmd_args}")

        if is_pass:
            cc.artis_ai_current_log = 'Parsing Success'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
            ###print(f"[Artis_AI] Parsing Success")

            if cmd_args[0] == 0x01:
                ret, time_list = k1.warm_up(cmd_args[1], cmd_args[2], cmd_args[3])
                if ret:
                    ret, dep_time = d1.warm_up(cmd_args[1], cmd_args[4])
                    time_list[2] = dep_time
                
                if ret:
                    send_str = "0x02" + "|" + "0x01" + "|" + "OK" + "|" + time_list[0] + "|" + time_list[1] + "|" + time_list[2] + "|" + "0x03"
                else:
                    send_str = "0x02" + "|" + "0x01" + "|" + "NG" + "|" + time_list[0] + "|" + time_list[1] + "|" + time_list[2] + "|" + "0x03"

            elif cmd_args[0] == 0x02:
                ###send_str, total_time, bbox_output = k1.inference(cmd_args[1], False, cmd_args[3])
                ###ret, dep_time = d1.inference(cmd_args[1], cmd_args[2], cmd_args[3], cmd_args[4], bbox_output)
                
                # ======================= Depth 이미지 중앙값 계산 (첫 번째 추론 시에만 실행) =======================
                loaded_offset = load_depth_offset_from_file()
                if loaded_offset is None:
                    cc.depth_offset = calculate_and_save_depth_offset(crop_lx, crop_ly, crop_rx, crop_ry, crop_width, crop_height, cmd_args)
                else:
                    cc.depth_offset = loaded_offset
                # ===========================================================================================

                ret, dep_time = d1.inference(cmd_args[1], cmd_args[2], cmd_args[3], cmd_args[4], cc.depth_offset)

                cc.artis_ai_current_log = f'OD & OC Start'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

                send_str, total_time, bbox_output, result_format, ai_combine_mode = k1.inference(cmd_args[1], False, cmd_args[3], cmd_args[4])

                total_time += dep_time

                if ai_combine_mode >= cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
                    feature_matching_score, feature_matching_id, feature_matching_time = f1.inference_by_feature_matching(cmd_args[1], bbox_output)
                    total_time += feature_matching_time
                    print(f"[feature_matching] Processing Time : {feature_matching_time} ms")
                    print(f"[feature_matching] Score Info\n{feature_matching_score}")
                    print(f"[feature_matching] Class Info\n{feature_matching_id}")

                cc.artis_ai_current_log = f'ret: {ret}'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

                cc.artis_ai_current_log = f'OD & OC Finish'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

                ###print(f"[Artis_AI] ret: {ret}")

                if (ret == "OK"):
                    send_str += str(round(dep_time, 2)) + "|" + str(round(total_time, 2)) + "|" + "0x03"
                    cc.artis_ai_current_log = f'총 추론 소요 시간 : {total_time:.2f} ms, 누적 추론 카운트 : {inf_cnt}'
                    inf_cnt += 1
                    make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
                    ###print(f"[Artis_AI] 총 추론 소요 시간 : {total_time:.2f} ms, 누적 추론 카운트 : {inf_cnt}")

                else:
                    send_str = "0x02" + "|" + "0x02" + "|" + "NG" + "|" + ret + "|" + "0x03"
                    cc.artis_ai_current_log = f'Failed to Depth Map Generation'
                    make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

            elif cmd_args[0] == 0xF5:
                send_str = k1.reinit(cmd_args[1])
            elif cmd_args[0] == 0xF6:
                left_x, left_y, right_x, right_y = cmd_args[1:5]

                cc.artis_ai_json_config["crop_lx"] = left_x
                cc.artis_ai_json_config["crop_ly"] = left_y
                cc.artis_ai_json_config["crop_rx"] = right_x
                cc.artis_ai_json_config["crop_ry"] = right_y

                d1.update_crop_settings(left_x, left_y, right_x, right_y)

                send_str = "0x02" + "|" + "0xF6" + "|" + "OK" + "|" + "0x03"
            elif cmd_args[0] == 0xEE:
                # Do check something
                send_str = "0x02" + "|" + "0xEE" + "|" + "OK" + "|" + "0x03"
            elif cmd_args[0] == 0xFF:
                send_str = "0x02" + "|" + "0xFF" + "|" + "OK" + "|" + "0x03"
                is_on = False
        else:
            cc.artis_ai_current_log = 'Parsing Fail'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
            ###print(f"[Artis_AI] Parsing Fail")

        if cmd_args[0] == 0x02:
            if not result_format:
                client_socket.send_string(send_str)
                print(f"{CurrentDateTime(0)} [Artis_AI] send result by normal format (Artis_AI -> JetsonMan)")
                print(f"{CurrentDateTime(0)} [Artis_AI] {send_str}")
            else:
                cc.artis_ai_current_log = 'send result by json format (Artis_AI -> JetsonMan)'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

                result_json = json.dumps(cc.artis_ai_result_json, ensure_ascii=False)
                send_result_json += "0x02" + "|" + "0x02" + "|" + "JSON" + "|" + result_json + "|" + "0x03"

                client_socket.send_string(send_result_json)
                ###print(f"[Artis_AI] send result by json format (Artis_AI -> JetsonMan)")
                ###print(f"[Artis_AI] {send_result_json}")
        else:
            client_socket.send_string(send_str)
            print(f"{CurrentDateTime(0)} [Artis_AI] send result by normal format (Artis_AI -> JetsonMan)")
            print(f"{CurrentDateTime(0)} [Artis_AI] {send_str}")

    print(f"{CurrentDateTime(0)} [Artis_AI] ZeroMQ Disconnect")
    client_socket.disconnect("tcp://localhost:" + port_num)

    if log_on_off == "On":
        # 표준 출력 복원
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__