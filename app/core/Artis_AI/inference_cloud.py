import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import common_config as cc
from common_config import CamInfo
from kisane import Kisane
from depth import Depth
from inference import make_default_result_json, make_artis_ai_log
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np

import pyzipper
from utils.crypto import openssl_decrypt
import shutil
import time
from feature_matching import Feature_Matching

from make_feature_db import MakeFeatureDB
from bbox_transformer import transform_all_bboxes, create_bbox_debug_image
from depth_validation import DepthValidation

class Inference:
    def __init__(self, config_file_path=None, cal_file_path=None):
        self.config_file_path = cc.path_to_config if config_file_path is None else config_file_path
        self.cal_file_path = cc.path_to_cal_file if cal_file_path is None else cal_file_path

        print(self.config_file_path, self.cal_file_path)

        cc.artis_ai_json_config = cc.get_config(self.config_file_path, cc.artis_ai_json_config)

        print(f"{cc.CurrentDateTime(0)} [Artis_AI] Success to parse json file\n")
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] {cc.artis_ai_json_config}\n")

        print(f"{cc.CurrentDateTime(0)} [Artis_AI] Version Information : {cc.artis_ai_version}\n")
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] DB Version Information : {cc.artis_ai_db_version}")
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] Model Version Information : {cc.artis_ai_model_version}")
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] Model Integrity Check : {cc.artis_ai_model_integrity}\n")
        if cc.artis_ai_model_integrity == "fail":
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] Model Integrity Fail List : {cc.artis_ai_model_fail_list}\n")

        #port_num = str(cc.artis_ai_json_config["zmq_port_ai_inference"])
        min_depth, max_depth = cc.artis_ai_json_config["depth_min"], cc.artis_ai_json_config["depth_max"]
        crop_lx, crop_ly = cc.artis_ai_json_config["crop_lx"], cc.artis_ai_json_config["crop_ly"]
        crop_rx, crop_ry = cc.artis_ai_json_config["crop_rx"], cc.artis_ai_json_config["crop_ry"]
        crop_width, crop_height = cc.artis_ai_json_config["crop_width"], cc.artis_ai_json_config["crop_height"]
        #log_on_off = cc.artis_ai_json_config["log_on_off"]

        if "ai_model_mode" in cc.artis_ai_json_config:
            ai_model_mode = cc.artis_ai_json_config["ai_model_mode"]
            if (ai_model_mode < cc.artis_ai_model_mode["mode_rgb_only"]
                or ai_model_mode > cc.artis_ai_model_mode["mode_seg_with_feature_matching"]):
                ai_model_mode = cc.artis_ai_model_mode["mode_rgb_with_depth"]
        else:
            ai_model_mode = cc.artis_ai_model_mode["mode_rgb_with_depth"]

        self.inf_kisane = Kisane(self.config_file_path)
        self.inf_depth = Depth(self.cal_file_path, [min_depth, max_depth], [crop_lx, crop_ly, crop_rx, crop_ry, crop_width, crop_height], self.config_file_path)
        self.inf_kisane.set_depth_instance(self.inf_depth) # Depth 인스턴스 연결
        self.inf_depth_validation = DepthValidation(self.config_file_path) # Depth 기반 검사 모듈 초기화

        if ai_model_mode >= cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
            ##### Load Feature Matching #####
            self.inf_feature_matching = Feature_Matching(self.config_file_path)
            self.inf_feature_db = MakeFeatureDB(self.config_file_path)
        else:
            self.inf_feature_matching = None
            self.inf_feature_db = None

            # WARM UP #
        self.inf_kisane.warm_up(5, 0.01, 0.01)
        self.inf_depth.warm_up(5, 0.01)
        if ai_model_mode >= cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
            self.inf_feature_matching.warm_up(5, 0.01)

        self.inf_cnt = 0

        self.crypto_key = openssl_decrypt()
        self.filelist = ["Cam_2_Depth.jpg", "Cam_2_Depth.bin", "artis_result_debug.json"]

        right_depth = cc.artis_ai_json_config.get("right_depth", False)
        if right_depth:
            self.filelist = ["Cam_1_Depth.jpg", "Cam_1_Depth.bin"] + self.filelist
        
        if not hasattr(cc, 'depth_offset'):
            cc.depth_offset = None
        
        self.item_list = self.get_item_list()

    def update_calibration_and_crop(self, cal_file_path=None, crop_settings=None, depth_offset=None):
        """
        Calibration 파일, crop 설정, depth_offset을 업데이트하는 함수
        
        Args:
            cal_file_path: 새로운 calibration_results.xml 경로 (None이면 기존 경로 유지)
            crop_settings: 새로운 crop 설정 딕셔너리 (None이면 기존 설정 유지)
            depth_offset: 새로운 depth_offset 값 (None이면 기존 값 유지)
        """
        start_time = time.time()
        
        if cal_file_path is None and crop_settings is None and depth_offset is None:
            return
        
        # cal_file_path가 주어진 경우: Depth 인스턴스 재생성
        if cal_file_path is not None:
            if depth_offset is not None:
                cc.depth_offset = depth_offset
            else:
                cc.depth_offset = 0.0
            
            # calibration 파일 경로 업데이트
            self.cal_file_path = cal_file_path
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] Calibration 파일 경로 업데이트: {cal_file_path}")
            
            # crop_settings 추출 (없으면 config.json 값 사용)
            if crop_settings:
                crop_lx = crop_settings.get("left_x")
                crop_ly = crop_settings.get("left_y")
                crop_rx = crop_settings.get("right_x")
                crop_ry = crop_settings.get("right_y")
                crop_width = crop_settings.get("width")
                crop_height = crop_settings.get("height")
            else:
                crop_lx, crop_ly = cc.artis_ai_json_config["crop_lx"], cc.artis_ai_json_config["crop_ly"]
                crop_rx, crop_ry = cc.artis_ai_json_config["crop_rx"], cc.artis_ai_json_config["crop_ry"]
                crop_width, crop_height = cc.artis_ai_json_config["crop_width"], cc.artis_ai_json_config["crop_height"]
            
            min_depth, max_depth = cc.artis_ai_json_config["depth_min"], cc.artis_ai_json_config["depth_max"]

            # Depth 인스턴스 재생성
            self.inf_depth = Depth(self.cal_file_path, [min_depth, max_depth], 
                                    [crop_lx, crop_ly, crop_rx, crop_ry, crop_width, crop_height], 
                                    self.config_file_path)
            self.inf_kisane.set_depth_instance(self.inf_depth) # Depth 인스턴스 연결
            self.inf_depth.warm_up(5, 0.01)
            
            elapsed_time = (time.time() - start_time) * 1000
            cc.artis_ai_current_log = f'Depth 업데이트 완료: LEFT({crop_lx},{crop_ly}) RIGHT({crop_rx},{crop_ry}) 크기({crop_width}x{crop_height}) 범위({min_depth}~{max_depth}mm) offset({cc.depth_offset}mm) ===> 소요시간 : {elapsed_time:.2f}ms'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
            return
        
        elapsed_time = None
        
        # crop_settings 업데이트
        if crop_settings:
            crop_lx = crop_settings.get("left_x")
            crop_ly = crop_settings.get("left_y")
            crop_rx = crop_settings.get("right_x")
            crop_ry = crop_settings.get("right_y")
            
            self.inf_depth.update_crop_settings(crop_lx, crop_ly, crop_rx, crop_ry)
            elapsed_time = (time.time() - start_time) * 1000
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] Crop 포인트 업데이트 완료 : {elapsed_time:.2f}ms")
        
        # depth_offset 업데이트
        if depth_offset is not None:
            cc.depth_offset = depth_offset
            if elapsed_time is None:
                elapsed_time = (time.time() - start_time) * 1000
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] Depth Offset 업데이트 완료 : {elapsed_time:.2f}ms")
        
        return

    def get_item_list(self):
        from glob import glob
        encodings_to_try = [
        "utf-8",
        "utf-8-sig",   # UTF-8 with BOM
        "cp949",       # ANSI or EUC-KR 대응
        "euc-kr",
        ]
        
        item_files = glob(cc.path_to_checkpoint + "item*korean*json")

        item_list = {}
        if len(item_files):
            for enc in encodings_to_try:
                try:
                    with open(item_files[0], encoding=enc) as json_file:
                        item_info = json.load(json_file)
                        if "item_list" in item_info:
                            for key, val in item_info["item_list"].items():
                                item_list[int(key)] = [key, val[3]]
                        else:
                            for category, ctg_val in item_info.items():
                                for item_dic in ctg_val:
                                    item_list[int(item_dic['item_code'])] = [item_dic['item_code'], item_dic['item_name']]
                        print(f"{cc.CurrentDateTime(0)} [Artis_AI] {item_files[0]} {enc} 파싱 성공")
                        return item_list
                except UnicodeDecodeError:
                    continue
                except json.JSONDecodeError:
                    print(f"{cc.CurrentDateTime(0)} [Artis_AI] {item_files[0]} Json 파싱 실패")
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] {item_files[0]} 모든 인코딩 시도 실패 : {encodings_to_try}")
        else:
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] item_info_korean.json이 없음")

        return item_list

    def update_config(self, processed_files):
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] Checking Config files...")
        for file in processed_files:
            file_name = file["original_name"]
            raw_data = file["data"]
            
            # 설정 파일이 아닌 경우 건너뛰기
            if not (".json" in file_name or ".xml" in file_name):
                continue
                
            try:
                processed_data = None
                if ".json" in file_name:
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
                elif ".xml" in file_name:
                    # XML 파일도 동일한 방식으로 처리
                    try:
                        text_data = raw_data.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            text_data = raw_data.decode('utf-8-sig')
                        except UnicodeDecodeError:
                            try:
                                text_data = raw_data.decode('latin-1')
                            except UnicodeDecodeError:
                                text_data = raw_data.decode('utf-8', errors='ignore')
                    
                    processed_data = ET.fromstring(text_data)       # str → XML Element
                #print(processed_data)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                print(f"{cc.CurrentDateTime(0)} [Artis_AI] 파일을 JSON으로 파싱할 수 없습니다: {e}")
                return False, f"파일을 JSON으로 파싱할 수 없습니다: {e}"
            except (UnicodeDecodeError, ET.ParseError) as e:
                print(f"{cc.CurrentDateTime(0)} [Artis_AI] XML 파싱 오류: {e}")
                return False, f"XML 파싱 오류: {e}"
            except Exception as e:
                print(f"{cc.CurrentDateTime(0)} [Artis_AI] 설정 파일 오류: {e}")
                return False, f"설정 파일 오류: {e}"

            if "Artis_EdgeManager_Config.json" in file_name:
                out_file_name = cc.path_to_config
            elif "calibration_results.xml" in file_name:
                out_file_name = cc.path_to_cal_file
            elif "item" in file_name:
                out_file_name = cc.path_to_checkpoint + "/" + file_name
            else:
                #print(f"{cc.CurrentDateTime(0)} [Artis_AI] 업데이트 불가 파일: {file_name}")
                #return False, f"업데이트 불가 파일: {file_name}"
                continue
            
            with open(out_file_name, 'wb') as f:
                    f.write(raw_data)
        
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] Updating Config files...")
        cc.artis_ai_json_config = {}
        cc.artis_ai_json_config = cc.get_config(self.config_file_path, cc.artis_ai_json_config)

        self.inf_kisane.reinit(self.config_file_path)
        self.inf_depth.reinit(self.cal_file_path, [cc.artis_ai_json_config["depth_min"], cc.artis_ai_json_config["depth_max"]], \
                              [cc.artis_ai_json_config["crop_lx"], cc.artis_ai_json_config["crop_ly"], \
                               cc.artis_ai_json_config["crop_rx"], cc.artis_ai_json_config["crop_ry"], \
                                cc.artis_ai_json_config["crop_width"], cc.artis_ai_json_config["crop_height"]], self.config_file_path)
        self.inf_kisane.set_depth_instance(self.inf_depth)
        if self.inf_feature_matching is not None:
            self.inf_feature_matching.init(self.config_file_path)
            self.inf_feature_db.load_model()
        elif cc.artis_ai_json_config["ai_model_mode"] >= cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
            self.inf_feature_matching = Feature_Matching(self.config_file_path)
            self.inf_feature_db = MakeFeatureDB(self.config_file_path)
        self.inf_cnt = 0        
        self.item_list = self.get_item_list()

        return True, ""

    def start_ai_training(self, processed_files, db_root_path):
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] AI 학습 시작")
        
        #NOTE::2025-09-21
        # 1) 비동기로 학습 스크립트 실행 하고 즉시 리턴 하여 
        #    EdgeMan 의 /iscan-start-ai-training POST 에 응답하도록 할 것
        # 2) 학습 스크립트를 통한 비동기 학습 실행 관련 정보를 
        #    EdgeMan 의 /api/edgeman/sync-status 앤드포인트로 JSON 형식으로 POST 할 것
        #    (예전 작업했던 AiModelingStatus 와 AiModelingComplete 커맨드 시 구현한 내용을
        #    참고하여 재사용 할 수 있도록 할 것)

        cc.artis_ai_json_config = cc.get_config(self.config_file_path, cc.artis_ai_json_config)
        if self.inf_feature_db is None:
            return False, f"ai_model_mode={cc.artis_ai_json_config['ai_model_mode']} 오류", None

        if self.inf_feature_db.ai_feature_mode > 1:
            return False, f"ai_feature_mode={self.inf_feature_db.ai_feature_mode} 오류", None

        is_pass, msg = self.inf_feature_db.set_db(processed_files, db_root_path)
        if not is_pass:
            return is_pass, msg, None

        from threading import Thread
        t = Thread(target=self.inf_feature_db.run, daemon=True)
        t.start()  # 백그라운드 실행
        
        return True, "", t

    def status_ai_training(self):
        train_log = self.inf_feature_db.train_log
        if self.inf_feature_db.eta == 3600 or self.inf_feature_db.eta == self.inf_feature_db.eta_pre:
            eta = self.inf_feature_db.eta - (time.time() - self.inf_feature_db.start_time)
        else:
            eta = self.inf_feature_db.eta
        self.inf_feature_db.start_time = time.time()
        self.inf_feature_db.eta_pre = self.inf_feature_db.eta
        if eta > 0:
            if eta / 3600 > 0:
                train_log["ai_model_stage_eta"][0] = str(int(eta / 3600))
                eta -= (int(eta / 3600) * 3600)
            if eta / 60 > 0:
                train_log["ai_model_stage_eta"][1] = str(int(eta / 60))
                eta -= (int(eta / 60) * 60)
            train_log["ai_model_stage_eta"][2] = str(int(eta))
        else:
            train_log["ai_model_stage_eta"] = ["0", "0", "0"]
            train_log["ai_model_stage"] = "TRAIN_END"
        return train_log

    def inference(self, processed_files, img_dir, run_mode, timestamp):
        start_time = time.time()
        cc.artis_ai_current_error_code = 0
        cc.artis_ai_current_error_reason = ''
        cc.artis_ai_current_log = ''
        cc.artis_ai_result_json = make_default_result_json(cc.artis_ai_result_json)

        cc.artis_ai_current_log = 'recv (EdgeMan -> Artis_AI)'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
        cc.artis_ai_current_log = f'recv parameter : \n' + \
                                  f'run_mode: {run_mode}\n' + \
                                  f'processed_files: {processed_files}\n' + \
                                  f'img_dir: {img_dir}\n'
        make_artis_ai_log(cc.artis_ai_current_log, 'debug', False)

        cc.artis_ai_current_log = 'Parsing Success'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

        # 입력 영상 처리 및 해상도 검사/리사이즈
        imgfile1 = None
        imgfile2 = None
        imgfile_single = None
        
        cc.artis_ai_current_log = f'처리할 파일 목록: {[f["original_name"] for f in processed_files]}'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
        
        for file_info in processed_files:
            file_name_lower = file_info["original_name"].lower()
            if "cam_1" in file_name_lower and file_info["original_name"].lower().endswith('.jpg'):
                imgfile1 = self._process_image_file(file_info, timestamp, "Cam_1", run_mode)
                cc.artis_ai_current_log = f'Cam_1 이미지 파일 처리 완료: {file_info["original_name"]}'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
            elif "cam_2" in file_name_lower and file_info["original_name"].lower().endswith('.jpg'):
                imgfile2 = self._process_image_file(file_info, timestamp, "Cam_2", run_mode)
                cc.artis_ai_current_log = f'Cam_2 이미지 파일 처리 완료: {file_info["original_name"]}'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
            elif "cam_single_1" in file_name_lower and file_info["original_name"].lower().endswith('.jpg'):
                imgfile_single = self._process_image_file(file_info, timestamp, "Cam_Single", run_mode)
                cc.artis_ai_current_log = f'Cam_Single 이미지 파일 처리 완료: {file_info["original_name"]}'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
        
        if imgfile1 is None or imgfile2 is None:
            cc.artis_ai_current_log = f'이미지 파일을 찾을 수 없습니다'
            make_artis_ai_log(cc.artis_ai_current_log, 'error', True)
            return None, None, None, None
            
        img_list = [imgfile2, imgfile2, imgfile1, imgfile_single]  # imgs[3] = Single (None 가능)

        # kisan_config.json 업데이트
        config_file_found = False
        for file_info in processed_files:
            if "Artis_EdgeManager_Config.json" in file_info["original_name"]:
                with open(file_info["path"], "rb") as src:
                    data = src.read()
                
                with open(cc.path_to_config, "wb") as dst:
                    dst.write(data)
                config_file_found = True
                cc.artis_ai_current_log = f'설정 파일 업데이트 완료: {file_info["original_name"]}'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
                break
        
        if not config_file_found:
            cc.artis_ai_current_log = f'설정 파일을 찾을 수 없습니다'
            make_artis_ai_log(cc.artis_ai_current_log, 'warning', True)

        time_for_prepare = (time.time() - start_time) * 1000
        cc.artis_ai_current_log = f'입력 이미지 처리 : {time_for_prepare} ms'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

        cc.artis_ai_current_log = f'kisan_config.json 업데이트'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

        # CalCam 모드는 추론 스킵
        if run_mode == "CalCam":
            cc.artis_ai_current_log = f'CalCam 모드: 추론 스킵'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
            ret = "OK"
            dep_time = 0
            time_for_depth = 0
            total_time = 0
            time_for_inf = 0
        else:
            start_time = time.time()
            ret, dep_time = self.inf_depth.inference(img_list, False, img_dir, run_mode, cc.depth_offset)
            time_for_depth = (time.time() - start_time) * 1000

            start_time = time.time()
            cc.artis_ai_current_log = f'OD & OC Start'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

            _, total_time, bbox, _, ai_mode, bbox_all = self.inf_kisane.inference(img_list, False, img_dir, run_mode)
            #total_time += dep_time
            time_for_inf = (time.time() - start_time) * 1000

            if ai_mode >= cc.artis_ai_model_mode["mode_depth_with_feature_matching"]:
                start_time = time.time()
                cc.artis_ai_current_log = f'Feature Matching Start'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
                fm_score, fm_id, fm_time = self.inf_feature_matching.inference_by_feature_matching(img_dir, bbox)
                time_for_inf_fm = (time.time() - start_time) * 1000
                time_for_inf += time_for_inf_fm

            # Depth 기반 검사
            start_time = time.time()
            cc.artis_ai_result_json = self.inf_depth_validation.check_validation(bbox, bbox_all, self.inf_depth, cc.artis_ai_result_json, cc.artis_ai_json_config)
            time_for_depth_check = (time.time() - start_time) * 1000

            # LEFT 기준 bbox를 RIGHT와 SINGLE로 좌표 변환
            bbox_transform_enable = cc.artis_ai_json_config.get("bbox_transform", False)
            if bbox_transform_enable:
                print(f"{cc.CurrentDateTime(0)} [Artis_AI] ===============================================")
                start_time = time.time()
                self._perform_bbox_transform(img_dir, imgfile1, imgfile2, imgfile_single)
                elapsed_time = (time.time() - start_time) * 1000
                print(f"{cc.CurrentDateTime(0)} [Artis_AI] ==============================================> 좌표변환 완료 ({elapsed_time:.2f} ms)")
            else:
                print(f"{cc.CurrentDateTime(0)} [Artis_AI] 좌표변환 스킵: bbox_transform=false")
                cc.artis_ai_result_json['artis_object_bbox_translate'] = {
                    'right': {},
                    'single': {}
                }

            start_time = time.time()
            cc.artis_ai_current_log = f'ret: {ret}'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

            cc.artis_ai_current_log = f'OD & OC Finish'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

            ###print(f"[Artis_AI] ret: {ret}")

            if (ret == "OK"):
                cc.artis_ai_current_log = f'총 추론 소요 시간 : {total_time + dep_time:.2f} ms, 누적 추론 카운트 : {self.inf_cnt}'
                self.inf_cnt += 1
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
                ###print(f"[Artis_AI] 총 추론 소요 시간 : {total_time:.2f} ms, 누적 추론 카운트 : {self.inf_cnt}")

            else:
                cc.artis_ai_current_log = f'Failed to Depth Map Generation'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

        #cc.artis_ai_current_log = 'send result by json format (Artis_AI -> JetsonMan)'
        #make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

        # depth_offset을 artis_result_debug.json에 포함
        cc.artis_ai_result_json["depth_offset"] = cc.depth_offset

        result_json = json.dumps(cc.artis_ai_result_json, ensure_ascii=False)
        with open(os.path.join(img_dir, "artis_result_debug.json"), 'w') as file:
            #json.dump(result_json, file, indent=3)
            json.dump(cc.artis_ai_result_json, file, indent=3)
        
        #output_path, output_filename = self.make_output(img_dir, timestamp)
        output_filename = timestamp + ".zip"
        output_path = os.path.join(cc.path_to_root, img_dir, output_filename).replace("\\", "/")
        
        # ZIP에 포함할 파일 목록 및 경로 매핑 구성
        file_path_map = self._prepare_zip_files(processed_files, img_dir, run_mode)

        # 디버깅 이미지 생성
        self._create_debug_image(os.path.join(cc.path_to_root, img_dir))

        # artis_combined_debug.jpg zip 파일에 포함
        fusion_debug_path = os.path.join(cc.path_to_root, img_dir, cc.artis_fusion_img_file)
        if os.path.exists(fusion_debug_path):
            file_path_map[cc.artis_fusion_img_file] = fusion_debug_path
        
        with pyzipper.AESZipFile(output_path, 'w', compression=pyzipper.ZIP_DEFLATED,
                                     encryption=pyzipper.WZ_AES) as zipf:
            zipf.setpassword(self.crypto_key)
            
            for filename, filepath in file_path_map.items():
                if os.path.exists(filepath):
                    #print(filepath)
                    zipf.write(filepath, arcname=filename)
                else:
                    cc.artis_ai_current_log = f'ZIP 파일 생성: {filename} 파일이 없어 건너뜀'
                    make_artis_ai_log(cc.artis_ai_current_log, 'warning', True)

        time_for_zip = (time.time() - start_time) * 1000

        return cc.artis_ai_result_json, output_path, output_filename, [time_for_prepare, time_for_depth, time_for_inf, time_for_zip]
    
    def make_msg(self, result_json, total_time):
        artis_msg = f""
        obj_cnt = int(result_json["artis_result_info"]["object_total_cnt"])
        if obj_cnt > 0:
            for obj_idx in range(obj_cnt):
                cls_id = int(result_json['artis_object_detail'][str(obj_idx)])
                if cls_id >= cc.artis_ai_class_invalid_item:
                    cls_code = str(cls_id)
                    cls_name = "예외클래스" if cls_id == cc.artis_ai_class_invalid_item else "미학습상품"
                else:
                    cls_code = self.item_list[cls_id][0] if cls_id in self.item_list else str(cls_id)
                    cls_name = self.item_list[cls_id][1] if cls_id in self.item_list else "미등록상품"
                artis_msg += f"[{obj_idx + 1}] {cls_code} {cls_name}\n"
        
        # total_time이 리스트인지 확인하고 적절히 처리
        if isinstance(total_time, list) and len(total_time) >= 4:
            artis_msg += f"• 입력 처리 시간 : {total_time[0]:.2f} ms\n"
            artis_msg += f"• Depth 생성 시간 : {total_time[1]:.2f} ms\n"
            artis_msg += f"• 추론 시간 : {total_time[2]:.2f} ms\n"
            artis_msg += f"• 압축 시간 : {total_time[3]:.2f} ms\n"
            artis_msg += f"⏱️ 총 추론기 시간 : {sum(total_time):.2f} ms\n"
        else:
            # total_time이 단일 값인 경우
            artis_msg += f"⏱️ 총 추론 시간 : {total_time:.2f} ms\n"

        return artis_msg

    def upload_output(self, input_dir, output_dir, files):
        for file in files:
            try:
                shutil.move(os.path.join(input_dir, file), os.path.join(output_dir, file))
            except FileNotFoundError:
                return False, f"추론 결과 파일이 존재하지 않습니다: {file}"
            except PermissionError:
                return False, f"NAS에 업로드할 권한이 없습니다: {os.path.join(output_dir, file)}"
            except Exception as e:
                return False, f"에러 발생 {e}"
        return True, ""

    def _process_image_file(self, file_info, timestamp, camera_name, run_mode):
        """
        이미지 파일을 처리하고 해상도 검사/리사이즈를 수행합니다.
        초기에 한 번만 해상도 검사를 하고, 필요시 리사이즈하여 저장합니다.
        CalCam 모드일 때는 리사이즈 없이 원본 그대로 저장합니다.
        """
        import cv2
        import numpy as np
        
        # 원본 파일 읽기
        with open(file_info["path"], "rb") as src:
            data = src.read()
        
        # 대상 파일 경로 설정
        target_path = file_info["path"].replace(timestamp + "_", "")
        
        if run_mode == "CalCam":
            with open(target_path, "wb") as dst:
                dst.write(data)
            return target_path
        
        # 이미지 해상도 검사
        img_array = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            # 이미지 로드 실패시 원본 그대로 저장
            with open(target_path, "wb") as dst:
                dst.write(data)
            return target_path

        current_height, current_width = img.shape[:2]
        target_height, target_width = cc.image_resolution_rgb[0], cc.image_resolution_rgb[1]
        
        # 해상도가 다른 경우에만 리사이즈
        if current_height != target_height or current_width != target_width:
            if camera_name == "Cam_2":
                camera_type = CamInfo.LEFT
            elif camera_name == "Cam_1":
                camera_type = CamInfo.RIGHT
            elif camera_name == "Cam_Single":
                camera_type = CamInfo.SINGLE
            else:
                camera_type = CamInfo.LEFT  # 기본값
            
            img = self.inf_depth.make_input_image(img, camera_type, True)

            cc.artis_ai_current_log = f'{camera_name} 이미지 해상도 변경: {current_width}x{current_height} -> {target_width}x{target_height}'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
            
            # 이미지 리사이즈
            resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # 리사이즈된 이미지를 JPEG로 인코딩하여 저장
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # JPEG 품질 70%
            _, encoded_img = cv2.imencode('.jpg', resized_img, encode_param)
            
            with open(target_path, "wb") as dst:
                dst.write(encoded_img.tobytes())
        else:
            # 해상도가 동일한 경우 원본 그대로 저장
            with open(target_path, "wb") as dst:
                dst.write(data)
        
        return target_path

    def _perform_bbox_transform(self, img_dir, imgfile1, imgfile2, imgfile_single):
        """
        좌표변환을 수행하고 결과를 JSON에 저장하며 디버깅 이미지를 생성합니다.
        """
        flow_rect = getattr(self.inf_depth, 'last_flow_rect', None)
        rect_info = getattr(self.inf_depth, 'last_rect_info', None)
        depth_map_rect = getattr(self.inf_depth, 'last_depth_rect', None)
        
        if flow_rect is None or rect_info is None:
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] 좌표변환 스킵: 필수 정보 부족 (flow_rect={flow_rect is not None}, rect_info={rect_info is not None})")
            return
        
        left_bbox_dict = cc.artis_ai_result_json.get('artis_object_bbox', {})
        right_bbox_dict = {} # 오른쪽 변환된 좌표
        single_bbox_dict = {} # 싱글 변환된 좌표

        if not left_bbox_dict:
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] 좌표변환 스킵: artis_object_bbox가 없거나 비어있음 (디버깅 이미지는 생성)")
        else:
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] [좌표변환 시작] 총 bbox 개수: {len(left_bbox_dict)}")

            try:
                min_depth = cc.artis_ai_json_config.get("depth_min", 200.0)
                max_depth = cc.artis_ai_json_config.get("depth_max", 500.0)

                # 좌표변환 함수 호출
                right_bbox_dict, single_bbox_dict = transform_all_bboxes(
                    left_bbox_dict,
                    flow_rect,
                    rect_info,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    depth_map_rect=depth_map_rect,
                    skip_single=(imgfile_single is None)
                )
            except Exception as e:
                print(f"{cc.CurrentDateTime(0)} [Artis_AI] 좌표변환 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        
        # JSON에 결과 저장
        cc.artis_ai_result_json['artis_object_bbox_translate'] = {
            'right': right_bbox_dict,
            'single': single_bbox_dict
        }
        
        # 디버깅 이미지 생성
        try:
            transform_debug_path = os.path.join(img_dir, cc.artis_transform_img_file)
            if left_bbox_dict:
                create_bbox_debug_image(imgfile1, imgfile2, imgfile_single, left_bbox_dict, right_bbox_dict, single_bbox_dict, transform_debug_path)
        except Exception as e:
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] 디버깅 이미지 생성 실패: {e}")

    def _create_debug_image(self, img_dir):
        """
        디버깅 이미지들을 합쳐서 artis_combined_debug.jpg로 저장
        """
        try:
            transform_path = os.path.join(img_dir, cc.artis_transform_img_file)
            fusion_path = os.path.join(img_dir, cc.artis_fusion_img_file)
            depth_path = os.path.join(img_dir, "Cam_1_Depth.jpg")
            output_path = os.path.join(img_dir, cc.artis_debug_img_file)

            transform_img = cv2.imread(transform_path) if os.path.exists(transform_path) else None
            fusion_img = cv2.imread(fusion_path) if os.path.exists(fusion_path) else None
            depth_img = cv2.imread(depth_path) if os.path.exists(depth_path) else None

            if transform_img is None and fusion_img is None and depth_img is None:
                return False

            right_images = []
            if fusion_img is not None:
                right_images.append(fusion_img)
            if depth_img is not None:
                right_images.append(depth_img)

            right_panel = None
            if right_images:
                max_w = max([img.shape[1] for img in right_images])
                resized_list = []
                for img in right_images:
                    h, w = img.shape[:2]
                    if w != max_w:
                        scale = max_w / w
                        new_h = int(h * scale)
                        img = cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_LINEAR)
                    resized_list.append(img)
                right_panel = np.vstack(resized_list)

            # 기준: 왼쪽 좌표변환 이미지 (transform)
            if transform_img is not None and right_panel is not None:
                th, tw = transform_img.shape[:2]
                rh, rw = right_panel.shape[:2]

                if len(right_images) == 1:
                    # 오른쪽 1장: 좌표변환 이미지 밑에 수직으로 붙임
                    if rw != tw:
                        scale = tw / rw
                        new_rh = int(rh * scale)
                        right_panel = cv2.resize(right_panel, (tw, new_rh), interpolation=cv2.INTER_LINEAR)
                    combined_image = np.vstack([transform_img, right_panel])
                else:
                    # 오른쪽 2장 이상: 수평 합침, 높이를 왼쪽(transform)에 맞춤
                    if th > rh:
                        # 오른쪽이 짧음 -> 아래에 흰색 여백 추가
                        padding = np.ones((th - rh, rw, 3), dtype=np.uint8) * 255
                        right_panel = np.vstack([right_panel, padding])
                    elif rh > th:
                        # 오른쪽이 김 -> 왼쪽 높이에 맞게 비율 유지하며 리사이즈
                        scale = th / rh
                        new_rw = int(rw * scale)
                        right_panel = cv2.resize(right_panel, (new_rw, th), interpolation=cv2.INTER_LINEAR)
                    combined_image = np.hstack([transform_img, right_panel])
            elif transform_img is not None:
                combined_image = transform_img
            elif right_panel is not None:
                combined_image = right_panel
            else:
                return False

            # 해상도를 1/5로 줄여서 저장
            h, w = combined_image.shape[:2]
            combined_image = cv2.resize(combined_image, (w // 5, h // 5), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, combined_image, [cv2.IMWRITE_JPEG_QUALITY, 70])

            return True

        except Exception as e:
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] 디버깅 이미지 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _prepare_zip_files(self, processed_files, img_dir, run_mode=None):
        """
        ZIP 파일에 포함할 파일 목록과 경로를 준비합니다.
        
        Args:
            processed_files: 처리된 파일 정보 리스트
            img_dir: 이미지 디렉토리 경로
            run_mode: 실행 모드 (CalCam 모드일 경우 processed_files, artis_result_debug.json 포함)
            
        Returns:
            dict: {파일명: 파일경로} 형태의 딕셔너리
        """
        file_map = {}
        
        # CalCam 모드일 경우 processed_files과 artis_result_debug.json 추가
        if run_mode == "CalCam":
            file_map = {
                f["original_name"]: f["path"]
                for f in processed_files
            }
            json_file_path = os.path.join(cc.path_to_root, img_dir, "artis_result_debug.json")
            if os.path.exists(json_file_path):
                file_map["artis_result_debug.json"] = json_file_path
        else:
            # 1. 기본 파일 리스트 추가 (self.filelist)
            for filename in self.filelist:
                file_map[filename] = os.path.join(cc.path_to_root, img_dir, filename)
            
            # 2. processed_files에서 원본 파일명과 경로 매핑
            processed_file_dict = {f["original_name"]: f["path"] for f in processed_files}
            
            for original_name, file_path in processed_file_dict.items():
                if original_name not in file_map:
                    file_map[original_name] = file_path
            
            # 3. 캘리브레이션 파일 경로 설정
            file_map["calibration_results.xml"] = self.cal_file_path
        
        file_list = sorted(file_map.keys())
        cc.artis_ai_current_log = f'ZIP 파일 포함 목록 ({len(file_list)}개): {", ".join(file_list)}'
        make_artis_ai_log(cc.artis_ai_current_log, 'info', True)
        
        return file_map

if __name__=="__main__":
    #inf = Inference("kisan_config_illi.json")
    inf = Inference()
    processed_files = []
    processed_files.append({
        "original_name": "Cam_1_Color.jpg",
        "saved_name": "Cam_1_Color.jpg",
        "size": 0,
        "path": "./samples/Cam_1_Color.jpg"
    })
    processed_files.append({
        "original_name": "Cam_2_Color.jpg",
        "saved_name": "Cam_2_Color.jpg",
        "size": 0,
        "path": "./samples/Cam_2_Color.jpg"
    })
    inf.inference(processed_files, "./samples/", "UserRun", "0")