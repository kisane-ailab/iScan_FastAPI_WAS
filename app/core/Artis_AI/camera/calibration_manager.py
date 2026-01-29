"""
캘리브레이션 데이터 관리 모듈
"""

import os
import shutil
import traceback
import glob
from typing import Dict, Any, Optional
from xml.etree.ElementTree import ElementTree

import cv2
import numpy as np

# 시리얼 넘버별 calibration 데이터
serial_calibration_data: Dict[str, Dict[str, Any]] = {}


def extract_serial_number(metadata_dict: Dict[str, Any]) -> str:
    """
    메타데이터 딕셔너리에서 시리얼 넘버를 추출하는 헬퍼 함수
    
    Args:
        metadata_dict: 메타데이터 딕셔너리
    
    Returns:
        str: 시리얼 넘버 (없으면 빈 문자열)
    """
    return metadata_dict.get("serialNumber", "")


def _normalize_dist(dist: np.ndarray) -> np.ndarray:
    """왜곡 계수를 올바른 shape으로 변환"""
    if dist is None:
        return np.zeros((5, 1), dtype=np.float64)
    if dist.ndim == 1:
        return dist.reshape(-1, 1) if dist.size > 0 else np.zeros((5, 1), dtype=np.float64)
    return dist


def _normalize_t(t: np.ndarray) -> np.ndarray:
    """변환 벡터를 (3, 1) shape으로 변환"""
    if t.ndim == 1:
        return t.reshape(3, 1)
    elif t.shape == (1, 3):
        return t.reshape(3, 1)
    return t


def _parse_stereo_calibration_params(cal_file_path: str) -> Optional[dict]:
    """
    스테레오 캘리브레이션 XML 파일을 파싱하여 딕셔너리로 반환
    
    Args:
        cal_file_path: calibration_results.xml 파일 경로
    
    Returns:
        dict: 스테레오 캘리브레이션 파라미터 딕셔너리
              - 기본 파라미터: mtx_l, mtx_r, K_l, K_r, dist_l, dist_r, R, t, origin_size
              - Rectify 정보: R1, P1, img_size (image_fusion에서 사용)
              파싱 실패 시 None 반환
    """
    try:
        from app.core.Artis_AI.camera.utils import load_calibration_results
        
        mtx_l, mtx_r, K_l, K_r, dist_l, dist_r, R, t, R_l, t_l, R_r, t_r = load_calibration_results(cal_file_path)
        origin_size = [1920, 1200]  # 기본값

        try:
            tree = ElementTree()
            tree.parse(cal_file_path)
            root = tree.getroot()
            
            if root.find("img_width") is not None:
                origin_size[0] = int(root.find("img_width").text)
            if root.find("img_height") is not None:
                origin_size[1] = int(root.find("img_height").text)
        except Exception as e:
            print(f"⚠️ origin_size 읽기 실패, 기본값 사용: {e}")

        # ========== Rectify 정보 계산 (image_fusion에서 사용) ==========
        img_size = (origin_size[0], origin_size[1])  # (width, height)

        # 왜곡 계수 및 변환 벡터 정규화
        dist_l_norm = _normalize_dist(dist_l).astype(np.float64)
        dist_r_norm = _normalize_dist(dist_r).astype(np.float64)
        t_norm = _normalize_t(t).astype(np.float64)

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            cameraMatrix1=K_l.astype(np.float64),
            distCoeffs1=dist_l_norm,
            cameraMatrix2=K_r.astype(np.float64),
            distCoeffs2=dist_r_norm,
            imageSize=img_size,
            R=R.astype(np.float64),
            T=t_norm,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        return {
            "mtx_l": mtx_l,      # LEFT 원본 카메라 매트릭스 (3×3)
            "mtx_r": mtx_r,      # RIGHT 원본 카메라 매트릭스 (3×3)
            "K_l": K_l,          # LEFT 정규화된 카메라 매트릭스 (3×3)
            "K_r": K_r,          # RIGHT 정규화된 카메라 매트릭스 (3×3)
            "dist_l": dist_l,    # LEFT 왜곡 계수
            "dist_r": dist_r,    # RIGHT 왜곡 계수
            "R": R,              # LEFT→RIGHT 회전 행렬 (3×3)
            "t": t,              # LEFT→RIGHT 이동 벡터 (3×1)
            "origin_size": origin_size,  # 원본 이미지 크기 [width, height]
            "R1": R1.astype(np.float64),  # Left rectification rotation matrix
            "P1": P1.astype(np.float64),  # Left rectification projection matrix
            "img_size": img_size,         # 이미지 크기 (width, height)
        }
    except Exception as e:
        print(f"⚠️ 스테레오 캘리브레이션 파라미터 파싱 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def _parse_single_calibration_params(cal_file_path_s: str) -> Optional[dict]:
    """
    싱글 캘리브레이션 XML 파일을 파싱하여 딕셔너리로 반환
    
    Args:
        cal_file_path_s: calibration_results_single.xml 파일 경로
    
    Returns:
        dict: 싱글 캘리브레이션 파라미터 딕셔너리 (mtx_s, K_s, dist_s, R_s, t_s 포함)
              파싱 실패 시 None 반환
    """
    try:
        from app.core.Artis_AI.camera.utils import load_calibration_results_single
        mtx_s, K_s, dist_s, R_s, t_s = load_calibration_results_single(cal_file_path_s)
        
        return {
            "mtx_s": mtx_s,  # SINGLE 원본 카메라 매트릭스 (3×3)
            "K_s": K_s,      # SINGLE 정규화된 카메라 매트릭스 (3×3)
            "dist_s": dist_s, # SINGLE 왜곡 계수
            "R_s": R_s,      # LEFT→SINGLE 회전 행렬 (3×3)
            "t_s": t_s       # LEFT→SINGLE 이동 벡터 (3×1)
        }
    except Exception as e:
        print(f"⚠️ 싱글 캘리브레이션 파라미터 파싱 실패: {e}")
        return None


# ============================================================================
# 메모리 관리 함수들
# ============================================================================
def load_all_serial_calibration_data():
    """
    부팅 시 모든 시리얼 넘버 폴더와 default 폴더를 탐색하여 calibration 데이터를 메모리에 로드
    
    1. camera/calibration 디렉토리 탐색
    2. 각 시리얼 넘버 폴더와 default 폴더에서:
       - calibration_results.xml 확인 및 스테레오 파라미터 파싱 (NumPy 배열로 메모리 저장)
       - calibration_results_single.xml 확인 및 싱글 파라미터 파싱 (있는 경우만, NumPy 배열로 메모리 저장)
       - temp/depth_offset.txt 파일에서 depth_offset 로드
    3. 파싱된 모든 파라미터를 serial_calibration_data 전역 딕셔너리에 저장
    
    특징:
    - 부팅 시 한 번만 실행되어 모든 파라미터를 메모리에 로드 (이후 파일 I/O 불필요)
    - 파라미터는 NumPy 배열로 저장되어 바로 사용 가능
    - 파싱 실패 시 경로만 저장하고 계속 진행 (에러 복구 가능)
    """
    # AI_INFERENCE_DISABLED 체크를 위해 control.py에서 가져옴
    try:
        from app.api.control import AI_INFERENCE_DISABLED
        if AI_INFERENCE_DISABLED:
            print("⚠️  AI 추론 비활성화 모드: calibration 데이터 로드 건너뜀")
            return
    except ImportError:
        # control.py를 import할 수 없는 경우 (순환 참조 방지)
        pass
    
    global serial_calibration_data
    serial_calibration_data.clear()
    
    # path_to_root 가져오기 (순환 참조 방지를 위해 직접 import)
    try:
        from app.core.Artis_AI.common_config import path_to_root
    except ImportError:
        # fallback: control.py를 통해 가져오기 (순환 참조 가능하지만 시도)
        try:
            from app.api.control import _get_inference_modules
            _, _, path_to_root, _, _ = _get_inference_modules()
            if path_to_root is None:
                print("⚠️  path_to_root를 가져올 수 없습니다")
                return
        except ImportError:
            print("⚠️  path_to_root를 가져올 수 없습니다")
            return
    
    calibration_base_dir = os.path.join(path_to_root, "camera", "calibration")
    
    if not os.path.exists(calibration_base_dir):
        print(f"⚠️ Calibration 기본 디렉토리가 없음: {calibration_base_dir}")
        return
    
    def _load_calibration_data(item_path, key_name):
        """
        calibration 데이터 로드 헬퍼 함수 (부팅 시 파라미터를 메모리에 적재)
        """
        cal_file_path = os.path.join(item_path, "calibration_results.xml")
        cal_file_path_s = os.path.join(item_path, "calibration_results_single.xml")
        
        # 1. 스테레오 캘리브레이션 XML 파일 존재 확인
        if not os.path.exists(cal_file_path):
            print(f"❌ '{key_name}': stereo=missing (calibration_results.xml 없음)")
            return False
        
        # 2. 스테레오 캘리브레이션 파라미터 파싱 (부팅 시 한 번, NumPy 배열로 메모리 저장)
        stereo_cal_params = _parse_stereo_calibration_params(cal_file_path)
        stereo_status = "ok" if stereo_cal_params else "parse_failed"
        
        # 3. 싱글 캘리브레이션 XML 확인 및 파라미터 파싱 (있는 경우만, 3개 카메라 시스템)
        single_cal_params = None
        has_single_xml = os.path.exists(cal_file_path_s)
        if has_single_xml:
            single_cal_params = _parse_single_calibration_params(cal_file_path_s)
            single_status = "ok" if single_cal_params else "parse_failed"
        else:
            single_status = "none"
        
        # 4. depth_offset 로드 (temp/depth_offset.txt)
        temp_dir = os.path.join(item_path, "temp")
        depth_offset_file = os.path.join(temp_dir, "depth_offset.txt")
        depth_offset = None
        
        if os.path.exists(depth_offset_file):
            try:
                with open(depth_offset_file, 'r') as f:
                    depth_offset_str = f.read().strip()
                    if depth_offset_str and depth_offset_str != "None":
                        depth_offset = float(depth_offset_str)
                        offset_status = f"{depth_offset}"
                    else:
                        offset_status = "invalid"
            except Exception:
                offset_status = "error"
        else:
            offset_status = "none"
        
        # 5. 모든 데이터를 serial_calibration_data 전역 딕셔너리에 저장
        serial_calibration_data[key_name] = {
            "cal_file_path": cal_file_path,                    # 스테레오 XML 경로
            "cal_file_path_s": cal_file_path_s if has_single_xml else None,  # 싱글 XML 경로 (없으면 None)
            "stereo_cal_params": stereo_cal_params,            # 스테레오 파라미터 (NumPy 배열 포함, 파싱 실패 시 None)
            "single_cal_params": single_cal_params,            # 싱글 파라미터 (NumPy 배열 포함, 없거나 파싱 실패 시 None)
            "depth_offset": depth_offset                       # depth_offset 값 (mm, 없으면 None)
        }
        print(
            f"✅ '{key_name}': stereo={stereo_status}, single={single_status}, depth_offset={offset_status}"
        )
        return True
    
    try:
        # 전체 폴더 개수 계산
        total_folders = 0
        for item in os.listdir(calibration_base_dir):
            item_path = os.path.join(calibration_base_dir, item)
            if os.path.isdir(item_path):
                total_folders += 1
        
        # 시리얼 넘버별 폴더 로드
        for item in os.listdir(calibration_base_dir):
            item_path = os.path.join(calibration_base_dir, item)
            
            if os.path.isdir(item_path) and item != "default":
                _load_calibration_data(item_path, item)
        
        # default 폴더 로드
        default_path = os.path.join(calibration_base_dir, "default")
        if os.path.isdir(default_path):
            _load_calibration_data(default_path, "default")
        
        # 로드 결과 출력
        loaded_count = len(serial_calibration_data)
        print(f"📊 총 {total_folders}개 폴더 중 {loaded_count}개 calibration 데이터 로드 완료")
        print_serial_calibration_data()
        
    except Exception as e:
        print(f"❌ calibration 데이터 로드 중 오류 발생: {e}")
        traceback.print_exc()


def print_serial_calibration_data():
    """
    현재 메모리에 로드된 시리얼 넘버별 calibration 데이터를 출력
    """
    if not serial_calibration_data:
        print("⚠️ 메모리에 calibration 데이터가 없습니다.")
        return
    
    print("\n" + "="*80)
    print("📋 현재 메모리에 로드된 시리얼 넘버별 Calibration 데이터 목록")
    print("="*80)
    for idx, (serial_num, cal_data) in enumerate(sorted(serial_calibration_data.items()), 1):
        print(f"\n[{idx}] 시리얼 넘버: {serial_num}")
        print(f"    📄 Stereo Calibration XML: {cal_data['cal_file_path']}")
        
        # 싱글 XML이 있으면 출력
        if cal_data.get('cal_file_path_s'):
            print(f"    📄 Single Calibration XML: {cal_data['cal_file_path_s']}")
        
        depth_offset_str = f"{cal_data['depth_offset']}" if cal_data['depth_offset'] is not None else "None (미설정)"
        print(f"    📏 Depth Offset: {depth_offset_str}")
    print("\n" + "="*80)


def update_serial_calibration_data(serial_number: str, cal_data_dict: Dict[str, Any]) -> None:
    """
    시리얼 넘버의 캘리브레이션 데이터를 메모리에 업데이트
    
    Args:
        serial_number: 시리얼 넘버
        cal_data_dict: 캘리브레이션 데이터 딕셔너리
    """
    global serial_calibration_data
    serial_calibration_data[serial_number] = cal_data_dict


def remove_serial_calibration_data(serial_number: str) -> bool:
    """
    시리얼 넘버의 캘리브레이션 데이터를 메모리에서 삭제
    
    Args:
        serial_number: 시리얼 넘버
    
    Returns:
        bool: 삭제 성공 여부 (데이터가 존재했으면 True, 없었으면 False)
    """
    global serial_calibration_data
    if serial_number in serial_calibration_data:
        del serial_calibration_data[serial_number]
        return True
    return False


# ============================================================================
# 캘리브레이션 적용 함수
# ============================================================================
def apply_serial_calibration_data(serial_number: str):
    """
    startcam 시 시리얼 넘버에 맞는 calibration 데이터를 메모리에서 찾아서 적용
    시리얼 넘버가 없거나 메모리에 데이터가 없으면 default 데이터 사용
    
    Returns:
        bool: 적용 성공 여부
    """
    try:
        from app.api.control import _get_artis_model, _get_inference_modules, AI_INFERENCE_DISABLED
        
        if AI_INFERENCE_DISABLED:
            return False
        
        _, cc, _, _, _ = _get_inference_modules()
        if cc is None:
            return False
        
        artis_model = _get_artis_model()
        if artis_model is None:
            return False
    except ImportError:
        print("⚠️ artis_model을 가져올 수 없습니다")
        return False
    
    # 사용할 calibration 데이터 키 결정
    if not serial_number:
        if "default" in serial_calibration_data:
            print("ℹ️ 시리얼 넘버가 없어 default calibration 데이터 사용")
            cal_key = "default"
        else:
            print("⚠️ 시리얼 넘버가 없고 default calibration 데이터도 없음")
            return False
    elif serial_number in serial_calibration_data:
        cal_key = serial_number
    elif "default" in serial_calibration_data:
        print(f"⚠️ 시리얼 넘버 '{serial_number}': 메모리에 calibration 데이터가 없어 default 사용")
        cal_key = "default"
    else:
        print(f"⚠️ 시리얼 넘버 '{serial_number}': 메모리에 calibration 데이터가 없고 default도 없음")
        return False
    
    cal_data = serial_calibration_data[cal_key]
    new_cal_file_path = cal_data["cal_file_path"]
    new_depth_offset = cal_data["depth_offset"]
    
    if not os.path.exists(new_cal_file_path):
        print(f"⚠️ '{cal_key}': calibration_results.xml 파일이 없음: {new_cal_file_path}")
        return False
    
    # calibration 경로 변경 여부 확인
    cal_file_changed = artis_model.cal_file_path != new_cal_file_path
    try:
        if cal_file_changed:
            artis_model.update_calibration_and_crop(new_cal_file_path, None, new_depth_offset)
        else:
            artis_model.update_calibration_and_crop(None, None, new_depth_offset)
        return True
        
    except Exception as e:
        print(f"❌ '{cal_key}': calibration 데이터 적용 실패: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# 복원 및 정리 함수
# ============================================================================
def _restore_calibration_from_nas(nas_path: str, serial_number: str, calibration_dir: str, is_serial_path_included: bool = False) -> bool:
    """
    캘리브레이션 실패 시 백업된 데이터로 복원하는 안전장치 함수
    NAS에 데이터가 없거나 시리얼 넘버가 없으면 WAS의 "default" 폴더를 확인하고, 그것도 없으면 False 반환
    
    Args:
        nas_path: NAS 기본 경로 (/mynas/{savePath} 또는 /mynas/{savePath}/{serial_number})
        serial_number: 시리얼 넘버 (없으면 NAS 복원 건너뛰고 default 폴더 참조)
        calibration_dir: WAS 로컬 calibration 디렉토리 경로
        is_serial_path_included: nas_path에 이미 시리얼 넘버가 포함되어 있는지 여부
    
    Returns:
        bool: 복원 성공 여부
    """
    try:
        # Inference 클래스는 control.py에서 관리
        from app.api.control import _get_inference_modules
        Inference, _, path_to_root, _, _ = _get_inference_modules()
        if Inference is None or path_to_root is None:
            print("⚠️ Inference 모듈을 가져올 수 없습니다")
            return False
    except ImportError:
        try:
            from app.core.Artis_AI.inference_cloud import Inference
            from app.core.Artis_AI.common_config import path_to_root
        except ImportError:
            print("⚠️ Inference 모듈을 가져올 수 없습니다")
            return False
    
    try:
        # 1단계: NAS에서 복원 시도 (시리얼 넘버가 있을 때만)
        nas_restore_success = False
        
        if serial_number:
            # nas_path에 이미 시리얼 넘버가 포함되어 있으면 그대로 사용, 아니면 추가
            if is_serial_path_included:
                nas_serial_path = nas_path
            else:
                nas_serial_path = os.path.join(nas_path, serial_number)
            
            if os.path.exists(nas_serial_path):
                timestamp_dirs = [
                    os.path.join(nas_serial_path, item)
                    for item in os.listdir(nas_serial_path)
                    if os.path.isdir(os.path.join(nas_serial_path, item)) and item.isdigit() and len(item) == 17
                ]
                
                if timestamp_dirs:
                    latest_timestamp_dir = max(timestamp_dirs, key=os.path.getmtime)
                    print(f"안전장치: 최신 타임스탬프 폴더 발견: {latest_timestamp_dir}")
                    
                    # calibration_results.xml 복원
                    nas_xml_path = os.path.join(latest_timestamp_dir, "calibration_results.xml")
                    if os.path.exists(nas_xml_path):
                        os.makedirs(calibration_dir, exist_ok=True)
                        output_file = os.path.join(calibration_dir, "calibration_results.xml")
                        shutil.copy2(nas_xml_path, output_file)
                        print(f"안전장치: NAS에서 calibration_results.xml 복원 완료: {output_file}")
                        
                        # calibration_results_single.xml 복원 (있는 경우만)
                        nas_xml_s_path = os.path.join(latest_timestamp_dir, "calibration_results_single.xml")
                        output_file_s = None
                        if os.path.exists(nas_xml_s_path):
                            output_file_s = os.path.join(calibration_dir, "calibration_results_single.xml")
                            shutil.copy2(nas_xml_s_path, output_file_s)
                            print(f"안전장치: NAS에서 calibration_results_single.xml 복원 완료: {output_file_s}")
                        
                        # temp 폴더 복원
                        nas_temp_dir = os.path.join(latest_timestamp_dir, "temp")
                        if os.path.exists(nas_temp_dir):
                            restore_temp_dir = os.path.join(calibration_dir, "temp")
                            if os.path.exists(restore_temp_dir):
                                shutil.rmtree(restore_temp_dir)
                            shutil.copytree(nas_temp_dir, restore_temp_dir)
                            print(f"안전장치: NAS에서 temp 폴더 복원 완료: {restore_temp_dir}")
                        
                        # ZIP 파일 복원 (타임스탬프 폴더에서 시리얼 넘버 폴더로)
                        zip_pattern = os.path.join(latest_timestamp_dir, "Cal_*.zip")
                        zip_files = glob.glob(zip_pattern)
                        zip_restored_count = 0
                        for zip_file in zip_files:
                            try:
                                zip_filename = os.path.basename(zip_file)
                                # NAS 시리얼 넘버 폴더로 복원
                                restore_zip_path = os.path.join(nas_serial_path, zip_filename)
                                shutil.copy2(zip_file, restore_zip_path)
                                zip_restored_count += 1
                                print(f"안전장치: NAS에서 ZIP 파일 복원 완료: {zip_filename} -> {restore_zip_path}")
                            except Exception as zip_error:
                                print(f"⚠️ ZIP 파일 복원 실패: {zip_file}, 오류: {zip_error}")
                        
                        if zip_restored_count > 0:
                            print(f"안전장치: 총 {zip_restored_count}개의 ZIP 파일 복원 완료")
                        
                        nas_restore_success = True
                    else:
                        print(f"안전장치: NAS에 calibration_results.xml이 없음: {nas_xml_path}")
                else:
                    print(f"안전장치: NAS에 타임스탬프 폴더가 없음: {nas_serial_path}")
            else:
                print(f"안전장치: NAS 시리얼 경로가 없음: {nas_serial_path}")
        else:
            print(f"안전장치: 시리얼 넘버가 없어 NAS 복원 건너뛰고 default 폴더 참조")
        
        # 2단계: NAS 복원 실패 시 WAS의 "default" 폴더 확인
        if not nas_restore_success:
            default_cal_dir = os.path.join(path_to_root, "camera", "calibration", "default")
            default_xml_path = os.path.join(default_cal_dir, "calibration_results.xml")
            
            if os.path.exists(default_xml_path):
                print(f"안전장치: NAS 복원 실패, WAS default 폴더에서 복원 시도: {default_xml_path}")
                os.makedirs(calibration_dir, exist_ok=True)
                output_file = os.path.join(calibration_dir, "calibration_results.xml")
                shutil.copy2(default_xml_path, output_file)
                print(f"안전장치: WAS default에서 calibration_results.xml 복원 완료: {output_file}")
                
                # calibration_results_single.xml 복원 (있는 경우만)
                default_xml_s_path = os.path.join(default_cal_dir, "calibration_results_single.xml")
                output_file_s = None
                if os.path.exists(default_xml_s_path):
                    output_file_s = os.path.join(calibration_dir, "calibration_results_single.xml")
                    shutil.copy2(default_xml_s_path, output_file_s)
                    print(f"안전장치: WAS default에서 calibration_results_single.xml 복원 완료: {output_file_s}")
                
                # temp 폴더 복원 (있는 경우)
                default_temp_dir = os.path.join(default_cal_dir, "temp")
                if os.path.exists(default_temp_dir):
                    restore_temp_dir = os.path.join(calibration_dir, "temp")
                    if os.path.exists(restore_temp_dir):
                        shutil.rmtree(restore_temp_dir)
                    shutil.copytree(default_temp_dir, restore_temp_dir)
                    print(f"안전장치: WAS default에서 temp 폴더 복원 완료: {restore_temp_dir}")
            else:
                print(f"안전장치: WAS default 폴더에도 calibration_results.xml이 없음: {default_xml_path}")
                return False
        
        # Inference 클래스 업데이트 및 depth_offset 로드
        try:
            expected_cal_file = os.path.join(calibration_dir, "calibration_results.xml")
            
            if os.path.exists(expected_cal_file):
                # depth_offset 로드
                depth_offset = None
                temp_dir = os.path.join(calibration_dir, "temp")
                depth_offset_file = os.path.join(temp_dir, "depth_offset.txt")
                if os.path.exists(depth_offset_file):
                    try:
                        with open(depth_offset_file, 'r') as f:
                            depth_offset_str = f.read().strip()
                            if depth_offset_str and depth_offset_str != "None":
                                depth_offset = float(depth_offset_str)
                                print(f"안전장치: depth_offset 로드 완료: {depth_offset}")
                    except Exception as e:
                        print(f"안전장치: depth_offset 로드 실패: {e}")
                
                # Inference 클래스 재생성 (control.py의 전역 변수 업데이트를 위해)
                try:
                    from app.api.control import artis_model as control_artis_model
                    # control.py의 전역 변수 업데이트
                    import app.api.control as control_module
                    from app.api.control import _get_inference_modules
                    
                    # crop_settings 구성 (cal 경로 변경 시 함께 전달하기 위해)
                    _, cc, _, _, _ = _get_inference_modules()
                    if cc:
                        cc.artis_ai_json_config = cc.get_config(cc.path_to_config, cc.artis_ai_json_config)
                        crop_settings = {
                            "left_x": cc.artis_ai_json_config.get("crop_lx", 180),
                            "left_y": cc.artis_ai_json_config.get("crop_ly", 0),
                            "right_x": cc.artis_ai_json_config.get("crop_rx", 130),
                            "right_y": cc.artis_ai_json_config.get("crop_ry", 0),
                            "width": cc.artis_ai_json_config.get("crop_width", 1600),
                            "height": cc.artis_ai_json_config.get("crop_height", 1200)
                        }
                    else:
                        crop_settings = None
                    
                    control_module.artis_model = Inference(cal_file_path=expected_cal_file)
                    print(f"안전장치: Inference 클래스 복원 완료")
                    
                    # cal 경로 변경 시 cal_file_path, crop_settings, depth_offset 모두 함께 업데이트
                    if depth_offset is not None and crop_settings:
                        control_module.artis_model.update_calibration_and_crop(expected_cal_file, crop_settings, depth_offset)
                    elif crop_settings:
                        control_module.artis_model.update_calibration_and_crop(expected_cal_file, crop_settings, None)
                except (ImportError, AttributeError):
                    # control.py를 import할 수 없는 경우 Inference만 재생성
                    try:
                        from app.core.Artis_AI import common_config as cc
                        cc.artis_ai_json_config = cc.get_config(cc.path_to_config, cc.artis_ai_json_config)
                        crop_settings = {
                            "left_x": cc.artis_ai_json_config.get("crop_lx", 180),
                            "left_y": cc.artis_ai_json_config.get("crop_ly", 0),
                            "right_x": cc.artis_ai_json_config.get("crop_rx", 130),
                            "right_y": cc.artis_ai_json_config.get("crop_ry", 0),
                            "width": cc.artis_ai_json_config.get("crop_width", 1600),
                            "height": cc.artis_ai_json_config.get("crop_height", 1200)
                        }
                    except Exception:
                        crop_settings = None
                    
                    artis_model = Inference(cal_file_path=expected_cal_file)
                    print(f"안전장치: Inference 클래스 복원 완료 (control.py 업데이트 불가)")
                    
                    # cal 경로 변경 시 cal_file_path, crop_settings, depth_offset 모두 함께 업데이트
                    if depth_offset is not None and crop_settings:
                        artis_model.update_calibration_and_crop(expected_cal_file, crop_settings, depth_offset)
                    elif crop_settings:
                        artis_model.update_calibration_and_crop(expected_cal_file, crop_settings, None)
                
                # 메모리 데이터 업데이트 (시리얼 넘버가 있을 때만)
                if serial_number:
                    try:
                        # 스테레오 캘리브레이션 파라미터 파싱 (메모리에 저장)
                        stereo_cal_params = _parse_stereo_calibration_params(expected_cal_file)
                        if stereo_cal_params:
                            print(f"✅ 안전장치: 시리얼 넘버 '{serial_number}': 스테레오 캘리브레이션 파라미터 메모리 로드 완료")
                        else:
                            print(f"⚠️ 안전장치: 시리얼 넘버 '{serial_number}': 스테레오 캘리브레이션 파라미터 파싱 실패 (경로만 저장)")
                        
                        # 싱글 캘리브레이션 파라미터 파싱 (있는 경우만)
                        single_cal_params = None
                        expected_cal_file_s = os.path.join(calibration_dir, "calibration_results_single.xml")
                        has_single_xml = os.path.exists(expected_cal_file_s)
                        if has_single_xml:
                            single_cal_params = _parse_single_calibration_params(expected_cal_file_s)
                            if single_cal_params:
                                print(f"✅ 안전장치: 시리얼 넘버 '{serial_number}': 싱글 캘리브레이션 파라미터 메모리 로드 완료")
                            else:
                                print(f"⚠️ 안전장치: 시리얼 넘버 '{serial_number}': 싱글 캘리브레이션 파라미터 파싱 실패 (경로만 저장)")
                        
                        # 메모리 데이터 업데이트
                        cal_data_dict = {
                            "cal_file_path": expected_cal_file,
                            "cal_file_path_s": expected_cal_file_s if has_single_xml else None,
                            "stereo_cal_params": stereo_cal_params,  # 스테레오 파라미터 메모리 저장
                            "single_cal_params": single_cal_params,  # 싱글 파라미터 메모리 저장
                            "depth_offset": depth_offset
                        }
                        update_serial_calibration_data(serial_number, cal_data_dict)
                        print(f"✅ 안전장치: 메모리에서 시리얼 넘버 '{serial_number}' calibration 데이터 복원 완료")
                    except Exception as mem_error:
                        print(f"⚠️ 안전장치: 메모리 데이터 업데이트 실패: {mem_error}")
                        traceback.print_exc()
                
                return True
            else:
                print(f"안전장치: calibration_results.xml이 복원되지 않아 Inference 클래스 업데이트 불가")
                return False
        except Exception as inference_error:
            print(f"안전장치: Inference 클래스 업데이트 실패: {inference_error}")
            return False
            
    except Exception as restore_error:
        print(f"안전장치: 이전 캘리브레이션 결과 복원 중 오류 발생: {restore_error}")
        traceback.print_exc()
        return False


def _cleanup_was_calibration_data(calibration_dir: str) -> None:
    """
    WAS 로컬 calibration_dir 폴더 내부의 모든 데이터 삭제
    
    Args:
        calibration_dir: calibration 디렉토리 경로
    """
    def _remove_readonly(func, path, exc):
        """읽기 전용 파일 삭제를 위한 헬퍼 함수"""
        try:
            os.chmod(path, 0o777)  # 권한 변경
            func(path)
        except Exception:
            pass
    
    try:
        if os.path.exists(calibration_dir):
            for item in os.listdir(calibration_dir):
                item_path = os.path.join(calibration_dir, item)
                try:
                    if os.path.isdir(item_path):
                        # 읽기 전용 파일 처리를 위한 onerror 추가
                        shutil.rmtree(item_path, onerror=_remove_readonly)
                        print(f"캘리브레이션 디렉토리 삭제: {item_path}")
                    else:
                        # 파일의 읽기 전용 속성 제거 시도
                        try:
                            os.chmod(item_path, 0o777)
                        except Exception:
                            pass
                        os.remove(item_path)
                        print(f"캘리브레이션 파일 삭제: {item_path}")
                except PermissionError as perm_error:
                    print(f"권한 오류: {item_path}, 오류: {perm_error}")
                except Exception as item_error:
                    print(f"항목 삭제 실패: {item_path}, 오류: {item_error}")
            print(f"캘리브레이션 디렉토리 정리 완료: {calibration_dir}")
        else:
            print(f"캘리브레이션 디렉토리가 존재하지 않음: {calibration_dir}")
    except Exception as cleanup_error:
        print(f"이미지 정리 중 오류 발생: {cleanup_error}")


def _classify_calibration_error(error_body: str) -> int:
    """
    캘리브레이션 오류 코드 분류
    
    Args:
        error_body: 오류 메시지
    
    Returns:
        int: 에러 코드
    """
    error_body_lower = error_body.lower()
    
    if "calibration_results.xml" in error_body_lower or "calibration_results" in error_body_lower:
        if "not found" in error_body_lower or "없습니다" in error_body:
            return 605  # ERROR_CODE_CALIBRATION_RESULT_FILE_NOT_FOUND
        elif "write" in error_body_lower or "저장" in error_body:
            return 602  # ERROR_CODE_CALIBRATION_FILE_WRITE_FAILED
    elif "depth" in error_body_lower or "depth_offset" in error_body_lower:
        return 607  # ERROR_CODE_CALIBRATION_DEPTH_VALIDATION_FAILED
    elif "inference" in error_body_lower or "restart" in error_body_lower:
        return 606  # ERROR_CODE_CALIBRATION_AI_RESTART_FAILED
    elif "connection" in error_body_lower or "connect" in error_body_lower:
        return 608  # ERROR_CODE_CALIBRATION_CONNECTION_FAILED
    elif "http" in error_body_lower:
        return 609  # ERROR_CODE_CALIBRATION_HTTP_ERROR
    elif "file" in error_body_lower and "not found" in error_body_lower:
        return 605  # ERROR_CODE_CALIBRATION_RESULT_FILE_NOT_FOUND
    
    return 600  # ERROR_CODE_CALIBRATION_UNKNOWN

