from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
# from app.core.logger import api_logger
import json
import os
import shutil
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
import zmq
import time
import asyncio
import sys
import glob
import traceback
import cv2
import shutil
# [임시] AI 추론 관련 import를 lazy import로 변경 (모듈 로드 시점에 실행되지 않도록)
# from app.core.Artis_AI.inference_cloud import Inference
# from app.core.Artis_AI import common_config as cc
# from app.core.Artis_AI.common_config import path_to_root
# from app.core.Artis_AI.depth import crop_calibration_images, calculate_and_save_depth_offset
from app.services.telegram_worker import TelegramBotManager

# NOTE: 2026-01-15
# 캘리브레이션 관련 함수들은 calibration_manager 모듈로 분리
from app.core.Artis_AI.camera.calibration_manager import (
    serial_calibration_data,
    extract_serial_number,
    _parse_stereo_calibration_params,
    _parse_single_calibration_params,
    load_all_serial_calibration_data,
    apply_serial_calibration_data,
    print_serial_calibration_data,
    _restore_calibration_from_nas,
    _cleanup_was_calibration_data,
    _classify_calibration_error,
    update_serial_calibration_data,
    remove_serial_calibration_data
)

router = APIRouter()

_edge_status_map: Dict[str, dict] = {}
ISCAN_INSTANCE_STATUS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'iScan_instance_status.json')

def _load_edge_status_map():
    """iScan Instance 상태 정보를 JSON 파일에서 로드"""
    global _edge_status_map
    try:
        if os.path.exists(ISCAN_INSTANCE_STATUS_FILE):
            with open(ISCAN_INSTANCE_STATUS_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                _edge_status_map.update(loaded_data)
                print(f"✅ iScan Instance 상태 정보 로드 완료: {ISCAN_INSTANCE_STATUS_FILE} ({len(_edge_status_map)}개 장비)")
        else:
            print(f"ℹ️ iScan Instance 상태 파일이 없습니다. 새로 생성합니다: {ISCAN_INSTANCE_STATUS_FILE}")
    except Exception as e:
        print(f"⚠️ iScan Instance 상태 정보 로드 실패 (기본값 사용): {e}")

def save_edge_status(server_id: str, status: dict):
    """
    EdgeMan 상태 정보를 저장합니다.
    
    Args:
        server_id: "vendorName/dbKey" 형식의 서버 ID
        status: EdgeMan 상태 정보 딕셔너리
    """
    global _edge_status_map
    
    # 마지막 활동 시간 추가
    status["lastSeen"] = datetime.now().timestamp() * 1000  # 밀리초
    _edge_status_map[server_id] = status
    
    # JSON 파일에 저장
    try:
        with open(ISCAN_INSTANCE_STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(_edge_status_map, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ iScan Instance 상태 정보 저장 실패: {e}")


def get_all_edge_servers() -> Dict[str, dict]:
    """
    저장된 모든 EdgeMan 정보를 반환합니다.
    
    Returns:
        server_id -> status 딕셔너리
    """
    return _edge_status_map.copy()

# [임시] SDUI 웹 서비스 테스트를 위한 플래그
AI_INFERENCE_DISABLED = False  # True로 설정하면 AI 추론 관련 엔드포인트가 비활성화됩니다

# [임시] AI 추론 관련 모듈을 lazy import하는 헬퍼 함수
def _get_inference_modules():
    """AI 추론 모듈을 lazy import"""
    if AI_INFERENCE_DISABLED:
        return None, None, None, None, None
    from app.core.Artis_AI.inference_cloud import Inference
    from app.core.Artis_AI import common_config as cc
    from app.core.Artis_AI.common_config import path_to_root
    from app.core.Artis_AI.depth import crop_calibration_images, calculate_and_save_depth_offset
    return Inference, cc, path_to_root, crop_calibration_images, calculate_and_save_depth_offset

async def process_db_deletion_background(remove_dir: str, metadata_dict: dict, current_time: str, start_time: float, telegram_manager: TelegramBotManager):
    """백그라운드에서 DB 삭제 및 텔레그램 메시지 전송 처리"""
    try:
        print(f"🔄 백그라운드 DB 삭제 시작: {remove_dir}")
        
        # 디렉토리 삭제 처리
        if os.path.exists(remove_dir):
            import shutil
            shutil.rmtree(remove_dir)
            print(f"✅ 디렉토리 삭제 완료: {remove_dir}")
        else:
            print(f"⚠️ 디렉토리가 존재하지 않습니다: {remove_dir}")

        # 소요 시간 계산 (밀리초 단위)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

        print(f"🗑️ DB 경로 삭제 완료: {remove_dir}, 소요시간: {elapsed_time:.2f}ms")
        
        # 텔레그램 메시지 전송
        try:
            telegram_message = f"""
🗑️ DB 경로 삭제 완료
📁 삭제 경로: {remove_dir}
⏱️ 소요시간: {elapsed_time:.2f}ms
🕐 처리시간: {current_time}
            """
            # chat_id가 있으면 전달, 없으면 기본값 사용
            chat_id = metadata_dict.get("chatID")
            await telegram_manager.send_message_async(telegram_message, chat_id)
            print(f"📤 텔레그램 메시지 전송 완료")
        except Exception as e:
            print(f"❌ 텔레그램 메시지 전송 실패: {e}")
            
    except Exception as e:
        print(f"❌ 백그라운드 DB 삭제 처리 중 오류 발생: {e}")
        # 에러 발생 시에도 텔레그램으로 알림
        try:
            error_message = f"""
❌ DB 경로 삭제 실패
📁 삭제 경로: {remove_dir}
🕐 처리시간: {current_time}
💥 오류: {str(e)}
            """
            chat_id = metadata_dict.get("chatID")
            await telegram_manager.send_message_async(error_message, chat_id)
        except Exception as telegram_error:
            print(f"❌ 에러 알림 텔레그램 전송도 실패: {telegram_error}")

async def send_telegram_message_by_vendor(telegram_manager: TelegramBotManager, vendor_name: str, message: str, log_prefix: str = "", chat_id: str = None, photo_path: Optional[str] = None):
    """
    vendorName과 일치하는 봇에게만 텔레그램 메시지를 전송합니다.
    chat_id가 제공된 경우에는 vendorName 매칭을 건너뛰고 모든 봇에 대해 해당 chat_id로 메시지를 전송합니다.

    Args:
        telegram_manager: TelegramBotManager 인스턴스
        vendor_name: 매칭할 vendorName 값
        message: 전송할 메시지
        log_prefix: 로그 접두사 (성공/에러 구분용)
        chat_id: 우선적으로 사용할 chat_id (메타데이터에서 전달받은 값)
        photo_path: 전송할 이미지 파일 경로 (선택적)
    """
    try:
        # chat_id가 제공된 경우, vendorName 매칭을 건너뛰고 모든 봇에 대해 해당 chat_id로 메시지 전송
        if chat_id:
            print(f"{log_prefix}메타데이터 chatID ({chat_id})")
            for bot_name, bot_config in telegram_manager.bots.items():
                try:
                    api_url = f"https://api.telegram.org/bot{bot_config['token']}"
                    # 이미지가 있으면 이미지와 함께 전송, 없으면 텍스트만 전송
                    if photo_path and os.path.exists(photo_path):
                        photo_success = await telegram_manager._send_photo(api_url, str(chat_id), photo_path, message)
                        if photo_success:
                            print(f"{log_prefix}텔레그램 메시지 및 이미지 전송 완료: {bot_name} -> chat_id: {chat_id}")
                        else:
                            # 이미지 전송 실패 시 텍스트만 전송
                            await telegram_manager._send_message(api_url, str(chat_id), message)
                            print(f"{log_prefix}텔레그램 메시지 전송 완료 (이미지 전송 실패): {bot_name} -> chat_id: {chat_id}")
                    else:
                        await telegram_manager._send_message(api_url, str(chat_id), message)
                        print(f"{log_prefix}텔레그램 메시지 전송 완료: {bot_name} -> chat_id: {chat_id}")
                except Exception as bot_error:
                    print(f"{log_prefix}봇 '{bot_name}'에서 메시지 전송 실패: {bot_error}")
            print(f"{log_prefix}총 {len(telegram_manager.bots)}개의 봇에 대해 chatID {chat_id}로 메시지 전송 완료")
        else:
            print(f"{log_prefix}vendorName: {vendor_name}")
            # 기존 로직: vendorName과 일치하는 봇에게만 메시지 전송
            for bot_name, bot_config in telegram_manager.bots.items():
                # 봇의 vendorName과 일치하는지 확인
                bot_vendor_name = bot_config.get("vendorName", "")

                if str(bot_vendor_name) == str(vendor_name):
                    # 기존 config.json의 chat_ids 사용
                    target_chat_ids = bot_config.get("chat_ids", [])
                    print(f"{log_prefix}기존 chat_ids 사용: {target_chat_ids}")

                    api_url = f"https://api.telegram.org/bot{bot_config['token']}"
                    for target_chat_id in target_chat_ids:
                        # 이미지가 있으면 이미지와 함께 전송, 없으면 텍스트만 전송
                        if photo_path and os.path.exists(photo_path):
                            photo_success = await telegram_manager._send_photo(api_url, str(target_chat_id), photo_path, message)
                            if photo_success:
                                print(f"{log_prefix}텔레그램 메시지 및 이미지 전송 완료: {bot_name} -> chat_id: {target_chat_id}")
                            else:
                                # 이미지 전송 실패 시 텍스트만 전송
                                await telegram_manager._send_message(api_url, str(target_chat_id), message)
                                print(f"{log_prefix}텔레그램 메시지 전송 완료 (이미지 전송 실패): {bot_name} -> chat_id: {target_chat_id}")
                        else:
                            await telegram_manager._send_message(api_url, str(target_chat_id), message)
                            print(f"{log_prefix}텔레그램 메시지 전송 완료: {bot_name} -> chat_id: {target_chat_id}")

                    if target_chat_ids:
                        print(f"{log_prefix}총 {len(target_chat_ids)}개의 chat_id에 메시지 전송 완료")
                    else:
                        print(f"{log_prefix}봇 '{bot_name}'에 chat_ids가 설정되지 않았습니다")
                else:
                    print(f"{log_prefix}vendorName '{vendor_name}'와 일치하는 봇이 없습니다 (현재 봇: {bot_name}, 봇 vendorName: {bot_vendor_name})")

    except Exception as e:
        print(f"{log_prefix}텔레그램 메시지 전송 실패: {e}")

# 수신 디렉토리 설정
RECEIVE_DIR = "receive"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 10MB
MAX_TOTAL_SIZE = 500 * 1024 * 1024  # 100MB

# Model (lazy initialization)
artis_model = None
artis_thread = None

def _get_artis_model():
    """AI 추론 모델을 lazy initialization"""
    global artis_model
    if AI_INFERENCE_DISABLED:
        return None
    if artis_model is None:
        Inference, _, _, _, _ = _get_inference_modules()
        if Inference is not None:
            artis_model = Inference()
    return artis_model

def ensure_receive_dir():
    """수신 디렉토리 생성 및 기존 파일 정리"""
    # 디렉토리가 존재하지 않으면 새로 생성
    if not os.path.exists(RECEIVE_DIR):
        os.makedirs(RECEIVE_DIR, exist_ok=True)
        print(f"새로운 {RECEIVE_DIR} 디렉토리를 생성했습니다.")
    else:
        # 디렉토리가 존재하면 내부 파일들만 삭제
        try:
            for filename in os.listdir(RECEIVE_DIR):
                file_path = os.path.join(RECEIVE_DIR, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"기존 파일 삭제: {filename}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"기존 하위 디렉토리 삭제: {filename}")
            print(f"기존 {RECEIVE_DIR} 디렉토리의 파일들을 정리했습니다.")
        except Exception as e:
            print(f"기존 파일 삭제 중 오류 발생: {e}")

def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """메타데이터 검증"""
    required_fields = ["publicIP", "companyName", "vendorName", "dbKey", "runMode", "totalScanCount"]

    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"필수 필드가 누락되었습니다: {field}")

    # runMode 검증 (경고만, 에러는 발생하지 않음)
    valid_run_modes = ["UserRun", "NewItem", "CalCam", "Base"]
    if metadata["runMode"] not in valid_run_modes:
        print(f"알 수 없는 runMode 값: {metadata['runMode']} (예상 값: {', '.join(valid_run_modes)})")

    # fileCount와 fileList가 있는 경우에만 검증
    if "fileCount" in metadata:
        # fileCount가 숫자인지 확인
        try:
            file_count = int(metadata["fileCount"])
            if file_count <= 0:
                raise ValueError("fileCount는 0보다 커야 합니다")
        except (ValueError, TypeError):
            raise ValueError("fileCount는 유효한 숫자여야 합니다")

        # fileList가 있는지 확인
        if "fileList" not in metadata:
            raise ValueError("fileCount가 있으면 fileList도 필요합니다")

        # fileList가 리스트인지 확인
        if not isinstance(metadata["fileList"], list):
            raise ValueError("fileList는 배열이어야 합니다")

        # fileCount와 fileList 길이가 일치하는지 확인
        if len(metadata["fileList"]) != file_count:
            raise ValueError(f"fileCount({file_count})와 fileList 길이({len(metadata['fileList'])})가 일치하지 않습니다")

        # 각 파일 정보 검증
        for i, file_info in enumerate(metadata["fileList"]):
            if not isinstance(file_info, dict):
                raise ValueError(f"fileList[{i}]는 객체여야 합니다")

            if "name" not in file_info or "size" not in file_info:
                raise ValueError(f"fileList[{i}]에 name과 size가 필요합니다")

            try:
                file_size = int(file_info["size"])
                if file_size <= 0:
                    raise ValueError(f"fileList[{i}].size는 0보다 커야 합니다")
            except (ValueError, TypeError):
                raise ValueError(f"fileList[{i}].size는 유효한 숫자여야 합니다")
    elif "fileList" in metadata:
        # fileCount가 없는데 fileList가 있는 경우
        raise ValueError("fileList가 있으면 fileCount도 필요합니다")

    return metadata

def validate_file_integrity(file: UploadFile, expected_size: int, expected_name: str) -> Dict[str, Any]:
    """파일 무결성 검증 - 스트리밍 방식"""
    # 파일 크기 검증
    file_size = 0
    file_content = b""
    
    # 파일 내용 읽기 - 스트리밍 방식
    chunk_size = 8192  # 8KB 청크
    while chunk := file.file.read(chunk_size):
        file_content += chunk
        file_size += len(chunk)
        
        # 메모리 사용량 제한 (50MB)
        if len(file_content) > MAX_FILE_SIZE:
            raise ValueError(f"파일 크기 초과: {len(file_content)}바이트 (최대 {MAX_FILE_SIZE}바이트)")
    
    # 파일 크기 검증
    if file_size != expected_size:
        raise ValueError(f"파일 크기 불일치: 예상 {expected_size}바이트, 실제 {file_size}바이트")
    
    # 파일명 검증
    if file.filename != expected_name:
        raise ValueError(f"파일명 불일치: 예상 '{expected_name}', 실제 '{file.filename}'")
    
    # 파일 크기 제한 검증
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"파일 크기 초과: {file_size}바이트 (최대 {MAX_FILE_SIZE}바이트)")
    
    return {
        "size": file_size,
        "content": file_content,
        "filename": file.filename
    }

def encode_file_to_base64(file_path: str, chunk_size: int = 8192) -> str:
    try:
        # 파일 전체를 한 번에 읽어서 Base64 인코딩
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        # 전체 파일을 Base64로 인코딩
        encoded_data = base64.b64encode(file_data).decode('utf-8')
        
        print(f"🔍 Base64 인코딩 완료: 원본 {len(file_data):,} bytes -> 인코딩 {len(encoded_data):,} 문자")
        
        return encoded_data
        
    except Exception as e:
        print(f"Base64 인코딩 실패: {e}")
        raise


@router.post("/iscan-input-images")
async def receive_images(
    request: Request,
    metadata: str = Form(...)
):
    """
    이미지 파일 수신 엔드포인트

    Args:
        request: FastAPI Request 객체
        metadata: JSON 형태의 메타데이터

    Returns:
        처리 결과 정보
    """
    # [임시] AI 추론 비활성화 체크
    if AI_INFERENCE_DISABLED:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "AI 추론 기능이 임시로 비활성화되어 있습니다 (SDUI 테스트 모드)",
                "data": {
                    "body": "이 기능은 현재 테스트 모드에서 사용할 수 없습니다."
                }
            }
        )
    
    start_time = time.time()
    telegram_manager = TelegramBotManager()

    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("=" * 100)
        print(f"🚀 /api/iscan-input-images 동작 시작 - 서버 시각: {current_time}")
        print("=" * 100)

        # 파라미터 검증
        print("파라미터 검증")
        if not metadata:
            print("메타데이터가 누락되었습니다")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "body": "메타데이터가 누락되었습니다"
                }
            }

        # 메타데이터 즉시 파싱하여 주요 정보 로그 출력
        try:
            metadata_dict = json.loads(metadata)
            print(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"메타데이터 JSON 파싱 실패: {e}")
            print(f"메타데이터 원본: {metadata}")
        except Exception as e:
            print(f"메타데이터 로그 출력 실패: {e}")
            print(f"메타데이터 원본: {metadata}")

        # 동적으로 파일들을 수집
        files = []
        form_data_start = time.time()
        form_data = await request.form()
        form_data_time = (time.time() - form_data_start) * 1000
        print(f"📊 Form 데이터 파싱 시간: {form_data_time:.2f}ms")
        
        # file0, file1, file2, ... 형태의 파일들을 찾아서 수집
        i = 0
        while True:
            file_key = f"file{i}"
            if file_key in form_data:
                file = form_data[file_key]
                if hasattr(file, 'filename') and file.filename:
                    files.append(file)
                    print(f"파일 {i}: {file.filename}, 크기: {file.size}, 타입: {file.content_type}")
                i += 1
            else:
                break

        if len(files) == 0:
            print("파일이 누락되었습니다")
            return {
                    "success": False,
                    "message": "파일 누락",
                    "data": {
                        "body": "파일이 누락되었습니다"
                    }
            }

        print(f"{metadata}")
        #print(f"메타데이터: {metadata}")
        print(f"파일 개수: {len(files)}")

        # 1. 메타데이터 파싱 및 검증
        metadata_start = time.time()
        try:
            metadata_dict = json.loads(metadata)
            validated_metadata = validate_metadata(metadata_dict)
            print(f"메타데이터 검증 완료: {validated_metadata['fileCount']}개 파일")
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return {
                "success": False,
                "message": "JSON 파싱 실패",
                "data": {
                    "body": f"잘못된 JSON 형식: {str(e)}"
                }
            }
        except ValueError as e:
            print(f"메타데이터 검증 오류: {e}")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "body": f"메타데이터 검증 실패: {str(e)}"
                }
            }
        metadata_time = (time.time() - metadata_start) * 1000
        print(f"📊 메타데이터 파싱/검증 시간: {metadata_time:.2f}ms")

        # iScan 장비 정보 저장
        try:
            vendor_name = validated_metadata.get("vendorName") or validated_metadata.get("vendor")
            db_key = validated_metadata.get("dbKey")
            
            if vendor_name and db_key:
                server_id = f"{vendor_name}/{db_key}"
                
                total_scan_count = 0
                try:
                    scan_count_value = validated_metadata.get("totalScanCount")
                    if scan_count_value is not None:
                        total_scan_count = int(scan_count_value)
                except (ValueError, TypeError):
                    total_scan_count = 0
                
                status = {
                    "totalScanCount": total_scan_count,
                    "serialNumber": validated_metadata.get("serialNumber", ""), 
                }
                
                save_edge_status(server_id, status)
        except Exception as e:
            print(f"⚠️ EdgeMan 정보 저장 실패 (무시하고 계속 진행): {e}")
        
        # 2. 파일 개수 검증
        expected_file_count = int(validated_metadata["fileCount"])
        actual_file_count = len(files)

        if actual_file_count != expected_file_count:
            print(f"파일 개수 불일치: 예상 {expected_file_count}개, 실제 {actual_file_count}개")
            return {
                "success": False,
                "message": "파일 개수 불일치",
                "data": {
                    "body": f"파일 개수 불일치: 예상 {expected_file_count}개, 실제 {actual_file_count}개"
                }
            }

        # 3. 수신 디렉토리 생성
        dir_start = time.time()
        ensure_receive_dir()
        dir_time = (time.time() - dir_start) * 1000
        print(f"📊 디렉토리 생성 시간: {dir_time:.2f}ms")
        
        # 4. 파일 처리 및 무결성 검증
        file_processing_start = time.time()
        total_size = 0
        processed_files = []

        timestamp = validated_metadata.get("timestamp")

        # timestamp를 읽기 쉬운 형태로 변환 (YY-MM-DD HH:MM:SS)
        readable_timestamp = ""
        if timestamp and len(timestamp) >= 14:
            try:
                # YYYYMMDDHHMMSS 형태를 YY-MM-DD HH:MM:SS로 변환
                year = timestamp[0:4]
                month = timestamp[4:6]
                day = timestamp[6:8]
                hour = timestamp[8:10]
                minute = timestamp[10:12]
                second = timestamp[12:14]
                readable_timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}"
            except:
                readable_timestamp = timestamp  # 변환 실패 시 원본 사용
        else:
            readable_timestamp = timestamp

        # 총 스캔 횟수
        totalScanCount = validated_metadata.get("totalScanCount", "0")
        try:
            totalScanCount_int = int(totalScanCount)
            if totalScanCount_int > 0:
                print(f"총 스캔 횟수: {totalScanCount_int}")
            else:
                print(f"총 스캔 횟수: 0")
        except (ValueError, TypeError):
            print(f"totalScanCount 값을 정수로 변환할 수 없습니다: {totalScanCount}")
            print(f"총 스캔 횟수: 0")

        for i, file in enumerate(files):
            try:
                file_start = time.time()
                # fileList에서 해당 파일 정보 가져오기
                file_info = validated_metadata["fileList"][i]
                expected_name = file_info["name"]
                expected_size = int(file_info["size"])

                # 파일 무결성 검증
                validation_start = time.time()
                file_data = validate_file_integrity(file, expected_size, expected_name)
                validation_time = (time.time() - validation_start) * 1000
                print(f"📊 파일 {i+1} 검증 시간: {validation_time:.2f}ms")
                
                total_size += file_data["size"]

                # 파일 저장
                save_start = time.time()
                #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                #safe_filename = f"{timestamp}_{i}_{expected_name}"
                safe_filename = f"{timestamp}_{expected_name}"
                file_path = os.path.join(RECEIVE_DIR, safe_filename)

                with open(file_path, "wb") as f:
                    f.write(file_data["content"])
                
                save_time = (time.time() - save_start) * 1000
                print(f"📊 파일 {i+1} 저장 시간: {save_time:.2f}ms")
                
                processed_files.append({
                    "original_name": expected_name,
                    "saved_name": safe_filename,
                    "size": file_data["size"],
                    "path": file_path
                })
                
                file_total_time = (time.time() - file_start) * 1000
                print(f"파일 {i+1}/{expected_file_count} 처리 완료: {expected_name} (총 {file_total_time:.2f}ms)")
                
            except Exception as e:
                print(f"파일 {i+1} 처리 오류: {e}")
                return {
                    "success": False,
                    "message": "파일 처리 오류",
                    "data": {
                        "body": f"파일 {i+1} 처리 실패: {str(e)}"
                    }
                }
        
        file_processing_time = (time.time() - file_processing_start) * 1000
        print(f"📊 전체 파일 처리 시간: {file_processing_time:.2f}ms")
        
        # 5. 전체 크기 검증
        if total_size > MAX_TOTAL_SIZE:
            print(f"전체 파일 크기 초과: {total_size}바이트 (최대 {MAX_TOTAL_SIZE}바이트)")
            return {
                "success": False,
                "message": "전체 파일 크기 초과",
                "data": {
                    "body": f"전체 파일 크기 초과: {total_size}바이트 (최대 {MAX_TOTAL_SIZE}바이트)"
                }
            }

        # 6. 추론 전 시리얼 넘버에 맞는 calibration 적용
        serial_number = extract_serial_number(validated_metadata)
        apply_success = apply_serial_calibration_data(serial_number)
        if not apply_success:
            print("⚠️ Calibration 데이터 적용 실패, 기본 모델로 추론 진행")

        # 7. 추론하기
        inference_start = time.time()
        run_mode = validated_metadata.get("runMode", "UserRun")
        
        try:
            artis_model = _get_artis_model()
            if artis_model is None:
                raise RuntimeError("AI 추론 모델을 사용할 수 없습니다")
            artis_result, zipfile, zipname, artis_time = artis_model.inference(processed_files, RECEIVE_DIR, run_mode, timestamp)
            #print(f"inference 성공: artis_result={type(artis_result)}, zipfile={zipfile}, zipname={zipname}, artis_time={type(artis_time)}")
        except Exception as e:
            #shutil.copytree(RECEIVE_DIR, RECEIVE_DIR + "_ERR", dirs_exist_ok=True)
            print(f"inference 에러: {e}")
            print(f"에러 타입: {type(e)}")
            print(f"에러 스택: {traceback.format_exc()}")
            raise
            
        inference_time = (time.time() - inference_start) * 1000
        print(f"📊 추론 시간: {inference_time:.2f}ms")

        inference_msg_start = time.time()
        
        try:
            artis_msg = artis_model.make_msg(artis_result, artis_time)
            #print(f"make_msg 성공: {artis_msg[:100]}...")  # 처음 100자만 출력
        except Exception as e:
            print(f"make_msg 에러: {e}")
            print(f"에러 타입: {type(e)}")
            print(f"에러 스택: {traceback.format_exc()}")
            raise

        inference_msg_time = (time.time() - inference_msg_start) * 1000
        print(f"📊 추론 디버깅 정보 생성 시간: {inference_msg_time:.2f}ms")
        
        # zip 파일 경로를 절대 경로로 변환
        if not os.path.isabs(zipfile):
            zipfile = os.path.join(RECEIVE_DIR, zipfile)

        print(f"zip 파일 경로: {zipfile}")
        print(f"zip 파일 존재 여부: {os.path.exists(zipfile)}")

        if not os.path.exists(zipfile):
            print(f"zip 파일이 존재하지 않습니다: {zipfile}")
            return {
                "success": False,
                "message": "zip 파일 오류",
                "data": {
                    "body": f"zip 파일이 존재하지 않습니다: {zipfile}"
                }
            }

        zipfilesize = os.path.getsize(zipfile)
        print(f"이미지 추론 완료, zip 파일 크기: {zipfilesize} 바이트")

        # 결과 디렉토리 경로 계산(실제 생성은 백그라운드에서)
        result_dir_start = time.time()
        if validated_metadata.get("savePath", None) is not None:
            # 경로 정규화
            safe_path = os.path.normpath(validated_metadata["savePath"])
            result_dir = os.path.join("/mynas/", safe_path)
        else:
            result_dir = os.path.join("/mynas/uploads/",
                                  validated_metadata.get("vendorName"),
                                  validated_metadata.get("dbKey"),
                                  "EdgeMan", run_mode,
                                  timestamp[0:4], timestamp[4:6], timestamp[6:8], timestamp[8:10])
        #os.makedirs(result_dir, exist_ok=True)
        result_dir_time = (time.time() - result_dir_start) * 1000
        print(f"📊 결과 디렉토리 경로 계산 시간: {result_dir_time:.2f}ms")

        # 7. 성공 응답 - 클라이언트가 기대하는 형식으로 수정
        # 생성된 zip 파일을 base64로 인코딩 - 스트리밍 방식
        base64_start = time.time()
        zip_base64 = ""
        try:
            zip_file_path = os.path.join(RECEIVE_DIR, zipname)

            print(f"응답 zip 파일 경로: {zip_file_path}")
            print(f"응답 zip 파일 존재 여부: {os.path.exists(zip_file_path)}")

            if os.path.exists(zip_file_path):
                # 스트리밍 방식으로 base64 인코딩
                zip_base64 = encode_file_to_base64(zip_file_path)
                print(f"zip 파일 base64 인코딩 완료: {len(zip_base64)} 문자")
            else:
                print(f"응답 zip 파일이 존재하지 않습니다: {zip_file_path}")
                # 원본 zip 파일에서 읽기 시도
                if os.path.exists(zipfile):
                    zip_base64 = encode_file_to_base64(zipfile)
                    print(f"원본 zip 파일에서 base64 인코딩 완료: {len(zip_base64)} 문자")
                else:
                    print(f"원본 zip 파일도 존재하지 않습니다: {zipfile}")
        except Exception as e:
            print(f"zip 파일 base64 인코딩 실패: {e}")
            zip_base64 = ""
        
        json_file_path = os.path.join(RECEIVE_DIR, "artis_result_debug.json")
        depth_file_path = os.path.join(RECEIVE_DIR, "Cam_2_Depth.jpg")

        # 변수 초기화
        json_file_base64 = ""
        depth_file_base64 = ""
            
        if os.path.exists(json_file_path):
            json_file_base64 = encode_file_to_base64(json_file_path)
        if os.path.exists(depth_file_path):
            depth_file_base64 = encode_file_to_base64(depth_file_path)

        base64_time = (time.time() - base64_start) * 1000
        print(f"📊 Base64 인코딩 시간: {base64_time:.2f}ms")
        
        # CalCam 모드: 결과 json + zip 파일 응답
        if run_mode == "CalCam":
            files_array = [json_file_base64, zip_base64]
            file_list_array = [
                {"name": "artis_result_debug.json", "size": os.path.getsize(json_file_path) if os.path.exists(json_file_path) else 0},
                {"name": zipname, "size": zipfilesize}
            ]
        # UserRun 모드: 결과 json + depth 이미지
        elif run_mode == "UserRun":
            files_array = [json_file_base64, depth_file_base64]
            file_list_array = [
                {"name": "artis_result_debug.json", "size": os.path.getsize(json_file_path) if os.path.exists(json_file_path) else 0},
                {"name": "Cam_2_Depth.jpg", "size": os.path.getsize(depth_file_path) if os.path.exists(depth_file_path) else 0}
            ]
        # 그 외 모드 (NewItem, Base 등): 결과 json + depth 이미지 + zip
        else:
            files_array = [json_file_base64, depth_file_base64, zip_base64]
            file_list_array = [
                {"name": "artis_result_debug.json", "size": os.path.getsize(json_file_path) if os.path.exists(json_file_path) else 0},
                {"name": "Cam_2_Depth.jpg", "size": os.path.getsize(depth_file_path) if os.path.exists(depth_file_path) else 0},
                {"name": zipname, "size": zipfilesize}
            ]

        #print("=== 클라이언트 조건 확인 ===")
        #print(f"files 배열 크기: {len(files_array)}")
        #print(f"fileList 배열 크기: {len(file_list_array)}")
        #print(f"files 배열 타입: {type(files_array)}")
        #print(f"fileList 배열 타입: {type(file_list_array)}")
        #print(f"조건 충족 여부: {len(files_array) == len(file_list_array)}")
        #print(f"zip DB 이름: {zipname}")
        #print("=== 클라이언트 조건 확인 끝 ===")

        # 소요 시간 계산 (밀리초 단위)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

        result = {
            "success": True,
            "message": "이미지 수신 처리 및 추론 완료",
            "data": {
                "metadata": validated_metadata,
                "files": files_array,  # base64로 인코딩된 파일 데이터
                "fileList": file_list_array,  # <timestamp>.zip 외 파일 리스트
                "elapsedTime": f"{elapsed_time:.2f}"
            }
        }

        # 응답을 먼저 반환하고 텔레그램 메시지는 백그라운드에서 전송
        print(f"이미지 수신 완료: {len(processed_files)}개 파일, {total_size}바이트, 소요시간: {elapsed_time:.2f}ms")
        
        # 📊 성능 분석 요약
        print("=" * 60)
        print("📊 성능 분석 요약")
        print("=" * 60)
        print(f"Form 데이터 파싱 : {form_data_time:.2f} ms")
        print(f"메타데이터 파싱/검증 : {metadata_time:.2f} ms")
        print(f"디렉토리 생성 : {dir_time:.2f} ms")
        print(f"파일 처리 : {file_processing_time:.2f} ms")
        print(f"추론 : {inference_time:.2f} ms")
        print(f"추론 디버깅 정보 생성 : {inference_msg_time:.2f} ms")
        print(f"결과 디렉토리 경로 계산 : {result_dir_time:.2f} ms")
        print(f"Base64 인코딩 : {base64_time:.2f} ms")
        print(f"총 소요 시간 : {elapsed_time:.2f} ms")
        print("=" * 60)
        
        # 텔레그램 메시지 생성 및 전송
        async def send_telegram_background():
            try:
                message = f"""📸 iScan 입력 영상 처리 완료
{artis_msg}

• {readable_timestamp}
• {run_mode}-{totalScanCount}
• ZIP 파일명 : {zipname}
• ZIP 파일 크기 : {zipfilesize:,} bytes
• ZIP 파일 경로 : {result_dir}
⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
"""
                '''
                                
                📊 프로파일링
                • Form 데이터 파싱 : {form_data_time:.2f} ms
                • 메타데이터 파싱/검증 : {metadata_time:.2f} ms
                • 디렉토리 생성 : {dir_time:.2f} ms
                • 파일 처리 : {file_processing_time:.2f} ms
                • 추론 : {inference_time:.2f} ms
                • 추론 디버깅 정보 생성 : {inference_msg_time:.2f} ms
                • 결과 디렉토리 경로 계산 : {result_dir_time:.2f} ms
                • Base64 인코딩 : {base64_time:.2f} ms
                """'''
                print(message)
                # vendorName과 일치하는 봇에게만 메시지 전송
                vendor_name = validated_metadata.get("vendorName", "")
                chat_id = validated_metadata.get("chatID", "")
                
                # artis_combined_debug.jpg 이미지 경로 (텔레그램으로 전송)
                from app.core.Artis_AI import common_config as cc
                combined_debug_path = os.path.join(RECEIVE_DIR, cc.artis_debug_img_file)
                photo_path = combined_debug_path if os.path.exists(combined_debug_path) else None
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, message, "[성공] ", chat_id, photo_path)
                
                # 텔레그램 전송 완료 후 디버그 이미지 삭제
                if photo_path and os.path.exists(photo_path):
                    try:
                        os.remove(photo_path)
                        print(f"디버그 이미지 삭제 완료: {cc.artis_debug_img_file}")
                    except Exception as delete_error:
                        print(f"디버그 이미지 삭제 중 오류 발생: {delete_error}")

            except Exception as telegram_error:
                print(f"텔레그램 메시지 전송 실패: {telegram_error}")

        # 백그라운드에서 텔레그램 메시지 전송
        asyncio.create_task(send_telegram_background())

        # 결과 ZIP 파일 백그라운드 NAS 디렉토리 생성 및 이동
        async def move_zip_to_nas_background():
            try:
                # NAS 디렉토리 생성
                nas_dir_start = time.time()
                os.makedirs(result_dir, exist_ok=True)
                nas_dir_time = (time.time() - nas_dir_start) * 1000
                print(f"📊 백그라운드 NAS 디렉토리 생성 시간: {nas_dir_time:.2f}ms")
                
                # 파일 업로드
                upload_start = time.time()
                #is_upload, msg = artis_model.upload_output(RECEIVE_DIR, result_dir, [zipname, processed_files[0]["saved_name"], processed_files[1]["saved_name"]])
                is_upload, msg = artis_model.upload_output(RECEIVE_DIR, result_dir, [zipname])
                upload_time = (time.time() - upload_start) * 1000
                print(f"📊 백그라운드 파일 업로드 시간: {upload_time:.2f}ms")

                if not is_upload:
                    print(f"NAS 이동 실패: {msg}")
                else:
                    print(f"추론 결과 ZIP 파일 및 RAW 데이터 NAS 이동 완료")
                    
                    # NAS 이동 성공 후 임시 파일들 삭제
                    cleanup_start = time.time()
                    deleted_count = 0
                    
                    try:
                        # processed_files의 모든 임시 파일 삭제
                        for file_info in processed_files:
                            file_path = file_info["path"]
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                deleted_count += 1
                                print(f"임시 파일 삭제 완료: {file_info['saved_name']}")
                        
                        # ZIP 파일도 삭제 (NAS에 이미 업로드됨)
                        zip_path = os.path.join(RECEIVE_DIR, zipname)
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                            deleted_count += 1
                            print(f"임시 ZIP 파일 삭제 완료: {zipname}")

                        # 디버그 이미지 파일들 삭제
                        from app.core.Artis_AI import common_config as cc
                        for debug_file in [cc.artis_transform_img_file, cc.artis_fusion_img_file]:
                            debug_path = os.path.join(RECEIVE_DIR, debug_file)
                            if os.path.exists(debug_path):
                                os.remove(debug_path)
                                deleted_count += 1
                                print(f"디버그 이미지 삭제 완료: {debug_file}")

                        cleanup_time = (time.time() - cleanup_start) * 1000
                        print(f"📊 임시 파일 정리 시간: {cleanup_time:.2f}ms, 삭제된 파일 수: {deleted_count}")
                        
                    except Exception as cleanup_error:
                        print(f"임시 파일 정리 중 오류 발생: {cleanup_error}")
                        
            except Exception as move_err:
                print(f"NAS 이동 중 오류 발생: {move_err}")

        if totalScanCount_int > 0:
            # 백그라운드에서 NAS 이동 실행
            try:
                asyncio.create_task(move_zip_to_nas_background())
            except Exception as task_error:
                print(f"백그라운드 NAS 이동 태스크 생성 실패: {task_error}")

        return JSONResponse(content=result, status_code=200)

    except HTTPException:
        # 에러 발생 시에도 텔레그램 메시지 전송 (백그라운드)
        async def send_error_telegram_background():
            try:
                # 소요 시간 계산 (밀리초 단위)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

                # metadata_dict에서 vendorName과 chatID 추출 시도
                vendor_name = ""
                chat_id = ""
                try:
                    metadata_dict = json.loads(metadata)
                    vendor_name = metadata_dict.get("vendorName", "")
                    chat_id = metadata_dict.get("chatID", "")
                except:
                    pass

                error_message = f"""❌ iScan 이미지 처리 실패

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
❌ 오류: HTTPException 발생

📊 처리 정보:
• 메타데이터: {metadata[:100]}... (첫 100자)
• 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

                # vendorName과 일치하는 봇에게만 메시지 전송
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, error_message, "[HTTPException] ", chat_id)

            except Exception as telegram_error:
                print(f"에러 텔레그램 메시지 전송 실패: {telegram_error}")

        # 백그라운드에서 에러 텔레그램 메시지 전송
        asyncio.create_task(send_error_telegram_background())

        raise
    except Exception as e:
        # 예상치 못한 오류 발생 시에도 텔레그램 메시지 전송 (백그라운드)
        async def send_unexpected_error_telegram_background(except_err):
            try:
                # 소요 시간 계산 (밀리초 단위)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

                # metadata_dict에서 vendorName과 chatID 추출 시도
                vendor_name = ""
                chat_id = ""
                try:
                    metadata_dict = json.loads(metadata)
                    vendor_name = metadata_dict.get("vendorName", "")
                    chat_id = metadata_dict.get("chatID", "")
                except:
                    pass

                error_message = f"""💥 iScan 이미지 처리 중 예상치 못한 오류

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
❌ 오류 타입: {type(except_err).__name__}
❌ 오류 내용: {str(except_err)}

📊 처리 정보:
• 메타데이터: {metadata[:100]}... (첫 100자)
• 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

                # vendorName과 일치하는 봇에게만 메시지 전송
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, error_message, "[Exception] ", chat_id)

            except Exception as telegram_error:
                print(f"에러 텔레그램 메시지 전송 실패: {telegram_error}")
        
        # 백그라운드에서 예상치 못한 오류 텔레그램 메시지 전송
        asyncio.create_task(send_unexpected_error_telegram_background(e))
        
        print(f"이미지 수신 중 예상치 못한 오류: {e}")
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 상세: {str(e)}")
        # 일관된 응답 형태로 에러 반환
        return {
            "success": False,
            "message": "요청 처리 실패",
            "data": {
                "body": f"요청 처리 실패: {str(e)}"
            }
        }

@router.post("/iscan-config-update")
async def update_config(
    request: Request,
    metadata: str = Form(...)
):
    """
    추론기 설정 갱신 엔드포인트

    Args:
        request: FastAPI Request 객체
        metadata: JSON 형태의 메타데이터

    Returns:
        처리 결과 정보
    """
    # [임시] AI 추론 비활성화 체크
    if AI_INFERENCE_DISABLED:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "AI 추론 기능이 임시로 비활성화되어 있습니다 (SDUI 테스트 모드)",
                "data": {
                    "body": "이 기능은 현재 테스트 모드에서 사용할 수 없습니다."
                }
            }
        )
    
    start_time = time.time()
    telegram_manager = TelegramBotManager()

    try:
        print("추론기 설정 갱신 요청 시작")

        print("파라미터 검증")
        if not metadata:
            print("메타데이터가 누락되었습니다")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "body": "메타데이터가 누락되었습니다"
                }
            }
        
        # 메타데이터 즉시 파싱하여 주요 정보 로그 출력
        try:
            metadata_dict = json.loads(metadata)
            print(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"메타데이터 JSON 파싱 실패: {e}")
            print(f"메타데이터 원본: {metadata}")
        except Exception as e:
            print(f"메타데이터 로그 출력 실패: {e}")
            print(f"메타데이터 원본: {metadata}")

        # 동적으로 파일들을 수집
        files = []
        form_data_start = time.time()
        form_data = await request.form()
        form_data_time = (time.time() - form_data_start) * 1000
        print(f"📊 Form 데이터 파싱 시간: {form_data_time:.2f}ms")
        
        # file0, file1, file2, ... 형태의 파일들을 찾아서 수집
        i = 0
        while True:
            file_key = f"file{i}"
            if file_key in form_data:
                file = form_data[file_key]
                if hasattr(file, 'filename') and file.filename:
                    files.append(file)
                    print(f"파일 {i}: {file.filename}, 크기: {file.size}, 타입: {file.content_type}")
                i += 1
            else:
                break

        if len(files) == 0:
            print("파일이 누락되었습니다")
            return {
                "success": False,
                "message": "파일 누락",
                "data": {
                    "body": "파일이 누락되었습니다"
                }
            }

        print(f"{metadata}")
        #print(f"메타데이터: {metadata}")
        print(f"파일 개수: {len(files)}")

        # 1. 메타데이터 파싱 및 검증
        metadata_start = time.time()
        try:
            metadata_dict = json.loads(metadata)
            validated_metadata = validate_metadata(metadata_dict)
            print(f"메타데이터 검증 완료: {validated_metadata['fileCount']}개 파일")
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return {
                "success": False,
                "message": "JSON 파싱 실패",
                "data": {
                    "body": f"잘못된 JSON 형식: {str(e)}"
                }
            }
        except ValueError as e:
            print(f"메타데이터 검증 오류: {e}")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "body": f"메타데이터 검증 실패: {str(e)}"
                }
            }
        metadata_time = (time.time() - metadata_start) * 1000
        print(f"📊 메타데이터 파싱/검증 시간: {metadata_time:.2f}ms")
        
        # 2. 파일 개수 검증
        expected_file_count = int(validated_metadata["fileCount"])
        actual_file_count = len(files)

        if actual_file_count != expected_file_count:
            print(f"파일 개수 불일치: 예상 {expected_file_count}개, 실제 {actual_file_count}개")
            return {
                "success": False,
                "message": "파일 개수 불일치",
                "data": {
                    "body": f"파일 개수 불일치: 예상 {expected_file_count}개, 실제 {actual_file_count}개"
                }
            }

        # 3. 파일 처리 및 무결성 검증
        file_processing_start = time.time()
        total_size = 0
        processed_files = []

        # 설정 파일 가져오기
        for i, file in enumerate(files):
            try:
                file_start = time.time()
                # fileList에서 해당 파일 정보 가져오기
                file_info = validated_metadata["fileList"][i]
                expected_name = file_info["name"]
                expected_size = int(file_info["size"])

                # 파일 무결성 검증
                validation_start = time.time()
                file_data = validate_file_integrity(file, expected_size, expected_name)
                validation_time = (time.time() - validation_start) * 1000
                print(f"📊 파일 {i+1} 검증 시간: {validation_time:.2f}ms")
                
                total_size += file_data["size"]

                # 파일 내용 리스트에 저장       
                processed_files.append({
                    "original_name": expected_name,
                    "size": file_data["size"],
                    "data": file_data["content"]
                })
                
                file_total_time = (time.time() - file_start) * 1000
                print(f"파일 {i+1}/{expected_file_count} 처리 완료: {expected_name} (총 {file_total_time:.2f}ms)")
                
            except Exception as e:
                print(f"파일 {i+1} 처리 오류: {e}")
                return {
                    "success": False,
                    "message": "파일 처리 오류",
                    "data": {
                        "body": f"파일 {i+1} 처리 실패: {str(e)}"
                    }
                }
        
        file_processing_time = (time.time() - file_processing_start) * 1000
        print(f"📊 전체 파일 처리 시간: {file_processing_time:.2f}ms")
        
        # 4. 전체 크기 검증
        if total_size > MAX_TOTAL_SIZE:
            print(f"전체 파일 크기 초과: {total_size}바이트 (최대 {MAX_TOTAL_SIZE}바이트)")
            return {
                "success": False,
                "message": "전체 파일 크기 초과",
                "data": {
                    "body": f"전체 파일 크기 초과: {total_size}바이트 (최대 {MAX_TOTAL_SIZE}바이트)"
                }
            }

        # 5. 추론기 설정 갱신하기 (기존 경로에 저장)
        update_start = time.time()
        artis_model = _get_artis_model()
        if artis_model is None:
            raise RuntimeError("AI 추론 모델을 사용할 수 없습니다")
        is_update, artis_msg = artis_model.update_config(processed_files)
        if not is_update:
            print(f"추론기 설정 갱신 실패: {artis_msg}")
            return {
                "success": False,
                "message": "추론기 설정 갱신 실패",
                "data": {
                    "body": f"추론기 설정 갱신 실패: {artis_msg}"
                }
            }
        update_time = (time.time() - update_start) * 1000
        print(f"📊 추론기 설정 갱신 시간: {update_time:.2f}ms")

        # 6. 성공 응답 - 클라이언트가 기대하는 형식으로 수정
        # 소요 시간 계산 (밀리초 단위)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

        result = {
            "success": True,
            "message": "추론기 설정 갱신 완료",
            "data": {
                "metadata": validated_metadata,
                "elapsedTime": f"{elapsed_time:.2f}"
            }
        }

        # 응답을 먼저 반환하고 NAS 저장 및 텔레그램 메시지는 백그라운드에서 처리
        print(f"추론기 설정 갱신 완료: {len(processed_files)}개 파일, {total_size}바이트, 소요시간: {elapsed_time:.2f}ms")

        
        # NAS 저장 및 텔레그램 메시지 전송 (백그라운드)
        async def save_nas_and_send_telegram_background():
            # 5-1. /mynas/ 하위에도 추가 저장
            nas_save_start = time.time()
            try:
                # 메타데이터에서 savePath 가져오기
                save_path = validated_metadata.get("savePath", "")
                if save_path:
                    # 경로 정규화 및 /mynas/ 하위 경로로 변환
                    safe_path = os.path.normpath(save_path)
                    base_dir = os.path.join("/mynas/", safe_path)
                    
                    for file_info in processed_files:
                        file_name = file_info["original_name"]
                        raw_data = file_info["data"]
                        
                        # 파일명에 경로가 포함된 경우를 대비해 경로 정규화 (Windows 스타일 \ -> Linux 스타일 /)
                        normalized_file_name = os.path.normpath(file_name).replace('\\', '/')
                        
                        # /mynas/ 하위 저장 경로
                        nas_file_path = os.path.join(base_dir, normalized_file_name)
                        
                        # 디렉토리가 없으면 생성
                        os.makedirs(os.path.dirname(nas_file_path), exist_ok=True)
                        
                        # 파일 저장
                        with open(nas_file_path, 'wb') as f:
                            f.write(raw_data)
                        
                        print(f"NAS 저장 완료: {nas_file_path}")
                    
                    nas_save_time = (time.time() - nas_save_start) * 1000
                    print(f"📊 NAS 저장 시간: {nas_save_time:.2f}ms")
                else:
                    print("savePath가 없어 NAS 저장을 건너뜁니다")
            except Exception as nas_error:
                # NAS 저장 실패는 경고만 출력하고 계속 진행
                print(f"⚠️ NAS 저장 중 오류 발생 (기존 저장은 정상 완료): {nas_error}")
            
            # 텔레그램 메시지 생성 및 전송
            try:
                # 처리된 파일들의 original_name 정보 수집 (한 줄씩 표시)
                file_names = [file_info["original_name"] for file_info in processed_files]
                if file_names:
                    file_names_str = "\n• " + "\n• ".join(file_names)
                else:
                    file_names_str = "없음"
                
                message = f"""✅ iScan 추론기 설정 완료
📁 설정 파일 {file_names_str}
⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
"""
                print(message)
                # vendorName과 일치하는 봇에게만 메시지 전송
                vendor_name = validated_metadata.get("vendorName", "")
                chat_id = validated_metadata.get("chatID", "")
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, message, "[성공] ", chat_id)

            except Exception as telegram_error:
                print(f"텔레그램 메시지 전송 실패: {telegram_error}")

        # 백그라운드에서 NAS 저장 및 텔레그램 메시지 전송
        asyncio.create_task(save_nas_and_send_telegram_background())

        return JSONResponse(content=result, status_code=200)

    except HTTPException:
        # 에러 발생 시에도 텔레그램 메시지 전송 (백그라운드)
        async def send_error_telegram_background():
            try:
                # 소요 시간 계산 (밀리초 단위)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

                # metadata_dict에서 vendorName과 chatID 추출 시도
                vendor_name = ""
                chat_id = ""
                try:
                    metadata_dict = json.loads(metadata)
                    vendor_name = metadata_dict.get("vendorName", "")
                    chat_id = metadata_dict.get("chatID", "")
                except:
                    pass

                error_message = f"""❌ iScan 추론기 설정 갱신 실패

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
❌ 오류: HTTPException 발생

📊 처리 정보:
• 메타데이터: {metadata[:100]}... (첫 100자)
• 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

                # vendorName과 일치하는 봇에게만 메시지 전송
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, error_message, "[HTTPException] ", chat_id)

            except Exception as telegram_error:
                print(f"에러 텔레그램 메시지 전송 실패: {telegram_error}")

        # 백그라운드에서 에러 텔레그램 메시지 전송
        asyncio.create_task(send_error_telegram_background())

        raise
    except Exception as e:
        # 예상치 못한 오류 발생 시에도 텔레그램 메시지 전송 (백그라운드)
        async def send_unexpected_error_telegram_background(except_err):
            try:
                # 소요 시간 계산 (밀리초 단위)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

                # metadata_dict에서 vendorName과 chatID 추출 시도
                vendor_name = ""
                chat_id = ""
                try:
                    metadata_dict = json.loads(metadata)
                    vendor_name = metadata_dict.get("vendorName", "")
                    chat_id = metadata_dict.get("chatID", "")
                except:
                    pass

                error_message = f"""💥 iScan 추론기 설정 갱신 중 예상치 못한 오류

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
❌ 오류 타입: {type(except_err).__name__}
❌ 오류 내용: {str(except_err)}

📊 처리 정보:
• 메타데이터: {metadata[:100]}... (첫 100자)
• 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

                # vendorName과 일치하는 봇에게만 메시지 전송
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, error_message, "[Exception] ", chat_id)

            except Exception as telegram_error:
                print(f"에러 텔레그램 메시지 전송 실패: {telegram_error}")
        
        # 백그라운드에서 예상치 못한 오류 텔레그램 메시지 전송
        asyncio.create_task(send_unexpected_error_telegram_background(e))
        
        print(f"추론기 설정 갱신 중 예상치 못한 오류: {e}")
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 상세: {str(e)}")
        # 일관된 응답 형태로 에러 반환
        return {
            "success": False,
            "message": "요청 처리 실패",
            "data": {
                "body": f"요청 처리 실패: {str(e)}"
            }
        }


@router.post("/iscan-iteminfo-update")
async def update_iteminfo(
    request: Request,
    metadata: str = Form(...)
):
    """
    iScan 아이템 정보 갱신 엔드포인트

    Args:
        request: FastAPI Request 객체
        metadata: JSON 형태의 메타데이터

    Returns:
        처리 결과 정보
    """
    start_time = time.time()
    telegram_manager = TelegramBotManager()

    try:
        print("iScan 아이템 정보 갱신 요청 시작")

        print("파라미터 검증")
        if not metadata:
            print("메타데이터가 누락되었습니다")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "body": "메타데이터가 누락되었습니다"
                }
            }
        
        # 메타데이터 즉시 파싱하여 주요 정보 로그 출력
        try:
            metadata_dict = json.loads(metadata)
            print(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"메타데이터 JSON 파싱 실패: {e}")
            print(f"메타데이터 원본: {metadata}")
        except Exception as e:
            print(f"메타데이터 로그 출력 실패: {e}")
            print(f"메타데이터 원본: {metadata}")

        # 동적으로 파일들을 수집
        files = []
        form_data_start = time.time()
        form_data = await request.form()
        form_data_time = (time.time() - form_data_start) * 1000
        print(f"📊 Form 데이터 파싱 시간: {form_data_time:.2f}ms")
        
        # file0, file1, file2, ... 형태의 파일들을 찾아서 수집
        i = 0
        while True:
            file_key = f"file{i}"
            if file_key in form_data:
                file = form_data[file_key]
                if hasattr(file, 'filename') and file.filename:
                    files.append(file)
                    print(f"파일 {i}: {file.filename}, 크기: {file.size}, 타입: {file.content_type}")
                i += 1
            else:
                break

        if len(files) == 0:
            print("파일이 누락되었습니다")
            return {
                "success": False,
                "message": "파일 누락",
                "data": {
                    "body": "파일이 누락되었습니다"
                }
            }

        print(f"{metadata}")
        print(f"파일 개수: {len(files)}")

        # 1. 메타데이터 파싱 및 검증
        metadata_start = time.time()
        try:
            metadata_dict = json.loads(metadata)
            validated_metadata = validate_metadata(metadata_dict)
            print(f"메타데이터 검증 완료: {validated_metadata['fileCount']}개 파일")
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return {
                "success": False,
                "message": "JSON 파싱 실패",
                "data": {
                    "body": f"잘못된 JSON 형식: {str(e)}"
                }
            }
        except ValueError as e:
            print(f"메타데이터 검증 오류: {e}")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "body": f"메타데이터 검증 실패: {str(e)}"
                }
            }
        metadata_time = (time.time() - metadata_start) * 1000
        print(f"📊 메타데이터 파싱/검증 시간: {metadata_time:.2f}ms")
        
        # 2. 파일 개수 검증
        expected_file_count = int(validated_metadata["fileCount"])
        actual_file_count = len(files)

        if actual_file_count != expected_file_count:
            print(f"파일 개수 불일치: 예상 {expected_file_count}개, 실제 {actual_file_count}개")
            return {
                "success": False,
                "message": "파일 개수 불일치",
                "data": {
                    "body": f"파일 개수 불일치: 예상 {expected_file_count}개, 실제 {actual_file_count}개"
                }
            }

        # 3. 파일 처리 및 무결성 검증 (아이템 정보 파일만 허용)
        file_processing_start = time.time()
        total_size = 0
        processed_files = []

        for i, file in enumerate(files):
            try:
                file_start = time.time()
                # fileList에서 해당 파일 정보 가져오기
                file_info = validated_metadata["fileList"][i]
                expected_name = file_info["name"]
                expected_size = int(file_info["size"])

                # 아이템 정보 파일인지 확인
                if "item" not in expected_name.lower():
                    print(f"파일 {i+1}은 아이템 정보 파일이 아닙니다: {expected_name}")
                    return {
                        "success": False,
                        "message": "잘못된 파일 타입",
                        "data": {
                            "body": f"아이템 정보 파일만 허용됩니다: {expected_name}"
                        }
                    }

                # 파일 무결성 검증
                validation_start = time.time()
                file_data = validate_file_integrity(file, expected_size, expected_name)
                validation_time = (time.time() - validation_start) * 1000
                print(f"📊 파일 {i+1} 검증 시간: {validation_time:.2f}ms")
                
                total_size += file_data["size"]

                # 파일 내용 리스트에 저장       
                processed_files.append({
                    "original_name": expected_name,
                    "size": file_data["size"],
                    "data": file_data["content"]
                })
                
                file_total_time = (time.time() - file_start) * 1000
                print(f"파일 {i+1}/{expected_file_count} 처리 완료: {expected_name} (총 {file_total_time:.2f}ms)")
                
            except Exception as e:
                print(f"파일 {i+1} 처리 오류: {e}")
                return {
                    "success": False,
                    "message": "파일 처리 오류",
                    "data": {
                        "body": f"파일 {i+1} 처리 실패: {str(e)}"
                    }
                }
        
        file_processing_time = (time.time() - file_processing_start) * 1000
        print(f"📊 전체 파일 처리 시간: {file_processing_time:.2f}ms")
        
        # 4. 전체 크기 검증
        if total_size > MAX_TOTAL_SIZE:
            print(f"전체 파일 크기 초과: {total_size}바이트 (최대 {MAX_TOTAL_SIZE}바이트)")
            return {
                "success": False,
                "message": "전체 파일 크기 초과",
                "data": {
                    "body": f"전체 파일 크기 초과: {total_size}바이트 (최대 {MAX_TOTAL_SIZE}바이트)"
                }
            }

        # 5. savePath 검증 (필수이므로 메인 스레드에서 검증)
        save_path = validated_metadata.get("savePath", "")
        if not save_path:
            print("savePath가 메타데이터에 없습니다")
            return {
                "success": False,
                "message": "savePath 누락",
                "data": {
                    "body": "메타데이터에 savePath가 없습니다"
                }
            }

        # 6. 성공 응답
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

        result = {
            "success": True,
            "message": "아이템 정보 파일 저장 완료",
            "data": {
                "metadata": validated_metadata,
                "elapsedTime": f"{elapsed_time:.2f}"
            }
        }

        # 응답을 먼저 반환하고 파일 저장 및 텔레그램 메시지는 백그라운드에서 처리
        print(f"아이템 정보 파일 저장 요청 완료: {len(processed_files)}개 파일, {total_size}바이트, 소요시간: {elapsed_time:.2f}ms")

        
        # 파일 저장 및 텔레그램 메시지 전송 (백그라운드)
        async def save_file_and_send_telegram_background():
            # 5-1. 아이템 정보 파일 저장
            update_start = time.time()
            try:
                # 경로 정규화 및 /mynas/ 하위 경로로 변환
                safe_path = os.path.normpath(save_path)
                base_dir = os.path.join("/mynas/", safe_path)
                
                for file_info in processed_files:
                    file_name = file_info["original_name"]
                    raw_data = file_info["data"]
                    
                    # 파일명에 경로가 포함된 경우를 대비해 경로 정규화 (Windows 스타일 \ -> Linux 스타일 /)
                    normalized_file_name = os.path.normpath(file_name).replace('\\', '/')
                    
                    # 아이템 정보 파일 저장 경로 (/mynas/ 하위)
                    out_file_name = os.path.join(base_dir, normalized_file_name)
                    
                    # 디렉토리가 없으면 생성
                    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
                    
                    # 파일 저장
                    with open(out_file_name, 'wb') as f:
                        f.write(raw_data)
                    
                    print(f"아이템 정보 파일 저장 완료: {out_file_name}")
                
                update_time = (time.time() - update_start) * 1000
                print(f"📊 아이템 정보 파일 저장 시간: {update_time:.2f}ms")
                
            except Exception as e:
                print(f"⚠️ 아이템 정보 파일 저장 중 오류 발생: {e}")
            
            # 텔레그램 메시지 생성 및 전송
            try:
                # 처리된 파일들의 original_name 정보 수집 (한 줄씩 표시)
                file_names = [file_info["original_name"] for file_info in processed_files]
                if file_names:
                    file_names_str = "\n• " + "\n• ".join(file_names)
                else:
                    file_names_str = "없음"
                
                message = f"""✅ iScan 아이템 정보 파일 저장 완료
📁 아이템 정보 파일 {file_names_str}
⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
"""
                print(message)
                # vendorName과 일치하는 봇에게만 메시지 전송
                vendor_name = validated_metadata.get("vendorName", "")
                chat_id = validated_metadata.get("chatID", "")
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, message, "[성공] ", chat_id)

            except Exception as telegram_error:
                print(f"텔레그램 메시지 전송 실패: {telegram_error}")

        # 백그라운드에서 파일 저장 및 텔레그램 메시지 전송
        asyncio.create_task(save_file_and_send_telegram_background())

        return JSONResponse(content=result, status_code=200)

    except HTTPException:
        # 에러 발생 시에도 텔레그램 메시지 전송 (백그라운드)
        async def send_error_telegram_background():
            try:
                # 소요 시간 계산 (밀리초 단위)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

                # metadata_dict에서 vendorName과 chatID 추출 시도
                vendor_name = ""
                chat_id = ""
                try:
                    metadata_dict = json.loads(metadata)
                    vendor_name = metadata_dict.get("vendorName", "")
                    chat_id = metadata_dict.get("chatID", "")
                except:
                    pass

                error_message = f"""❌ iScan 아이템 정보 파일 저장 실패

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
❌ 오류: HTTPException 발생

📊 처리 정보:
• 메타데이터: {metadata[:100]}... (첫 100자)
• 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

                # vendorName과 일치하는 봇에게만 메시지 전송
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, error_message, "[HTTPException] ", chat_id)

            except Exception as telegram_error:
                print(f"에러 텔레그램 메시지 전송 실패: {telegram_error}")

        # 백그라운드에서 에러 텔레그램 메시지 전송
        asyncio.create_task(send_error_telegram_background())

        raise
    except Exception as e:
        # 예상치 못한 오류 발생 시에도 텔레그램 메시지 전송 (백그라운드)
        async def send_unexpected_error_telegram_background(except_err):
            try:
                # 소요 시간 계산 (밀리초 단위)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

                # metadata_dict에서 vendorName과 chatID 추출 시도
                vendor_name = ""
                chat_id = ""
                try:
                    metadata_dict = json.loads(metadata)
                    vendor_name = metadata_dict.get("vendorName", "")
                    chat_id = metadata_dict.get("chatID", "")
                except:
                    pass

                error_message = f"""💥 iScan 아이템 정보 파일 저장 중 예상치 못한 오류

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
❌ 오류 타입: {type(except_err).__name__}
❌ 오류 내용: {str(except_err)}

📊 처리 정보:
• 메타데이터: {metadata[:100]}... (첫 100자)
• 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

                # vendorName과 일치하는 봇에게만 메시지 전송
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, error_message, "[Exception] ", chat_id)

            except Exception as telegram_error:
                print(f"에러 텔레그램 메시지 전송 실패: {telegram_error}")
        
        # 백그라운드에서 예상치 못한 오류 텔레그램 메시지 전송
        asyncio.create_task(send_unexpected_error_telegram_background(e))
        
        print(f"아이템 정보 파일 저장 중 예상치 못한 오류: {e}")
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 상세: {str(e)}")
        # 일관된 응답 형태로 에러 반환
        return {
            "success": False,
            "message": "요청 처리 실패",
            "data": {
                "body": f"요청 처리 실패: {str(e)}"
            }
        }


@router.post("/iscan-start-ai-training")
async def start_ai_training(
    request: Request,
    metadata: str = Form(...)
):
    """
    AI 학습 시작 엔드포인트

    Args:
        request: FastAPI Request 객체
        metadata: JSON 형태의 메타데이터

    Returns:
        처리 결과 정보
    """
    # [임시] AI 추론 비활성화 체크
    if AI_INFERENCE_DISABLED:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "AI 추론 기능이 임시로 비활성화되어 있습니다 (SDUI 테스트 모드)",
                "data": {
                    "body": "이 기능은 현재 테스트 모드에서 사용할 수 없습니다."
                }
            }
        )
    
    start_time = time.time()
    telegram_manager = TelegramBotManager()

    try:
        print("AI 학습 시작 요청 시작")

        # 파라미터 검증
        print("파라미터 검증")
        if not metadata:
            print("메타데이터가 누락되었습니다")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "body": "메타데이터가 누락되었습니다"
                }
            }

        # 메타데이터 즉시 파싱하여 주요 정보 로그 출력
        try:
            metadata_dict = json.loads(metadata)
            print(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"메타데이터 JSON 파싱 실패: {e}")
            print(f"메타데이터 원본: {metadata}")
        except Exception as e:
            print(f"메타데이터 로그 출력 실패: {e}")
            print(f"메타데이터 원본: {metadata}")

        # 동적으로 파일들을 수집
        files = []
        form_data_start = time.time()
        form_data = await request.form()
        form_data_time = (time.time() - form_data_start) * 1000
        print(f"📊 Form 데이터 파싱 시간: {form_data_time:.2f}ms")
        
        # file0, file1, file2, ... 형태의 파일들을 찾아서 수집
        i = 0
        while True:
            file_key = f"file{i}"
            if file_key in form_data:
                file = form_data[file_key]
                if hasattr(file, 'filename') and file.filename:
                    files.append(file)
                    print(f"파일 {i}: {file.filename}, 크기: {file.size}, 타입: {file.content_type}")
                i += 1
            else:
                break

        if len(files) == 0:
            print("파일이 누락되었습니다")
            return {
                "success": False,
                "message": "파일 누락",
                "data": {
                    "body": "파일이 누락되었습니다"
                }
            }

        print(f"{metadata}")
        #print(f"메타데이터: {metadata}")
        print(f"파일 개수: {len(files)}")

        # 1. 메타데이터 파싱 및 검증
        metadata_start = time.time()
        try:
            metadata_dict = json.loads(metadata)
            validated_metadata = validate_metadata(metadata_dict)
            print(f"메타데이터 검증 완료: {validated_metadata['fileCount']}개 파일")
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return {
                "success": False,
                "message": "JSON 파싱 실패",
                "data": {
                    "body": f"잘못된 JSON 형식: {str(e)}"
                }
            }
        except ValueError as e:
            print(f"메타데이터 검증 오류: {e}")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "body": f"메타데이터 검증 실패: {str(e)}"
                }
            }
        metadata_time = (time.time() - metadata_start) * 1000
        print(f"📊 메타데이터 파싱/검증 시간: {metadata_time:.2f}ms")
        
        # 2. 파일 개수 검증
        expected_file_count = int(validated_metadata["fileCount"])
        actual_file_count = len(files)

        if actual_file_count != expected_file_count:
            print(f"파일 개수 불일치: 예상 {expected_file_count}개, 실제 {actual_file_count}개")
            return {
                "success": False,
                "message": "파일 개수 불일치",
                "data": {
                    "body": f"파일 개수 불일치: 예상 {expected_file_count}개, 실제 {actual_file_count}개"
                }
            }

        # 3. 파일 처리 및 무결성 검증
        file_processing_start = time.time()
        total_size = 0
        processed_files = []

        # 설정 파일 가져오기
        for i, file in enumerate(files):
            try:
                file_start = time.time()
                # fileList에서 해당 파일 정보 가져오기
                file_info = validated_metadata["fileList"][i]
                expected_name = file_info["name"]
                expected_size = int(file_info["size"])

                # 파일 무결성 검증
                validation_start = time.time()
                file_data = validate_file_integrity(file, expected_size, expected_name)
                validation_time = (time.time() - validation_start) * 1000
                print(f"📊 파일 {i+1} 검증 시간: {validation_time:.2f}ms")
                
                total_size += file_data["size"]

                # 파일 내용 리스트에 저장       
                processed_files.append({
                    "original_name": expected_name,
                    "size": file_data["size"],
                    "data": file_data["content"]
                })
                
                file_total_time = (time.time() - file_start) * 1000
                print(f"파일 {i+1}/{expected_file_count} 처리 완료: {expected_name} (총 {file_total_time:.2f}ms)")
                
            except Exception as e:
                print(f"파일 {i+1} 처리 오류: {e}")
                return {
                    "success": False,
                    "message": "파일 처리 오류",
                    "data": {
                        "body": f"파일 {i+1} 처리 실패: {str(e)}"
                    }
                }
        
        file_processing_time = (time.time() - file_processing_start) * 1000
        print(f"📊 전체 파일 처리 시간: {file_processing_time:.2f}ms")
        
        # 4. 전체 크기 검증
        if total_size > MAX_TOTAL_SIZE:
            print(f"전체 파일 크기 초과: {total_size}바이트 (최대 {MAX_TOTAL_SIZE}바이트)")
            return {
                "success": False,
                "message": "전체 파일 크기 초과",
                "data": {
                    "body": f"전체 파일 크기 초과: {total_size}바이트 (최대 {MAX_TOTAL_SIZE}바이트)"
                }
            }

        # 5. AI 학습 시작하기
        ai_training_start = time.time()
        db_root_path = os.path.join("/mynas/uploads/", validated_metadata.get("vendorName"), "db_key",
                                    validated_metadata.get("dbKey"))
        global artis_thread
        is_ai_training_start, artis_msg, artis_thread = artis_model.start_ai_training(processed_files, db_root_path)
        if not is_ai_training_start:
            print(f"AI 학습 시작 실패: {artis_msg}")
            return {
                "success": False,
                "message": "AI 학습 시작 실패",
                "data": {
                    "body": f"AI 학습 시작 실패: {artis_msg}"
                }
            }

        ai_training_time = (time.time() - ai_training_start) * 1000
        print(f"📊 AI 학습 시작 시간: {ai_training_time:.2f}ms")

        # 6. 성공 응답 - 클라이언트가 기대하는 형식으로 수정
        # 소요 시간 계산 (밀리초 단위)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

        result = {
            "success": True,
            "message": "AI 학습 시작 완료",
            "data": {
                "metadata": validated_metadata,
                "elapsedTime": f"{elapsed_time:.2f}"
            }
        }

        # 응답을 먼저 반환하고 텔레그램 메시지는 백그라운드에서 전송
        print(f"AI 학습 시작 완료: {len(processed_files)}개 파일, {total_size}바이트, 소요시간: {elapsed_time:.2f}ms")
        
        # 텔레그램 메시지 생성 및 전송
        async def send_telegram_background():
            try:
                # 처리된 파일들의 original_name 정보 수집 (한 줄씩 표시)
                file_names = [file_info["original_name"] for file_info in processed_files]
                if file_names:
                    file_names_str = "\n• " + "\n• ".join(file_names)
                else:
                    file_names_str = "없음"
                
                message = f"""✅ iScan AI 학습 시작 완료
📁 설정 파일 {file_names_str}
⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
"""
                print(message)
                # vendorName과 일치하는 봇에게만 메시지 전송
                vendor_name = validated_metadata.get("vendorName", "")
                chat_id = validated_metadata.get("chatID", "")
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, message, "[성공] ", chat_id)

            except Exception as telegram_error:
                print(f"텔레그램 메시지 전송 실패: {telegram_error}")

        # 백그라운드에서 텔레그램 메시지 전송
        asyncio.create_task(send_telegram_background())

        async def check_train_background():
            '''import httpx
            async with httpx.AsyncClient(timeout=20.0) as client:
                #import requests

                start_time = time.time()
                time_interval = time.time()
                publicIP = validated_metadata.get("publicIP", "")

                url =  f"http://{publicIP}:18081/api/sync-status"
                while artis_thread.is_alive():
                    await asyncio.sleep(10)

                    sync_status_message = artis_model.status_ai_training(time.time() - time_interval)
                    time_interval = time.time()
                    data = {
                        "data": {
                            "body": sync_status_message
                        }
                    }
                    print(json.dumps(data, indent=4))
                    #response = requests.post(url, json=data, timeout=20)
                    try:
                        response = await client.post(url, json=data)
                        response.raise_for_status()
                        print("응답:", response.text)
                    except httpx.RequestError as e:
                        print(f"요청 실패: {e}")
                    except httpx.HTTPStatusError as e:
                        print(f"서버 오류: {e.response.status_code} {e.response.text}")

                is_update, artis_msg = artis_model.update_config([])
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환
                sync_status_message = artis_model.status_ai_training(11111111)
                sync_status_message["ai_model_stage"] = "TRAIN_END"
                data = {
                    "data": {
                        "body": sync_status_message
                    }
                }
                print(json.dumps(data, indent=4))
                #response = requests.post(url, json=data, timeout=20)
                try:
                    response = await client.post(url, json=data)
                    response.raise_for_status()
                    print("응답:", response.text)
                except httpx.RequestError as e:
                    print(f"요청 실패: {e}")
                except httpx.HTTPStatusError as e:
                    print(f"서버 오류: {e.response.status_code} {e.response.text}")'''

            while artis_thread.is_alive():
                await asyncio.sleep(10)
                print(f"{artis_model.status_ai_training()}")
            vendor_name = validated_metadata.get("vendorName", "")
            chat_id = validated_metadata.get("chatID", "")

            is_update, artis_msg = artis_model.update_config([])
            if not is_update:
                print(f"추론기 설정 갱신 실패: {artis_msg}")
                error_message = f"""💥 iScan 추론기 설정 갱신 중 예상치 못한 오류

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
❌ 오류 내용: {artis_msg}"""
                # vendorName과 일치하는 봇에게만 메시지 전송
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, error_message, "[Exception] ", chat_id)
            else:
                try:
                    message = f"""✅ iScan AI 학습 및 업데이트 완료

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms"""
                    print(message)

                    await send_telegram_message_by_vendor(telegram_manager, vendor_name, message, "[성공] ", chat_id)

                except Exception as telegram_error:
                    print(f"텔레그램 메시지 전송 실패: {telegram_error}")

        asyncio.create_task(check_train_background())

        return JSONResponse(content=result, status_code=200)

    except HTTPException:
        # 에러 발생 시에도 텔레그램 메시지 전송 (백그라운드)
        async def send_error_telegram_background():
            try:
                # 소요 시간 계산 (밀리초 단위)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

                # metadata_dict에서 vendorName과 chatID 추출 시도
                vendor_name = ""
                chat_id = ""
                try:
                    metadata_dict = json.loads(metadata)
                    vendor_name = metadata_dict.get("vendorName", "")
                    chat_id = metadata_dict.get("chatID", "")
                except:
                    pass

                error_message = f"""❌ iScan AI 학습 시작 실패

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
❌ 오류: HTTPException 발생

📊 처리 정보:
• 메타데이터: {metadata[:100]}... (첫 100자)
• 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

                # vendorName과 일치하는 봇에게만 메시지 전송
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, error_message, "[HTTPException] ", chat_id)

            except Exception as telegram_error:
                print(f"에러 텔레그램 메시지 전송 실패: {telegram_error}")

        # 백그라운드에서 에러 텔레그램 메시지 전송
        asyncio.create_task(send_error_telegram_background())

        raise
    except Exception as e:
        # 예상치 못한 오류 발생 시에도 텔레그램 메시지 전송 (백그라운드)
        async def send_unexpected_error_telegram_background(except_err):
            try:
                # 소요 시간 계산 (밀리초 단위)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

                # metadata_dict에서 vendorName과 chatID 추출 시도
                vendor_name = ""
                chat_id = ""
                try:
                    metadata_dict = json.loads(metadata)
                    vendor_name = metadata_dict.get("vendorName", "")
                    chat_id = metadata_dict.get("chatID", "")
                except:
                    pass

                error_message = f"""💥 iScan AI 학습 시작 중 예상치 못한 오류

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
❌ 오류 타입: {type(except_err).__name__}
❌ 오류 내용: {str(except_err)}

📊 처리 정보:
• 메타데이터: {metadata[:100]}... (첫 100자)
• 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

                # vendorName과 일치하는 봇에게만 메시지 전송
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, error_message, "[Exception] ", chat_id)

            except Exception as telegram_error:
                print(f"에러 텔레그램 메시지 전송 실패: {telegram_error}")
        
        # 백그라운드에서 예상치 못한 오류 텔레그램 메시지 전송
        asyncio.create_task(send_unexpected_error_telegram_background(e))
        
        print(f"AI 학습 시작 중 예상치 못한 오류: {e}")
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 상세: {str(e)}")
        # 일관된 응답 형태로 에러 반환
        return {
            "success": False,
            "message": "요청 처리 실패",
            "data": {
                "body": f"요청 처리 실패: {str(e)}"
            }
        }


@router.post("/iscan-sync-status")
async def sync_status(
    request: Request,
    metadata: str = Form(...)
):
    """
    AI 학습 상태 조회 엔드포인트

    Args:
        request: FastAPI Request 객체
        metadata: JSON 형태의 메타데이터

    Returns:
        처리 결과 정보
    """
    start_time = time.time()
    telegram_manager = TelegramBotManager()

    try:
        print("AI 학습 상태 조회 요청 시작")

        # 파라미터 검증
        print("파라미터 검증")
        if not metadata:
            print("메타데이터가 누락되었습니다")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "body": "메타데이터가 누락되었습니다"
                }
            }

        # 메타데이터 즉시 파싱하여 주요 정보 로그 출력
        print("메타데이터 즉시 파싱하여 주요 정보 로그 출력")
        try:
            metadata_dict = json.loads(metadata)
            print(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"메타데이터 JSON 파싱 실패: {e}")
            print(f"메타데이터 원본: {metadata}")
        except Exception as e:
            print(f"메타데이터 로그 출력 실패: {e}")
            print(f"메타데이터 원본: {metadata}")

        # 1. 메타데이터 파싱 및 검증
        print("1. 메타데이터 파싱 및 검증")
        metadata_start = time.time()
        try:
            metadata_dict = json.loads(metadata)
            validated_metadata = validate_metadata(metadata_dict)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return {
                "success": False,
                "message": "JSON 파싱 실패",
                "data": {
                    "body": f"잘못된 JSON 형식: {str(e)}"
                }
            }
        except ValueError as e:
            print(f"메타데이터 검증 오류: {e}")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "body": f"메타데이터 검증 실패: {str(e)}"
                }
            }
        metadata_time = (time.time() - metadata_start) * 1000
        print(f"📊 메타데이터 파싱/검증 시간: {metadata_time:.2f}ms")

        # 2. AI 상태 조회하기
        print("2. AI 상태 조회하기")
        ai_status_start = time.time()
        global artis_thread

        if artis_thread is None:
            print(f"AI 학습 상태 조회 실패 : 학습이 진행중이 아님")
            return {
                "success": False,
                "message": "AI 학습 상태 조회 실패",
                "data": {
                    "body": "AI 학습 상태 조회 실패 : 학습이 진행중이 아님"
                }
            }

        sync_status_message = artis_model.status_ai_training()

        ai_training_time = (time.time() - ai_status_start) * 1000
        print(f"📊 AI 학습 상태 조회 시간: {ai_training_time:.2f}ms")

        # 3. 성공 응답 - 클라이언트가 기대하는 형식으로 수정
        # 소요 시간 계산 (밀리초 단위)
        print("3. 성공 응답 - 클라이언트가 기대하는 형식으로 수정")
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

        result = {
            "success": True,
            "message": "AI 학습 상태 조회 완료",
            "data": {
                "body": sync_status_message
            }
        }

        # 응답을 먼저 반환하고 텔레그램 메시지는 백그라운드에서 전송
        print(f"AI 학습 상태 조회 완료: 소요시간: {elapsed_time:.2f}ms")

        # 텔레그램 메시지 생성 및 전송
        async def send_telegram_background():
            try:
                message = f"""✅ iScan AI 상태 조회 완료
⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
"""
                print(message)
                # vendorName과 일치하는 봇에게만 메시지 전송
                vendor_name = validated_metadata.get("vendorName", "")
                chat_id = validated_metadata.get("chatID", "")
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, message, "[성공] ", chat_id)

            except Exception as telegram_error:
                print(f"텔레그램 메시지 전송 실패: {telegram_error}")

        # 백그라운드에서 텔레그램 메시지 전송
        asyncio.create_task(send_telegram_background())

        return JSONResponse(content=result, status_code=200)

    except HTTPException:
        # 에러 발생 시에도 텔레그램 메시지 전송 (백그라운드)
        async def send_error_telegram_background():
            try:
                # 소요 시간 계산 (밀리초 단위)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

                # metadata_dict에서 vendorName과 chatID 추출 시도
                vendor_name = ""
                chat_id = ""
                try:
                    metadata_dict = json.loads(metadata)
                    vendor_name = metadata_dict.get("vendorName", "")
                    chat_id = metadata_dict.get("chatID", "")
                except:
                    pass

                error_message = f"""❌ iScan AI 학습 상태 조회 실패

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
❌ 오류: HTTPException 발생

📊 처리 정보:
• 메타데이터: {metadata[:100]}... (첫 100자)
• 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

                # vendorName과 일치하는 봇에게만 메시지 전송
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, error_message, "[HTTPException] ", chat_id)

            except Exception as telegram_error:
                print(f"에러 텔레그램 메시지 전송 실패: {telegram_error}")

        # 백그라운드에서 에러 텔레그램 메시지 전송
        asyncio.create_task(send_error_telegram_background())

        raise
    except Exception as e:
        # 예상치 못한 오류 발생 시에도 텔레그램 메시지 전송 (백그라운드)
        async def send_unexpected_error_telegram_background(except_err):
            try:
                # 소요 시간 계산 (밀리초 단위)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms 단위로 변환

                # metadata_dict에서 vendorName과 chatID 추출 시도
                vendor_name = ""
                chat_id = ""
                try:
                    metadata_dict = json.loads(metadata)
                    vendor_name = metadata_dict.get("vendorName", "")
                    chat_id = metadata_dict.get("chatID", "")
                except:
                    pass

                error_message = f"""💥 iScan AI 학습 상태 조회 중 예상치 못한 오류

⏱️ 총 처리 시간 : {elapsed_time:.2f} ms
❌ 오류 타입: {type(except_err).__name__}
❌ 오류 내용: {str(except_err)}

📊 처리 정보:
• 메타데이터: {metadata[:100]}... (첫 100자)
• 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

                # vendorName과 일치하는 봇에게만 메시지 전송
                await send_telegram_message_by_vendor(telegram_manager, vendor_name, error_message, "[Exception] ", chat_id)

            except Exception as telegram_error:
                print(f"에러 텔레그램 메시지 전송 실패: {telegram_error}")

        # 백그라운드에서 예상치 못한 오류 텔레그램 메시지 전송
        asyncio.create_task(send_unexpected_error_telegram_background(e))

        print(f"AI 학습 상태 조회 중 예상치 못한 오류: {e}")
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 상세: {str(e)}")
        # 일관된 응답 형태로 에러 반환
        return {
            "success": False,
            "message": "요청 처리 실패",
            "data": {
                "body": f"요청 처리 실패: {str(e)}"
            }
        }


@router.post("/remove-db-path")
async def remove_db_path(
    request: Request,
    metadata: str = Form(...)
):
    """
    DB 경로 삭제 엔드포인트

    Args:
        request: FastAPI Request 객체
        metadata: JSON 형태의 메타데이터

    Returns:
        처리 결과 정보
    """
    start_time = time.time()
    telegram_manager = TelegramBotManager()

    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("=" * 100)
        print(f"🚀 /api/remove-db-path 동작 시작 - 서버 시각: {current_time}")
        print("=" * 100)

        # 파라미터 검증
        print("파라미터 검증")
        if not metadata:
            print("메타데이터가 누락되었습니다")
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": "메타데이터 검증 실패",
                    "data": {
                        "body": "메타데이터가 누락되었습니다"
                    }
                }
            )

        # 메타데이터 파싱
        try:
            metadata_dict = json.loads(metadata)
            print(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"메타데이터 JSON 파싱 실패: {e}")
            raise HTTPException(
                status_code=422,
                detail={
                    "success": False,
                    "message": "JSON 파싱 실패",
                    "data": {
                        "body": f"잘못된 JSON 형식: {str(e)}"
                    }
                }
            )

        # remove_dir 구성: /mynas/uploads/vendorName/dbKey/removeDbPath
        vendor_name = metadata_dict.get("vendorName")
        db_key = metadata_dict.get("dbKey")
        remove_db_path = metadata_dict.get("removeDbPath")
        
        if not all([vendor_name, db_key, remove_db_path]):
            print("필수 필드가 누락되었습니다")
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": "필수 필드 누락",
                    "data": {
                        "body": "vendorName, dbKey, removeDbPath 중 누락된 필드가 있습니다"
                    }
                }
            )

        remove_dir = os.path.join("/mynas/uploads", vendor_name, "db_key", db_key, remove_db_path)
        print(f"삭제할 디렉토리: {remove_dir}")

        # 즉시 응답 반환 (실시간성 향상)
        result = {
            "success": True,
            "message": "DB 경로 삭제 요청 접수",
            "data": {
                "metadata": metadata_dict,
                "removeDir": remove_dir,
                "status": "processing"
            }
        }

        # 백그라운드에서 디렉토리 삭제 및 텔레그램 메시지 전송
        asyncio.create_task(process_db_deletion_background(
            remove_dir, metadata_dict, current_time, start_time, telegram_manager
        ))

        return result

    except Exception as e:
        print(f"DB 경로 삭제 처리 중 오류 발생: {e}")
        return {
            "success": False,
            "message": "처리 중 오류 발생",
            "data": {
                "body": f"오류: {str(e)}"
            }
        }


@router.post("/iscan-update-crop-point")
async def update_crop_point(
    request: Request,
    metadata: str = Form(...)
):
    """
    Crop 포인트 업데이트 전용 엔드포인트

    Args:
        request: FastAPI Request 객체
        metadata: JSON 형태의 메타데이터 (command: "UpdateCropPoint", crop 정보 포함)

    Returns:
        처리 결과 정보
    """
    start_time = time.time()

    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("=" * 100)
        print(f"🚀 /api/iscan-update-crop-point 동작 시작 - 서버 시각: {current_time}")
        print("=" * 100)
        
        print("Crop 포인트 업데이트 요청 시작")

        # 파라미터 검증
        if not metadata:
            print("메타데이터가 누락되었습니다")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "body": "메타데이터가 누락되었습니다"
                }
            }

        # 메타데이터 파싱
        try:
            metadata_dict = json.loads(metadata)
            print(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"메타데이터 JSON 파싱 실패: {e}")
            print(f"메타데이터 원본: {metadata}")
            return {
                "success": False,
                "message": "JSON 파싱 실패",
                "data": {
                    "body": f"잘못된 JSON 형식: {str(e)}"
                }
            }

        # Crop 정보 추출
        crop_info = metadata_dict.get("crop", {})
        left_x = crop_info.get("left_x", 0)
        left_y = crop_info.get("left_y", 0)
        right_x = crop_info.get("right_x", 0)
        right_y = crop_info.get("right_y", 0)

        # kisan_config.json 파일에 crop 포인트 업데이트
        try:
            from app.core.Artis_AI import common_config as cc
            
            if os.path.exists(cc.path_to_config):
                with open(cc.path_to_config, "r", encoding="utf-8") as f:
                    json_content = f.read()
                    json_content = cc.remove_json_comments(json_content)
                    config_json = json.loads(json_content)
                
                if "edge_manager_setting" not in config_json:
                    config_json["edge_manager_setting"] = {}
                if "crop" not in config_json["edge_manager_setting"]:
                    config_json["edge_manager_setting"]["crop"] = {}
                
                config_json["edge_manager_setting"]["crop"]["left_x"] = left_x
                config_json["edge_manager_setting"]["crop"]["left_y"] = left_y
                config_json["edge_manager_setting"]["crop"]["right_x"] = right_x
                config_json["edge_manager_setting"]["crop"]["right_y"] = right_y
                
                with open(cc.path_to_config, "w", encoding="utf-8") as f:
                    json.dump(config_json, f, indent=4, ensure_ascii=False)
                
                cc.artis_ai_json_config = cc.get_config(cc.path_to_config, cc.artis_ai_json_config)
                print(f"✅ kisan_config.json 업데이트 완료: LEFT({left_x}, {left_y}), RIGHT({right_x}, {right_y})")
            else:
                print(f"⚠️ kisan_config.json 파일이 없습니다: {cc.path_to_config}")
        except Exception as e:
            print(f"⚠️ kisan_config.json 업데이트 실패 (무시하고 계속): {e}")
            import traceback
            traceback.print_exc()

        # Artis AI의 depth 모듈에 crop 설정 업데이트
        artis_model = _get_artis_model()
        if artis_model and artis_model.inf_depth:
            artis_model.inf_depth.update_crop_settings(left_x, left_y, right_x, right_y)

            # 성공 응답
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000

            result = {
                "success": True,
                "message": "Crop 포인트 업데이트 완료",
                "data": {
                    "crop": {
                        "left_x": left_x,
                        "left_y": left_y,
                        "right_x": right_x,
                        "right_y": right_y
                    },
                    "elapsedTime": f"{elapsed_time:.2f}"
                }
            }

            print(f"Crop 포인트 업데이트 완료: 소요시간: {elapsed_time:.2f}ms")
            return JSONResponse(content=result, status_code=200)
        else:
            print("Artis AI 모델이 초기화되지 않았습니다")
            return {
                "success": False,
                "message": "Artis AI 모델 초기화 실패",
                "data": {
                    "body": "Artis AI 모델이 초기화되지 않았습니다"
                }
            }

    except Exception as e:
        print(f"UpdateCropPoint 처리 중 오류: {e}")
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 상세: {str(e)}")
        return {
            "success": False,
            "message": "UpdateCropPoint 처리 실패",
            "data": {
                "body": f"UpdateCropPoint 처리 중 오류: {str(e)}"
            }
        }


@router.post("/iscan-calibration")
async def process_calibration(
    request: Request,
    metadata: str = Form(...)
):
    start_time = time.time()
    telegram_manager = TelegramBotManager()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=" * 100)
    print(f"🚀 /api/iscan-calibration 동작 시작 - 서버 시각: {current_time}")
    print("=" * 100)
    
    serial_number = None
    nas_path = None
    calibration_dir = None
    
    try:
        # 파라미터 검증
        if not metadata:
            print("메타데이터가 누락되었습니다")
            return {
                "success": False,
                "message": "메타데이터 검증 실패",
                "data": {
                    "error_code": 600,  # ERROR_CODE_CALIBRATION_UNKNOWN
                    "body": "메타데이터가 누락되었습니다"
                }
            }

        # 메타데이터 파싱
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError as e:
            print(f"메타데이터 JSON 파싱 실패: {e}")
            return {
                "success": False,
                "message": "JSON 파싱 실패",
                "data": {
                    "error_code": 600,  # ERROR_CODE_CALIBRATION_UNKNOWN
                    "body": f"잘못된 JSON 형식: {str(e)}"
                }
            }

        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core', 'Artis_AI', 'camera'))
        
        # 시리얼 넘버 추출하여 경로 분리
        serial_number = extract_serial_number(metadata_dict)
        if not serial_number:
            return {
                "success": False,
                "message": "시리얼 넘버 누락",
                "data": {
                    "error_code": 600,
                    "body": "캘리브레이션을 수행하려면 시리얼 넘버가 필요합니다."
                }
            }
        
        # 시리얼 넘버별 경로 설정
        _, cc, path_to_root, _, _ = _get_inference_modules()
        if path_to_root is None:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "message": "AI 추론 기능이 임시로 비활성화되어 있습니다 (SDUI 테스트 모드)",
                    "data": {
                        "body": "이 기능은 현재 테스트 모드에서 사용할 수 없습니다."
                    }
                }
            )
        calibration_dir = os.path.join(path_to_root, "camera", "calibration", serial_number)
        cal_1_dir = os.path.join(calibration_dir, "Cal_1")
        cal_2_dir = os.path.join(calibration_dir, "Cal_2")
        cal_3_dir = os.path.join(calibration_dir, "Cal_3")
        temp_dir = os.path.join(calibration_dir, "temp")
        
        # 기존 캘리브레이션 관련 디렉토리 전체 삭제 후 재생성
        for dir_path in [cal_1_dir, cal_2_dir, cal_3_dir, temp_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"기존 캘리브레이션 이미지 삭제 완료 (경로: {calibration_dir})")

        # NAS에서 ZIP 파일 찾아서 압축 해제
        import pyzipper
        
        # 메타데이터에서 savePath 추출
        save_path = metadata_dict.get("savePath", "")
        if not save_path:
            return {
                "success": False,
                "message": "savePath 누락",
                "data": {
                    "error_code": 600,
                    "body": "메타데이터에 savePath가 없습니다"
                }
            }
        
        # NAS 경로 구성
        nas_path = os.path.join("/mynas/", os.path.normpath(save_path))
        print(f"NAS 경로에서 ZIP 파일 검색: {nas_path}")
        
        if not os.path.exists(nas_path):
            return {
                "success": False,
                "message": "NAS 경로 없음",
                "data": {
                    "error_code": 600,
                    "body": f"NAS 경로가 존재하지 않습니다: {nas_path}"
                }
            }
        
        # Cal_XX.zip 파일 찾기
        zip_pattern = os.path.join(nas_path, "Cal_*.zip")
        zip_files = sorted(glob.glob(zip_pattern))
        
        if len(zip_files) == 0:
            return {
                "success": False,
                "message": "ZIP 파일 없음",
                "data": {
                    "error_code": 600,
                    "body": f"NAS 경로에 Cal_XX.zip 파일이 없습니다: {nas_path}"
                }
            }
        
        print(f"발견된 ZIP 파일 개수: {len(zip_files)}")
        
        artis_model = _get_artis_model()
        if artis_model is None:
            return {
                "success": False,
                "message": "AI 추론 모델을 사용할 수 없습니다",
                "data": {
                    "error_code": 600,
                    "body": "AI 추론 모델을 초기화할 수 없습니다"
                }
            }

        # 각 ZIP 파일 압축 해제
        crypto_key = artis_model.crypto_key
        for zip_file in zip_files:
            zip_filename = os.path.basename(zip_file)
            # Cal_XX.zip에서 Cal_XX 추출
            cal_number = zip_filename.replace(".zip", "")
            
            try:
                with pyzipper.AESZipFile(zip_file, 'r', encryption=pyzipper.WZ_AES) as zipf:
                    zipf.setpassword(crypto_key)
                    
                    # ZIP 파일 내용 확인
                    file_list = zipf.namelist()
                    if "Cam_1_Color.jpg" in file_list:
                        cal_1_path = os.path.join(cal_1_dir, f"{cal_number}.jpg")
                        with zipf.open("Cam_1_Color.jpg") as source:
                            with open(cal_1_path, "wb") as target:
                                target.write(source.read())
                        print(f"Cam_1_Color.jpg -> {cal_1_path}")
                    
                    # Cam_2_Color.jpg 찾기 (Cal_2 폴더로)
                    if "Cam_2_Color.jpg" in file_list:
                        cal_2_path = os.path.join(cal_2_dir, f"{cal_number}.jpg")
                        with zipf.open("Cam_2_Color.jpg") as source:
                            with open(cal_2_path, "wb") as target:
                                target.write(source.read())
                        print(f"Cam_2_Color.jpg -> {cal_2_path}")
                    
                    # Cam_Single_1.jpg 찾기 (Cal_3 폴더로)
                    if "Cam_Single_1.jpg" in file_list:
                        cal_3_path = os.path.join(cal_3_dir, f"{cal_number}.jpg")
                        with zipf.open("Cam_Single_1.jpg") as source:
                            with open(cal_3_path, "wb") as target:
                                target.write(source.read())
                        print(f"Cam_Single_1.jpg -> {cal_3_path}")
                    
            except Exception as e:
                print(f"ZIP 파일 압축 해제 실패: {zip_file}, 오류: {e}")
                return {
                    "success": False,
                    "message": "ZIP 파일 압축 해제 실패",
                    "data": {
                        "error_code": 600,
                        "body": f"ZIP 파일 압축 해제 실패: {zip_file}, 오류: {str(e)}"
                    }
                }

        cal_1_images = sorted([f for f in os.listdir(cal_1_dir) if f.endswith('.jpg')])
        cal_2_images = sorted([f for f in os.listdir(cal_2_dir) if f.endswith('.jpg')])
        cal_3_images = sorted([f for f in os.listdir(cal_3_dir) if f.endswith('.jpg')])
        
        print(f"Cal_1 이미지 전송 완료: {len(cal_1_images)}개")
        print(f"Cal_2 이미지 전송 완료: {len(cal_2_images)}개")
        print(f"Cal_3 이미지 전송 완료: {len(cal_3_images)}개")
        
        if "Cal_00.jpg" not in cal_1_images and len(cal_1_images) > 0:
            first_image = os.path.join(cal_1_dir, cal_1_images[0])
            shutil.copy2(first_image, os.path.join(cal_1_dir, "Cal_00.jpg"))
            cal_1_images.append("Cal_00.jpg")
        
        if "Cal_00.jpg" not in cal_2_images and len(cal_2_images) > 0:
            first_image = os.path.join(cal_2_dir, cal_2_images[0])
            shutil.copy2(first_image, os.path.join(cal_2_dir, "Cal_00.jpg"))
            cal_2_images.append("Cal_00.jpg")
        
        if "Cal_00.jpg" not in cal_3_images and len(cal_3_images) > 0:
            first_image = os.path.join(cal_3_dir, cal_3_images[0])
            shutil.copy2(first_image, os.path.join(cal_3_dir, "Cal_00.jpg"))
            cal_3_images.append("Cal_00.jpg")

        try:
            from app.core.Artis_AI.camera.utils import calibration, compute_relative_pose, save_calibration_results, save_calibration_results_single
            
            right_images = sorted([f for f in glob.glob(os.path.join(cal_1_dir, "Cal*.jpg")) if not os.path.basename(f) == "Cal_00.jpg"])
            left_images = sorted([f for f in glob.glob(os.path.join(cal_2_dir, "Cal*.jpg")) if not os.path.basename(f) == "Cal_00.jpg"])
            
            if len(left_images) == 0 or len(right_images) == 0:
                error_code = 604  # ERROR_CODE_CALIBRATION_PYTHON_EXECUTION_FAILED
                error_msg = "캘리브레이션 이미지가 부족합니다 (왼쪽 또는 오른쪽 이미지가 없음)"
                error = ValueError(error_msg)
                error.error_code = error_code
                raise error
            
            if len(left_images) != len(right_images):
                error_code = 604  # ERROR_CODE_CALIBRATION_PYTHON_EXECUTION_FAILED
                error_msg = f"왼쪽과 오른쪽 이미지 개수가 일치하지 않습니다 (왼쪽: {len(left_images)}, 오른쪽: {len(right_images)})"
                error = ValueError(error_msg)
                error.error_code = error_code
                raise error
            
            NUMBER_OF_CALIBRATION_IMAGES = len(left_images)
            
            # 캘리브레이션 처리 (Cal_1 = RIGHT, Cal_2 = LEFT)
            checker_size = 25.0  # 체커보드 크기 (mm)
            mtx_l, K_l, R_l, t_l, dist_l, _, image_size = calibration(cal_2_dir, NUMBER_OF_CALIBRATION_IMAGES, checker_size)  # LEFT
            mtx_r, K_r, R_r, t_r, dist_r, _, _ = calibration(cal_1_dir, NUMBER_OF_CALIBRATION_IMAGES, checker_size)  # RIGHT
            
            # print(f">>>> Left Camera (Cal_2):")
            # print(f" mtx =\n{mtx_l}\n Intrinsic =\n{K_l}\n Rotation =\n{cv2.Rodrigues(R_l[0])[0]}\n translation = \n{t_l[0]}")
            # print(f">>>> Right Camera (Cal_1):")
            # print(f" mtx =\n{mtx_r}\n Intrinsic =\n{K_r}\n Rotation =\n{cv2.Rodrigues(R_r[0])[0]}\n translation = \n{t_r[0]}")
            
            R, t = compute_relative_pose(R_l[0], t_l[0], R_r[0], t_r[0])
            print("============================================================")
            print(f">>>> Stereo Relative Pose (LEFT-RIGHT):")
            print(f" Rotation =\n{R}\n translation =\n{t}")
            print("============================================================")
            
            cal_output_dir = calibration_dir
            os.makedirs(cal_output_dir, exist_ok=True)
            output_file = os.path.join(cal_output_dir, "calibration_results.xml")
            
            try:
                save_calibration_results(output_file, image_size, mtx_l, mtx_r, K_l, K_r, dist_l, dist_r, R, t, R_l[0], t_l[0], R_r[0], t_r[0])
            except Exception as save_error:
                error_code = 602  # ERROR_CODE_CALIBRATION_FILE_WRITE_FAILED
                error_msg = f"calibration_results.xml 저장 실패: {str(save_error)}"
                error = ValueError(error_msg)
                error.error_code = error_code
                raise error
            
            # calibration_results.xml 파일 확인
            if not os.path.exists(output_file):
                error_code = 605  # ERROR_CODE_CALIBRATION_RESULT_FILE_NOT_FOUND
                error_msg = f"calibration_results.xml 파일이 생성되지 않았습니다: {output_file}"
                error = ValueError(error_msg)
                error.error_code = error_code
                raise error
            else:
                print(f"calibration_results.xml 생성 완료: {output_file}")
            
            # ========== 싱글 카메라 캘리브레이션 (LEFT-SINGLE) - Cal3가 있는 경우만 ==========
            output_file_s = None
            single_images = sorted([f for f in glob.glob(os.path.join(cal_3_dir, "Cal*.jpg")) if not os.path.basename(f) == "Cal_00.jpg"])
            
            if len(single_images) > 0:
                print(f"싱글 카메라 이미지 발견: {len(single_images)}개 - calibration_results_single.xml 생성 시작")
                
                if len(single_images) != len(left_images):
                    print(f"⚠️ 싱글 카메라 이미지 개수({len(single_images)})가 LEFT 이미지 개수({len(left_images)})와 다릅니다. 싱글 캘리브레이션을 건너뜁니다.")
                else:
                    try:
                        # 싱글 카메라 캘리브레이션 (Cal_3 = SINGLE)
                        mtx_s, K_s, R_s_list, t_s_list, dist_s, _, _ = calibration(cal_3_dir, NUMBER_OF_CALIBRATION_IMAGES, checker_size)  # SINGLE
                        R_s_rel, t_s_rel = compute_relative_pose(R_l[0], t_l[0], R_s_list[0], t_s_list[0])
                        print("============================================================")
                        print(f">>>> Single Camera Relative Pose (LEFT-SINGLE):")
                        print(f" Rotation =\n{R_s_rel}\n translation =\n{t_s_rel}")
                        print("============================================================")
                        
                        output_file_s = os.path.join(cal_output_dir, "calibration_results_single.xml")
                        
                        try:
                            save_calibration_results_single(output_file_s, image_size, mtx_l, mtx_s, K_l, K_s, dist_l, dist_s, R_s_rel, t_s_rel, R_l[0], t_l[0], R_s_list[0], t_s_list[0])
                        except Exception as save_error_s:
                            print(f"⚠️ calibration_results_single.xml 저장 실패: {save_error_s} - 싱글 캘리브레이션을 건너뜁니다.")
                            output_file_s = None
                        
                        if output_file_s and os.path.exists(output_file_s):
                            print(f"calibration_results_single.xml 생성 완료: {output_file_s}")
                        else:
                            print(f"⚠️ calibration_results_single.xml 파일이 생성되지 않음 - 싱글 캘리브레이션을 건너뜁니다.")
                            output_file_s = None
                    except Exception as single_cal_error:
                        print(f"⚠️ 싱글 카메라 캘리브레이션 실패: {single_cal_error} - 싱글 캘리브레이션을 건너뜁니다.")
                        output_file_s = None
            else:
                print(f"싱글 카메라 이미지 없음 - calibration_results_single.xml 생성 건너뜀")
            
            # ========== 새로운 캘리브레이션 데이터를 메모리에 업데이트 ==========
            try:
                # 스테레오 캘리브레이션 파라미터 파싱
                stereo_cal_params = _parse_stereo_calibration_params(output_file)
                if stereo_cal_params:
                    print(f"✅ 시리얼 넘버 '{serial_number}': 스테레오 캘리브레이션 파라미터 메모리 로드 완료 (XML 생성 직후)")
                else:
                    print(f"⚠️ 시리얼 넘버 '{serial_number}': 스테레오 캘리브레이션 파라미터 파싱 실패 (경로만 저장)")
                
                # 싱글 캘리브레이션 파라미터 파싱 (있는 경우만)
                single_cal_params = None
                has_single_xml = output_file_s and os.path.exists(output_file_s)
                if has_single_xml:
                    single_cal_params = _parse_single_calibration_params(output_file_s)
                    if single_cal_params:
                        print(f"✅ 시리얼 넘버 '{serial_number}': 싱글 캘리브레이션 파라미터 메모리 로드 완료 (XML 생성 직후)")
                    else:
                        print(f"⚠️ 시리얼 넘버 '{serial_number}': 싱글 캘리브레이션 파라미터 파싱 실패 (경로만 저장)")
                
                # 메모리 데이터 업데이트 (depth_offset은 나중에 계산 후 업데이트)
                cal_data_dict = {
                    "cal_file_path": output_file,
                    "cal_file_path_s": output_file_s if has_single_xml else None,
                    "stereo_cal_params": stereo_cal_params,  # 스테레오 파라미터 메모리 저장
                    "single_cal_params": single_cal_params,  # 싱글 파라미터 메모리 저장
                    "depth_offset": None  # depth_offset은 나중에 계산 후 업데이트
                }
                
                update_serial_calibration_data(serial_number, cal_data_dict)
                
                if has_single_xml:
                    print(f"✅ 시리얼 넘버 '{serial_number}': 메모리 calibration 데이터 업데이트 완료 (스테레오 + 싱글, depth_offset 제외)")
                    print(f"   - calibration_results.xml: {output_file}")
                    print(f"   - calibration_results_single.xml: {output_file_s}")
                else:
                    print(f"✅ 시리얼 넘버 '{serial_number}': 메모리 calibration 데이터 업데이트 완료 (스테레오만, depth_offset 제외)")
                    print(f"   - calibration_results.xml: {output_file}")
            except Exception as mem_error:
                print(f"⚠️ 메모리 calibration 데이터 업데이트 실패 (XML 생성 직후): {mem_error}")
                traceback.print_exc()
            
            # ========== Inference 클래스 업데이트 (calibration_results.xml 반영) ==========
            try:
                expected_cal_file = os.path.join(calibration_dir, "calibration_results.xml")
                
                cc.artis_ai_json_config = cc.get_config(cc.path_to_config, cc.artis_ai_json_config)
                
                # 이미 업데이트된 crop 포인트를 inf_depth에서 가져오기 (RequestUpdateCropPointToWAS에서 업데이트한 값)
                if artis_model and artis_model.inf_depth:
                    crop_settings = {
                        "left_x": artis_model.inf_depth.ori_crop_lx,
                        "left_y": artis_model.inf_depth.ori_crop_ly,
                        "right_x": artis_model.inf_depth.ori_crop_rx,
                        "right_y": artis_model.inf_depth.ori_crop_ry,
                        "width": artis_model.inf_depth.crop_w,
                        "height": artis_model.inf_depth.crop_h
                    }
                    print(f"현재 Depth 인스턴스의 crop 포인트 사용: LEFT({crop_settings['left_x']}, {crop_settings['left_y']}), RIGHT({crop_settings['right_x']}, {crop_settings['right_y']}), 크기({crop_settings['width']}x{crop_settings['height']})")
                else:
                    # inf_depth가 없는 경우 config에서 기본값 사용
                    crop_settings = {
                        "left_x": cc.artis_ai_json_config.get("crop_lx", 180),
                        "left_y": cc.artis_ai_json_config.get("crop_ly", 0),
                        "right_x": cc.artis_ai_json_config.get("crop_rx", 130),
                        "right_y": cc.artis_ai_json_config.get("crop_ry", 0),
                        "width": cc.artis_ai_json_config.get("crop_width", 1600),
                        "height": cc.artis_ai_json_config.get("crop_height", 1200)
                    }
                    print(f"inf_depth가 없어 config 기본값 사용: LEFT({crop_settings['left_x']}, {crop_settings['left_y']}), RIGHT({crop_settings['right_x']}, {crop_settings['right_y']}), 크기({crop_settings['width']}x{crop_settings['height']})")
                
                # calibration 및 crop 업데이트 (depth_offset은 나중에 계산 후 업데이트)
                artis_model.update_calibration_and_crop(expected_cal_file, crop_settings, None)
                
                cal_00_right = os.path.join(cal_1_dir, "Cal_00.jpg")  # Cal_1 = RIGHT
                cal_00_left = os.path.join(cal_2_dir, "Cal_00.jpg")   # Cal_2 = LEFT
                cal_left_file = os.path.join(temp_dir, "Cal_left.jpg")
                cal_right_file = os.path.join(temp_dir, "Cal_right.jpg")
                
                if not os.path.exists(cal_00_left) or not os.path.exists(cal_00_right):
                    error_code = 607  # ERROR_CODE_CALIBRATION_DEPTH_VALIDATION_FAILED
                    error_msg = f"Cal_00.jpg (왼쪽, Cal_2) 또는 (오른쪽, Cal_1) 파일이 없습니다"
                    error = ValueError(error_msg)
                    error.error_code = error_code
                    raise error
                
                # 파일 복사
                shutil.copy2(cal_00_left, cal_left_file)
                shutil.copy2(cal_00_right, cal_right_file)
                
                _, _, _, crop_calibration_images, _ = _get_inference_modules()
                if crop_calibration_images is None:
                    raise RuntimeError("crop_calibration_images를 사용할 수 없습니다")
                img_l, img_r = crop_calibration_images(crop_settings, temp_dir)
                if img_l is None or img_r is None:
                    error_code = 607  # ERROR_CODE_CALIBRATION_DEPTH_VALIDATION_FAILED
                    error_msg = "전처리 실패: Cal_left.jpg 또는 Cal_right.jpg가 없습니다"
                    error = ValueError(error_msg)
                    error.error_code = error_code
                    raise error
                
                depth_file_bin = os.path.join(temp_dir, "Cam_2_Depth.bin")
                artis_model = _get_artis_model()
                if artis_model is None or not hasattr(artis_model, 'inf_depth'):
                    raise RuntimeError("AI 추론 모델의 depth inference를 사용할 수 없습니다")
                ret, dep_time = artis_model.inf_depth.inference([0, img_l, img_r], False, temp_dir, None, 0) # depth_offset=0 (보정 없음)
                
                if not os.path.exists(depth_file_bin):
                    error_code = 607  # ERROR_CODE_CALIBRATION_DEPTH_VALIDATION_FAILED
                    error_msg = f"Depth inference 실패: {depth_file_bin} 파일이 생성되지 않음"
                    error = ValueError(error_msg)
                    error.error_code = error_code
                    raise error
                print(f"Depth inference 완료: {depth_file_bin}")
                
                _, _, _, _, calculate_and_save_depth_offset = _get_inference_modules()
                if calculate_and_save_depth_offset is None:
                    raise RuntimeError("calculate_and_save_depth_offset를 사용할 수 없습니다")
                depth_offset = calculate_and_save_depth_offset(crop_settings, temp_dir)
                if depth_offset is None:
                    error_code = 607  # ERROR_CODE_CALIBRATION_DEPTH_VALIDATION_FAILED
                    error_msg = "depth_offset 계산 실패: None을 반환"
                    error = ValueError(error_msg)
                    error.error_code = error_code
                    raise error
                
                # depth_offset 계산 후 동기화
                artis_model.update_calibration_and_crop(None, None, depth_offset)
                
            except Exception as depth_error:
                print(f"depth_offset.txt 생성 실패: {depth_error}")
                traceback.print_exc()
                if not hasattr(depth_error, 'error_code'):
                    depth_error.error_code = 607  # ERROR_CODE_CALIBRATION_DEPTH_VALIDATION_FAILED
                raise
            
            # ========== 메모리 캐시에 depth_offset 업데이트 ==========
            try:
                from app.core.Artis_AI.camera.calibration_manager import serial_calibration_data
                existing_cal_data = serial_calibration_data.get(serial_number, {}).copy()
                
                if existing_cal_data:
                    existing_cal_data["depth_offset"] = depth_offset
                    update_serial_calibration_data(serial_number, existing_cal_data)
                    print(f"✅ 시리얼 넘버 '{serial_number}': 메모리 calibration 데이터의 depth_offset 업데이트 완료")
                    print(f"   - depth_offset: {depth_offset}")
                    print_serial_calibration_data()
                else:
                    print(f"⚠️ 시리얼 넘버 '{serial_number}': 메모리 캐시에 데이터가 없어 depth_offset 업데이트 실패")
            except Exception as mem_error:
                print(f"⚠️ 메모리 calibration 데이터의 depth_offset 업데이트 실패: {mem_error}")
                traceback.print_exc()
            
            # ========== WAS에 모든 데이터 저장 완료 후 NAS에 저장 ==========
            now = datetime.now()
            timestamp_str = now.strftime('%Y%m%d%H%M%S') + f"{now.microsecond // 1000:03d}"
            
            nas_serial_path = nas_path
            os.makedirs(nas_serial_path, exist_ok=True)
            nas_timestamp_dir = os.path.join(nas_serial_path, timestamp_str)
            os.makedirs(nas_timestamp_dir, exist_ok=True)
            
            # 1) calibration_results.xml을 NAS 타임스탬프 폴더로 복사
            nas_xml_path = os.path.join(nas_timestamp_dir, "calibration_results.xml")
            shutil.copy2(output_file, nas_xml_path)
            print(f"calibration_results.xml NAS 저장 완료: {nas_xml_path}")
            
            # 1-1) calibration_results_single.xml을 NAS 타임스탬프 폴더로 복사 (있는 경우만)
            if output_file_s and os.path.exists(output_file_s):
                nas_xml_s_path = os.path.join(nas_timestamp_dir, "calibration_results_single.xml")
                shutil.copy2(output_file_s, nas_xml_s_path)
                print(f"calibration_results_single.xml NAS 저장 완료: {nas_xml_s_path}")
            
            # 2) Cal_*.zip 파일들을 NAS 타임스탬프 폴더로 복사 (기존 경로에도 유지)
            for zip_file in zip_files:
                zip_filename = os.path.basename(zip_file)
                nas_zip_path = os.path.join(nas_timestamp_dir, zip_filename)
                shutil.copy2(zip_file, nas_zip_path)
                print(f"ZIP 파일 NAS 복사 완료: {zip_filename}")
            
            # 3) temp 폴더를 NAS 타임스탬프 폴더로 복사 (depth_offset 계산 완료 후)
            if os.path.exists(temp_dir):
                nas_temp_dir = os.path.join(nas_timestamp_dir, "temp")
                if os.path.exists(nas_temp_dir):
                    shutil.rmtree(nas_temp_dir)
                shutil.copytree(temp_dir, nas_temp_dir)
                print(f"temp 폴더 NAS 복사 완료: {nas_temp_dir}")
            else:
                print(f"temp 폴더가 존재하지 않아 복사를 건너뜁니다: {temp_dir}")
            
            print(f"캘리브레이션 결과 NAS 저장 완료: {nas_timestamp_dir}")
            # =================================================================
            
            elapsed_time = (time.time() - start_time) * 1000
            print(f"캘리브레이션 적용 완료 - 소요시간: {elapsed_time:.2f}ms")
            
            return {
                "success": True,
                "message": "캘리브레이션 적용 성공",
                "data": {
                    "error_code": 0,
                    "elapsedTime": f"{elapsed_time:.2f}ms",
                    "timestamp": timestamp_str
                }
            }
            
        except Exception as cal_error:
            print(f"캘리브레이션 적용 실패: {cal_error}")
            print(f"예외 타입: {type(cal_error).__name__}")
            traceback.print_exc()
            
            # 캘리브레이션 실패 시 저장된 이미지들 정리
            if serial_number and calibration_dir:
                try:
                    _cleanup_was_calibration_data(calibration_dir)
                except Exception as cleanup_err:
                    print(f"캘리브레이션 데이터 정리 실패: {cleanup_err}")
            
            # 안전장치: NAS에서 최신 캘리브레이션 결과 복원 시도
            if nas_path and serial_number and calibration_dir:
                try:
                    _restore_calibration_from_nas(nas_path, serial_number, calibration_dir, is_serial_path_included=True)
                except Exception as restore_err:
                    print(f"NAS에서 캘리브레이션 복원 실패: {restore_err}")
            
            # 에러 코드 추출 (예외 객체에 error_code 속성이 있으면 사용, 없으면 분류)
            if hasattr(cal_error, 'error_code'):
                error_code = cal_error.error_code
            else:
                error_code = _classify_calibration_error(str(cal_error))
                if error_code == 600:  # 기본값이면 Python 실행 실패로 변경
                    error_code = 604  # ERROR_CODE_CALIBRATION_PYTHON_EXECUTION_FAILED
            
            return {
                "success": False,
                "message": "캘리브레이션 적용 실패",
                "data": {
                    "error_code": error_code,
                    "body": str(cal_error)
                }
            }

    except Exception as e:
        print(f"캘리브레이션 적용 실패: {e}")
        print(f"예외 타입: {type(e).__name__}")
        traceback.print_exc()
        
        # 캘리브레이션 실패 시 저장된 이미지들 정리
        if serial_number and calibration_dir:
            try:
                _cleanup_was_calibration_data(calibration_dir)
            except Exception as cleanup_err:
                print(f"캘리브레이션 데이터 정리 실패: {cleanup_err}")
        
        # 안전장치: NAS에서 최신 캘리브레이션 결과 복원 시도
        if nas_path and serial_number and calibration_dir:
            try:
                _restore_calibration_from_nas(nas_path, serial_number, calibration_dir, is_serial_path_included=True)
            except Exception as restore_err:
                print(f"NAS에서 캘리브레이션 복원 실패: {restore_err}")
        
        # 에러 코드 추출 (예외 객체에 error_code 속성이 있으면 사용, 없으면 분류)
        if hasattr(e, 'error_code'):
            error_code = e.error_code
        else:
            error_code = _classify_calibration_error(str(e))
            if error_code == 600:  # 기본값이면 Python 실행 실패로 변경
                error_code = 604  # ERROR_CODE_CALIBRATION_PYTHON_EXECUTION_FAILED
        
        return {
            "success": False,
            "message": "캘리브레이션 적용 실패",
            "data": {
                "error_code": error_code,
                "body": str(e)
            }
        }


@router.post("/iscan-restore-calibration-data")
async def restore_calibration_data(
    request: Request,
    metadata: str = Form(...)
):
    """
    Calibration 데이터 복구 엔드포인트
    
    Args:
        request: FastAPI Request 객체
        metadata: JSON 형태의 메타데이터 (vendorName, dbKey, savePath, serialNumber 포함)
    
    Returns:
        처리 결과 정보
    """
    start_time = time.time()
    
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("=" * 100)
        print(f"🚀 /api/iscan-restore-calibration-data 동작 시작 - 서버 시각: {current_time}")
        print("=" * 100)
        
        # 파라미터 검증
        if not metadata:
            return {
                "success": False,
                "message": "메타데이터 누락",
                "data": {
                    "error_code": 600,
                    "body": "메타데이터가 누락되었습니다"
                }
            }
        
        # 메타데이터 파싱
        try:
            metadata_dict = json.loads(metadata)
            print(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"메타데이터 JSON 파싱 실패: {e}")
            return {
                "success": False,
                "message": "JSON 파싱 실패",
                "data": {
                    "error_code": 600,
                    "body": f"잘못된 JSON 형식: {str(e)}"
                }
            }
        
        # savePath 추출
        save_path = metadata_dict.get("savePath", "")
        if not save_path:
            return {
                "success": False,
                "message": "savePath 누락",
                "data": {
                    "error_code": 600,
                    "body": "메타데이터에 savePath가 없습니다"
                }
            }
        
        # NAS 경로 구성
        nas_path = os.path.join("/mynas/", os.path.normpath(save_path))
        print(f"NAS 경로에서 캘리브레이션 데이터 복구: {nas_path}")
        
        # 메타데이터에서 시리얼 넘버 추출
        serial_number = extract_serial_number(metadata_dict)
        
        restore_success = False
        
        # 시리얼 넘버가 있을 때만 복구 수행
        if serial_number:
            # WAS 로컬 calibration 디렉토리 경로
            _, _, path_to_root, _, _ = _get_inference_modules()
            if path_to_root is None:
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "message": "AI 추론 기능이 임시로 비활성화되어 있습니다 (SDUI 테스트 모드)",
                        "data": {
                            "body": "이 기능은 현재 테스트 모드에서 사용할 수 없습니다."
                        }
                    }
                )
            calibration_dir = os.path.join(path_to_root, "camera", "calibration", serial_number)
            
            try:
                # nas_path는 이미 /mynas/{savePath} 형태
                # _restore_calibration_from_nas는 nas_path에 serial_number를 추가하거나 포함된 경로를 받음
                # 여기서는 nas_path에 serial_number가 포함되지 않았으므로 is_serial_path_included=False
                restore_success = _restore_calibration_from_nas(nas_path, serial_number, calibration_dir, is_serial_path_included=False)
                if restore_success:
                    print(f"✅ NAS에서 최신 캘리브레이션 데이터 복구 완료: {serial_number}")
                else:
                    print(f"⚠️ NAS에서 캘리브레이션 데이터 복구 실패 (default 폴더에서 복구 시도됨 또는 복구 불가)")
            except Exception as restore_error:
                print(f"⚠️ NAS에서 캘리브레이션 데이터 복구 중 오류 발생: {restore_error}")
                traceback.print_exc()
        else:
            print("⚠️ 시리얼 넘버가 없어 캘리브레이션 데이터 복구를 건너뜁니다")
        
        elapsed_time = (time.time() - start_time) * 1000
        print(f"캘리브레이션 데이터 복구 완료: 복구 성공: {restore_success}, 소요시간: {elapsed_time:.2f}ms")

        if not restore_success:
            return {
                "success": False,
                "message": "캘리브레이션 데이터 복구 실패",
                "data": {
                    "error_code": 600,
                    "restore_success": False,
                    "elapsedTime": f"{elapsed_time:.2f}ms"
                }
            }

        return {
            "success": True,
            "message": "캘리브레이션 데이터 복구 완료",
            "data": {
                "error_code": 0,
                "restore_success": True,
                "elapsedTime": f"{elapsed_time:.2f}ms"
            }
        }
        
    except Exception as e:
        print(f"캘리브레이션 데이터 복구 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "message": "처리 중 오류 발생",
            "data": {
                "error_code": 600,
                "body": f"오류: {str(e)}"
            }
        }


@router.post("/iscan-delete-calibration-data")
async def reset_calibration_data(
    request: Request,
    metadata: str = Form(...)
):
    """
    Calibration 데이터 초기화 엔드포인트
    
    Args:
        request: FastAPI Request 객체
        metadata: JSON 형태의 메타데이터 (vendorName, dbKey, savePath, serialNumber 포함)
    
    Returns:
        처리 결과 정보
    """
    start_time = time.time()
    
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("=" * 100)
        print(f"🚀 /api/iscan-delete-calibration-data 동작 시작 - 서버 시각: {current_time}")
        print("=" * 100)

        # 파라미터 검증
        if not metadata:
            return {
                "success": False,
                "message": "메타데이터 누락",
                "data": {
                    "error_code": 600,
                    "body": "메타데이터가 누락되었습니다"
                }
            }

        # 메타데이터 파싱
        try:
            metadata_dict = json.loads(metadata)
            print(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"메타데이터 JSON 파싱 실패: {e}")
            return {
                "success": False,
                "message": "JSON 파싱 실패",
                "data": {
                    "error_code": 600,
                    "body": f"잘못된 JSON 형식: {str(e)}"
                }
            }

        # savePath 추출
        save_path = metadata_dict.get("savePath", "")
        if not save_path:
            return {
                "success": False,
                "message": "savePath 누락",
                "data": {
                    "error_code": 600,
                    "body": "메타데이터에 savePath가 없습니다"
                }
            }

        # NAS 경로 구성
        nas_path = os.path.join("/mynas/", os.path.normpath(save_path))
        print(f"NAS 경로에서 ZIP 파일 삭제: {nas_path}")

        if not os.path.exists(nas_path):
            return {
                "success": False,
                "message": "NAS 경로 없음",
                "data": {
                    "error_code": 600,
                    "body": f"NAS 경로가 존재하지 않습니다: {nas_path}"
                }
            }
        
        # 메타데이터에서 시리얼 넘버 추출 (한 번만)
        serial_number = extract_serial_number(metadata_dict)
        
        deleted_count = 0
        was_data_reset = False
        
        # 시리얼 넘버가 있을 때만 NAS ZIP 파일 삭제 및 WAS 로컬 데이터 초기화
        if serial_number:
            # 1) NAS 시리얼 넘버 폴더 안의 ZIP 파일 삭제: /mynas/{savePath}/{serial_number}/Cal_*.zip
            nas_serial_path = os.path.join(nas_path, serial_number)
            if os.path.exists(nas_serial_path):
                zip_pattern = os.path.join(nas_serial_path, "Cal_*.zip")
                zip_files = glob.glob(zip_pattern)
                for zip_file in zip_files:
                    try:
                        os.remove(zip_file)
                        deleted_count += 1
                        print(f"ZIP 파일 삭제 완료: {os.path.basename(zip_file)}")
                    except Exception as e:
                        print(f"ZIP 파일 삭제 실패: {zip_file}, 오류: {e}")
            else:
                print(f"⚠️ NAS 시리얼 경로가 없음: {nas_serial_path}")
            
            # 2) WAS 로컬 캘리브레이션 데이터 초기화
            _, _, path_to_root, _, _ = _get_inference_modules()
            if path_to_root is None:
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "message": "AI 추론 기능이 임시로 비활성화되어 있습니다 (SDUI 테스트 모드)",
                        "data": {
                            "body": "이 기능은 현재 테스트 모드에서 사용할 수 없습니다."
                        }
                    }
                )
            calibration_dir = os.path.join(path_to_root, "camera", "calibration", serial_number)
            try:
                _cleanup_was_calibration_data(calibration_dir)
                was_data_reset = True
                print(f"WAS 로컬 캘리브레이션 데이터 초기화 완료: {calibration_dir}")
            except Exception as cleanup_error:
                print(f"WAS 로컬 데이터 초기화 실패: {cleanup_error}")
            
            # 3) 메모리에서 해당 시리얼 넘버 calibration 데이터 삭제
            try:
                if remove_serial_calibration_data(serial_number):
                    print(f"✅ 메모리에서 시리얼 넘버 '{serial_number}' calibration 데이터 삭제 완료")
                    print_serial_calibration_data()
            except Exception as mem_error:
                print(f"⚠️ 메모리 데이터 삭제 실패: {mem_error}")
        else:
            print("⚠️ 시리얼 넘버가 없어 NAS ZIP 파일 삭제 및 WAS 로컬 데이터 초기화를 건너뜁니다 (default 폴더 보호)")
        
        elapsed_time = (time.time() - start_time) * 1000
        print(f"캘리브레이션 데이터 초기화 완료: NAS ZIP 파일 {deleted_count}개 삭제, WAS 로컬 데이터 초기화: {was_data_reset}, 소요시간: {elapsed_time:.2f}ms")
        
        return {
            "success": True,
            "message": "캘리브레이션 데이터 초기화 완료",
            "data": {
                "error_code": 0,
                "nas_zip_deleted_count": deleted_count,
                "was_data_reset": was_data_reset,
                "elapsedTime": f"{elapsed_time:.2f}ms"
            }
        }
        
    except Exception as e:
        print(f"캘리브레이션 데이터 초기화 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "message": "처리 중 오류 발생",
            "data": {
                "error_code": 600,
                "body": f"오류: {str(e)}"
            }
        }