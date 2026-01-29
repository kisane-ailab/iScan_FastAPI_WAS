#!/usr/bin/env python3
"""
간단한 API 테스트 스크립트
"""

import requests
import json
import os
from datetime import datetime

def test_basic_endpoints():
    """기본 API 엔드포인트들을 테스트합니다."""
    base_url = "http://localhost:50000"
    
    print("🔍 기본 API 테스트 시작")
    print("=" * 40)
    
    # 테스트할 엔드포인트들
    endpoints = [
        ("/", "루트"),
        ("/health", "헬스 체크"),
        ("/api/status", "시스템 업타임"),
        ("/api/system-info", "시스템 정보"),
        ("/api/bots", "봇 상태"),
        ("/docs", "API 문서")
    ]
    
    for endpoint, description in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ {description}: {response.status_code}")
                try:
                    data = response.json()
                    print(f"   응답: {json.dumps(data, ensure_ascii=False, indent=2)}")
                except:
                    print(f"   응답: {response.text[:100]}...")
            else:
                print(f"❌ {description}: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {description}: 연결 실패 - {e}")
        
        print("-" * 40)
    
    print("=" * 40)
    print("🏁 기본 테스트 완료")

def test_iscan_input_images():
    """이미지 수신 엔드포인트를 테스트합니다."""
    base_url = "http://localhost:50000"
    
    print("🔍 이미지 수신 API 테스트 시작")
    print("=" * 40)
    
    # 테스트용 이미지 파일들 (실행 파일 경로에 있다고 가정)
    test_images = [
        "sample/Cam_1_Color.jpg",  # RightCam
        "sample/Cam_2_Color.jpg",   # TopCam, LeftCam
        "sample/EdgeMan/Artis_EdgeManager_Config.json"
    ]
    
    # 실제 파일이 있는지 확인
    existing_images = []
    for img in test_images:
        if os.path.exists(img):
            existing_images.append(img)
        else:
            print(f"⚠️  테스트 이미지 파일이 없습니다: {img}")
    
    if not existing_images:
        print("❌ 테스트할 이미지 파일이 없습니다.")
        print("실행 파일 경로에 test_image_1.jpg, test_image_2.jpg 파일을 생성해주세요.")
        return
    
    # 메타데이터 생성
    metadata = {
        "publicIP": "192.168.1.100",
        "companyName": "TestCompany",
        "vendorName": "TestVendor",
        "dbKey": "test_db_key_123",
        "fileCount": len(existing_images),
        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "runMode": "UserRun",
        "fileList": [],
        "totalScanCount": 0
    }
    # 파일 정보 추가
    for i, img in enumerate(existing_images):
        file_size = os.path.getsize(img)
        metadata["fileList"].append({
            "name": img.split("/")[-1],
            "size": file_size
        })
    
    try:
        url = f"{base_url}/api/iscan-input-images"
        
        # multipart/form-data로 요청 준비
        files = []
        for img_idx, img in enumerate(existing_images):
            # 파일을 바이너리로 읽어서 메모리에 저장
            with open(img, 'rb') as f:
                file_content = f.read()
            file_format = "image/jpeg" if ".jpg" in img else "application/json"
            files.append((f'file{img_idx}', (img.split("/")[-1], file_content, file_format)))
        
        data = {
            'metadata': json.dumps(metadata, ensure_ascii=False)
        }
        
        print(f"📤 요청 전송 중...")
        print(f"   URL: {url}")
        print(f"   파일: {existing_images}")
        print(f"   메타데이터: {json.dumps(metadata, ensure_ascii=False, indent=2)}")
        
        response = requests.post(url, files=files, data=data, timeout=30)
        
        print(f"📥 응답 수신 완료")
        print(f"   상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 이미지 수신 성공!")
            try:
                result = response.json()
                print(f"   응답: {json.dumps(result, ensure_ascii=False, indent=2)}")
            except:
                print(f"   응답: {response.text[:200]}...")
        else:
            print(f"❌ 이미지 수신 실패: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   오류: {json.dumps(error_detail, ensure_ascii=False, indent=2)}")
            except:
                print(f"   오류: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 실패: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
    
    print("=" * 40)
    print("🏁 이미지 수신 테스트 완료")

def test_config_update():
    """추론기 설정 엔드포인트를 테스트합니다."""
    base_url = "http://localhost:50000"
    
    print("🔍 추론기 설정 API 테스트 시작")
    print("=" * 40)

    # 테스트용 설정 파일들 (실행 파일 경로에 있다고 가정)
    test_files = [
        "samples/EdgeMan/Artis_EdgeManager_Config.json",    # kisan_config.json
        "samples/EdgeMan/camera/calibration_results.xml",   # stereo camera calibration
        "samples/EdgeMan/item/item_info_korean.json",       # item info
    ]

    test_file_format = [
        "application/json",
        "application/xml",
        "application/json"
    ]
    
    # 실제 파일이 있는지 확인
    existing_files = []
    for file in test_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            print(f"⚠️  업데이트할 파일이 없습니다: {file}")
    
    if not existing_files:
        print("❌ 업데이트할 파일이 없습니다.")
        print("실행 파일 경로에 파일을 생성해주세요.")
        print(test_files)
        return
    
    # 메타데이터 생성
    metadata = {
        "publicIP": "192.168.1.100",
        "companyName": "TestCompany",
        "vendorName": "TestVendor",
        "dbKey": "test_db_key_123",
        "fileCount": len(existing_files),
        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "runMode": "UserRun",
        "fileList": [],
        "totalScanCount": 0
    }
    # 파일 정보 추가
    for i, file_name in enumerate(existing_files):
        file_size = os.path.getsize(file_name)
        metadata["fileList"].append({
            "name": file_name.split("/")[-1],
            "size": file_size
        })
    
    try:
        url = f"{base_url}/api/iscan-config-update"
        
        # multipart/form-data로 요청 준비
        files = []
        for file_idx, file in enumerate(existing_files):
            # 파일을 바이너리로 읽어서 메모리에 저장
            with open(file, 'rb') as f:
                file_content = f.read()
            files.append((f'file{file_idx}', (file.split("/")[-1], file_content, test_file_format[file_idx])))
        
        data = {
            'metadata': json.dumps(metadata, ensure_ascii=False)
        }
        
        print(f"📤 요청 전송 중...")
        print(f"   URL: {url}")
        print(f"   파일: {existing_files}")
        print(f"   메타데이터: {json.dumps(metadata, ensure_ascii=False, indent=2)}")
        
        response = requests.post(url, files=files, data=data, timeout=30)
        
        print(f"📥 응답 수신 완료")
        print(f"   상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 추론기 설정 갱신 성공!")
            try:
                result = response.json()
                print(f"   응답: {json.dumps(result, ensure_ascii=False, indent=2)}")
            except:
                print(f"   응답: {response.text[:200]}...")
        else:
            print(f"❌ 추론기 설정 갱신 실패: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   오류: {json.dumps(error_detail, ensure_ascii=False, indent=2)}")
            except:
                print(f"   오류: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 실패: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
    
    print("=" * 40)
    print("🏁 추론기 설정 테스트 완료")

def test_start_ai_training():
    """추론기 설정 엔드포인트를 테스트합니다."""
    base_url = "http://localhost:50000"

    print("🔍 추론기 학습 API 테스트 시작")
    print("=" * 40)

    # 테스트용 설정 파일들 (실행 파일 경로에 있다고 가정)
    test_files = [
        "sample/EdgeMan/db/db_sync_report.json"
    ]

    test_file_format = [
        "application/json"
    ]

    # 실제 파일이 있는지 확인
    existing_files = []
    for file in test_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            print(f"⚠️  업데이트할 파일이 없습니다: {file}")

    if not existing_files:
        print("❌ 업데이트할 파일이 없습니다.")
        print("실행 파일 경로에 파일을 생성해주세요.")
        print(test_files)
        return

    # 메타데이터 생성
    metadata = {
        "publicIP": "192.168.5.10",
        "companyName": "TestCompany",
        "vendorName": "TestVendor",
        "dbKey": "test_db_key_123",
        "fileCount": len(existing_files),
        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "runMode": "UserRun",
        "fileList": [],
        "totalScanCount": 0
    }
    # 파일 정보 추가
    for i, file_name in enumerate(existing_files):
        file_size = os.path.getsize(file_name)
        metadata["fileList"].append({
            "name": file_name.split("/")[-1],
            "size": file_size
        })

    try:
        url = f"{base_url}/api/iscan-start-ai-training"

        # multipart/form-data로 요청 준비
        files = []
        for file_idx, file in enumerate(existing_files):
            # 파일을 바이너리로 읽어서 메모리에 저장
            with open(file, 'rb') as f:
                file_content = f.read()
            files.append((f'file{file_idx}', (file.split("/")[-1], file_content, test_file_format[file_idx])))

        data = {
            'metadata': json.dumps(metadata, ensure_ascii=False)
        }

        print(f"📤 요청 전송 중...")
        print(f"   URL: {url}")
        print(f"   파일: {existing_files}")
        print(f"   메타데이터: {json.dumps(metadata, ensure_ascii=False, indent=2)}")

        response = requests.post(url, files=files, data=data, timeout=90)

        print(f"📥 응답 수신 완료")
        print(f"   상태 코드: {response.status_code}")

        if response.status_code == 200:
            print("✅ 추론기 학습 시작 성공!")
            try:
                result = response.json()
                print(f"   응답: {json.dumps(result, ensure_ascii=False, indent=2)}")
            except:
                print(f"   응답: {response.text[:200]}...")
        else:
            print(f"❌ 추론기 학습 시작 실패: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   오류: {json.dumps(error_detail, ensure_ascii=False, indent=2)}")
            except:
                print(f"   오류: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 실패: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

    print("=" * 40)
    print("🏁 추론기 학습 테스트 완료")

def test_sync_status():
    """추론기 설정 엔드포인트를 테스트합니다."""
    base_url = "http://localhost:50000"

    print("🔍 학습 상태 조회 API 테스트 시작")
    print("=" * 40)

    # 메타데이터 생성
    metadata = {
        "publicIP": "192.168.5.10",
        "companyName": "TestCompany",
        "vendorName": "TestVendor",
        "dbKey": "test_db_key_123",
        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "runMode": "UserRun",
        "totalScanCount": 0
    }

    try:
        url = f"{base_url}/api/iscan-sync-status"

        data = {
            'metadata': json.dumps(metadata, ensure_ascii=False)
        }

        print(f"📤 요청 전송 중...")
        print(f"   URL: {url}")
        print(f"   메타데이터: {json.dumps(metadata, ensure_ascii=False, indent=2)}")

        response = requests.post(url, data=data, timeout=90)

        print(f"📥 응답 수신 완료")
        print(f"   상태 코드: {response.status_code}")

        if response.status_code == 200:
            print("✅ 학습 상태 조회  성공!")
            try:
                result = response.json()
                print(f"   응답: {json.dumps(result, ensure_ascii=False, indent=2)}")
            except:
                print(f"   응답: {response.text[:200]}...")
        else:
            print(f"❌ 학습 상태 조회 실패: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   오류: {json.dumps(error_detail, ensure_ascii=False, indent=2)}")
            except:
                print(f"   오류: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 실패: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

    print("=" * 40)
    print("🏁 학습 상태 조회 테스트 완료")

def test_api():
    """전체 API 테스트를 실행합니다."""
    print("🚀 API 테스트 시작")
    print("=" * 50)
    
    # 기본 엔드포인트 테스트
    #test_basic_endpoints()
    
    #print("\n" + "=" * 50)
    
    # 이미지 수신 테스트
    #test_iscan_input_images()

    # 추론기 설정 업데이트
    #test_config_update()

    # 이미지 수신 테스트
    #test_iscan_input_images()

    # 학습하기 테스트
    test_start_ai_training()

    # 학습 상태 조회 테스트
    test_sync_status()
    
    print("\n" + "=" * 50)
    print("🎉 모든 테스트 완료!")

if __name__ == "__main__":
    test_api()