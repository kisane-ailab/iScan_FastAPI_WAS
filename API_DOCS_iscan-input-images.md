# iScan Input Images API 문서

## 엔드포인트 개요

**POST** `/api/iscan-input-images`

이미지 파일을 수신하고 처리하는 엔드포인트입니다. multipart/form-data 형식으로 메타데이터와 이미지 파일들을 전송받아 검증 후 저장합니다.

## 요청 형식

### Content-Type
```
multipart/form-data; boundary=----WebKitFormBoundary...
```

### 요청 구조

#### Part 1: metadata
- **name**: `metadata`
- **Content-Type**: `application/json`
- **내용**: 파일 리스트 및 부가 정보가 담긴 JSON

#### Part 2~N: 파일 데이터
- **name**: `file0`, `file1`, `file2`, ...
- **filename**: 실제 파일명
- **Content-Type**: `application/octet-stream`
- **내용**: 파일 바이너리 데이터

## 메타데이터 JSON 구조

### 필수 필드

```json
{
  "publicIP": "1.2.3.4",
  "companyName": "company",
  "vendorName": "vendor", 
  "dbKey": "key",
  "fileCount": "2",
  "timestamp": "2025-07-21 18:00:00",
  "runMode": "UserRun",
  "fileList": [
    {
      "name": "image1.jpg",
      "size": 123456
    },
    {
      "name": "image2.jpg", 
      "size": 78910
    }
  ]
}
```

### 필드 설명

| 필드명 | 타입 | 필수 | 설명 |
|--------|------|------|------|
| `publicIP` | string | ✓ | 공개 IP 주소 |
| `companyName` | string | ✓ | 회사명 |
| `vendorName` | string | ✓ | 벤더명 |
| `dbKey` | string | ✓ | 데이터베이스 키 |
| `fileCount` | string | ✓ | 파일 개수 (숫자 문자열) |
| `timestamp` | string | ✓ | 타임스탬프 (YYYY-MM-DD HH:MM:SS) |
| `runMode` | string | ✓ | 실행 모드 |
| `fileList` | array | ✓ | 파일 정보 배열 |

### runMode 값

- `"UserRun"`: 사용자 실행 모드
- `"NewItem"`: 새 아이템 모드  
- `"CalCam"`: 카메라 보정 모드
- `"Base"`: 기본 모드

> **참고**: runMode가 위 4개 값 중 하나가 아닌 경우에도 요청은 처리되며, 경고 로그만 기록됩니다.

### fileList 구조

```json
[
  {
    "name": "image1.jpg",
    "size": 123456
  }
]
```

| 필드명 | 타입 | 필수 | 설명 |
|--------|------|------|------|
| `name` | string | ✓ | 파일명 |
| `size` | number | ✓ | 파일 크기 (바이트) |

## 파일 업로드 규칙

### 파일 매칭
- `file0` → `fileList[0].name`
- `file1` → `fileList[1].name`
- `file2` → `fileList[2].name`
- ...

### 파일 크기 제한
- **개별 파일**: 최대 50MB
- **전체 파일**: 최대 500MB

### 파일 무결성 검증
1. **파일 크기 검증**: 예상 크기와 실제 크기 비교
2. **파일명 검증**: 예상 파일명과 실제 파일명 비교
3. **파일 개수 검증**: `fileCount`와 실제 업로드된 파일 개수 비교

## 응답 형식

### 성공 응답 (200 OK)

```json
{
  "success": true,
  "message": "이미지 수신 완료",
  "data": {
    "metadata": {
      "publicIP": "1.2.3.4",
      "companyName": "company",
      "vendorName": "vendor",
      "dbKey": "key",
      "runMode": "UserRun",
      "fileCount": "2",
      "timestamp": "20250721180000",
      "fileList": [
        {
          "name": "Cam_1_Color.jpg",
          "size": 123456
        },
        {
          "name": "Cam_2_Color.jpg",
          "size": 78910
        }
      ]
    },
    "result": [
      {
        "name": "20250721180000.zip",
        "size": 1234567
      }
    ]
  }
}
```

### result 필드 상세 내용 (예시)

```json
{
  "artis_error_info": {
    "version": {
      "Artis_AI": "0.2.6",
      "Artis_AI_Model": "0.1.6"
    },
    "error_code": "0x00000000",
    "error_reason": "OK",
    "error_code_detail": [],
    "error_reason_detail": [],
    "object_total_cnt": 0,
    "object_bbox": {}
  },
  "artis_result_info": {
    "object_total_cnt": 4
  },
  "artis_object_bbox": {
    "0": [519, 230, 624, 411],
    "1": [264, 280, 515, 528],
    "2": [666, 284, 836, 606],
    "3": [461, 488, 640, 671]
  },
  "artis_object_detail": {
    "0": 768,
    "1": 360,
    "2": 792,
    "3": 686
  },
  "artis_object_score": {
    "0": "0.9839587",
    "1": "0.9927134",
    "2": "0.98438925",
    "3": "0.99061453"
  },
  "depth": {
    "object_bbox": {
      "0": [518, 230, 626, 408],
      "1": [264, 288, 516, 528],
      "2": [666, 280, 838, 606],
      "3": [462, 492, 636, 668]
    },
    "object_detail": {
      "0": 230,
      "1": 230,
      "2": 230,
      "3": 230
    },
    "object_score": {
      "0": "0.8345",
      "1": "0.864",
      "2": "0.8228",
      "3": "0.7783"
    },
    "result_info": {
      "object_total_cnt": 4
    }
  },
  "seg": {
    "object_bbox": {},
    "object_detail": {},
    "object_score": {},
    "object_contour": {},
    "result_info": {
      "object_total_cnt": 0
    }
  },
  "rgb": {
    "object_bbox": {
      "0": [519, 230, 624, 411],
      "1": [264, 280, 515, 528],
      "2": [666, 284, 836, 606],
      "3": [461, 488, 640, 671]
    },
    "object_detail": {
      "0": 768,
      "1": 360,
      "2": 792,
      "3": 686
    },
    "object_score": {
      "0": "0.9839587",
      "1": "0.9927134",
      "2": "0.98438925",
      "3": "0.99061453"
    },
    "result_info": {
      "object_total_cnt": 4
    }
  }
}
```

## 에러 응답

### 400 Bad Request

#### JSON 파싱 오류
```json
{
  "detail": "잘못된 JSON 형식: Expecting property name enclosed in double quotes"
}
```

#### 메타데이터 검증 실패
```json
{
  "detail": "메타데이터 검증 실패: 필수 필드가 누락되었습니다: publicIP"
}
```

#### 파일 개수 불일치
```json
{
  "detail": "파일 개수 불일치: 예상 2개, 실제 1개"
}
```

#### 파일 크기 초과
```json
{
  "detail": "파일 크기 초과: 52428800바이트 (최대 52428800바이트)"
}
```

#### 전체 파일 크기 초과
```json
{
  "detail": "전체 파일 크기 초과: 524288000바이트 (최대 524288000바이트)"
}
```

#### 파일 무결성 검증 실패
```json
{
  "detail": "파일 1 처리 실패: 파일 크기 불일치: 예상 123456바이트, 실제 123000바이트"
}
```

### 500 Internal Server Error
```json
{
  "detail": "서버 내부 오류"
}
```

## 사용 예시

### cURL 예시

```bash
curl -X POST "http://localhost:8000/api/iscan-input-images" \
  -H "Content-Type: multipart/form-data" \
  -F "metadata={\"publicIP\":\"1.2.3.4\",\"companyName\":\"company\",\"vendorName\":\"vendor\",\"dbKey\":\"key\",\"fileCount\":\"2\",\"timestamp\":\"2025-07-21 18:00:00\",\"runMode\":\"UserRun\",\"fileList\":[{\"name\":\"image1.jpg\",\"size\":123456},{\"name\":\"image2.jpg\",\"size\":78910}]}" \
  -F "file0=@image1.jpg" \
  -F "file1=@image2.jpg"
```

### Python 예시

```python
import requests
import json

# 메타데이터 준비
metadata = {
    "publicIP": "1.2.3.4",
    "companyName": "company",
    "vendorName": "vendor",
    "dbKey": "key",
    "fileCount": "2",
    "timestamp": "2025-07-21 18:00:00",
    "runMode": "UserRun",
    "fileList": [
        {"name": "image1.jpg", "size": 123456},
        {"name": "image2.jpg", "size": 78910}
    ]
}

# 파일 업로드
files = {
    'metadata': (None, json.dumps(metadata)),
    'file0': ('image1.jpg', open('image1.jpg', 'rb')),
    'file1': ('image2.jpg', open('image2.jpg', 'rb'))
}

response = requests.post(
    'http://localhost:8000/api/iscan-input-images',
    files=files
)

print(response.json())
```

## 파일 저장 위치

업로드된 파일은 `receive/` 디렉토리에 다음 형식으로 저장됩니다:

```
receive/
├── 20250730_143022_0_image1.jpg
├── 20250730_143022_1_image2.jpg
└── ...
```

파일명 형식: `{YYYYMMDD_HHMMSS}_{인덱스}_{원본파일명}`

## 로그

### 성공 로그
```
[INFO] 이미지 수신 요청 시작
[INFO] 메타데이터 검증 완료: 2개 파일
[INFO] 파일 1/2 처리 완료: image1.jpg
[INFO] 파일 2/2 처리 완료: image2.jpg
[INFO] 이미지 수신 완료: 2개 파일, 202366바이트
```

### 경고 로그
```
[WARNING] 알 수 없는 runMode 값: CustomMode (예상 값: UserRun, NewItem, CalCam, Base)
```

### 에러 로그
```
[ERROR] JSON 파싱 오류: Expecting property name enclosed in double quotes
[ERROR] 메타데이터 검증 오류: 필수 필드가 누락되었습니다: publicIP
[ERROR] 파일 개수 불일치: 예상 2개, 실제 1개
[ERROR] 파일 1 처리 오류: 파일 크기 불일치: 예상 123456바이트, 실제 123000바이트
``` 