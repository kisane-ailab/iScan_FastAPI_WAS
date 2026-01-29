# iScan Config Update API 문서

## 엔드포인트 개요

**POST** `/api/iscan-config-update`

추론기 설정을 갱신하는 엔드포인트입니다. multipart/form-data 형식으로 메타데이터와 설정파일들을 전송받아 검증 후 추론기를 갱신합니다.

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
      "name": "Artis_EdgeManager_Config.json",
      "size": 2281
    },
    {
      "name": "calibration_results.xml",
      "size": 2329
    },
    {
      "name": "item_info_korean.json",
      "size": 177168
    }
  ],
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
    "name": "Artis_EdgeManager_Config.json",
    "size": 2281
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
  "message": "추론기 설정 갱신 완료",
  "data": {
    "metadata": {
      "publicIP": "1.2.3.4",
      "companyName": "company",
      "vendorName": "vendor",
      "dbKey": "key",
      "runMode": "UserRun",
      "success": true,
      "message": "추론기 설정 갱신 완료",
      "fileList": [
        {
          "name": "Artis_EdgeManager_Config.json",
          "size": 2281
        },
        {
          "name": "calibration_results.xml",
          "size": 2329
        },
        {
          "name": "item_info_korean.json",
          "size": 177168
        }
      ]
    }
  }
}
```

### result 필드 상세 내용 (예시)
현재 없음


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
    "fileCount": "3",
    "timestamp": "20250818171231",
    "runMode": "UserRun",
    "fileList": [
        { "name": "Artis_EdgeManager_Config.json", "size": 2281 },
        { "name": "calibration_results.xml", "size": 2329 },
        { "name": "item_info_korean.json", "size": 177168 }
    ]
}

# 파일 업로드
files = {
    'metadata': (None, json.dumps(metadata)),
    'file0': ('Artis_EdgeManager_Config.json', open('Artis_EdgeManager_Config.json', 'rb')),
    'file1': ('calibration_results.xml', open('calibration_results.xml', 'rb')),
    'file2': ('item_info_korean.json', open('item_info_korean.json', 'rb'))
}

response = requests.post(
    'http://localhost:8000/api/iscan-config-update',
    files=files
)

print(response.json())
```

## 파일 저장 위치

업로드된 파일은 `Artis_AI/` 디렉토리에 다음 형식으로 저장됩니다:

```
camera
└── calibration_results.xml
checkpoints/
└── item_info_korean.json
kisan_config.json
```

## 로그

### 성공 로그
```
[INFO] app.api: 추론기 설정 갱신 요청 시작
[INFO] app.api: 📊 Form 데이터 파싱 시간: 0.00ms
[INFO] app.api: 파일 0: Artis_EdgeManager_Config.json, 크기: 2281, 타입: application/json
[INFO] app.api: 파일 1: calibration_results.xml, 크기: 2329, 타입: application/xml
[INFO] app.api: 파일 2: item_info_korean.json, 크기: 177168, 타입: application/json
[INFO] app.api: 파일 개수: 3
[INFO] app.api: 메타데이터 검증 완료: 3개 파일
[INFO] 파일 1/2 처리 완료: image1.jpg
[INFO] 파일 2/2 처리 완료: image2.jpg
[INFO] app.api: 📊 메타데이터 파싱/검증 시간: 0.00ms
[INFO] app.api: 📊 파일 1 검증 시간: 0.00ms
[INFO] app.api: 파일 1/3 처리 완료: Artis_EdgeManager_Config.json (총 0.00ms)
[INFO] app.api: 📊 파일 2 검증 시간: 0.00ms
[INFO] app.api: 파일 2/3 처리 완료: calibration_results.xml (총 0.00ms)
[INFO] app.api: 📊 파일 3 검증 시간: 0.00ms
[INFO] app.api: 파일 3/3 처리 완료: item_info_korean.json (총 0.00ms)
[INFO] app.api: 📊 전체 파일 처리 시간: 0.00ms
[INFO] app.api: 📊 추론기 설정 갱신 시간: 1356.70ms
[INFO] app.api: 추론기 설정 갱신 완료: 3개 파일, 181778바이트, 소요시간: 1356.70ms
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