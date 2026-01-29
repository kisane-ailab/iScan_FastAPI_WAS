# iScan FastAPI WAS

추론 시스템 제어 기능을 제공하는 FastAPI 기반 웹 애플리케이션

## 🚀 주요 기능

- **다중 Telegram 봇 지원**: 하나의 애플리케이션에서 여러 봇을 동시 운영
- **다중 채팅 그룹 지원**: 각 봇이 여러 채팅 그룹에서 명령을 받을 수 있음
- **실시간 명령 로깅**: 어떤 봇, 어떤 채팅 그룹에서 어떤 명령이 들어왔는지 상세 로깅
- **REST API**: 시스템 정보 조회 및 제어
- **실시간 모니터링**: CPU, 메모리 사용률 및 업타임
- **보안 검증**: 권한 기반 시스템 제어
- **확장성**: 새로운 봇 토큰 추가 시 쉽게 확장 가능
- **SDUI 웹 서비스**: 서버 기반 UI 구성 및 동적 렌더링
  - 키오스크/POS 해상도별 반응형 디자인 (1024x768, 1024x1920)
  - 해상도별 버튼 및 텍스트 크기 자동 조정
  - 터치스크린 환경 최적화
- **상품 관리**: MySQL 데이터베이스 연동을 통한 상품 정보 관리
  - 상품 목록 전체 조회 (제한 없음)
  - 상품 목록 테이블 컬럼: 상품코드, 상품명, 가격, 스캔 이미지 개수, 대표 이미지
  - 상품 상세 정보 팝업 (행 클릭 시)
    - 상품 코드/상품명 클릭 시 ServMan 프로그램 실행 (선택된 상품 정보 포함)
    - 대표 이미지 자동 로드 및 표시 (EdgeMan 서버 연동)
    - EdgeMan 서버 통신 실패 시 에러 메시지 표시
  - 상품 검색 및 터치 키보드 기능 (상품코드, 상품명, 바코드)
- **EdgeMan 연동**: 브라우저 종료 및 ServMan 실행 요청
  - 클라이언트 IP 기반 EdgeMan 서버 통신
  - 통신 실패 시 에러 팝업 표시

## 📋 요구사항

- Python 3.8+
- Telegram Bot Token(s)
- 시스템 제어 권한 (sudo)
- MySQL 5.7+ (상품 관리 기능 사용 시)

## 🛠️ 설치 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 설정 파일 구성

#### config.json 파일 설정

프로젝트 루트의 `config.json` 파일을 편집하여 설정을 구성합니다:

```json
{
  "telegram_bots": {
    "bot1": {
      "token": "your_bot1_token_here",
      "chat_ids": ["chat_id1", "chat_id2"],
      "vendorName": "vendor1"
    },
    "bot2": {
      "token": "your_bot2_token_here", 
      "chat_ids": ["chat_id3", "chat_id4"],
      "vendorName": "vendor2"
    }
  },
  "server": {
    "host": "0.0.0.0",
    "port": 19901,
    "debug": false
  },
  "client": {
    "edgeman_port": 18081
  },
  "system": {
    "base_409_wait_ms": 3000,
    "max_consecutive_409": 5
  },
  "database": {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "iscan_test",
    "charset": "utf8mb4",
    "pool_size": 10
  }
}
```

#### 설정 항목 설명

- **telegram_bots**: Telegram 봇 설정
  - `token`: Telegram Bot API 토큰
  - `chat_ids`: 봇이 명령을 받을 채팅 ID 배열
  - `vendorName`: 벤더 이름 (메시지 전송 시 매칭에 사용)

- **server**: 서버 설정
  - `host`: 서버 호스트 (기본값: "0.0.0.0")
  - `port`: 서버 포트 (기본값: 19901)
  - `debug`: 디버그 모드 (기본값: false)

- **client**: 클라이언트 설정
  - `edgeman_port`: EdgeMan 서버 포트 (브라우저 종료 및 ServMan 실행 요청용, 기본값: 18081)

- **system**: 시스템 설정
  - `base_409_wait_ms`: 409 에러 대기 시간 (밀리초)
  - `max_consecutive_409`: 최대 연속 409 에러 허용 횟수

- **database**: 데이터베이스 설정 (상품 관리 기능 사용 시)
  - `host`: MySQL 서버 호스트 (기본값: "localhost")
  - `port`: MySQL 서버 포트 (기본값: 3306)
  - `user`: MySQL 사용자 이름 (기본값: "root")
  - `password`: MySQL 비밀번호
  - `database`: 데이터베이스 이름 (기본값: "iscan_test")
  - `charset`: 문자 인코딩 (기본값: "utf8mb4")
  - `pool_size`: 연결 풀 크기 (기본값: 10)

### 3. Telegram 봇 설정

1. [@BotFather](https://t.me/botfather)에서 봇 생성 (⭐ privacy mode : off)
2. 봇 토큰을 `config.json`의 `telegram_bots` 섹션에 추가
3. 채팅 ID들을 `chat_ids` 배열에 추가 (⭐ 채팅 ID 확인 : https://api.telegram.org/bot<BOT_TOKEN>/getUpdates)

### 4. 채팅 그룹 설정

채팅 그룹의 제목을 다음 형식으로 설정해야 합니다:
```
chat_description|public_ip
```

예시:
- `챗그룹설명|공인IP`

## 🚀 실행 방법 (uvicorn 으로 싱글 코어만 사용)

### 개발 환경

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 19901
```

### 운영 환경

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 19901
```

### Docker 사용

```bash
# Docker Compose로 실행
docker-compose up -d

# 또는 Docker 직접 실행
docker build -t iscan-fastapi-was .
docker run -p 19901:19901 -v $(pwd)/config.json:/app/config.json:ro iscan-fastapi-was
```

### 서비스 등록 및 실행

```bash
cp Artis_CS.service /etc/systemd/system/Artis_CS.service

# 시스템 데몬 재로드
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

# 서비스 등록
sudo systemctl enable Artis_CS

# 서비스 시작
sudo systemctl start Artis_CS

# 상태 확인
sudo systemctl status Artis_CS

# 로그 확인
journalctl -u Artis_CS -f
```

## 📡 API 엔드포인트

### 상태 조회
- `GET /api/status` - 시스템 상태 조회
- `GET /api/system-info` - 시스템 상세 정보 조회
- `GET /api/bots` - Telegram 봇 상태 조회

### 상품 관리
- `GET /api/products` - 상품 목록 조회
  - Query Parameters:
    - `vendor_id` (optional): 벤더 ID로 필터링
    - `category` (optional): 카테고리로 필터링
    - `is_deleted` (optional, default: false): 삭제된 상품 포함 여부
    - `limit` (optional): 조회 개수 (None이면 전체 조회, 지정 시 1-1000)
    - `offset` (optional, default: 0): 시작 위치
    - `search` (optional): 검색어 (상품코드, 상품명, 바코드에서 검색)
  - 응답 형식:
    ```json
    {
      "success": true,
      "data": [...],
      "total": 100,
      "limit": 100,
      "offset": 0
    }
    ```
  - 에러 응답 (데이터베이스 연결 실패 시):
    ```json
    {
      "success": false,
      "error": "데이터베이스 서버에 연결할 수 없습니다. MySQL 서버가 실행 중인지 확인해주세요."
    }
    ```
- `GET /api/products/{product_id}` - 상품 상세 정보 조회

### EdgeMan 연동
- `POST /api/edgeman/terminate-browser` - 브라우저 종료 요청
  - 클라이언트 IP 기반으로 EdgeMan 서버로 브라우저 종료 요청 전달
  - EdgeMan 서버 통신 실패 시 에러 팝업 표시
- `POST /api/edgeman/run-servman` - ServMan 프로그램 실행 요청
  - 클라이언트 IP 기반으로 EdgeMan 서버로 ServMan 실행 요청 전달
  - 선택된 상품의 상세 정보를 JSON으로 함께 전달
  - EdgeMan 서버 통신 실패 시 에러 팝업 표시
  - 요청 본문 예시:
    ```json
    {
      "product": {
        "id": 123,
        "vendor_id": 1,
        "vendor_name": "cheonsang_seongsu",
        "item_code": "0000012",
        "item_name_default": "테스트 상품명",
        "barcode": "8801234567890",
        "base_amount": 5000.0,
        "is_pos_use": true,
        "scan_image_count": 5,
        ...
      }
    }
    ```
- `POST /api/edgeman/request-thumbnail` - 대표 이미지 요청
  - 클라이언트 IP 기반으로 EdgeMan 서버로 대표 이미지 요청 전달
  - 선택된 상품의 상세 정보를 JSON으로 함께 전달 (run-servman과 동일한 형식)
  - EdgeMan 서버 응답 형식: JSON (status/data 구조)
  - 성공 응답 예시:
    ```json
    {
      "status": "success",
      "message": "Thumbnail image retrieved successfully",
      "data": {
        "image_data": "data:image/png;base64,iVBORw0KGgoAAAANS...",
        "content_type": "image/png"
      }
    }
    ```
  - 실패 응답 예시:
    ```json
    {
      "status": "error",
      "message": "Thumbnail image file not found"
    }
    ```
  - EdgeMan 서버 통신 실패 시 에러 메시지 표시

### 추론기 입력 이미지 파일 수신
- `POST /api/iscan-input-images` - 추론기 입력 이미지 파일 수신 (multipart/form-data)

### 웹 UI
- `GET /` - SDUI 메인 페이지
- `GET /web` - SDUI 메인 페이지 (별칭)
- `GET /product` - 상품 관리 페이지
- `GET /settings` - 환경설정 페이지 (비밀번호 인증 필요)
- `GET /password` - 비밀번호 입력 페이지
- `GET /api/ui/config` - UI 구성 정보 조회 (JSON)

### 헬스 체크
- `GET /health` - 서비스 상태 확인

## 🤖 Telegram 봇 명령어

- `/status` - 시스템 상태 조회
- `/uptime` - 시스템 업타임 조회
- `/info` - 상세 시스템 정보 조회
- `/help` - 도움말 표시

## 📁 추론기 입력 이미지 파일 수신 API

### 엔드포인트
```
POST /api/iscan-input-images
```

### 요청 형식
- **Content-Type**: `multipart/form-data`
- **Method**: POST

### 요청 구조

#### Part 1: 메타데이터 (JSON)
- **name**: `metadata`
- **Content-Type**: `application/json`
- **내용**: 파일 리스트 및 부가 정보

#### Part 2~N: 파일 데이터
- **name**: `file0`, `file1`, ...
- **Content-Type**: `application/octet-stream`
- **내용**: 파일 바이너리 데이터

### 메타데이터 JSON 예시
```json
{
  "publicIP": "1.2.3.4",
  "companyName": "company",
  "vendorName": "vendor",
  "dbKey": "key",
  "runMode": "UserRun",
  "fileCount": "2",
  "timestamp": "20250108112409858",
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

### runMode 값
- `"UserRun"`: 사용자 실행 모드
- `"NewItem"`: 새 아이템 모드
- `"CalCam"`: 카메라 보정 모드
- `"Base"`: 기본 모드

> **참고**: `runMode`는 필수 필드이지만, 위 4개 값과 일치하지 않아도 업로드가 실패하지 않습니다. 알 수 없는 값의 경우 경고 로그만 남깁니다.

### 서버 처리 과정
1. **multipart 파싱**: 메타데이터와 파일 분리
2. **무결성 검증**: 
   - 각 파일의 실제 크기와 JSON의 `size` 값 비교
   - 파일명 매칭 (`file0` → `image1.jpg`, `file1` → `image2.jpg`)
3. **용량 체크**: 
   - 개별 파일 최대 50MB
   - 전체 파일 최대 500MB
4. **파일 저장**: 타임스탬프가 포함된 안전한 파일명으로 저장

### 응답 예시
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
      "fileCount": "1",
      "timestamp": "20250108112409858"
    }
	"file": "<BASE64_인코딩된_iscan_input_result_20250108112409858.zip>",
    "fileList": [
        {
          "name": "iscan_input_result_20250108112409858.zip",
          "size": 123456
        }
    ]
  }
}
```

## 📊 로깅

로그는 `logs/` 디렉토리에 일별로 저장됩니다:
- `app_YYYYMMDD.log` - 애플리케이션 로그

### 명령어 로깅 예시
```
📥 Bot: bot1 | Chat: 개발팀 채팅방 (123456789) | User: @username | Command: /status
📥 Bot: bot2 | Chat: 운영팀 채팅방 (987654321) | User: @admin | Command: /system
```

## 🔧 의존성 패키지

- **fastapi**: REST API 및 WebSocket 서버용 메인 프레임워크
- **uvicorn**: ASGI 서버
- **aiohttp**: 비동기 HTTP 통신 (Telegram API long polling)
- **python-multipart**: multipart/form-data 파싱 (파일 업로드용)
- **requests**: HTTP 클라이언트 라이브러리
- **psutil**: 시스템 정보 수집 (CPU, 메모리, 업타임 등)
- **pyzmq**: ZeroMQ 통신 (추론기)
- **aiomysql**: 비동기 MySQL 클라이언트 (상품 관리 기능)
- **pymysql**: MySQL 클라이언트 (aiomysql 의존성)

## 🔄 확장성

### 새로운 봇 추가 방법

`config.json`의 `telegram_bots` 섹션에 새로운 봇을 추가:

```json
{
  "telegram_bots": {
    "bot1": {
      "token": "existing_token",
      "chat_ids": ["existing_chat_id"],
      "vendorName": "existing_vendor"
    },
    "new_bot": {
      "token": "new_token",
      "chat_ids": ["new_chat_id"],
      "vendorName": "new_vendor"
    }
  }
}
```

### 새로운 채팅 그룹 추가 방법

기존 봇의 `chat_ids` 배열에 새로운 채팅 ID를 추가하면 됩니다.

### 설정 파일 권한

Docker 환경에서 `config.json` 파일은 읽기 전용으로 마운트되어 보안을 강화합니다.

## 🛡️ 보안 고려사항

- 시스템 재부팅은 sudo 권한이 필요합니다
- Telegram 봇은 허용된 채팅 ID에서만 명령을 받습니다
- 모든 API 요청과 명령어는 상세히 로깅됩니다
- 각 봇은 독립적으로 관리됩니다
- `config.json` 파일은 읽기 전용으로 마운트되어 무단 수정을 방지합니다
