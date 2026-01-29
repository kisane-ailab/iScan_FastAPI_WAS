# iScan FastAPI WAS API 테스트 가이드

이 프로젝트에는 API를 테스트할 수 있는 여러 가지 방법이 있습니다.

## 📋 테스트 방법

### 1. Python 스크립트 테스트 (권장)

#### 전체 테스트 (상세)
```bash
python test_api.py
```

#### 간단한 테스트
```bash
python simple_api_test.py
```

#### 다른 서버 주소로 테스트
```bash
python test_api.py http://your-server:19901
```

### 2. Bash 스크립트 테스트 (curl 사용)

```bash
# 실행 권한 부여
chmod +x test_api.sh

# 테스트 실행
./test_api.sh
```

### 3. 수동 테스트 (curl)

```bash
# 기본 엔드포인트
curl http://localhost:19901/
curl http://localhost:19901/health

# 상태 API
curl http://localhost:19901/api/status
curl http://localhost:19901/api/system-info
curl http://localhost:19901/api/bots

# API 문서
curl http://localhost:19901/docs
curl http://localhost:19901/openapi.json
```

## 🚀 서버 실행

테스트하기 전에 서버를 실행해야 합니다:

```bash
# 개발 모드로 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 19901

# 또는 Docker로 실행
docker-compose up
```

## 📊 테스트 결과

### 성공적인 응답 예시

#### 루트 엔드포인트 (/)
```json
{
  "message": "iScan FastAPI WAS",
  "version": "1.0.0",
  "status": "running"
}
```

#### 시스템 업타임 (/api/status)
```json
{
  "uptime": "2 days, 3 hours, 45 minutes"
}
```

#### 시스템 정보 (/api/system-info)
```json
{
  "cpu_usage": 15.2,
  "memory_usage": 45.8,
  "disk_usage": 23.1,
  "uptime": "2 days, 3 hours, 45 minutes"
}
```

#### 봇 상태 (/api/bots)
```json
{
  "total_bots": 2,
  "total_chats": 5,
  "bots": {
    "bot1": {
      "name": "MyBot1",
      "chat_ids": ["123456789", "987654321"],
      "chat_count": 2
    },
    "bot2": {
      "name": "MyBot2", 
      "chat_ids": ["111222333"],
      "chat_count": 1
    }
  }
}
```

## ⚠️ 주의사항

1. **서버 실행**: 테스트하기 전에 서버가 실행 중인지 확인하세요.

2. **포트 확인**: 기본 포트는 19901입니다. 다른 포트를 사용하는 경우 URL을 수정하세요.

3. **재부팅 API**: `/api/reboot` 엔드포인트는 테스트에서 제외되었습니다. 실제로 시스템을 재부팅하므로 수동으로 테스트할 때 주의하세요.

## 🔧 문제 해결

### 서버 연결 실패
- 서버가 실행 중인지 확인
- 포트 번호 확인 (19901)
- 방화벽 설정 확인

### 권한 오류
```bash
chmod +x test_api.sh
```

### Python 패키지 설치
```bash
pip install requests
```

## 📝 테스트 결과 파일

`test_api.py`를 실행하면 `api_test_results.json` 파일이 생성됩니다. 이 파일에는 모든 테스트 결과가 상세히 기록됩니다.

## 🌐 웹 브라우저에서 테스트

API 문서를 웹 브라우저에서 확인할 수 있습니다:
- Swagger UI: http://localhost:19901/docs
- ReDoc: http://localhost:19901/redoc