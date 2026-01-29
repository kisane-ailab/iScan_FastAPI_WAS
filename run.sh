#!/bin/bash

# iScan FastAPI WAS 실행 스크립트 (config.json 기반 설정)

echo "🚀 Starting iScan FastAPI WAS..."

# config.json 파일 확인
if [ ! -f config.json ]; then
    echo "❌ config.json file not found. Please create config.json with proper configuration."
    exit 1
fi

# 설정 테스트
#echo "🔧 Testing configuration..."
#python test_config.py

# 로그 디렉토리 생성
mkdir -p logs

# 운영 환경에서 uvicorn으로 실행
echo "📡 Starting with uvicorn..."
python -m uvicorn app.main:app --host 0.0.0.0 --port 50000