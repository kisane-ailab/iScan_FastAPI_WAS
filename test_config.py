#!/usr/bin/env python3
"""
TELEGRAM_BOTS 설정 테스트 스크립트
"""

import os
import sys
import json

# app 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from core.config import get_telegram_bots, validate_config

def create_config_file():
    """예제 config.json 파일을 생성합니다."""
    config_content = {
        "telegram_bots": {
            "test_bot": {
                "token": "your_bot_token_here",
                "chat_ids": ["your_chat_id_here"],
                "description": "테스트용 봇"
            }
        },
        "server": {
            "host": "0.0.0.0",
            "port": 50000,
            "debug": False
        },
        "system": {
            "base_409_wait_ms": 3000,
            "max_consecutive_409": 5
        }
    }
    
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not os.path.exists(config_path):
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_content, f, indent=2, ensure_ascii=False)
            print(f"✅ config.json 파일이 생성되었습니다: {config_path}")
            return True
        except Exception as e:
            print(f"❌ config.json 파일 생성 실패: {e}")
            return False
    else:
        print(f"ℹ️  config.json 파일이 이미 존재합니다: {config_path}")
        return True

def test_config():
    """설정을 테스트합니다."""
    print("=== TELEGRAM_BOTS 설정 테스트 ===")
    
    # config.json 파일 확인 및 생성
    config_created = create_config_file()
    
    # 설정 가져오기
    bots = get_telegram_bots()
    print(f"\n파싱된 봇 설정:")
    print(json.dumps(bots, indent=2, ensure_ascii=False))
    
    # 설정 검증
    try:
        validate_config()
        print("\n✅ 설정이 유효합니다!")
    except ValueError as e:
        print(f"\n❌ 설정 오류: {e}")
        print("\n💡 해결 방법:")
        print("1. config.json 파일을 수정하여 실제 Telegram 봇 토큰과 채팅 ID를 설정하세요.")
        print("\n📝 config.json 파일 예시:")
        print('{')
        print('  "telegram_bots": {')
        print('    "my_bot": {')
        print('      "token": "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz",')
        print('      "chat_ids": ["123456789", "-987654321"],')
        print('      "description": "내 봇"')
        print('    }')
        print('  },')
        print('  "server": {')
        print('    "host": "0.0.0.0",')
        print('    "port": 50000,')
        print('    "debug": false')
        print('  },')
        print('  "system": {')
        print('    "base_409_wait_ms": 3000,')
        print('    "max_consecutive_409": 5')
        print('  }')
        print('}')
    
    # 각 봇 정보 출력
    print(f"\n=== 봇 상세 정보 ===")
    for bot_name, bot_config in bots.items():
        print(f"\n봇 이름: {bot_name}")
        print(f"  토큰: {bot_config.get('token', 'N/A')}")
        print(f"  채팅 ID: {bot_config.get('chat_ids', [])}")
        print(f"  설명: {bot_config.get('description', 'N/A')}")

if __name__ == "__main__":
    test_config()