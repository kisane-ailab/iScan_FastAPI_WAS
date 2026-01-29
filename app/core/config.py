import os
import json
import re
from typing import Optional, Dict, List

# 설정 파일 경로
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')

def _remove_comments(json_str: str) -> str:
    """JSON 문자열에서 주석을 제거합니다."""
    # // 주석 제거 (줄 끝까지)
    json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
    
    # /* */ 주석 제거 (여러 줄)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # 빈 줄 제거
    lines = json_str.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:  # 빈 줄이 아닌 경우만 유지
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def _read_json_with_comments(file_path: str) -> Dict:
    """주석이 포함된 JSON 파일을 읽습니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 주석 제거
        cleaned_content = _remove_comments(content)
        
        # JSON 파싱
        return json.loads(cleaned_content)
    except Exception as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        raise

# Telegram 설정 - JSON 파일 기반
def get_telegram_bots() -> Dict[str, Dict]:
    """Telegram 봇 설정을 반환합니다."""
    return _read_bots_from_config_file()

def _read_bots_from_config_file() -> Dict[str, Dict]:
    """JSON 설정 파일에서 봇 설정을 읽습니다."""
    try:
        if not os.path.exists(CONFIG_FILE):
            print(f"Warning: 설정 파일이 없습니다: {CONFIG_FILE}")
            return {}
        
        config = _read_json_with_comments(CONFIG_FILE)
        return config.get('telegram_bots', {})
    except Exception as e:
        print(f"Warning: 설정 파일을 읽을 수 없습니다: {e}")
        return {}

def _get_config_value(key: str, default=None):
    """config.json에서 설정값을 가져옵니다."""
    try:
        if not os.path.exists(CONFIG_FILE):
            return default
        
        config = _read_json_with_comments(CONFIG_FILE)
        
        # server 섹션에서 찾기
        if key in config.get('server', {}):
            return config['server'][key]
        
        # client 섹션에서 찾기
        if key in config.get('client', {}):
            return config['client'][key]
        
        # system 섹션에서 찾기
        if key in config.get('system', {}):
            return config['system'][key]
        
        # 최상위에서 찾기
        return config.get(key, default)
    except Exception:
        return default

# 시스템 설정
BASE_409_WAIT_MS = _get_config_value('base_409_wait_ms', 3000)
MAX_CONSECUTIVE_409 = _get_config_value('max_consecutive_409', 5)

# 서버 설정
HOST = _get_config_value('host', '0.0.0.0')
PORT = _get_config_value('port', 19901)
DEBUG = _get_config_value('debug', False)

def validate_config() -> bool:
    """필수 설정 검증"""
    bots = get_telegram_bots()
    
    if not bots:
        raise ValueError("No Telegram bots configured. Please create config.json with telegram_bots section")
    
    for bot_name, bot_config in bots.items():
        if not bot_config.get("token") or bot_config.get("token") == "your_bot_token_here":
            raise ValueError(f"Missing or invalid token for bot: {bot_name}. Please set a valid Telegram bot token.")
        if not bot_config.get("chat_ids") or bot_config.get("chat_ids") == ["your_chat_id_here"]:
            raise ValueError(f"No valid chat IDs configured for bot: {bot_name}. Please set valid chat IDs.")
    
    return True

def get_all_chat_ids() -> List[str]:
    """모든 봇의 모든 채팅 ID를 반환합니다."""
    chat_ids = []
    bots = get_telegram_bots()
    for bot_config in bots.values():
        chat_ids.extend(bot_config.get("chat_ids", []))
    return list(set(chat_ids))  # 중복 제거

def get_bot_by_chat_id(chat_id: str) -> Optional[Dict]:
    """채팅 ID로 봇 설정을 찾습니다."""
    bots = get_telegram_bots()
    for bot_name, bot_config in bots.items():
        if chat_id in bot_config.get("chat_ids", []):
            return {"name": bot_name, **bot_config}
    return None

def get_bot_by_token(token: str) -> Optional[Dict]:
    """토큰으로 봇 설정을 찾습니다."""
    bots = get_telegram_bots()
    for bot_name, bot_config in bots.items():
        if bot_config.get("token") == token:
            return {"name": bot_name, **bot_config}
    return None

def get_server_monitoring_config() -> Dict:
    """iScan Instance 모니터링 설정을 반환합니다."""
    try:
        if not os.path.exists(CONFIG_FILE):
            return {"chat_id": "", "interval_min": 60}
        
        config = _read_json_with_comments(CONFIG_FILE)
        server_config = config.get('server', {})
        
        return {
            "chat_id": server_config.get("chat_id", ""),
            "interval_min": server_config.get("interval_min", 60)
        }
    except Exception as e:
        print(f"Warning: iScan Instance 모니터링 설정을 읽을 수 없습니다: {e}")
        return {"chat_id": "", "interval_min": 60}

def get_database_config() -> Dict:
    """데이터베이스 설정을 반환합니다."""
    try:
        if not os.path.exists(CONFIG_FILE):
            return {
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "",
                "database": "iscan_test",
                "charset": "utf8mb4",
                "pool_size": 10
            }
        
        config = _read_json_with_comments(CONFIG_FILE)
        db_config = config.get('database', {})
        
        return {
            "host": db_config.get("host", "localhost"),
            "port": db_config.get("port", 3306),
            "user": db_config.get("user", "root"),
            "password": db_config.get("password", ""),
            "database": db_config.get("database", "iscan_test"),
            "charset": db_config.get("charset", "utf8mb4"),
            "pool_size": db_config.get("pool_size", 10)
        }
    except Exception as e:
        print(f"Warning: 데이터베이스 설정을 읽을 수 없습니다: {e}")
        return {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "",
            "database": "iscan_test",
            "charset": "utf8mb4",
            "pool_size": 10
        }
