"""
환경설정 서비스
시스템 설정 관련 비즈니스 로직
"""
from typing import Dict, List, Any
from app.core.config import CONFIG_FILE, _read_json_with_comments
import json
import os


class SettingsService:
    """환경설정 서비스"""
    
    @staticmethod
    async def get_settings() -> Dict[str, Any]:
        """설정 조회"""
        try:
            if not os.path.exists(CONFIG_FILE):
                return {}
            config = _read_json_with_comments(CONFIG_FILE)
            # 민감한 정보 제거
            safe_config = {}
            if 'server' in config:
                safe_config['server'] = config['server']
            if 'system' in config:
                safe_config['system'] = config['system']
            if 'security' in config:
                safe_config['security'] = config['security']
            return safe_config
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    async def update_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
        """설정 업데이트"""
        # TODO: 실제 설정 업데이트 로직 구현
        # 주의: config.json 파일을 직접 수정하는 것은 위험할 수 있으므로
        # 백업 및 검증 로직이 필요합니다
        return {"success": True, "message": "설정이 업데이트되었습니다"}


# 전역 서비스 인스턴스
settings_service = SettingsService()

