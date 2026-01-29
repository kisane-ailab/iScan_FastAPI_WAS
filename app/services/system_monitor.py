"""
iScan Instance 모니터링 서비스
n분마다 iScan Instance 상태와 연결된 iScan 장비 정보를 텔레그램으로 전송합니다.
"""

import asyncio
import os
import re
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
from app.services.telegram_worker import bot_manager
from app.core.config import get_server_monitoring_config
from app.core.constants import VERSION
from app.api.control import get_all_edge_servers


class ServerMonitoringService:
    """iScan Instance 모니터링 서비스"""
    
    def __init__(self):
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.config = {
            "interval_min": 60,  # 기본 1시간
            "chat_id": ""  # iScan Instance 모니터링용 그룹방 chat_id
        }
    
    def start(self):
        """모니터링 서비스 시작"""
        if self.is_running:
            print("⚠️ iScan Instance 모니터링 서비스가 이미 실행 중")
            return
        
        # config.json에서 설정 읽기
        monitoring_config = get_server_monitoring_config()
        self.config.update(monitoring_config)
        
        # chat_id가 설정되지 않았으면 시작하지 않음
        if not self.config.get("chat_id"):
            print("⚠️ iScan Instance 모니터링 chat_id가 설정되지 않아 모니터링을 시작하지 않음")
            return
        
        self.is_running = True
        print(f"🚀 iScan Instance 모니터링 시작 (간격: {self.config['interval_min']}분)")
        
        # 모니터링 루프 시작
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """모니터링 루프 (n분마다 리포트 전송)"""
        # 즉시 첫 리포트 전송
        await self.send_monitoring_report()
        
        # 이후 주기적으로 리포트 전송
        interval_seconds = self.config["interval_min"] * 60
        
        while self.is_running:
            await asyncio.sleep(interval_seconds)
            if self.is_running:
                await self.send_monitoring_report()
    
    async def send_monitoring_report(self):
        """iScan Instance 모니터링 리포트 전송"""
        try:
            # 연결된 EdgeMan 정보 수집
            edge_servers = get_all_edge_servers()
            
            # 리포트 메시지 생성
            report = self._generate_report(edge_servers)
            
            # 텔레그램으로 전송
            chat_id = self.config.get("chat_id")
            if not chat_id:
                print("⚠️ iScan Instance 모니터링 chat_id가 설정되지 않음")
                return
            
            sent = False
            for bot_name, bot_config in bot_manager.bots.items():
                try:
                    await bot_manager._send_message(
                        f"https://api.telegram.org/bot{bot_config['token']}",
                        str(chat_id),
                        report
                    )
                    print(f"📊 iScan Instance 모니터링 리포트 전송 완료: {bot_name} -> {chat_id}")
                    sent = True
                    break
                except Exception as bot_error:
                    print(f"❌ iScan Instance 모니터링 리포트 전송 실패 ({bot_name}): {bot_error}")
                    continue
            
            if not sent:
                print("❌ 모든 봇에서 iScan Instance 모니터링 리포트 전송 실패")
            
        except Exception as err:
            print(f"❌ iScan Instance 모니터링 리포트 전송 실패: {err}")
    
    def _format_time_ago(self, minutes: int) -> str:
        """분을 일, 시간, 분 형식으로 변환 (예: 61분 → "1시간 1분", 1500분 → "1일 1시간")"""
        if minutes < 60:
            return f"{minutes}분"
        
        # 24시간(1440분) 이상이면 일 단위 추가
        if minutes >= 1440:
            days = minutes // 1440
            remaining_minutes = minutes % 1440
            
            hours = remaining_minutes // 60
            remaining_minutes_after_hours = remaining_minutes % 60
            
            parts = [f"{days}일"]
            
            if hours > 0:
                parts.append(f"{hours}시간")
            
            if remaining_minutes_after_hours > 0:
                parts.append(f"{remaining_minutes_after_hours}분")
            
            return " ".join(parts)
        
        # 60분 이상 1440분 미만: 시간과 분
        hours = minutes // 60
        remaining_minutes = minutes % 60
        
        if remaining_minutes == 0:
            return f"{hours}시간"
        else:
            return f"{hours}시간 {remaining_minutes}분"
    
    def _get_docker_name(self) -> str:
        """경로에서 도커 컨테이너 이름 추출"""
        try:
            pattern = r'/docker/([^/]+)/iScan_FastAPI_WAS'
            docker_name = None
            
            # /proc/self/mountinfo에서 마운트 정보 확인
            mountinfo_path = "/proc/self/mountinfo"
            if os.path.exists(mountinfo_path):
                try:
                    with open(mountinfo_path, 'r') as f:
                        for line in f:
                            if '/iScan_FastAPI_WAS' in line:
                                match = re.search(pattern, line)
                                if match:
                                    docker_name = match.group(1)
                                    break
                except Exception:
                    pass
            
            if not docker_name:
                current_path = str(Path(__file__).resolve())
                match = re.search(pattern, current_path)
                if match:
                    docker_name = match.group(1)
            
            if not docker_name:
                cwd = os.getcwd()
                match = re.search(pattern, cwd)
                if match:
                    docker_name = match.group(1)
            
            if not docker_name:
                return "unknown"
            
            if docker_name.startswith("iScanInstance."):
                docker_name = docker_name[len("iScanInstance."):]
            
            return docker_name
        except Exception:
            return "unknown"
    
    def _generate_report(self, edge_servers: Dict[str, dict]) -> str:
        """모니터링 리포트 메시지 생성"""
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        docker_name = self._get_docker_name()
        
        message = f"🔍 iScan Instance 모니터링 🔍\n\n"
        message += f"[ {docker_name} ] - v{VERSION}\n\n"
        
        # 연결된 iScan 정보 섹션
        edge_count = len(edge_servers)
        status_emoji = "✅" if edge_count > 0 else "❌"
        message += f"{status_emoji} 연결된 iScan 장비 ({edge_count}대)\n\n"
        
        if edge_servers:
            # 알파벳 순 정렬
            sorted_servers = sorted(edge_servers.items())
            
            for idx, (server_id, status) in enumerate(sorted_servers, 1):
                try:
                    vendor, db_key = server_id.split("/", 1) if "/" in server_id else (server_id, "")
                    
                    last_seen = status.get("lastSeen", 0)
                    if last_seen and last_seen > 0:
                        current_time_ms = int(datetime.now().timestamp() * 1000)
                        minutes_ago = max(0, int((current_time_ms - last_seen) / (1000 * 60)))
                    else:
                        minutes_ago = 0  # lastSeen이 없거나 0이면 0분으로 표시
                    
                    time_ago_str = self._format_time_ago(minutes_ago)
                    
                    message += f"[{idx}]\n"
                    message += f"• 매장 정보 : {vendor}\n"
                    message += f"• 장비 정보 : {db_key}\n"
                    
                    serial_number = status.get("serialNumber", "")
                    if serial_number:
                        message += f"• 장비 번호 : {serial_number}\n"
                    else:
                        message += f"• 장비 번호 : -\n"
                    
                    try:
                        scan_count = status.get("totalScanCount")
                        if scan_count is not None:
                            scan_count_int = int(scan_count) if isinstance(scan_count, (int, str)) else 0
                            message += f"• 스캔 수: {scan_count_int:,}회\n"
                        else:
                            message += f"• 스캔 수: 0회\n"
                    except (ValueError, TypeError):
                        message += f"• 스캔 수: 0회\n"
                    
                    message += f"• 활동: {time_ago_str} 전\n\n"
                    
                except Exception as e:
                    print(f"⚠️ iScan 장비 정보 처리 실패 ({server_id}): {e}")
                    continue
        
        message += f"📅 {time_str}"
        
        return message
    

server_monitoring_service = ServerMonitoringService()

