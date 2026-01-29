import asyncio
import os
import subprocess
import platform
import psutil
import time
from typing import Optional, Dict, Any
# from app.core.logger import system_logger
from app.core.constants import HELP_TEXT

class SystemControlError(Exception):
    """시스템 제어 관련 예외"""
    pass

class SystemControlService:
    """시스템 제어 서비스 클래스"""
    
    def __init__(self):
        self.supported_commands = {
            "/status": self._get_system_status,
            "/uptime": self._get_uptime_status,
            "/info": self._get_detailed_info,
            "/help": self._get_help_info
        }
    
    async def parse_command(self, chat_description: str, public_ip: str, 
                          command: str, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        텔레그램 명령어를 파싱하고 처리
        
        Args:
            vendor: 벤더 정보
            db_key: 데이터베이스 키
            public_ip: 공인 IP 주소
            command: 실행할 명령어
            user_info: 사용자 정보
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            print(f"명령어 파싱 시작: {command} (chat_description={chat_description}, public_ip={public_ip})")
            
            # 명령어 정리 (앞뒤 공백 제거, 소문자 변환)
            clean_command = command.strip().lower()
            
            # 지원되는 명령어인지 확인
            if clean_command in self.supported_commands:
                # 해당 명령어 처리 함수 호출
                handler = self.supported_commands[clean_command]
                result = await handler(chat_description, public_ip, user_info)
                
                print(f"명령어 처리 완료: {command}")
                return {
                    "success": True,
                    "message": result.get("message", f"✅ 명령 실행 완료: {command}"),
                    "data": result.get("data", {}),
                    "command": command
                }
            else:
                # 지원되지 않는 명령어
                error_msg = f"❌ 지원되지 않는 명령어입니다: {command}\n\n사용 가능한 명령어:\n"
                error_msg += "\n".join([f"• {cmd}" for cmd in self.supported_commands.keys()])
                
                print(f"지원되지 않는 명령어: {command}")
                return {
                    "success": False,
                    "error": error_msg,
                    "command": command
                }
                
        except Exception as e:
            print(f"명령어 파싱 중 오류: {e}")
            return {
                "success": False,
                "error": f"❌ 명령 처리 중 오류가 발생했습니다: {str(e)}",
                "command": command
            }
    
    async def _get_uptime(self) -> str:
        """시스템 업타임 정보 조회 (크로스 플랫폼)"""
        try:
            system = platform.system().lower()
            
            if system == "windows":
                # Windows에서 업타임 조회
                try:
                    proc = await asyncio.create_subprocess_shell(
                        "net statistics server | findstr /C:\"Statistics since\"",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    
                    if proc.returncode == 0:
                        result = stdout.decode().strip()
                        print("Windows uptime retrieved successfully")
                        return result
                except Exception as e:
                    print(f"Windows net statistics failed: {e}")
                
                # 대안: PowerShell 사용
                try:
                    proc = await asyncio.create_subprocess_shell(
                        "powershell -Command \"(Get-Date) - (Get-CimInstance Win32_OperatingSystem).LastBootUpTime\"",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    
                    if proc.returncode == 0:
                        result = stdout.decode().strip()
                        print("Windows uptime retrieved via PowerShell")
                        return f"Windows uptime: {result}"
                except Exception as e:
                    print(f"Windows PowerShell uptime failed: {e}")
                
                # psutil 사용 (Windows fallback)
                boot_time = psutil.boot_time()
                uptime_seconds = time.time() - boot_time
                days = int(uptime_seconds // 86400)
                hours = int((uptime_seconds % 86400) // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                print("Windows uptime retrieved via psutil")
                return f"Windows uptime: {days} days, {hours} hours, {minutes} minutes"
            else:
                # Linux/Unix에서 업타임 조회
                try:
                    proc = await asyncio.create_subprocess_shell(
                        "uptime",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    
                    if proc.returncode == 0:
                        result = stdout.decode().strip()
                        print("Linux uptime retrieved successfully")
                        return result
                except Exception as e:
                    print(f"Linux uptime command failed: {e}")
                
                # psutil 사용 (Linux fallback)
                boot_time = psutil.boot_time()
                uptime_seconds = time.time() - boot_time
                days = int(uptime_seconds // 86400)
                hours = int((uptime_seconds % 86400) // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                print("Linux uptime retrieved via psutil")
                return f"Linux uptime: {days} days, {hours} hours, {minutes} minutes"
                
        except Exception as e:
            print(f"Error getting uptime: {e}")
            # psutil을 사용한 최종 대안
            try:
                boot_time = psutil.boot_time()
                uptime_seconds = time.time() - boot_time
                days = int(uptime_seconds // 86400)
                hours = int((uptime_seconds % 86400) // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                print("Uptime retrieved via psutil fallback")
                return f"Uptime: {days} days, {hours} hours, {minutes} minutes"
            except Exception as fallback_error:
                print(f"Fallback uptime also failed: {fallback_error}")
                raise SystemControlError(f"Failed to get uptime: {e}")



    async def _get_system_info(self) -> dict:
        """시스템 정보 조회 (크로스 플랫폼)"""
        try:
            system = platform.system().lower()
            
            if system == "windows":
                # Windows에서 시스템 정보 조회
                try:
                    # psutil을 사용한 크로스 플랫폼 방식
                    cpu_usage = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    mem_usage = memory.percent
                    
                    return {
                        "cpu_usage": f"{cpu_usage:.1f}%",
                        "memory_usage": f"{mem_usage:.1f}%",
                        "uptime": await self._get_uptime()
                    }
                except Exception as e:
                    print(f"Windows system info error: {e}")
                    return {"error": str(e)}
            else:
                # Linux/Unix에서 시스템 정보 조회
                try:
                    # CPU 사용률
                    cpu_proc = await asyncio.create_subprocess_shell(
                        "top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    cpu_stdout, _ = await cpu_proc.communicate()
                    cpu_usage = cpu_stdout.decode().strip() if cpu_stdout else "N/A"
                    
                    # 메모리 사용률
                    mem_proc = await asyncio.create_subprocess_shell(
                        "free | grep Mem | awk '{printf \"%.1f\", $3/$2 * 100.0}'",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    mem_stdout, _ = await mem_proc.communicate()
                    mem_usage = mem_stdout.decode().strip() if mem_stdout else "N/A"
                    
                    # 명령어가 실패하면 psutil 사용
                    if cpu_usage == "N/A" or mem_usage == "N/A":
                        cpu_usage = psutil.cpu_percent(interval=1)
                        memory = psutil.virtual_memory()
                        mem_usage = memory.percent
                        return {
                            "cpu_usage": f"{cpu_usage:.1f}%",
                            "memory_usage": f"{mem_usage:.1f}%",
                            "uptime": await self._get_uptime()
                        }
                    
                    return {
                        "cpu_usage": f"{cpu_usage}%",
                        "memory_usage": f"{mem_usage}%",
                        "uptime": await self._get_uptime()
                    }
                except Exception as e:
                    print(f"Linux system info error: {e}")
                    # psutil을 사용한 대안
                    try:
                        cpu_usage = psutil.cpu_percent(interval=1)
                        memory = psutil.virtual_memory()
                        mem_usage = memory.percent
                        return {
                            "cpu_usage": f"{cpu_usage:.1f}%",
                            "memory_usage": f"{mem_usage:.1f}%",
                            "uptime": await self._get_uptime()
                        }
                    except Exception as fallback_error:
                        print(f"Fallback system info also failed: {fallback_error}")
                        return {"error": str(e)}
            
        except Exception as e:
            print(f"Error getting system info: {e}")
            return {"error": str(e)}
    
    async def _get_system_status(self, chat_description: str, public_ip: str, 
                                user_info: Dict[str, Any]) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            system_info = await self._get_system_info()
            
            message = f"🖥️ **시스템 상태**\n\n"
            message += f"**채팅방 제목:** {chat_description}\n"
            message += f"**Public IP:** {public_ip}\n"
            message += f"**CPU 사용률:** {system_info.get('cpu_usage', 'N/A')}\n"
            message += f"**메모리 사용률:** {system_info.get('memory_usage', 'N/A')}\n"
            message += f"**업타임:** {system_info.get('uptime', 'N/A')}"
            
            return {
                "message": message,
                "data": system_info
            }
        except Exception as e:
            print(f"시스템 상태 조회 오류: {e}")
            raise
    
    async def _get_uptime_status(self, chat_description: str, public_ip: str, 
                                user_info: Dict[str, Any]) -> Dict[str, Any]:
        """업타임 정보 조회"""
        try:
            uptime = await self._get_uptime()
            
            message = f"⏰ **시스템 업타임**\n\n"
            message += f"**채팅방 제목:** {chat_description}\n"
            message += f"**Public IP:** {public_ip}\n"
            message += f"**업타임:** {uptime}"
            
            return {
                "message": message,
                "data": {"uptime": uptime}
            }
        except Exception as e:
            print(f"업타임 조회 오류: {e}")
            raise
    

    
    async def _get_detailed_info(self, chat_description: str, public_ip: str, 
                                user_info: Dict[str, Any]) -> Dict[str, Any]:
        """상세 시스템 정보 조회"""
        try:
            system_info = await self._get_system_info()
            
            message = f"📊 **상세 시스템 정보**\n\n"
            message += f"**채팅방 제목:** {chat_description}\n"
            message += f"**Public IP:** {public_ip}\n"
            message += f"**CPU 사용률:** {system_info.get('cpu_usage', 'N/A')}\n"
            message += f"**메모리 사용률:** {system_info.get('memory_usage', 'N/A')}\n"
            message += f"**업타임:** {system_info.get('uptime', 'N/A')}\n\n"
            message += "**사용자 정보:**\n"
            message += f"• 사용자 ID: {user_info.get('id', 'N/A')}\n"
            message += f"• 사용자명: {user_info.get('username', 'N/A')}\n"
            message += f"• 권한: {'관리자' if self._is_admin_user(user_info) else '일반 사용자'}"
            
            return {
                "message": message,
                "data": {**system_info, "user_info": user_info}
            }
        except Exception as e:
            print(f"상세 정보 조회 오류: {e}")
            raise
    
    async def _get_help_info(self, chat_description: str, public_ip: str, 
                            user_info: Dict[str, Any]) -> Dict[str, Any]:
        """도움말 정보"""
        message = f"🤖 **iScan FastAPI WAS 도움말**\n\n"
        message += HELP_TEXT + "\n\n"
        message += f"**현재 시스템:**\n"
        message += f"• 채팅방 제목: {chat_description}\n"
        message += f"• Public IP: {public_ip}"
        
        return {
            "message": message,
            "data": {"help": True}
        }
    

    
    # API에서 사용할 수 있는 공개 메서드들
    async def get_system_info(self) -> dict:
        """시스템 정보 조회 (API용)"""
        return await self._get_system_info()
    
    async def get_uptime(self) -> str:
        """시스템 업타임 조회 (API용)"""
        return await self._get_uptime()
    


# 전역 인스턴스 생성
system_control_service = SystemControlService()
