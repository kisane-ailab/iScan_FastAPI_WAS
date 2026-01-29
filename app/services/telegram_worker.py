import aiohttp
import asyncio
import time
import os
from typing import Dict, List, Optional
from app.core.config import get_telegram_bots, get_bot_by_chat_id, BASE_409_WAIT_MS, MAX_CONSECUTIVE_409
# from app.core.logger import telegram_logger
from app.services.system_control import SystemControlError, system_control_service
from app.core.constants import HELP_TEXT

class TelegramBotManager:
    """다중 Telegram 봇을 관리하는 클래스"""
    
    def __init__(self):
        self.bots = get_telegram_bots()
        self.last_update_ids = {bot_name: 0 for bot_name in self.bots.keys()}
        self.consecutive_409 = {bot_name: 0 for bot_name in self.bots.keys()}
        self.processed_commands = {}
        
    async def start_all_bots(self):
        """모든 봇의 폴링을 시작합니다."""
        print(f"🤖 {len(self.bots)}개의 텔레그램 봇을 시작합니다...")
        
        tasks = []
        for bot_name in self.bots.keys():
            task = asyncio.create_task(self._poll_bot(bot_name))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def _poll_bot(self, bot_name: str):
        """개별 봇의 폴링 루프"""
        bot_config = self.bots[bot_name]
        token = bot_config["token"]
        api_url = f"https://api.telegram.org/bot{token}"
        
        print(f"🤖 봇 폴링 시작: {bot_name}")
        
        while True:
            params = {}
            if self.last_update_ids[bot_name]:
                params["offset"] = self.last_update_ids[bot_name] + 1

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{api_url}/getUpdates", params=params, timeout=100) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            if result.get("ok"):
                                self.consecutive_409[bot_name] = 0
                                updates = result.get("result", [])
                                print(f"봇 {bot_name}: {len(updates)}개의 업데이트를 받았습니다")
                                
                                for update in updates:
                                    await self._handle_update(update, bot_name, api_url)
                        elif resp.status == 409:
                            self.consecutive_409[bot_name] += 1
                            wait = min(BASE_409_WAIT_MS * self.consecutive_409[bot_name], 300_000)
                            print(f"봇 {bot_name}: [409] 충돌 발생. {wait}ms 대기")
                            await asyncio.sleep(wait / 1000)
                        elif resp.status == 429:
                            print(f"봇 {bot_name}: [429] 속도 제한. 10초 대기")
                            await asyncio.sleep(10)
                        else:
                            print(f"봇 {bot_name}: 알 수 없는 상태 {resp.status}")
                            await asyncio.sleep(5)
            except asyncio.TimeoutError:
                print(f"봇 {bot_name}: API 타임아웃")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"봇 {bot_name}: 폴링 오류: {e}")
                await asyncio.sleep(3)

    async def _handle_update(self, update: Dict, bot_name: str, api_url: str):
        """
        {
          "ok": true,
          "result": [
            {
              "update_id": 956914109,
              "message": {
                "message_id": 10061,
                "from": {
                  "id": 7858224631,
                  "is_bot": false,
                  "first_name": "용제",
                  "last_name": "홍"
                },
                "chat": {
                  "id": -1002209883790,
                  "title": "yjhong_test|yjhong_test|localhost",
                  "type": "supergroup"
                },
                "date": 1753774644,
                "text": "test"
              }
            }
          ]
        }
        """
        
        """Telegram 업데이트 처리"""
        message = update.get("message")
        if not message:
            return

        chat_id = str(message.get("chat", {}).get("id"))
        text = message.get("text", "")
        user_id = str(message.get("from", {}).get("id", ""))
        
        # first_name과 last_name을 조합하여 username 생성
        from_user = message.get("from", {})
        first_name = from_user.get("first_name", "")
        last_name = from_user.get("last_name", "")
        username = f"{first_name} {last_name}".strip() if first_name or last_name else "알 수 없음"
        
        chat_title = message.get("chat", {}).get("title", "개인 채팅")

        # 명령어가 아니면 무시
        if not text.startswith("/"):
            return

        # 해당 봇의 채팅 ID인지 확인
        bot_config = self.bots[bot_name]
        if chat_id not in bot_config.get("chat_ids", []):
            print(f"봇 {bot_name}: 권한이 없는 채팅 {chat_id}에서 온 메시지를 무시했습니다")
            return

        command_key = f"{text}:{user_id}:{bot_name}"
        now = int(time.time() * 1000)
        if command_key in self.processed_commands and now - self.processed_commands[command_key] < 5000:
            print(f"봇 {bot_name}: 중복 명령어 무시: {text}")
            return
        self.processed_commands[command_key] = now

        # 명령어 로깅
        print(f"📥 봇: {bot_name} | 챗제목: {chat_title} | 챗ID: {chat_id} | 사용자: {username} | 명령어: {text}")

        # chat_title 을 파싱해서 동작 하도록 수정
        # chat_description|public_ip
        try:
            if "|" in chat_title:
                parts = chat_title.split("|")
                if len(parts) == 2:
                    chat_description, public_ip = parts
                    # 각 파트가 비어있거나 공백만 있는지 검사
                    if chat_description.strip():
                        # public_ip는 빈 값이어도 허용 (기본값 사용)
                        if not public_ip.strip():
                            public_ip = "localhost"
                        print(f"✅ 파싱 성공: chat_description={chat_description}, public_ip={public_ip}")
                        
                        # API 호출
                        user_info = {
                            "user_id": user_id,
                            "username": username,
                            "bot_name": bot_name,
                            "chat_id": chat_id
                        }
                        
                        # 명령어에 따른 API 호출
                        if text == "/help":
                            help_text = f"🤖 iScan FastAPI WAS\n\n{HELP_TEXT}"
                            await self._send_message(api_url, chat_id, help_text)
                        else:
                            # /help 이외의 명령어를 API로 전송
                            result = await system_control_service.parse_command(
                                chat_description=chat_description,
                                public_ip=public_ip,
                                command=text,
                                user_info=user_info
                            )
                            
                            if result["success"]:
                                # 포맷된 메시지가 있으면 사용, 없으면 기본 메시지 사용
                                if "message" in result:
                                    response_message = result["message"]
                                    #print(f"📤 포맷된 메시지 사용: \n{response_message[:100]}...")
                                else:
                                    response_message = f"✅ 명령 실행 완료: {text}\n{result.get('data', {}).get('message', '')}"
                                    #print(f"📤 기본 메시지 사용: \n{response_message}")
                                
                                print(f"📤 텔레그램 메시지 전송: \n{response_message}")
                                await self._send_message(api_url, chat_id, response_message)
                            else:
                                response_message = f"❌ 명령 실행 실패: {result.get('error', '알 수 없는 오류')}"
                                print(f"❌ 오류 메시지: \n{response_message}")
                                await self._send_message(api_url, chat_id, response_message)
                    else:
                        print(f"❌ chat_title에 빈 값이 포함됨: {chat_title} (chat_description='{chat_description}', public_ip='{public_ip}')")
                        chat_description, public_ip = "unknown", "unknown"
                else:
                    print(f"❌ chat_title 형식 오류: {chat_title} (필요: 2개 파트, 실제: {len(parts)}개)")
                    chat_description, public_ip = "unknown", "unknown"
            else:
                print(f"❌ chat_title에 구분자 '|' 없음: {chat_title}")
                chat_description, public_ip = "unknown", "unknown"
        except Exception as e:
            print(f"❌ chat_title 파싱 중 오류: {e}, chat_title: {chat_title}")
            chat_description, public_ip = "unknown", "unknown"

        # 파싱된 값이 유효하지 않은 경우 기본 응답
        if chat_description == "unknown" or public_ip == "unknown":
            await self._send_message(api_url, chat_id, "❌ 채팅방 제목 형식이 올바르지 않습니다. (형식: chat_description|public_ip)")
            self.last_update_ids[bot_name] = update["update_id"]
            return

        # update_id 업데이트 (모든 처리 완료 후)
        self.last_update_ids[bot_name] = update["update_id"]

    async def _send_message(self, api_url: str, chat_id: str, text: str):
        """Telegram 메시지 전송 (4096자 제한 적용)"""
        try:
            # 텔레그램 메시지 최대 길이 (4096자)
            MAX_MESSAGE_LENGTH = 4096
            
            if len(text) <= MAX_MESSAGE_LENGTH:
                # 단일 메시지로 전송
                await self._send_single_message(api_url, chat_id, text)
            else:
                # 긴 메시지를 분할하여 전송
                await self._send_split_messages(api_url, chat_id, text, MAX_MESSAGE_LENGTH)
                
        except Exception as e:
            print(f"메시지 전송 중 오류: {e}")
    
    async def _send_single_message(self, api_url: str, chat_id: str, text: str):
        """단일 메시지 전송"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{api_url}/sendMessage", json={
                    "chat_id": chat_id,
                    "text": text
                }) as resp:
                    if resp.status != 200:
                        print(f"메시지 전송 실패: {resp.status}")
                    else:
                        print("메시지가 성공적으로 전송되었습니다")
        except Exception as e:
            print(f"단일 메시지 전송 중 오류: {e}")
    
    async def _send_photo(self, api_url: str, chat_id: str, photo_path: str, caption: str = ""):
        """Telegram 사진 전송"""
        try:
            if not os.path.exists(photo_path):
                print(f"이미지 파일이 존재하지 않습니다: {photo_path}")
                return False
            
            async with aiohttp.ClientSession() as session:
                with open(photo_path, 'rb') as photo_file:
                    form_data = aiohttp.FormData()
                    form_data.add_field('chat_id', str(chat_id))
                    form_data.add_field('photo', photo_file, filename=os.path.basename(photo_path))
                    if caption:
                        form_data.add_field('caption', caption)
                    
                    async with session.post(f"{api_url}/sendPhoto", data=form_data) as resp:
                        if resp.status == 200:
                            print(f"이미지 전송 성공: {os.path.basename(photo_path)}")
                            return True
                        else:
                            error_text = await resp.text()
                            print(f"이미지 전송 실패: {resp.status}, {error_text}")
                            return False
        except Exception as e:
            print(f"이미지 전송 중 오류: {e}")
            return False
    
    async def _send_split_messages(self, api_url: str, chat_id: str, text: str, max_length: int):
        """긴 메시지를 분할하여 전송"""
        try:
            # 메시지를 줄 단위로 분할
            lines = text.split('\n')
            current_message = ""
            message_count = 0
            messages = []  # 모든 메시지를 저장할 리스트
            
            for line in lines:
                # 현재 줄을 추가했을 때 길이 확인
                test_message = current_message + line + '\n' if current_message else line + '\n'
                
                if len(test_message) > max_length:
                    # 현재 메시지가 최대 길이를 초과하면 저장
                    if current_message:
                        message_count += 1
                        messages.append(current_message)
                        current_message = line + '\n'
                    else:
                        # 단일 줄이 너무 길면 강제로 분할
                        chunks = [line[i:i+max_length-10] for i in range(0, len(line), max_length-10)]
                        for chunk in chunks:
                            message_count += 1
                            messages.append(chunk)
                else:
                    current_message = test_message
            
            # 마지막 메시지 추가
            if current_message:
                message_count += 1
                messages.append(current_message)
            
            # 총 메시지 수 계산 후 전송
            total_messages = len(messages)
            for i, message in enumerate(messages, 1):
                header = f"📄 메시지 {i}/{total_messages}\n"
                await self._send_single_message(api_url, chat_id, header + message)
            
            print(f"긴 메시지를 {total_messages}개로 분할하여 전송 완료")
            
        except Exception as e:
            print(f"분할 메시지 전송 중 오류: {e}")

    async def send_message_async(self, text: str, chat_id: str = None):
        """공개 메시지 전송 메서드"""
        try:
            # chat_id가 제공되지 않으면 첫 번째 봇의 첫 번째 채팅 ID 사용
            if not chat_id:
                if not self.bots:
                    print("사용 가능한 봇이 없습니다")
                    return False
                
                # 첫 번째 봇의 첫 번째 채팅 ID 사용
                first_bot_name = list(self.bots.keys())[0]
                first_bot = self.bots[first_bot_name]
                chat_ids = first_bot.get("chat_ids", [])
                if not chat_ids:
                    print("사용 가능한 채팅 ID가 없습니다")
                    return False
                chat_id = chat_ids[0]
            
            # 해당 chat_id에 매핑된 봇 찾기
            target_bot_name = None
            for bot_name, bot_config in self.bots.items():
                if chat_id in bot_config.get("chat_ids", []):
                    target_bot_name = bot_name
                    break
            
            if not target_bot_name:
                print(f"chat_id {chat_id}에 매핑된 봇을 찾을 수 없습니다")
                return False
            
            # 봇의 API URL 구성
            bot_config = self.bots[target_bot_name]
            token = bot_config["token"]
            api_url = f"https://api.telegram.org/bot{token}"
            
            # 메시지 전송
            await self._send_message(api_url, chat_id, text)
            return True
            
        except Exception as e:
            print(f"메시지 전송 중 오류: {e}")
            return False

# 전역 봇 매니저 인스턴스
bot_manager = TelegramBotManager()

async def telegram_poll_loop():
    """Telegram 봇 폴링 루프 (다중 봇 지원)"""
    await bot_manager.start_all_bots()
