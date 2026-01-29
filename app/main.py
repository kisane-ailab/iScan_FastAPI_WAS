from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from app.api import status, control
from app.core.worker import start_background_workers
from app.core.config import validate_config, HOST, PORT, DEBUG, get_telegram_bots, _get_config_value
from app.core.database import init_db_pool, close_db_pool
# from app.core.logger import logger
from app.core.constants import HELP_TEXT, VERSION, TITLE, DEFAULT_EDGEMAN_SERVER_PORT
import aiohttp
import asyncio
import atexit
import json
import sys
from datetime import datetime
import ipaddress
from typing import Optional


def get_actual_host() -> str:
    """명령줄 인자에서 실제 실행 호스트를 가져오거나, 없으면 config 값 사용"""
    for i, arg in enumerate(sys.argv):
        if arg == "--host" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        elif arg.startswith("--host="):
            return arg.split("=", 1)[1]
    return HOST


def get_actual_port() -> int:
    """명령줄 인자에서 실제 실행 포트를 가져오거나, 없으면 config 값 사용"""
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                pass
        elif arg.startswith("--port="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                pass
    return PORT

app = FastAPI(
    title=TITLE,
    description="추론 시스템 제어 기능 및 웹 서비스를 제공하는 FastAPI 기반 웹 애플리케이션",
    version=VERSION
)

async def collect_request_info(request: Request) -> dict:
    """요청 정보를 수집하여 딕셔너리로 반환"""
    try:
        # 클라이언트 IP 주소
        client_ip = request.client.host if request.client else "unknown"
        
        # User-Agent
        user_agent = request.headers.get("user-agent", "unknown")
        
        # 요청 메서드와 URL
        method = request.method
        url = str(request.url)
        
        # 헤더 정보 (민감한 정보 제외)
        headers = {}
        for key, value in request.headers.items():
            if key.lower() not in ['authorization', 'cookie', 'x-api-key']:
                headers[key] = value
        
        # 쿼리 파라미터
        query_params = dict(request.query_params)
        
        # 요청 본문 읽기 (content-length만큼)
        request_body = ""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 0:
            try:
                # content-length만큼만 읽기
                body_bytes = await request.body()
                if body_bytes:
                    # UTF-8로 디코딩 시도, 실패하면 hex로 표시
                    try:
                        request_body = body_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        request_body = f"<Binary data: {body_bytes.hex()}>"
            except Exception as body_error:
                request_body = f"<Error reading body: {str(body_error)}>"
        
        # 요청 시간
        request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            "client_ip": client_ip,
            "user_agent": user_agent,
            "method": method,
            "url": url,
            "headers": headers,
            "query_params": query_params,
            "request_body": request_body,
            "content_length": content_length,
            "request_time": request_time
        }
    except Exception as e:
        return {
            "error": f"요청 정보 수집 실패: {str(e)}",
            "request_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def get_allowed_ips():
    """config.json에서 허용된 IP 목록을 가져옵니다."""
    try:
        # 주석이 포함된 JSON 파일을 읽기 위해 config.py의 함수 사용
        from app.core.config import _read_json_with_comments, CONFIG_FILE
        config = _read_json_with_comments(CONFIG_FILE)
        return config.get('security', {}).get('allowed_ips', [])
    except Exception as e:
        print(f"IP 화이트리스트 설정 로드 실패: {e}")
        return []

def is_ip_allowed(client_ip: str) -> bool:
    """클라이언트 IP가 허용 목록에 있는지 확인합니다."""
    try:
        # localhost는 항상 허용
        if client_ip in ['127.0.0.1', '::1', 'localhost']:
            return True
        
        allowed_ips = get_allowed_ips()
        
        # IP 화이트리스트가 비활성화된 경우 모든 IP 허용
        from app.core.config import _read_json_with_comments, CONFIG_FILE
        try:
            config = _read_json_with_comments(CONFIG_FILE)
            if not config.get('security', {}).get('enable_ip_whitelist', True):
                return True
        except Exception:
            pass  # 설정 파일 읽기 실패 시 계속 진행
        
        # 허용된 IP 목록이 비어있으면 모든 IP 허용
        if not allowed_ips:
            return True
        
        # 클라이언트 IP를 IPAddress 객체로 변환
        try:
            client_ip_obj = ipaddress.ip_address(client_ip)
        except ValueError:
            print(f"유효하지 않은 IP 주소: {client_ip}")
            return False
        
        # 허용된 IP 목록과 비교
        for allowed_ip in allowed_ips:
            try:
                # IPv4/IPv6 주소 또는 CIDR 블록 처리
                if '/' in allowed_ip:
                    # CIDR 블록 처리
                    allowed_network = ipaddress.ip_network(allowed_ip, strict=False)
                    if client_ip_obj in allowed_network:
                        return True
                else:
                    # 단일 IP 주소 처리
                    allowed_ip_obj = ipaddress.ip_address(allowed_ip)
                    if client_ip_obj == allowed_ip_obj:
                        return True
            except ValueError:
                print(f"유효하지 않은 허용 IP: {allowed_ip}")
                continue
        
        return False
        
    except Exception as e:
        print(f"IP 화이트리스트 검증 중 오류: {e}")
        return True  # 오류 발생 시 허용 (보안상 주의 필요)

async def send_blocked_ip_telegram_notification(request: Request):
    """차단된 IP 접근 시 텔레그램으로 알림 전송"""
    try:
        from app.services.telegram_worker import TelegramBotManager
        telegram_manager = TelegramBotManager()
        request_info = await collect_request_info(request)
        
        # 요청 본문 정보 추가
        body_info = ""
        if request_info.get('request_body'):
            body_info = f"""

🔍 요청 본문 (Content-Length: {request_info.get('content_length', 'unknown')}):
{request_info.get('request_body', '')}"""
        
        # 텔레그램 메시지 생성
        message = f"""🛡️  iScan 인스턴스 IP 차단 - 허용되지 않은 IP에서의 접근 시도

📊 차단된 요청 정보:
• 클라이언트 IP: {request_info.get('client_ip', 'unknown')}
• 요청 메서드: {request_info.get('method', 'unknown')}
• 요청 URL: {request_info.get('url', 'unknown')}
• User-Agent: {request_info.get('user_agent', 'unknown')}
• 요청 시간: {request_info.get('request_time', 'unknown')}

🔍 헤더 정보:
{json.dumps(request_info.get('headers', {}), indent=2, ensure_ascii=False)}

🔍 쿼리 파라미터:
{json.dumps(request_info.get('query_params', {}), indent=2, ensure_ascii=False)}{body_info}

⚠️ 이 IP는 화이트리스트에 등록되지 않은 IP입니다.
"""

        # 모든 봇에게 알림 전송 (보안 알림이므로 모든 봇에 전송)
        for bot_name, bot_config in telegram_manager.bots.items():
            try:
                target_chat_ids = bot_config.get("chat_ids", [])
                for target_chat_id in target_chat_ids:
                    await telegram_manager._send_message(
                        f"https://api.telegram.org/bot{bot_config['token']}",
                        str(target_chat_id),
                        message
                    )
                    print(f"IP 차단 알림 전송 완료: {bot_name} -> chat_id: {target_chat_id}")
            except Exception as bot_error:
                print(f"IP 차단 알림 전송 실패 ({bot_name}): {bot_error}")

    except Exception as e:
        print(f"IP 차단 텔레그램 알림 전송 실패: {e}")

async def send_404_telegram_notification(request: Request):
    """404 에러 발생 시 텔레그램으로 알림 전송"""
    try:
        from app.services.telegram_worker import TelegramBotManager
        telegram_manager = TelegramBotManager()
        request_info = await collect_request_info(request)
        
        # 요청 본문 정보 추가
        body_info = ""
        if request_info.get('request_body'):
            body_info = f"""

🔍 요청 본문 (Content-Length: {request_info.get('content_length', 'unknown')}):
{request_info.get('request_body', '')}"""
        
        # 텔레그램 메시지 생성
        message = f"""🚨  iScan 인스턴스 404 에러 발생 - 정의되지 않은 API 요청

📊 요청 정보:
• 클라이언트 IP: {request_info.get('client_ip', 'unknown')}
• 요청 메서드: {request_info.get('method', 'unknown')}
• 요청 URL: {request_info.get('url', 'unknown')}
• User-Agent: {request_info.get('user_agent', 'unknown')}
• 요청 시간: {request_info.get('request_time', 'unknown')}

🔍 헤더 정보:
{json.dumps(request_info.get('headers', {}), indent=2, ensure_ascii=False)}

🔍 쿼리 파라미터:
{json.dumps(request_info.get('query_params', {}), indent=2, ensure_ascii=False)}{body_info}
"""

        # 모든 봇에게 알림 전송 (404는 긴급 알림이므로 모든 봇에 전송)
        for bot_name, bot_config in telegram_manager.bots.items():
            try:
                target_chat_ids = bot_config.get("chat_ids", [])
                for target_chat_id in target_chat_ids:
                    await telegram_manager._send_message(
                        f"https://api.telegram.org/bot{bot_config['token']}",
                        str(target_chat_id),
                        message
                    )
                    print(f"404 알림 전송 완료: {bot_name} -> chat_id: {target_chat_id}")
            except Exception as bot_error:
                print(f"404 알림 전송 실패 ({bot_name}): {bot_error}")

    except Exception as e:
        print(f"404 텔레그램 알림 전송 실패: {e}")

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """404 Not Found 예외 핸들러"""
    try:
        # 무시할 경로 목록 (브라우저가 자동으로 요청하는 파일들)
        ignored_paths = [
            '/.well-known/',
            '/favicon.ico',
            '/robots.txt',
            '/sitemap.xml',
            '/apple-touch-icon',
            '/manifest.json',
        ]
        
        request_path = str(request.url.path)
        
        # 무시할 경로인지 확인
        should_ignore = any(request_path.startswith(ignored_path) for ignored_path in ignored_paths)
        
        if should_ignore:
            # 무시할 경로는 조용히 404 반환 (로그 및 알림 없음)
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "Not Found"
                }
            )
        
        # 요청 정보 수집 및 상세 로깅
        request_info = await collect_request_info(request)
        
        print("=" * 80)
        print("🚨 404 Not Found - 정의되지 않은 API 요청")
        print("=" * 80)
        print(f"📊 요청 정보:")
        print(f"• 클라이언트 IP: {request_info.get('client_ip', 'unknown')}")
        print(f"• 요청 메서드: {request_info.get('method', 'unknown')}")
        print(f"• 요청 URL: {request_info.get('url', 'unknown')}")
        print(f"• User-Agent: {request_info.get('user_agent', 'unknown')}")
        print(f"• 요청 시간: {request_info.get('request_time', 'unknown')}")
        print(f"• 헤더 정보: {json.dumps(request_info.get('headers', {}), indent=2, ensure_ascii=False)}")
        print(f"• 쿼리 파라미터: {json.dumps(request_info.get('query_params', {}), indent=2, ensure_ascii=False)}")
        
        # 요청 본문 정보 출력
        if request_info.get('request_body'):
            print(f"• 요청 본문 (Content-Length: {request_info.get('content_length', 'unknown')}):")
            print(f"{request_info.get('request_body', '')}")
        
        print("=" * 80)
        
        # 텔레그램 알림 전송 (백그라운드)
        asyncio.create_task(send_404_telegram_notification(request))
        
    except Exception as e:
        print(f"404 핸들러에서 오류 발생: {e}")
    
    # 404 응답 반환
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "Not Found",
            "data": {
                "body": "요청하신 엔드포인트를 찾을 수 없습니다."
            }
        }
    )

# IP 화이트리스트 미들웨어
@app.middleware("http")
async def ip_whitelist_middleware(request: Request, call_next):
    """IP 화이트리스트 검증 미들웨어"""
    try:
        # 클라이언트 IP 가져오기
        client_ip = request.client.host if request.client else "unknown"
        
        # X-Forwarded-For 헤더 확인 (프록시/로드밸런서 뒤에 있는 경우)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # 첫 번째 IP를 실제 클라이언트 IP로 사용
            client_ip = forwarded_for.split(",")[0].strip()
        
        # IP 화이트리스트 검증
        try:
            is_allowed = is_ip_allowed(client_ip)
        except Exception as ip_check_error:
            print(f"⚠️ IP 검증 중 오류 발생 (허용으로 처리): {ip_check_error}")
            import traceback
            traceback.print_exc()
            is_allowed = True  # 오류 발생 시 허용 (보안상 주의 필요)
        
        if not is_allowed:
            print("=" * 80)
            print("🛡️ IP 차단 - 허용되지 않은 IP에서의 접근 시도")
            print("=" * 80)
            print(f"차단된 IP: {client_ip}")
            print(f"요청 URL: {request.url}")
            print(f"요청 메서드: {request.method}")
            print(f"User-Agent: {request.headers.get('user-agent', 'unknown')}")
            print("=" * 80)
            
            # 텔레그램 알림 전송 (백그라운드, 오류 무시)
            try:
                asyncio.create_task(send_blocked_ip_telegram_notification(request))
            except Exception as telegram_error:
                print(f"⚠️ 텔레그램 알림 전송 실패 (무시): {telegram_error}")
            
            # 403 Forbidden 응답 반환
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "message": "Access Denied",
                    "data": {
                        "body": "접근이 거부되었습니다. 허용되지 않은 IP에서의 접근입니다."
                    }
                }
            )
        
        # 허용된 IP인 경우 다음 미들웨어로 진행
        try:
            response = await call_next(request)
            return response
        except Exception as request_error:
            print(f"❌ 요청 처리 중 오류: {request_error}")
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Internal Server Error",
                    "error": str(request_error)
                }
            )
        
    except Exception as e:
        print(f"❌ IP 화이트리스트 미들웨어 오류: {e}")
        import traceback
        traceback.print_exc()
        # 오류 발생 시 요청을 계속 진행 (보안상 주의 필요)
        try:
            response = await call_next(request)
            return response
        except Exception as inner_error:
            print(f"❌ 요청 처리 중 오류: {inner_error}")
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Internal Server Error",
                    "error": str(inner_error)
                }
            )

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app/main.py 위치
STATIC_DIR = os.path.join(BASE_DIR, "static")  # app/static

# 정적 파일 디렉토리 존재 확인
if not os.path.exists(STATIC_DIR):
    print(f"⚠️  경고: 정적 파일 디렉토리를 찾을 수 없습니다: {STATIC_DIR}")
else:
    print(f"✅ 정적 파일 디렉토리: {STATIC_DIR}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# API 라우터 등록
app.include_router(status.router, prefix="/api", tags=["status"])
app.include_router(control.router, prefix="/api", tags=["control"])

async def send_welcome_messages():
    """모든 봇의 모든 채팅 ID에 웰컴 메시지를 전송합니다."""
    bots = get_telegram_bots()
    
    if not bots:
        print("No Telegram bots configured, skipping welcome messages")
        return
    
    welcome_message = f"""
🚀 {TITLE} 시작 (v{VERSION})

{HELP_TEXT}
    """.strip()
    
    for bot_name, bot_config in bots.items():
        token = bot_config.get("token")
        chat_ids = bot_config.get("chat_ids", [])
        
        if not token or not chat_ids:
            print(f"Bot {bot_name}: Missing token or chat_ids")
            continue
        
        api_url = f"https://api.telegram.org/bot{token}"
        
        for chat_id in chat_ids:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{api_url}/sendMessage", json={
                        "chat_id": chat_id,
                        "text": welcome_message
                    }) as resp:
                        if resp.status == 200:
                            print(f"✅ Welcome message sent to {bot_name} -> {chat_id}")
                        else:
                            error_text = await resp.text()
                            print(f"❌ Failed to send welcome message to {bot_name} -> {chat_id}: {resp.status} - {error_text}")
            except Exception as e:
                print(f"❌ Error sending welcome message to {bot_name} -> {chat_id}: {e}")

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    try:
        # 데이터베이스 연결 풀 초기화
        await init_db_pool()
    except Exception as e:
        print(f"⚠️ 데이터베이스 연결 풀 초기화 실패 (계속 진행): {e}")
    
    try:
        # AI 추론 기능 활성화 여부 확인
        from app.api.control import AI_INFERENCE_DISABLED
        
        if AI_INFERENCE_DISABLED:
            print("⚠️  [임시 모드] AI 추론 기능이 비활성화되어 있습니다. SDUI 웹 서비스만 실행됩니다.")
            
            # 설정 검증 (텔레그램 봇 검증 제외)
            # validate_config()  # 텔레그램 봇 검증 주석 처리
            print("✅ Configuration validation skipped (test mode)")
            
            # 시리얼 넘버별 calibration 데이터 로드 (AI 추론 관련)
            # from app.core.Artis_AI.camera.calibration_manager import load_all_serial_calibration_data
            # load_all_serial_calibration_data()
            print("⚠️  Serial number calibration data loading skipped (test mode)")
            
            # 웰컴 메시지 전송 (텔레그램 관련)
            # await send_welcome_messages()
            print("⚠️  Welcome messages skipped (test mode)")
            
            # 백그라운드 워커 시작 (텔레그램 폴링)
            # await start_background_workers()
            print("⚠️  Background workers skipped (test mode)")
        else:
            print("✅ AI 추론 기능이 활성화되어 있습니다. 전체 시스템을 초기화합니다.")
            
            # 설정 검증
            validate_config()
            print("✅ Configuration validation completed")
            
            # 시리얼 넘버별 calibration 데이터 로드 (AI 추론 관련)
            from app.core.Artis_AI.camera.calibration_manager import load_all_serial_calibration_data
            load_all_serial_calibration_data()
            print("✅ Serial number calibration data loaded")

            # 추론 모델 로드 및 warm up
            from app.api.control import _get_artis_model
            _get_artis_model()
            print("✅ Inference model loaded")
            
            # 웰컴 메시지 전송 (텔레그램 관련)
            await send_welcome_messages()
            print("✅ Welcome messages sent")
            
            # 백그라운드 워커 시작 (텔레그램 폴링)
            await start_background_workers()
            print("✅ Background workers started")

            # EdgeMan 상태 정보 로드
            from app.api.control import _load_edge_status_map
            _load_edge_status_map()
            print("✅ iScan Instance status loaded")
            
            # iScan Instance 모니터링 서비스 시작
            from app.services.system_monitor import server_monitoring_service
            server_monitoring_service.start()
            print("✅ iScan Instance monitoring service started")
        
        actual_host = get_actual_host()
        actual_port = get_actual_port()
        # 0.0.0.0은 모든 인터페이스를 의미하므로 표시용으로는 localhost 사용
        display_host = "localhost" if actual_host == "0.0.0.0" else actual_host
        print("✅ SDUI 웹 서비스가 준비되었습니다.")
        print(f"   - 웹 인터페이스: http://{display_host}:{actual_port}/ 또는 http://{display_host}:{actual_port}/web")
        print(f"   - UI 구성 API: http://{display_host}:{actual_port}/api/ui/config")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        # AI 추론이 비활성화된 경우에만 테스트 모드로 계속 진행
        from app.api.control import AI_INFERENCE_DISABLED
        if AI_INFERENCE_DISABLED:
            print(f"⚠️  Continuing in test mode despite error: {e}")
        else:
            raise

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행"""
    try:
        # 데이터베이스 연결 풀 종료
        await close_db_pool()
    except Exception as e:
        print(f"⚠️ 데이터베이스 연결 풀 종료 중 오류: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """SDUI 메인 페이지"""
    try:
        index_path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"index.html not found at {index_path}")
        return FileResponse(index_path)
    except Exception as e:
        print(f"❌ Error serving index.html: {e}")
        # HTML 파일이 없으면 기본 JSON 응답
        return JSONResponse({
            "message": TITLE,
            "version": VERSION,
            "status": "running",
            "error": str(e)
        })

@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """SDUI 웹 인터페이스"""
    try:
        index_path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"index.html not found at {index_path}")
        return FileResponse(index_path)
    except Exception as e:
        print(f"❌ Error serving index.html: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load index.html: {str(e)}")

@app.get("/product", response_class=HTMLResponse)
async def product_page():
    """상품 관리 페이지"""
    try:
        product_path = os.path.join(STATIC_DIR, "product.html")
        if not os.path.exists(product_path):
            raise FileNotFoundError(f"product.html not found at {product_path}")
        return FileResponse(product_path)
    except Exception as e:
        print(f"❌ Error serving product.html: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load product.html: {str(e)}")

@app.get("/password", response_class=HTMLResponse)
async def password_page():
    """비밀번호 입력 페이지"""
    try:
        password_path = os.path.join(STATIC_DIR, "password.html")
        if not os.path.exists(password_path):
            raise FileNotFoundError(f"password.html not found at {password_path}")
        return FileResponse(password_path)
    except Exception as e:
        print(f"❌ Error serving password.html: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load password.html: {str(e)}")

@app.get("/settings", response_class=HTMLResponse)
async def settings_page():
    """환경설정 페이지"""
    try:
        settings_path = os.path.join(STATIC_DIR, "settings.html")
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"settings.html not found at {settings_path}")
        return FileResponse(settings_path)
    except Exception as e:
        print(f"❌ Error serving settings.html: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load settings.html: {str(e)}")

@app.post("/api/edgeman/terminate-browser")
async def terminate_browser(request: Request):
    """브라우저 종료 요청 처리
    
    웹 브라우저가 실행되는 PC의 HTTP 서버로 브라우저 종료 요청을 전달합니다.
    클라이언트 IP를 기반으로 EdgeMan 서버 주소를 구성하여 요청을 전달합니다.
    """
    try:
        # 클라이언트 IP 가져오기
        client_ip = request.client.host if request.client else "unknown"
        
        # X-Forwarded-For 헤더 확인 (프록시/로드밸런서 뒤에 있는 경우)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        print(f"🔄 브라우저 종료 요청 수신: {client_ip}")
        
        # EdgeMan 서버 주소 구성 (클라이언트 IP 기반)
        # config.json에서 포트 가져오기 (기본값: DEFAULT_EDGEMAN_SERVER_PORT)
        edgeman_port = _get_config_value('edgeman_port', DEFAULT_EDGEMAN_SERVER_PORT)
        edgeman_url = f"http://{client_ip}:{edgeman_port}/api/edgeman/terminate-browser"
        
        print(f"📤 EdgeMan 서버로 브라우저 종료 요청 전송: {edgeman_url}")
        
        # EdgeMan 서버로 요청 전달
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    edgeman_url,
                    json={},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ EdgeMan 서버 응답: {result}")
                        return JSONResponse(
                            status_code=200,
                            content={
                                "success": True,
                                "message": "브라우저 종료 요청이 EdgeMan 서버로 전달되었습니다.",
                                "edgeman_response": result
                            }
                        )
                    else:
                        error_text = await response.text()
                        print(f"⚠️ EdgeMan 서버 응답 오류 ({response.status}): {error_text}")
                        return JSONResponse(
                            status_code=response.status,
                            content={
                                "success": False,
                                "message": f"EdgeMan 서버에서 오류가 발생했습니다 (HTTP {response.status})",
                                "error": error_text
                            }
                        )
        except aiohttp.ClientError as e:
            print(f"❌ EdgeMan 서버 통신 실패: {e}")
            # EdgeMan 서버와 통신할 수 없는 경우에도 성공으로 처리
            # (브라우저가 실행되는 PC가 다른 서버일 수 있음)
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "브라우저 종료 요청이 처리되었습니다.",
                    "warning": f"EdgeMan 서버 통신 실패 (무시됨): {str(e)}"
                }
            )
        
    except Exception as e:
        print(f"❌ 브라우저 종료 요청 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "브라우저 종료 요청 처리 중 오류가 발생했습니다.",
                "error": str(e)
            }
        )

@app.post("/api/edgeman/run-servman")
async def run_servman(request: Request):
    """ServMan 프로그램 실행 요청 처리
    
    웹 브라우저가 실행되는 PC의 HTTP 서버로 ServMan 프로그램 실행 요청을 전달합니다.
    클라이언트 IP를 기반으로 EdgeMan 서버 주소를 구성하여 요청을 전달합니다.
    선택된 상품의 상세 정보를 함께 전달합니다.
    
    요청 본문 예시:
    {
        "product": {
            "id": 123,
            "vendor_id": 1,
            "vendor_name": "cheonsang_seongsu",
            "vendor_display_name": "천상성수",
            "item_code": "0000012",
            "barcode": "8801234567890",
            "item_name_default": "테스트 상품명",
            "item_description_default": "상품 설명입니다.",
            "category_top": "식품",
            "category_mid": "과자",
            "category_low": "스낵",
            "currency_code": "KRW",
            "base_amount": 5000.0,
            "vat_included": true,
            "is_pos_use": true,
            "is_deleted": false,
            "stock": 100,
            "is_out_of_stock": false,
            "item_type": 0,
            "order_unit": 0,
            "disp_priority": 1,
            "is_discounted": false,
            "discount_rate": 0.0,
            "scan_image_count": 5,
            "thumb_image_file": "thumb_item001.jpg",
            "similar_item_group": null,
            "option_groups": null,
            "created_at": "2025-12-03T10:30:00",
            "updated_at": "2025-12-03T15:45:00"
        }
    }
    
    주요 필드:
    - id: 상품 고유 ID
    - vendor_id: 벤더 ID
    - vendor_name: 벤더명
    - vendor_display_name: 벤더 표시명
    - item_code: 상품 코드
    - barcode: 바코드
    - item_name_default: 상품명
    - item_description_default: 상품 설명
    - category_top/mid/low: 대/중/소분류
    - currency_code: 통화 코드 (예: "KRW")
    - base_amount: 기본 가격 (float)
    - vat_included: 부가세 포함 여부 (boolean)
    - is_pos_use: POS 사용 여부 (boolean)
    - stock: 재고 수량
    - is_out_of_stock: 품절 여부 (boolean)
    - item_type: 상품 유형 (0: 자체제작, 1: 유통상품, 2: 선택상품, 3: Tray)
    - order_unit: 주문 단위 (0: 낱개, 1: 세트)
    - scan_image_count: 스캔 이미지 개수
    - thumb_image_file: 대표 이미지 파일명
    - similar_item_group: 유사 상품 그룹 (JSON 또는 null)
    - option_groups: 옵션 그룹 (JSON 또는 null)
    - created_at: 생성일시
    - updated_at: 수정일시
    """
    try:
        # 요청 본문에서 상품 정보 가져오기
        request_data = await request.json()
        product_info = request_data.get('product', {})
        
        # 클라이언트 IP 가져오기
        client_ip = request.client.host if request.client else "unknown"
        
        # X-Forwarded-For 헤더 확인 (프록시/로드밸런서 뒤에 있는 경우)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        print(f"🔄 ServMan 실행 요청 수신: {client_ip}")
        if product_info:
            product_name = product_info.get('item_name_default', 'N/A')
            product_code = product_info.get('item_code', 'N/A')
            print(f"📦 선택된 상품: {product_code} - {product_name}")
        
        # EdgeMan 서버 주소 구성 (클라이언트 IP 기반)
        # config.json에서 포트 가져오기 (기본값: DEFAULT_EDGEMAN_SERVER_PORT)
        edgeman_port = _get_config_value('edgeman_port', DEFAULT_EDGEMAN_SERVER_PORT)
        edgeman_url = f"http://{client_ip}:{edgeman_port}/api/edgeman/run-servman"
        
        print(f"📤 EdgeMan 서버로 ServMan 실행 요청 전송: {edgeman_url}")
        
        # EdgeMan 서버로 요청 전달 (상품 정보 포함)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    edgeman_url,
                    json={
                        'product': product_info
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ EdgeMan 서버 응답: {result}")
                        return JSONResponse(
                            status_code=200,
                            content={
                                "success": True,
                                "message": "ServMan 실행 요청이 EdgeMan 서버로 전달되었습니다.",
                                "edgeman_response": result
                            }
                        )
                    else:
                        error_text = await response.text()
                        print(f"⚠️ EdgeMan 서버 응답 오류 ({response.status}): {error_text}")
                        return JSONResponse(
                            status_code=response.status,
                            content={
                                "success": False,
                                "message": f"EdgeMan 서버에서 오류가 발생했습니다 (HTTP {response.status})",
                                "error": error_text
                            }
                        )
        except aiohttp.ClientError as e:
            print(f"❌ EdgeMan 서버 통신 실패: {e}")
            # EdgeMan 서버와 통신할 수 없는 경우에도 성공으로 처리
            # (브라우저가 실행되는 PC가 다른 서버일 수 있음)
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "ServMan 실행 요청이 처리되었습니다.",
                    "warning": f"EdgeMan 서버 통신 실패 (무시됨): {str(e)}"
                }
            )
        
    except Exception as e:
        print(f"❌ ServMan 실행 요청 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "ServMan 실행 요청 처리 중 오류가 발생했습니다.",
                "error": str(e)
            }
        )

@app.post("/api/edgeman/request-thumbnail")
async def request_thumbnail(request: Request):
    """대표 이미지 요청 처리
    
    웹 브라우저가 실행되는 PC의 HTTP 서버로 대표 이미지 요청을 전달합니다.
    클라이언트 IP를 기반으로 EdgeMan 서버 주소를 구성하여 요청을 전달합니다.
    선택된 상품의 상세 정보를 함께 전달합니다.
    
    요청 본문은 run-servman과 동일합니다:
    {
        "product": {
            "id": 123,
            "vendor_id": 1,
            "vendor_name": "cheonsang_seongsu",
            "vendor_display_name": "천상성수",
            "item_code": "0000012",
            "barcode": "8801234567890",
            "item_name_default": "테스트 상품명",
            "item_description_default": "상품 설명입니다.",
            "category_top": "식품",
            "category_mid": "과자",
            "category_low": "스낵",
            "currency_code": "KRW",
            "base_amount": 5000.0,
            "vat_included": true,
            "is_pos_use": true,
            "is_deleted": false,
            "stock": 100,
            "is_out_of_stock": false,
            "item_type": 0,
            "order_unit": 0,
            "disp_priority": 1,
            "is_discounted": false,
            "discount_rate": 0.0,
            "scan_image_count": 5,
            "thumb_image_file": "thumb_item001.jpg",
            "similar_item_group": null,
            "option_groups": null,
            "created_at": "2025-12-03T10:30:00",
            "updated_at": "2025-12-03T15:45:00"
        }
    }
    
    EdgeMan 서버 응답 형식 (JSON):
    
    Success response example:
    {
        "status": "success",
        "message": "Thumbnail image retrieved successfully",
        "data": {
            "image_data": "data:image/png;base64,iVBORw0KGgoAAAANS...",
            "content_type": "image/png"
        }
    }
    
    Error response example:
    {
        "status": "error",
        "message": "Thumbnail image file not found"
    }
    """
    try:
        # 요청 본문에서 상품 정보 가져오기
        request_data = await request.json()
        product_info = request_data.get('product', {})
        
        # 클라이언트 IP 가져오기
        client_ip = request.client.host if request.client else "unknown"
        
        # X-Forwarded-For 헤더 확인 (프록시/로드밸런서 뒤에 있는 경우)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        print(f"🖼️ 대표 이미지 요청 수신: {client_ip}")
        if product_info:
            product_name = product_info.get('item_name_default', 'N/A')
            product_code = product_info.get('item_code', 'N/A')
            thumb_image = product_info.get('thumb_image_file', 'N/A')
            print(f"📦 선택된 상품: {product_code} - {product_name}, 이미지: {thumb_image}")
        
        # EdgeMan 서버 주소 구성 (클라이언트 IP 기반)
        # config.json에서 포트 가져오기 (기본값: DEFAULT_EDGEMAN_SERVER_PORT)
        edgeman_port = _get_config_value('edgeman_port', DEFAULT_EDGEMAN_SERVER_PORT)
        edgeman_url = f"http://{client_ip}:{edgeman_port}/api/edgeman/request-thumbnail"
        
        print(f"📤 EdgeMan 서버로 대표 이미지 요청 전송: {edgeman_url}")
        
        # EdgeMan 서버로 요청 전달 (상품 정보 포함)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    edgeman_url,
                    json={
                        'product': product_info
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        # EdgeMan 서버는 항상 JSON 형식으로 응답
                        edgeman_result = await response.json()
                        
                        # EdgeMan 응답을 클라이언트가 기대하는 형식으로 변환
                        if edgeman_result.get('status') == 'success':
                            # 성공 응답 변환
                            result = {
                                "success": True,
                                "message": edgeman_result.get('message', '대표 이미지를 성공적으로 가져왔습니다.'),
                                "image_data": edgeman_result.get('data', {}).get('image_data'),
                                "content_type": edgeman_result.get('data', {}).get('content_type')
                            }
                        else:
                            # 실패 응답 변환
                            result = {
                                "success": False,
                                "message": edgeman_result.get('message', '대표 이미지를 불러올 수 없습니다.')
                            }
                        
                        print(f"✅ EdgeMan 서버 응답: {result.get('message', 'Success')}")
                        return JSONResponse(
                            status_code=200,
                            content=result
                        )
                    else:
                        error_text = await response.text()
                        print(f"⚠️ EdgeMan 서버 응답 오류 ({response.status}): {error_text}")
                        return JSONResponse(
                            status_code=response.status,
                            content={
                                "success": False,
                                "message": f"EdgeMan 서버에서 오류가 발생했습니다 (HTTP {response.status})",
                                "error": error_text
                            }
                        )
        except aiohttp.ClientError as e:
            print(f"❌ EdgeMan 서버 통신 실패: {e}")
            # EdgeMan 서버와 통신할 수 없는 경우에도 실패로 처리
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "message": "EdgeMan 서버와 통신할 수 없습니다.",
                    "error": str(e)
                }
            )
        
    except Exception as e:
        print(f"❌ 대표 이미지 요청 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "대표 이미지 요청 처리 중 오류가 발생했습니다.",
                "error": str(e)
            }
        )

@app.post("/api/auth/verify-password")
async def verify_password(request: Request):
    """비밀번호 검증"""
    from fastapi import HTTPException
    import json
    
    try:
        data = await request.json()
        password = data.get("password", "")
        redirect_url = data.get("redirect_url", "/settings")
        
        # 기본 비밀번호: 7890
        correct_password = "7890"
        
        if password == correct_password:
            return {
                "success": True,
                "message": "비밀번호가 일치합니다.",
                "redirect_url": redirect_url
            }
        else:
            raise HTTPException(
                status_code=401,
                detail="비밀번호가 일치하지 않습니다."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            #detail=f"비밀번호 검증 중 오류가 발생했습니다: {str(e)}"
            detail="비밀번호가 일치하지 않습니다."
        )

@app.get("/api/products")
async def get_products(
    vendor_id: Optional[int] = Query(None, description="벤더 ID"),
    category: Optional[str] = Query(None, description="카테고리"),
    is_deleted: bool = Query(False, description="삭제된 상품 포함 여부"),
    limit: Optional[int] = Query(None, ge=1, description="조회 개수 (None이면 전체 조회)"),
    offset: int = Query(0, ge=0, description="시작 위치"),
    search: Optional[str] = Query(None, description="검색어 (상품코드, 상품명, 바코드)")
):
    """상품 목록 조회 API"""
    try:
        from app.services.product_service import product_service
        
        products = await product_service.get_product_list(
            vendor_id=vendor_id,
            category=category,
            is_deleted=is_deleted,
            limit=limit,
            offset=offset,
            search=search
        )
        
        total_count = await product_service.get_product_count(
            vendor_id=vendor_id,
            category=category,
            is_deleted=is_deleted,
            search=search
        )
        
        return {
            "success": True,
            "data": products,
            "total": total_count,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        error_message = str(e)
        print(f"❌ 상품 목록 조회 실패: {e}")
        
        # 데이터베이스 연결 에러인 경우 더 친절한 메시지
        if "Can't connect" in error_message or "2003" in error_message:
            error_message = "데이터베이스 서버에 연결할 수 없습니다. MySQL 서버가 실행 중인지 확인해주세요."
        elif "Access denied" in error_message or "1045" in error_message:
            error_message = "데이터베이스 인증에 실패했습니다. 사용자 이름과 비밀번호를 확인해주세요."
        elif "Unknown database" in error_message or "1049" in error_message:
            error_message = "데이터베이스를 찾을 수 없습니다. 데이터베이스 이름을 확인해주세요."
        
        # 클라이언트에 친절한 에러 메시지 반환
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": error_message,
                "data": [],
                "total": 0
            }
        )


@app.get("/api/products/{product_id}")
async def get_product_detail(product_id: int):
    """상품 상세 정보 조회 API"""
    try:
        from app.services.product_service import product_service
        
        product = await product_service.get_product_detail(product_id)
        
        if not product:
            raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다")
        
        return {
            "success": True,
            "data": product
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 상품 상세 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"상품 상세 정보 조회 실패: {str(e)}")


@app.get("/api/ui/config")
async def get_ui_config(page: str = "main", redirect: str = "/settings"):
    """UI 구성 정보를 JSON으로 반환"""
    from app.services.web_service import web_ui_service
    if page == "password":
        config = await web_ui_service._generate_password_page_config(redirect_url=redirect)
    else:
        config = await web_ui_service.generate_ui_config(page=page)
    return config

@app.post("/api/ui/action")
async def handle_ui_action(request: Request):
    """UI 액션 처리"""
    from app.services.web_service import web_ui_service
    try:
        payload = await request.json()
        action_type = payload.get("type", "unknown")
        result = await web_ui_service.handle_ui_action(action_type, payload)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)}
        )

@app.get("/favicon.ico")
async def favicon():
    """Favicon 제공"""
    try:
        favicon_path = os.path.join(STATIC_DIR, "favicon.ico")
        if os.path.exists(favicon_path):
            return FileResponse(favicon_path)
        else:
            # favicon이 없으면 404 반환 (에러 없이)
            return JSONResponse(
                status_code=404,
                content={"message": "Favicon not found"}
            )
    except Exception as e:
        # 에러 발생 시 조용히 404 반환
        return JSONResponse(
            status_code=404,
            content={"message": "Favicon not found"}
        )

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy"}
