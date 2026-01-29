from fastapi import APIRouter, HTTPException
from app.services.system_control import system_control_service, SystemControlError
from app.core.config import get_telegram_bots, get_all_chat_ids
# from app.core.logger import api_logger

router = APIRouter()

@router.get("/status")
async def status():
    """시스템 업타임 조회"""
    try:
        uptime = await system_control_service.get_uptime()
        print("Status endpoint called successfully")
        return {"uptime": uptime}
    except SystemControlError as e:
        print(f"Status endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Unexpected error in status endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/system-info")
async def system_info():
    """시스템 상세 정보 조회"""
    try:
        info = await system_control_service.get_system_info()
        print("System info endpoint called successfully")
        return info
    except SystemControlError as e:
        print(f"System info endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Unexpected error in system info endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/bots")
async def bots_status():
    """Telegram 봇 상태 조회"""
    try:
        bots = get_telegram_bots()
        all_chat_ids = get_all_chat_ids()
        
        bots_info = {}
        for bot_name, bot_config in bots.items():
            bots_info[bot_name] = {
                "name": bot_config.get("name", bot_name),
                "chat_ids": bot_config.get("chat_ids", []),
                "chat_count": len(bot_config.get("chat_ids", []))
            }
        
        print("Bots status endpoint called successfully")
        return {
            "total_bots": len(bots),
            "total_chats": len(all_chat_ids),
            "bots": bots_info
        }
    except Exception as e:
        print(f"Bots status endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
