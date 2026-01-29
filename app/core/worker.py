import asyncio
from app.services.telegram_worker import telegram_poll_loop
#from app.core.logger import logger

async def start_background_workers():
    print("🔧 Starting background workers...")
    #asyncio.create_task(telegram_poll_loop())
