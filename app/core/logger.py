import logging
import os
from datetime import datetime

# 로그 디렉토리 생성
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 로그 포맷 설정
log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# 기본 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt=date_format,
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        logging.FileHandler(
            os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding='utf-8'
        )  # 파일 출력
    ]
)

logger = logging.getLogger("app")

# 특정 모듈별 로거
telegram_logger = logging.getLogger("app.telegram")
system_logger = logging.getLogger("app.system")
api_logger = logging.getLogger("app.api")
