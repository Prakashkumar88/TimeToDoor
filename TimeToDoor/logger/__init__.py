# Auto-generated file.
import logging
import os, sys
from datetime import datetime

LOG_DIR = "logs"
LOG_DIR = os.path.join(os.getcwd(), LOG_DIR)

os.makedirs(LOG_DIR, exist_ok=True)

# .log
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
file_name = f"log_{CURRENT_TIME_STAMP}.log"

#output
LOG_FILE_PATH = os.path.join(LOG_DIR, file_name)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filemode='w'
)