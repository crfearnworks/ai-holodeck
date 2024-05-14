
#from constants import LOGS_DIR
from loguru import logger
import sys

def setup_logger():
    logger.remove()  # Removes all handlers associated with the logger
    logger.add(
        sys.stdout,
        colorize=True,
        level="DEBUG",  # Set to capture all log levels starting from INFO
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
