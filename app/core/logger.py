import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from app.core.config import settings

def setup_logging():
    """Configure logging to both file and console with rotation"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(settings.LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=10485760,  # 10MB
        backupCount=5,      # Keep 5 backup files
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Set logging levels for specific loggers if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.WARNING)

    return root_logger