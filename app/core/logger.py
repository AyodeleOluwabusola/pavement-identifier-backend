import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from app.core.config import settings


def setup_logging(name: str = None) -> logging.Logger:
    """
    Configure logging with both file and console handlers.
    
    Args:
        name: Logger name (optional). If None, returns root logger.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)

    # Create formatter with process information
    formatter = logging.Formatter(
        '[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get or create logger
    logger = logging.getLogger(name) if name else logging.getLogger()
    logger.setLevel(logging.getLevelName(settings.LOG_LEVEL))

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        # Add file handler with rotation
        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        console_handler.setLevel(logging.ERROR)
        logger.error(f"Failed to create file handler: {e}")

    # Prevent propagation to root logger if this is a named logger
    if name:
        logger.propagate = False

    # Set levels for third-party loggers
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('pika').setLevel(logging.WARNING)

    return logger
