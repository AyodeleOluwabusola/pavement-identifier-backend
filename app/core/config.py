from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str
    DEBUG: bool

    # Database Settings
    DATABASE_URL: str

    # RabbitMQ Settings
    RABBITMQ_URL: str
    QUEUE_NAME: str
    EXCHANGE_NAME: str
    ROUTING_KEY: str
    RABBITMQ_HOST: str
    RABBITMQ_PORT: int
    RABBITMQ_USER: str
    RABBITMQ_PASSWORD: str
    RABBITMQ_VHOST: str
    RABBITMQ_NUM_PRODUCERS: int
    RABBITMQ_NUM_CONSUMERS: int

    # Model Settings
    MODEL_URI_TENSORFLOW: str
    MODEL_URI_PYTORCH: str
    MLFLOW_TRACKING_URI: str
    CONFIDENCE_THRESHOLD: float
    FRAMEWORK_IN_USE: str

    # Directory Settings
    CATEGORIZED_IMAGES_DIR: str
    RESULTS_DIR: str
    EXCEL_RESULTS_PATH: str
    BATCH_PROCESSING_STARTUP_DIRECTORY: str

    # Results Queue Settings
    RESULTS_EXCHANGE_NAME: str
    RESULTS_QUEUE_NAME: str
    RESULTS_ROUTING_KEY: str

    # Process Management Settings
    CLEANUP_GRACE_PERIOD: int
    PROCESS_SHUTDOWN_TIMEOUT: int

    # Logging Settings
    LOG_FILE: str
    LOG_LEVEL: str

    # Image Organization Settings
    ORGANIZE_IMAGES_INTO_FOLDERS: bool

    # AWS Settings
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create settings instance
settings = Settings()
