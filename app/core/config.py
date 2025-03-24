from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Pavement Identifier"
    DEBUG: bool = False
    DATABASE_URL: str = ''
    RABBITMQ_URL: str = 'amqp://guest:guest@localhost:5672/'
    QUEUE_NAME: str = 'pavement_identifier_queue01'
    EXCHANGE_NAME: str = 'pavement_identifier_queue01'
    ROUTING_KEY: str = 'pavement_identifier_queue01'
    RABBITMQ_HOST: str = 'localhost'
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = 'guest'
    RABBITMQ_PASSWORD: str = 'guest'
    RABBITMQ_VHOST: str = '/'
    MODEL_URI_TENSORFLOW: str = 'runs:/5540bb07536845eeab2e06fab951f1fb/model'
    MODEL_URI_PYTORCH: str = 'runs:/b76e4133aee04487acedf5708b66d7af/model'

    CONFIDENCE_THRESHOLD: float = 0.70
    EXCEL_RESULTS_PATH: str = '/Users/ayodele/Documents/data/image_processing_results.xlsx'
    # BATCH_PROCESSING_STARTUP_DIRECTORY: str = '/Users/ayodele/Documents/data/test'  # Empty string means no startup processing
    BATCH_PROCESSING_STARTUP_DIRECTORY: str = ''  # Empty string means no startup processing
    CATEGORIZED_IMAGES_DIR: str='/Users/ayodele/Documents/data'
    ORGANIZED_IMAGES_INTO_FOLDERS: bool=True
    RABBITMQ_NUM_PRODUCERS: int = 5
    RABBITMQ_NUM_CONSUMERS: int = 12
    LOG_FILE: str = 'logs/pavement_identifier.log'
    LOG_LEVEL: str = 'INFO'
    FRAMEWORK_IN_USE: str = 'pytorch'

    # Add AWS credentials settings
    AWS_ACCESS_KEY_ID: str = ''
    AWS_SECRET_ACCESS_KEY: str = ''
    AWS_DEFAULT_REGION: str = 'us-west-2'

    class Config:
        env_file = ".env"


# Instantiate settings
settings = Settings()
