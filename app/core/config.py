from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Pavement Identifier"
    DEBUG: bool = False
    DATABASE_URL: str = ''
    RABBITMQ_URL: str = 'amqp://guest:guest@localhost:5672/'
    QUEUE_NAME: str = 'pavement_identifier6'
    RABBITMQ_HOST: str = 'localhost'
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = 'guest'
    RABBITMQ_PASSWORD: str = 'guest'
    RABBITMQ_VHOST: str = '/'

    RESULTS_DIR: str = 'results'  # Default directory for results
    EXCEL_RESULTS_PATH: str = '/Users/ayodele/Documents/data/image_processing_results.xlsx'
    BATCH_PROCESSING_STARTUP_DIRECTORY: str = '/Users/ayodele/Documents/data/new_unlabeled_data'  # Empty string means no startup processing
    CATEGORIZED_IMAGES_DIR: str='/Users/ayodele/Documents/data'
    ORGANIZED_IMAGES_INTO_FOLDERS: bool=True
    RABBITMQ_NUM_PRODUCERS: int = 4
    RABBITMQ_NUM_CONSUMERS: int = 5
    LOG_FILE: str = 'logs/pavement_identifier.log'
    LOG_LEVEL: str = 'INFO'

    # Add AWS credentials settings
    AWS_ACCESS_KEY_ID: str = ''
    AWS_SECRET_ACCESS_KEY: str = ''
    AWS_DEFAULT_REGION: str = 'us-west-2'

    class Config:
        env_file = ".env"


# Instantiate settings
settings = Settings()
