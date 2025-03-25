from pydantic_settings import BaseSettings

# TODO revert the rabbitMQ 
class Settings(BaseSettings):
    APP_NAME: str = "Pavement Identifier"
    DEBUG: bool = False
    DATABASE_URL: str = ''
    RABBITMQ_URL: str = 'amqp://guest:guest@localhost:5673/'
    QUEUE_NAME: str = 'pavement_identifier_queue'
    EXCHANGE_NAME: str = 'pavement_identifier_exchange'
    ROUTING_KEY: str = 'pavement_identifier_key'
    RABBITMQ_HOST: str = 'localhost'
    RABBITMQ_PORT: int = 5673
    RABBITMQ_USER: str = 'guest'
    RABBITMQ_PASSWORD: str = 'guest'
    RABBITMQ_VHOST: str = '/'
    MODEL_URI_TENSORFLOW: str = 'runs:/5540bb07536845eeab2e06fab951f1fb/model'
    MODEL_URI_PYTORCH: str = 'runs:/b76e4133aee04487acedf5708b66d7af/model'

    MLFLOW_TRACKING_URI: str = 'http://52.42.208.9:5000/'
    CONFIDENCE_THRESHOLD: float = 0.70
    CATEGORIZED_IMAGES_DIR: str='/Users/ayodele/Documents/data'
    RESULTS_DIR: str = 'results'  # Default directory for results
    EXCEL_RESULTS_PATH: str = '/Users/oyewolz/Downloads/image_processing_results_1.xlsx'
    # Empty string means no startup processing
    BATCH_PROCESSING_STARTUP_DIRECTORY: str = '/Users/oyewolz/Downloads/data'
    CATEGORIZED_IMAGES_DIR: str = '/Users/oyewolz/Downloads/data'
    ORGANIZED_IMAGES_INTO_FOLDERS: bool=True
    RABBITMQ_NUM_PRODUCERS: int = 7
    RABBITMQ_NUM_CONSUMERS: int = 7
    LOG_FILE: str = 'logs/pavement_identifier.log'
    LOG_LEVEL: str = 'INFO'
    FRAMEWORK_IN_USE: str = 'pytorch'

    # Add AWS credentials settings
    AWS_ACCESS_KEY_ID: str = ''
    AWS_SECRET_ACCESS_KEY: str = ''
    AWS_DEFAULT_REGION: str = 'us-west-2'

    # Results queue settings
    RESULTS_EXCHANGE_NAME: str = "results_exchange"
    RESULTS_QUEUE_NAME: str = "results_queue"
    RESULTS_ROUTING_KEY: str = "results_key"

    class Config:
        env_file = ".env"


# Instantiate settings
settings = Settings()
