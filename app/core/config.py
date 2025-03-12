from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Pavement Identifier"
    DEBUG: bool = False
    DATABASE_URL: str = ''
    RABBITMQ_URL: str = 'amqp://guest:guest@localhost:5672/'
    QUEUE_NAME: str = 'pavement_identifier6'
    RABBITMQ_HOST: str = 'localhost'

    class Config:
        env_file = ".env"

# Instantiate settings
settings = Settings()