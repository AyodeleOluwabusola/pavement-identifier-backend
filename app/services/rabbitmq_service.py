import pika
import json
import logging
from typing import Optional
from app.core.config import settings
from contextlib import contextmanager

logger = logging.getLogger(__name__)

RABBITMQ_HOST = settings.RABBITMQ_HOST
EXCHANGE_NAME = "image_exchange"
QUEUE_NAME = "image_queue"
ROUTING_KEY = "image_routing_key"
CONNECTION_TIMEOUT = 5  # seconds

@contextmanager
def get_rabbitmq_connection():
    """Context manager for handling RabbitMQ connections"""
    connection = None
    try:
        # Set up connection parameters with timeout
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            connection_attempts=3,
            retry_delay=1,
            socket_timeout=CONNECTION_TIMEOUT
        )
        connection = pika.BlockingConnection(parameters)
        yield connection
    except Exception as e:
        logger.error(f"RabbitMQ connection error: {e}")
        raise
    finally:
        if connection and not connection.is_closed:
            connection.close()

def publish_message(message: dict) -> bool:
    """
    Publish a message to RabbitMQ queue
    Returns True if successful, False otherwise
    """
    try:
        with get_rabbitmq_connection() as connection:
            channel = connection.channel()
            
            # Declare exchange and queue
            channel.exchange_declare(
                exchange=EXCHANGE_NAME,
                exchange_type='direct',
                durable=True
            )
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.queue_bind(
                exchange=EXCHANGE_NAME,
                queue=QUEUE_NAME,
                routing_key=ROUTING_KEY
            )
            
            # Publish with mandatory flag to ensure message is routable
            channel.basic_publish(
                exchange=EXCHANGE_NAME,
                routing_key=ROUTING_KEY,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # make message persistent
                    content_type='application/json'
                ),
                mandatory=True
            )
            logger.info(f"Successfully published message to exchange {EXCHANGE_NAME} for file: {message.get('file_name', 'unknown')}")
            return True
            
    except pika.exceptions.AMQPChannelError as e:
        logger.error(f"Channel error: {e}")
        return False
    except pika.exceptions.AMQPConnectionError as e:
        logger.error(f"Connection error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while publishing message: {e}")
        return False