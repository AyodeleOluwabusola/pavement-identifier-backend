import json
import logging
from contextlib import contextmanager

import pika

from app.core.config import settings

logger = logging.getLogger(__name__)

RABBITMQ_HOST = settings.RABBITMQ_HOST
EXCHANGE_NAME = settings.EXCHANGE_NAME
QUEUE_NAME = settings.QUEUE_NAME
ROUTING_KEY = settings.ROUTING_KEY
CONNECTION_TIMEOUT = 5  # seconds


def get_rabbitmq_connection(use_context_manager=False):
    """
    Get a RabbitMQ connection.
    Args:
        use_context_manager (bool): If True, returns a context manager. If False, returns direct connection.
    """
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        credentials=pika.PlainCredentials(
            settings.RABBITMQ_USER,
            settings.RABBITMQ_PASSWORD
        ),
        virtual_host=settings.RABBITMQ_VHOST,
        connection_attempts=3,
        retry_delay=1,
        socket_timeout=CONNECTION_TIMEOUT
    )

    if use_context_manager:
        @contextmanager
        def connection_context():
            connection = None
            try:
                connection = pika.BlockingConnection(parameters)
                yield connection
            finally:
                if connection and not connection.is_closed:
                    connection.close()
        return connection_context()
    else:
        return pika.BlockingConnection(parameters)


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
            logger.info(
                f"Successfully published message to exchange {EXCHANGE_NAME} for file: {message.get('file_name', 'unknown')}")
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
