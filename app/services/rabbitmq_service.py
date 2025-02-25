import pika
import json
from app.core.config import settings


RABBITMQ_HOST = settings.RABBITMQ_HOST
QUEUE_NAME = settings.QUEUE_NAME

def publish_message(message: dict):
    
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAME,
        body=json.dumps(message),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    connection.close()