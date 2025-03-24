import json
import logging
import os
from typing import Dict, Any
import pika
from openpyxl import Workbook, load_workbook
from app.core.config import settings
from app.core.logger import setup_logging

logger = setup_logging(__name__)

class ExcelWriterService:
    def __init__(self):
        self.excel_file = settings.EXCEL_RESULTS_PATH
        self._ensure_excel_file_exists()
        self._setup_rabbitmq()

    def _ensure_excel_file_exists(self):
        """Ensure Excel file exists with headers"""
        if not os.path.exists(self.excel_file):
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Image Processing Results"
            sheet.append(["File Name", "Status", "Predicted Class", "Confidence"])
            workbook.save(self.excel_file)

    def _setup_rabbitmq(self):
        """Setup RabbitMQ exchanges and queues"""
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=settings.RABBITMQ_HOST,
                    port=settings.RABBITMQ_PORT,
                    credentials=pika.PlainCredentials(
                        settings.RABBITMQ_USER,
                        settings.RABBITMQ_PASSWORD
                    ),
                    virtual_host=settings.RABBITMQ_VHOST
                )
            )
            channel = connection.channel()

            # Declare exchange
            channel.exchange_declare(
                exchange=settings.RESULTS_EXCHANGE_NAME,
                exchange_type='direct',
                durable=True
            )

            # Declare queue
            channel.queue_declare(
                queue=settings.RESULTS_QUEUE_NAME,
                durable=True
            )

            # Bind queue to exchange
            channel.queue_bind(
                exchange=settings.RESULTS_EXCHANGE_NAME,
                queue=settings.RESULTS_QUEUE_NAME,
                routing_key=settings.RESULTS_ROUTING_KEY
            )

            logger.info("RabbitMQ setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to setup RabbitMQ: {e}")
            raise

    def write_result(self, data: Dict[str, Any]):
        """Write a single result to Excel"""
        try:
            workbook = load_workbook(self.excel_file)
            sheet = workbook.active
            sheet.append([
                data.get('file_name', 'Unknown'),
                data.get('status', 'Error'),
                data.get('predicted_class', 'Unknown'),
                data.get('confidence', 0.0)
            ])
            workbook.save(self.excel_file)
            logger.info(f"Successfully wrote results for {data.get('file_name')} to Excel")
        except Exception as e:
            logger.error(f"Failed to write to Excel: {e}")
            raise

def run_excel_writer():
    """Consumer function to listen for results and write to Excel"""
    writer = ExcelWriterService()
    logger.info("Starting Excel writer service")

    def callback(ch, method, properties, body):
        try:
            data = json.loads(body)
            writer.write_result(data)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)

    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=settings.RABBITMQ_HOST,
                port=settings.RABBITMQ_PORT,
                credentials=pika.PlainCredentials(
                    settings.RABBITMQ_USER,
                    settings.RABBITMQ_PASSWORD
                ),
                virtual_host=settings.RABBITMQ_VHOST
            )
        )
        channel = connection.channel()

        # Declare exchange and queue for results
        channel.exchange_declare(
            exchange=settings.RESULTS_EXCHANGE_NAME,
            exchange_type='direct',
            durable=True
        )
        channel.queue_declare(
            queue=settings.RESULTS_QUEUE_NAME,
            durable=True
        )
        channel.queue_bind(
            exchange=settings.RESULTS_EXCHANGE_NAME,
            queue=settings.RESULTS_QUEUE_NAME,
            routing_key=settings.RESULTS_ROUTING_KEY
        )

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(
            queue=settings.RESULTS_QUEUE_NAME,
            on_message_callback=callback
        )

        logger.info("Excel writer service is ready to consume messages")
        channel.start_consuming()

    except Exception as e:
        logger.error(f"Excel writer service error: {e}")
        raise
