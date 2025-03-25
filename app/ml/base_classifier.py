from abc import ABC, abstractmethod
import json
import time
from typing import Dict, Any, Optional
from threading import Lock, Event
import os
import logging
import mlflow
from openpyxl import Workbook
import openpyxl
import pika

from app.core.config import settings
from app.services.rabbitmq_service import get_rabbitmq_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasePavementClassifier(ABC):
    def __init__(self):
        self.model = None
        self.is_ready = Event()
        self.lock = Lock()
        self.IMG_SIZE = (256, 256)
        self.classes = ['asphalt', 'chip-sealed', 'gravel']
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Initialize RabbitMQ connection and channel as None
        self._connection = None
        self._channel = None

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    @abstractmethod
    def initialize(self) -> None:
        pass

    def _get_rabbitmq_channel(self):
        """Get or create a RabbitMQ channel"""
        try:
            # Check if we have a working connection and channel
            if (self._connection is None or not self._connection.is_open or
                    self._channel is None or not self._channel.is_open):

                # Create new connection if needed
                if self._connection is None or not self._connection.is_open:
                    self._connection = get_rabbitmq_connection()

                # Create new channel if needed
                if self._channel is None or not self._channel.is_open:
                    self._channel = self._connection.channel()

                    # Declare exchange (idempotent operation)
                    self._channel.exchange_declare(
                        exchange=settings.RESULTS_EXCHANGE_NAME,
                        exchange_type='direct',
                        durable=True
                    )

            return self._channel

        except Exception as e:
            logger.error(f"Error getting RabbitMQ channel: {e}")
            # Clean up any partially initialized resources
            self._cleanup_rabbitmq()
            raise

    def publish_result(self, file_name: str, result: Dict[str, Any]) -> None:
        """Publish classification results to queue"""
        message = {
            "file_name": file_name,
            "status": result.get('Status', 'Error'),
            "predicted_class": result.get('Predicted Class', 'Unknown'),
            "confidence": result.get('Confidence', 0.0)
        }

        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                channel = self._get_rabbitmq_channel()

                channel.basic_publish(
                    exchange=settings.RESULTS_EXCHANGE_NAME,
                    routing_key=settings.RESULTS_ROUTING_KEY,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                        content_type='application/json'
                    )
                )

                logger.info(f"Published results for {file_name} to queue")
                return

            except (pika.exceptions.AMQPConnectionError,
                    pika.exceptions.AMQPChannelError) as e:
                logger.warning(
                    f"RabbitMQ connection error on attempt {attempt + 1}: {e}")
                self._cleanup_rabbitmq()  # Clean up the failed connection

                if attempt < max_retries - 1:
                    # Exponential backoff
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    logger.error("Failed to publish message after all retries")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error publishing results: {e}")
                raise

    def _cleanup_rabbitmq(self):
        """Clean up RabbitMQ resources"""
        try:
            if self._channel and self._channel.is_open:
                self._channel.close()
        except Exception:
            pass

        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
        except Exception:
            pass

        self._channel = None
        self._connection = None



    @abstractmethod
    def transform_image(self, image_path: Optional[str] = None,
                       image_base64: Optional[str] = None) -> Any:
        pass

    @abstractmethod
    def run_inference(self, img_data: Any) -> tuple:
        pass

    def classify_image(self, image_path: Optional[str] = None,
                      image_data: Optional[str] = None) -> Dict:
        try:
            if not self.is_ready.is_set():
                return {
                    'Status': 'Error',
                    'Message': 'Model not initialized'
                }

            img_data = self.transform_image(image_path, image_data)
            if img_data is None:
                return {
                    'Status': 'Error',
                    'Message': 'Failed to process image'
                }

            with self.lock:
                confidence, predicted_class_idx = self.run_inference(img_data)

            status = "Success" if confidence >= self.confidence_threshold else "Uncertain"
            predicted_class = self.classes[predicted_class_idx] if confidence >= self.confidence_threshold else "Uncertain"

            result = {
                'Status': status,
                'Predicted Class': predicted_class,
                'Confidence': confidence
            }

            return result

        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return {
                'Status': 'Error',
                'Message': f'Classification failed: {str(e)}'
            }

    @abstractmethod
    def shutdown(self):
        pass
