import json
import logging
import os
import sys
from threading import Thread, Event
from typing import List

import pika

from app.core.config import settings
from app.ml.pavement_classifier import PavementClassifier
from app.services.image_organizer import ImageOrganizer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Issue with reading torch import this resolves it.
try:
    import torch
    print("PyTorch version:", torch.__version__)
except ImportError as e:
    print("Failed to import torch:", e)
    print("Checking if torch is in site-packages...")
    python_path = sys.executable
    site_packages = os.path.join(os.path.dirname(
        os.path.dirname(python_path)), 'lib/python3.11/site-packages')
    print("Site packages location:", site_packages)
    if os.path.exists(os.path.join(site_packages, 'torch')):
        print("torch directory exists in site-packages")
    else:
        print("torch directory not found in site-packages")



class RabbitMQConsumer:
    def __init__(self, classifier: PavementClassifier):
        self.classifier = classifier
        self.connection = None
        self.channel = None
        self._stopped = False
        self.queue_name = "image_queue"
        self.exchange_name = "image_exchange"
        self.routing_key = "image_routing_key"
        self.image_organizer = ImageOrganizer(settings.CATEGORIZED_IMAGES_DIR)

    def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            self.connection = pika.BlockingConnection(
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
            self.channel = self.connection.channel()

            # Declare exchange and queue
            self.channel.exchange_declare(
                exchange=self.exchange_name,
                exchange_type='direct',
                durable=True
            )
            self.channel.queue_declare(
                queue=self.queue_name,
                durable=True
            )
            self.channel.queue_bind(
                exchange=self.exchange_name,
                queue=self.queue_name,
                routing_key=self.routing_key
            )
            self.channel.basic_qos(prefetch_count=1)
            logger.info("Successfully connected to RabbitMQ")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self.stop()
            raise

    def start_consuming(self):
        """Start consuming messages"""
        if not self.channel or self.channel.is_closed:
            logger.error("Cannot start consuming - channel is not open")
            return

        try:
            self.channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=self._process_message
            )
            logger.info("Started consuming messages")
            self.channel.start_consuming()

        except Exception as e:
            logger.error(f"Error while consuming messages: {e}")
            self.stop()

    def _process_message(self, ch, method, properties, body):
        """Process incoming messages"""
        try:
            # Log message received
            logger.info("Received message from RabbitMQ queue")

            # Check if classifier is ready
            if not self.classifier.is_ready.is_set():
                logger.warning("Classifier not ready, rejecting message")
                ch.basic_reject(delivery_tag=method.delivery_tag, requeue=True)
                return

            # Process the message
            message_data = json.loads(body)
            file_name = message_data.get("file_name")
            image_data = message_data.get("image_data")
            image_path = message_data.get("image_path")

            logger.info(f"Processing message for file: {file_name}")

            # Classify the image
            logger.debug(f"Starting classification for {file_name}")
            result = self.classifier.classify_image(image_data=image_data)

            # Organize the image if path is provided
            print(f"image_path here: {image_path}")
            # print(f"image_data here: {image_data}")
            if image_path or image_data and os.path.exists(image_path) and settings.ORGANIZED_IMAGES_INTO_FOLDERS:
                print(f"Organizing image..., image_path: {image_path}")
                organization_result = self.image_organizer.organize_image(
                    image_path,
                    result
                )
                result['organization_result'] = organization_result
            logger.debug(
                f"Classification result for {file_name}: {result.get('Status')} (Class: {result.get('Predicted Class')}, Confidence: {result.get('Confidence', 0):.2f})")

            # Write results to Excel
            if file_name and result:
                logger.info(f"Writing results to Excel for {file_name}")
                self.classifier.write_to_excel(file_name, result)

            # Log classification and organization results
            logger.info(
                f"Processed {file_name}: "
                f"Class={result.get('Predicted Class', 'unknown')}, "
                f"Confidence={result.get('Confidence', 0):.2f}"
            )

            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"Successfully processed and acknowledged message for {file_name}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid message format: {e}")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=True)

    def stop(self):
        """Stop consuming messages and close connection"""
        self._stopped = True
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.stop_consuming()
                self.channel.close()
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            logger.info("Consumer stopped successfully")
        except Exception as e:
            logger.error(f"Error while stopping consumer: {e}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop()


class ConsumerManager:
    def __init__(self, pavement_classifier: PavementClassifier, num_consumers: int = 3):
        self.num_consumers = num_consumers
        self.consumers: List[RabbitMQConsumer] = []
        self.consumer_threads: List[Thread] = []
        self.classifier = pavement_classifier
        self.initialization_thread = None
        self._ready = Event()
        self._stopped = Event()

    def start(self):
        """Start all consumers and initialize classifier asynchronously"""
        try:
            # Start classifier initialization in a separate thread
            self.initialization_thread = Thread(target=self._initialize)
            self.initialization_thread.daemon = True
            self.initialization_thread.start()
            logger.info("Started classifier initialization")

        except Exception as e:
            logger.error(f"Error starting consumer Manager: {e}")
            self.stop()
            raise

    def _initialize(self):
        """Initialize the classifier and start consumers"""
        try:
            # Start consumers
            for i in range(self.num_consumers):
                if self._stopped.is_set():
                    break

                consumer = RabbitMQConsumer(self.classifier)
                consumer.connect()
                thread = Thread(target=consumer.start_consuming)
                thread.daemon = True
                thread.start()
                self.consumers.append(consumer)
                self.consumer_threads.append(thread)
                logger.info(f"Started consumer {i+1}")

            # Mark as ready if we have active consumers
            if self.consumers and not self._stopped.is_set():
                self._ready.set()
                logger.info("Consumer manager is ready")

        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            self.stop()

    def stop(self):
        """Stop all consumers gracefully"""
        self._stopped.set()
        self._ready.clear()
        for consumer in self.consumers:
            consumer.stop()
        self.consumers.clear()
        self.consumer_threads.clear()
        logger.info("Consumer manager stopped")

    def is_ready(self) -> bool:
        """Check if the service is ready"""
        return self._ready.is_set() and any(thread.is_alive() for thread in self.consumer_threads)


def start_listener(classifier: PavementClassifier, num_consumers: int = 3) -> ConsumerManager:
    """Start the listener with multiple consumers"""
    try:
        manager = ConsumerManager(classifier, num_consumers)
        manager.start()
        logger.info("Started consumer manager")
        return manager
    except Exception as e:
        logger.error(f"Critical error in listener: {e}")
        raise
