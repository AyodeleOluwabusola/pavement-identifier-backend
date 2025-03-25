import json
import logging
import os
import sys
import signal
from multiprocessing import Process, Event
from typing import List, Optional
import multiprocessing
from contextlib import contextmanager
import threading
import time

import pika

from app.core.config import settings
from app.ml.classifier_factory import create_classifier
from app.services.image_organizer import ImageOrganizer
from app.core.logger import setup_logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(
            f"Received signal {signum}. Initiating graceful shutdown...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


class ProcessManager:
    """Manages process lifecycle and resource cleanup"""

    def __init__(self):
        self._processes: List[multiprocessing.Process] = []
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

    def add_process(self, process: multiprocessing.Process):
        with self._lock:
            self._processes.append(process)

    def remove_process(self, process: multiprocessing.Process):
        with self._lock:
            if process in self._processes:
                self._processes.remove(process)

    def cleanup_resources(self):
        """Clean up any remaining resources"""
        try:
            # Get the internal resource tracker module
            resource_tracker = multiprocessing.resource_tracker._resource_tracker
            if resource_tracker is not None:
                # Clear the resource tracker's internal state
                resource_tracker._resource_tracker = None
                resource_tracker._pid = None
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")


def run_consumer():
    """Function to run in each consumer process"""
    process_logger = setup_logging(f"consumer_process_{os.getpid()}")
    process_logger.info(f"Starting consumer process with PID: {os.getpid()}")

    connection = None
    channel = None
    classifier = None

    def cleanup():
        nonlocal classifier, channel, connection
        try:
            if classifier:
                process_logger.info("Shutting down classifier...")
                # Safely call shutdown if it exists
                if hasattr(classifier, 'shutdown'):
                    classifier.shutdown()
                classifier = None

            if channel and channel.is_open:
                process_logger.info("Closing RabbitMQ channel...")
                channel.close()
                channel = None

            if connection and not connection.is_closed:
                process_logger.info("Closing RabbitMQ connection...")
                connection.close()
                connection = None

            process_logger.info("Cleanup completed successfully")
        except Exception as e:
            process_logger.error(f"Error during cleanup: {e}")

    try:
        setup_signal_handlers()
        classifier = create_classifier()
        classifier.initialize()

        # Initialize image organizer
        image_organizer = ImageOrganizer(settings.CATEGORIZED_IMAGES_DIR)

        # Initialize RabbitMQ connection
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=settings.RABBITMQ_HOST,
                port=settings.RABBITMQ_PORT,
                credentials=pika.PlainCredentials(
                    settings.RABBITMQ_USER,
                    settings.RABBITMQ_PASSWORD
                ),
                virtual_host=settings.RABBITMQ_VHOST,
                heartbeat=600,
                blocked_connection_timeout=300
            )
        )
        channel = connection.channel()

        # Declare exchange and queue
        channel.exchange_declare(
            exchange=settings.EXCHANGE_NAME,
            exchange_type='direct',
            durable=True
        )
        channel.queue_declare(
            queue=settings.QUEUE_NAME,
            durable=True
        )
        channel.queue_bind(
            exchange=settings.EXCHANGE_NAME,
            queue=settings.QUEUE_NAME,
            routing_key=settings.ROUTING_KEY
        )
        channel.basic_qos(prefetch_count=1)

        def process_message(ch, method, properties, body):
            try:
                # Log message received
                logger.info("Received message from RabbitMQ queue")

                # Check if classifier is ready
                if not classifier.is_ready.is_set():
                    logger.warning("Classifier not ready, rejecting message")
                    ch.basic_reject(
                        delivery_tag=method.delivery_tag, requeue=True)
                    return

                # Process the message
                message_data = json.loads(body)
                file_name = message_data.get("file_name")
                image_data = message_data.get("image_data")
                image_path = message_data.get("image_path")

                logger.info(f"Processing message for file: {file_name}")

                # Classify the image
                logger.debug(f"Starting classification for {file_name}")
                result = classifier.classify_image(image_data=image_data)

                # Organize the image if path is provided
                if image_path and os.path.exists(image_path) and settings.ORGANIZED_IMAGES_INTO_FOLDERS:
                    organization_result = image_organizer.organize_image(
                        image_path,
                        result
                    )
                    result['organization_result'] = organization_result

                # Publish result to queue
                if file_name and result:
                    classifier.publish_result(file_name, result)

                # Log classification results
                logger.info(
                    f"Processed {file_name}: "
                    f"Class={result.get('Predicted Class', 'unknown')}, "
                    f"Confidence={result.get('Confidence', 0):.2f}"
                )

                # Acknowledge message
                ch.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(
                    f"Successfully processed and acknowledged message for {file_name}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid message format: {e}")
                ch.basic_reject(
                    delivery_tag=method.delivery_tag, requeue=False)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                ch.basic_reject(delivery_tag=method.delivery_tag, requeue=True)

        # Start consuming
        channel.basic_consume(
            queue=settings.QUEUE_NAME,
            on_message_callback=process_message
        )

        logger.info(f"Process {os.getpid()} started consuming messages")
        channel.start_consuming()

    except (KeyboardInterrupt, SystemExit):
        logger.info(f"Process {os.getpid()} received shutdown signal")
    except Exception as e:
        logger.error(f"Process {os.getpid()} failed: {e}")
    finally:
        cleanup()


class ConsumerManager:
    def __init__(self, num_consumers: int = 3):
        self.num_consumers = num_consumers
        self.consumer_processes = []
        self._ready = Event()
        self._stopped = Event()
        self.process_manager = ProcessManager()

    @contextmanager
    def _managed_process(self, target_func, *args, **kwargs):
        """Context manager for process lifecycle management"""
        process = multiprocessing.Process(
            target=target_func, args=args, kwargs=kwargs)
        process.daemon = False

        try:
            self.process_manager.add_process(process)
            process.start()
            yield process
        finally:
            self.process_manager.remove_process(process)

    def start(self):
        """Start all consumers in separate processes"""
        try:
            for i in range(self.num_consumers):
                if self._stopped.is_set():
                    break

                with self._managed_process(run_consumer) as process:
                    self.consumer_processes.append(process)
                    logger.info(
                        f"Started consumer process {i+1} with PID {process.pid}")

            if self.consumer_processes and not self._stopped.is_set():
                self._ready.set()
                logger.info("Consumer manager is ready")

        except Exception as e:
            logger.error(f"Error starting consumer Manager: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop all consumer processes"""
        self._stopped.set()
        self._ready.clear()

        # First attempt graceful shutdown
        for process in self.consumer_processes[:]:  # Create a copy of the list
            if process.is_alive():
                try:
                    os.kill(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    self.consumer_processes.remove(process)
                    continue

        # Wait for processes to finish
        timeout = settings.CLEANUP_GRACE_PERIOD
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not any(p.is_alive() for p in self.consumer_processes):
                break
            time.sleep(0.1)

        # Force terminate any remaining processes
        for process in self.consumer_processes[:]:
            if process.is_alive():
                try:
                    process.terminate()
                    process.join(timeout=settings.PROCESS_SHUTDOWN_TIMEOUT)
                    if process.is_alive():
                        os.kill(process.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                finally:
                    self.consumer_processes.remove(process)

        # Clean up resources
        self.process_manager.cleanup_resources()
        logger.info("Consumer manager stopped and resources cleaned up")

    def is_ready(self) -> bool:
        """Check if the service is ready"""
        return self._ready.is_set() and any(process.is_alive() for process in self.consumer_processes)


def start_listener(num_consumers: int = 3) -> ConsumerManager:
    """Start the listener with multiple consumers"""
    try:
        manager = ConsumerManager(num_consumers)
        manager.start()
        logger.info("Started consumer manager")
        return manager
    except Exception as e:
        logger.error(f"Critical error in listener: {e}")
        raise
