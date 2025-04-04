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
import prometheus_client
from prometheus_client import multiprocess
from prometheus_client import CollectorRegistry, generate_latest

from app.core.logger import setup_logging
from app.core.config import settings  # Update this import
from app.core.metrics import (
    CLASSIFICATION_DISTRIBUTION,
    CLASSIFICATION_LATENCY,
    CLASSIFICATION_REQUESTS,
    PROCESS_MODEL_STATUS,
    ACTIVE_CLASSIFICATIONS,
    PROCESS_LAST_HEARTBEAT,
    ACTIVE_PROCESSES,
    clear_process_metrics
)
from app.ml.classifier_factory import create_classifier
from app.services.image_organizer import ImageOrganizer

# Initialize logger using setup_logging
logger = setup_logging(__name__)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        if not shutdown_event.is_set():
            shutdown_event.set()
            logger.info(
                f"Received signal {signum}. Initiating graceful shutdown...")
            # Instead of sys.exit(0), we'll use the event to coordinate shutdown
            raise SystemExit()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    return shutdown_event


class ProcessManager:
    """Manages process lifecycle and resource cleanup"""

    def __init__(self):
        self._processes: List[multiprocessing.Process] = []
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # Configure prometheus multiprocess mode
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc"
        if not os.path.exists("/tmp/prometheus_multiproc"):
            os.makedirs("/tmp/prometheus_multiproc")

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
            # Clean up prometheus multiprocess files
            multiprocess.mark_process_dead(process_number)

            # Get the internal resource tracker module
            resource_tracker = multiprocessing.resource_tracker._resource_tracker
            if resource_tracker is not None:
                resource_tracker._resource_tracker = None
                resource_tracker._pid = None
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")


def run_consumer(process_number: str):
    """Function to run in each consumer process"""
    process_logger = setup_logging(f"consumer_process_{process_number}")
    process_logger.info(f"Starting consumer process number: {process_number}")

    # Clear any existing metrics for this process number first
    PROCESS_MODEL_STATUS.labels(
        process_id=process_number).set(-1)  # Reset to error state
    ACTIVE_CLASSIFICATIONS.labels(process_id=process_number).set(0)
    PROCESS_LAST_HEARTBEAT.labels(process_id=process_number).set(time.time())

    # Now set initial state
    PROCESS_MODEL_STATUS.labels(
        process_id=process_number).set(0)  # 0 = initializing
    ACTIVE_PROCESSES.inc()

    shutdown_event = setup_signal_handlers()

    def cleanup():
        nonlocal classifier, channel, connection
        try:
            if classifier:
                process_logger.info("Shutting down classifier...")
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

    @contextmanager
    def track_active_classification():
        """Context manager to track active classifications"""
        ACTIVE_CLASSIFICATIONS.labels(process_id=process_number).inc()
        try:
            yield
        finally:
            ACTIVE_CLASSIFICATIONS.labels(process_id=process_number).dec()

    try:
        classifier = create_classifier()
        classifier.initialize()

        # Update model status to ready
        PROCESS_MODEL_STATUS.labels(
            process_id=process_number).set(1)  # 1 = ready

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
                # Update heartbeat
                PROCESS_LAST_HEARTBEAT.labels(
                    process_id=process_number).set(time.time())

                # Log message received
                logger.info("Received message from RabbitMQ queue")

                # Check if classifier is ready
                if not classifier.is_ready.is_set():
                    logger.warning("Classifier not ready, rejecting message")
                    PROCESS_MODEL_STATUS.labels(
                        process_id=process_number).set(0)
                    ch.basic_reject(
                        delivery_tag=method.delivery_tag, requeue=True)
                    return

                # Process the message
                message_data = json.loads(body)
                file_name = message_data.get("file_name")
                image_data = message_data.get("image_data")
                image_path = message_data.get("image_path")

                logger.info(f"Processing message for file: {file_name}")

                # Use the context manager to ensure proper tracking
                with track_active_classification():
                    logger.debug(f"Starting classification for {file_name}")
                    result = classifier.classify_image(image_data=image_data)

                    # Record classification result
                    if result.get('Status') == 'Success':
                        CLASSIFICATION_REQUESTS.labels(status="success").inc()
                        CLASSIFICATION_DISTRIBUTION.labels(
                            predicted_class=result.get(
                                'Predicted Class', 'unknown').lower()
                        ).inc()
                    else:
                        CLASSIFICATION_REQUESTS.labels(status="error").inc()

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

        logger.info(f"Process {process_number} started consuming messages")
        channel.start_consuming()

        while not shutdown_event.is_set():
            try:
                connection.process_data_events(timeout=1.0)
            except Exception as e:
                if not shutdown_event.is_set():
                    process_logger.error(f"Error processing messages: {e}")
                    time.sleep(1)

    except (KeyboardInterrupt):
        process_logger.info(
            f"Process {process_number} received shutdown signal")
    except Exception as e:
        process_logger.error(f"Process {process_number} failed: {e}")
        PROCESS_MODEL_STATUS.labels(
            process_id=process_number).set(-1)  # -1 = error
    finally:
        # Cleanup metrics
        PROCESS_MODEL_STATUS.labels(process_id=process_number).set(-1)
        ACTIVE_CLASSIFICATIONS.labels(process_id=process_number).set(0)
        ACTIVE_PROCESSES.dec()
        cleanup()


class ConsumerManager:
    def __init__(self, num_consumers: int = 3):
        self.num_consumers = num_consumers
        self.consumer_processes = []
        self._ready = Event()
        self._stopped = Event()
        self.process_manager = ProcessManager()

    @contextmanager
    def _managed_process(self, target_func, process_number, *args, **kwargs):
        """Context manager for process lifecycle management"""
        process = multiprocessing.Process(
            target=target_func,
            args=(process_number, *args),
            kwargs=kwargs
        )
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

                # Convert to string for consistency
                process_number = str(i + 1)
                with self._managed_process(run_consumer, process_number) as process:
                    self.consumer_processes.append(process)
                    logger.info(
                        f"Started consumer process {process_number} with PID {process.pid}")

            if self.consumer_processes and not self._stopped.is_set():
                self._ready.set()
                logger.info("Consumer manager is ready")

        except Exception as e:
            logger.error(f"Error starting consumer Manager: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop all consumer processes"""
        if self._stopped.is_set():
            return

        self._stopped.set()
        self._ready.clear()

        logger.info("Initiating graceful shutdown of consumer processes...")

        # First attempt graceful shutdown
        for i, process in enumerate(self.consumer_processes[:]):
            if process.is_alive():
                try:
                    process_number = str(i + 1)
                    os.kill(process.pid, signal.SIGTERM)
                    # Clear metrics immediately
                    clear_process_metrics(process_number)
                except ProcessLookupError:
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
                    pid = str(process.pid)
                    process.terminate()
                    process.join(timeout=settings.PROCESS_SHUTDOWN_TIMEOUT)
                    if process.is_alive():
                        os.kill(process.pid, signal.SIGKILL)
                    # Ensure metrics are cleaned up
                    clear_process_metrics(pid)
                except (OSError, ProcessLookupError):
                    pass

        # Clean up resources
        self.process_manager.cleanup_resources()
        self.consumer_processes.clear()
        logger.info("Consumer manager stopped and resources cleaned up")

    def is_ready(self) -> bool:
        """Check if the service is ready"""
        return self._ready.is_set() and any(process.is_alive() for process in self.consumer_processes)


def start_listener(num_consumers: int = settings.RABBITMQ_NUM_CONSUMERS) -> ConsumerManager:
    """Start the listener with multiple consumers"""
    try:
        manager = ConsumerManager(num_consumers)
        manager.start()
        logger.info(f"Started consumer manager with {num_consumers} consumers")
        return manager
    except Exception as e:
        logger.error(f"Critical error in listener: {e}")
        raise
