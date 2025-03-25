
import json
import logging
import os
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Dict, Any, Optional, Tuple
from io import BytesIO
from threading import Lock, Event
import base64

import mlflow
import pika
import torch
from PIL import Image
from torchvision import transforms

from app.core.config import settings
from app.core.logger import setup_logging

logger = setup_logging(__name__)


class PavementClassifier:
    def __init__(self):
        print("Initializing PavementClassifier...")
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.model = None

        print("Checking device availability...")
        # Use MPS if available (for Apple Silicon), fall back to CPU
        try:

            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using MPS device")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA device")
            else:
                self.device = torch.device("cpu")
                logger.warning("Using CPU device")
        except Exception as e:
            logger.error(f"Error during device selection: {e}")
            self.device = torch.device("cpu")
            logger.warning("Falling back to CPU device")

        logger.info(f"Selected device: {self.device}")

        self.lock = Lock()
        self.classes = ['asphalt', 'chip-sealed', 'gravel']
        self.model_uri = 'runs:/b76e4133aee04487acedf5708b66d7af/model'
        self.confidence_threshold = 0.9
        self.img_size = (256, 256)
        self.is_ready = Event()

        logger.info("Setting up inference transform...")
        # Define the inference transform
        self.inference_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("L")),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        logger.info("PavementClassifier initialization completed")

        # Initialize RabbitMQ connection and channel as None
        self._connection = None
        self._channel = None

    def initialize(self):
        """Initialize MLflow and load model asynchronously"""
        try:
            self._setup_mlflow()
            self._load_model()
            self.is_ready.set()
            logger.info("PavementClassifier initialization completed")
        except Exception as e:
            logger.error(f"Failed to initialize PavementClassifier: {e}")
            raise

    def _setup_mlflow(self) -> None:
        """Setup MLflow connection with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                mlflow.set_tracking_uri("http://52.42.208.9:5000/")
                mlflow.set_experiment(
                    "Pytorch_CNN_from_Scratch_Pavement_Surface_Classification")
                logger.info("MLflow connection established successfully")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to connect to MLflow after {max_retries} attempts: {e}")
                    raise
                logger.warning(
                    f"MLflow connection attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)

    def _load_model(self) -> None:
        """Load the model with proper error handling"""
        try:
            logger.info("Loading model from MLflow...")
            self.model = mlflow.pytorch.load_model(
                self.model_uri,
                map_location=self.device
            )

            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait until the classifier is ready to process images"""
        return self.is_ready.wait(timeout=timeout)

    def transform_image(self, image_path: Optional[str] = None, image_base64: Optional[str] = None) -> Optional[torch.Tensor]:
        """Transform image for model input"""
        try:
            if image_base64:
                image_data = base64.b64decode(image_base64)
                img = Image.open(BytesIO(image_data))  # Use PIL.Image instead of mlflow.Image
            elif image_path:
                img = Image.open(image_path)  # Use PIL.Image instead of mlflow.Image
            else:
                logger.error("Neither image path nor base64 data provided")
                return None

            img_tensor = self.inference_transform(img)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            return img_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Error transforming image: {e}")
            return None

    @torch.no_grad()
    def run_inference(self, img_tensor: torch.Tensor) -> Tuple[float, int]:
        """Run model inference"""
        try:
            logger.info(f"Running inference on image: {img_tensor}")
            outputs = self.model(img_tensor)
            logger.info(f"Inference results: {outputs}")
            probs = torch.softmax(outputs, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
            return max_prob.item(), pred_idx.item()
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def classify_image(self, image_path: Optional[str] = None, image_data: Optional[str] = None) -> Dict[str, Any]:
        """Classify a single image"""
        try:
            img_tensor = self.transform_image(image_path, image_data)
            if img_tensor is None:
                return {
                    'Image Path': image_path,
                    'Message': "Error processing Image",
                    'Status': "Error"
                }

            max_prob, pred_idx = self.run_inference(img_tensor)

            logger.debug(
                f"Classification result: {self.classes[pred_idx]} with confidence {max_prob}")

            status = "Success" if max_prob >= self.confidence_threshold else "Uncertain"
            predicted_class = self.classes[pred_idx] if max_prob >= self.confidence_threshold else "Uncertain"

            return {
                'Image Path': image_path,
                'Message': status,
                'Predicted Class': predicted_class,
                'Confidence': max_prob,
                'Status': status
            }

        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return {
                'Image Path': image_path,
                'Message': f"Error: {str(e)}",
                'Status': "Error"
            }

    def _get_rabbitmq_channel(self):
        """Get or create a RabbitMQ channel"""
        try:
            # Check if we have a working connection and channel
            if (self._connection is None or not self._connection.is_open or
                    self._channel is None or not self._channel.is_open):

                # Create new connection if needed
                if self._connection is None or not self._connection.is_open:
                    self._connection = pika.BlockingConnection(
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

    def shutdown(self):
        """Cleanup resources used by the classifier"""
        try:
            # Shutdown the thread pool executor
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)

            # Clear the model from memory
            if hasattr(self, 'model'):
                self.model = None

            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clear the ready flag
            self.is_ready.clear()

            # Add RabbitMQ cleanup
            self._cleanup_rabbitmq()

            logger.info("PavementClassifier shutdown completed")
        except Exception as e:
            logger.error(f"Error during PavementClassifier shutdown: {e}")
