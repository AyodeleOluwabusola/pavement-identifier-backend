from contextlib import contextmanager
import time
import mlflow.pytorch
import mlflow
from PIL import Image
from io import BytesIO
from torchvision import transforms
from app.core.config import settings
from concurrent.futures import ThreadPoolExecutor, Future
from openpyxl import Workbook
import openpyxl
import pika
from typing import Optional, Dict, Any, Tuple, List
from threading import Lock, Thread, Event
import logging
import json
import base64
import sys
import os

# Print environment information
print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Python path:", sys.path)
print("Current working directory:", os.getcwd())

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

# Rest of your imports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PavementClassifier:
    def __init__(self):
        print("Initializing PavementClassifier...")
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.model = None

        print("Checking device availability...")
        # Use MPS if available (for Apple Silicon), fall back to CPU
        try:
            print("MPS available:", torch.backends.mps.is_available())
            print("CUDA available:", torch.cuda.is_available())

            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using MPS device")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA device")
            else:
                self.device = torch.device("cpu")
                print("Using CPU device")
        except Exception as e:
            print(f"Error during device selection: {e}")
            self.device = torch.device("cpu")
            print("Falling back to CPU device")

        print(f"Selected device: {self.device}")

        self.lock = Lock()
        self.classes = ['asphalt', 'chip-sealed', 'gravel']
        self.model_uri = 'runs:/b76e4133aee04487acedf5708b66d7af/model'
        self.confidence_threshold = 0.55
        self.img_size = (256, 256)
        self.is_ready = Event()

        print("Setting up inference transform...")
        # Define the inference transform
        self.inference_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("L")),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        print("PavementClassifier initialization completed")

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
            # self.model = mlflow.pyfunc.load_model(self.model_uri)
            # print("Model dependencies ", mlflow.pyfunc.get_model_dependencies(self.model_uri))
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
            if image_path:
                img = Image.open(image_path)
            else:
                image_data = base64.b64decode(image_base64)
                img = Image.open(BytesIO(image_data))

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

            if max_prob >= self.confidence_threshold:
                return {
                    'Image Path': image_path,
                    'Message': "Success",
                    'Predicted Class': self.classes[pred_idx],
                    'Confidence': max_prob,
                    'Status': "Success"
                }
            else:
                return {
                    'Image Path': image_path,
                    'Message': "Uncertain",
                    'Predicted Class': "Uncertain",
                    'Confidence': max_prob,
                    'Status': "Uncertain"
                }

        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return {
                'Image Path': image_path,
                'Message': f"Error: {str(e)}",
                'Status': "Error"
            }

    def write_to_excel(self, file_name: str, result: Dict[str, Any]) -> None:
        """Write classification results to Excel"""
        with self.lock:
            try:
                logger.info("Writing results to Excel...")
                file_path = "image_processing_results.xlsx"
                if os.path.exists(file_path):
                    workbook = openpyxl.load_workbook(file_path)
                else:
                    workbook = Workbook()
                    sheet = workbook.active
                    sheet.title = "Image Processing Results"
                    sheet.append(
                        ["File Name", "Status", "Predicted Class", "Confidence"])

                sheet = workbook.active
                sheet.append([
                    file_name,
                    result.get('Status', 'Error'),
                    result.get('Predicted Class', 'Unknown'),
                    result.get('Confidence', 0.0)
                ])
                workbook.save(file_path)
            except Exception as e:
                logger.error(f"Error writing to Excel: {e}")


class RabbitMQConsumer:
    def __init__(self, classifier: PavementClassifier):
        self.classifier = classifier
        self.connection = None
        self.channel = None
        self._stopped = False
        self.queue_name = "image_queue"
        self.exchange_name = "image_exchange"
        self.routing_key = "image_routing_key"

    def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host='localhost')
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

            logger.info(f"Processing message for file: {file_name}")

            # Classify the image
            logger.info(f"Starting classification for {file_name}")
            result = self.classifier.classify_image(image_data=image_data)
            logger.info(
                f"Classification result for {file_name}: {result.get('Status')} (Class: {result.get('Predicted Class')}, Confidence: {result.get('Confidence', 0):.2f})")

            # Write results to Excel
            if file_name and result:
                logger.info(f"Writing results to Excel for {file_name}")
                self.classifier.write_to_excel(file_name, result)

            # Acknowledge successful processing
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(
                f"Successfully processed and acknowledged message for {file_name}")

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
    def __init__(self, num_consumers: int = 3):
        self.num_consumers = num_consumers
        self.consumers: List[RabbitMQConsumer] = []
        self.consumer_threads: List[Thread] = []
        self.classifier = PavementClassifier()
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
            logger.error(f"Error starting initialization: {e}")
            self.stop()
            raise

    def _initialize(self):
        """Initialize the classifier and start consumers"""
        try:
            # Initialize the classifier
            self.classifier.initialize()

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


if __name__ == "__main__":
    start_listener()
