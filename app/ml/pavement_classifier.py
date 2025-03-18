
# Configure logging
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
import os
from threading import Event, Lock
import time
from typing import Any, Dict, Optional, Tuple

from torchvision import transforms
from openpyxl import Workbook
import openpyxl
import torch

import time
import mlflow.pytorch
import mlflow
from PIL import Image

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

            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS device")
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
        self.confidence_threshold = 0.55
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
