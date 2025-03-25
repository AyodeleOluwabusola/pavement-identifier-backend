import torch
from torchvision import transforms
import mlflow.pytorch
from PIL import Image
from io import BytesIO
import base64
import logging
from typing import Optional, Any

from app.ml.base_classifier import BasePavementClassifier
from app.core.config import settings

logger = logging.getLogger(__name__)

class PyTorchPavementClassifier(BasePavementClassifier):
    def __init__(self):
        super().__init__()
        mlflow.set_experiment("Pytorch_CNN_from_Scratch_Pavement_Surface_Classification")
        self.model_uri = settings.MODEL_URI_PYTORCH
        
        try:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        except:
            self.device = torch.device("cpu")
            
        self.inference_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("L")),
            transforms.Resize(self.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def initialize(self) -> None:
        try:
            logger.info("Loading PyTorch model from MLflow...")
            with self.lock:
                self.model = mlflow.pytorch.load_model(
                    self.model_uri,
                    map_location=self.device
                )
                self.model.to(self.device)
                self.model.eval()
                self.is_ready.set()
                logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_ready.clear()
            raise

    def transform_image(self, image_path: Optional[str] = None,
                       image_base64: Optional[str] = None) -> Any:
        try:
            if image_base64:
                image_data = base64.b64decode(image_base64)
                img = Image.open(BytesIO(image_data))
            elif image_path:
                img = Image.open(image_path)
            else:
                return None

            img_tensor = self.inference_transform(img)
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Error transforming image: {e}")
            return None

    def run_inference(self, img_tensor: Any) -> tuple:
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
            return max_prob.item(), pred_idx.item()