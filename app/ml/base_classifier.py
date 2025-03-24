from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from threading import Lock, Event
import os
import logging
import mlflow
from openpyxl import Workbook
import openpyxl

from app.core.config import settings

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
        
        mlflow.set_tracking_uri("http://52.42.208.9:5000/")

    @abstractmethod
    def initialize(self) -> None:
        pass

    def write_to_excel(self, file_name: str, result: Dict[str, Any]) -> None:
        """Write classification results to Excel"""
        with self.lock:
            try:
                logger.info("Writing results to Excel...")
                file_path = settings.EXCEL_RESULTS_PATH
                if os.path.exists(file_path):
                    workbook = openpyxl.load_workbook(file_path)
                else:
                    workbook = Workbook()
                    sheet = workbook.active
                    sheet.title = "Image Processing Results"
                    sheet.append(
                        ["File Name", "Status", "Predicted Class", "Confidence"]
                    )

                sheet = workbook.active
                sheet.append([
                    file_name,
                    result.get('Status', 'Error'),
                    result.get('Predicted Class', 'Unknown'),
                    result.get('Confidence', 0.0)
                ])
                workbook.save(file_path)
                logger.info(f"Successfully wrote results for {file_name} to Excel")
            except Exception as e:
                logger.error(f"Error writing to Excel: {e}")

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

            # Write results to Excel if image_path is provided
            if image_path:
                file_name = os.path.basename(image_path)
                self.write_to_excel(file_name, result)

            return result

        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return {
                'Status': 'Error',
                'Message': f'Classification failed: {str(e)}'
            }