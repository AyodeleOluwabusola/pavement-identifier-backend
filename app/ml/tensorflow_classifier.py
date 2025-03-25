from typing import Optional
import os

import numpy as np
try:
    import tensorflow as tf
    # Force TensorFlow to use the legacy keras path
    os.environ['TF_KERAS'] = '1'
    import mlflow.tensorflow
except ImportError as e:
    raise ImportError(
        "TensorFlow dependencies not found. "
        "Please install tensorflow and mlflow[tensorflow] packages"
    ) from e

from PIL import Image
from io import BytesIO
import base64
import logging

from app.ml.base_classifier import BasePavementClassifier
from app.core.config import settings

logger = logging.getLogger(__name__)

class TensorFlowPavementClassifier(BasePavementClassifier):
    def __init__(self):
        super().__init__()
        self._check_tensorflow_installation()
        mlflow.set_experiment("Tensorflow-Inference-CNN-Pavement_Surface_Classification")
        self.model_uri = settings.MODEL_URI_TENSORFLOW

    def _check_tensorflow_installation(self):
        """Verify TensorFlow installation and GPU availability"""
        try:
            tf_version = tf.__version__
            logger.info(f"TensorFlow version: {tf_version}")

            # Configure TensorFlow to use legacy keras implementation
            tf.keras.backend.clear_session()

            # Check GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"TensorFlow GPU devices available: {gpus}")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                logger.warning("No GPU devices available, using CPU")

        except Exception as e:
            logger.error(f"Error checking TensorFlow installation: {e}")
            raise

    def initialize(self) -> None:
        try:
            logger.info("Loading TensorFlow model from MLflow...")
            with self.lock:
                # Set specific TensorFlow configurations for model loading
                tf.keras.backend.clear_session()

                # Load the model with custom configuration
                custom_objects = {}  # Add any custom objects if needed
                self.model = mlflow.tensorflow.load_model(
                    self.model_uri,
                    keras_model_kwargs={
                        'compile': False,
                        'custom_objects': custom_objects
                    }
                )

                logger.info("Model loaded successfully")
                self.model.summary()
                self.is_ready.set()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            self.is_ready.clear()
            raise

    def transform_image(self, image_path:  Optional[str] = None,
                       image_base64: Optional[str] = None) -> np.ndarray:
        try:
            if image_base64:
                image_data = base64.b64decode(image_base64)
                img = Image.open(BytesIO(image_data)).convert('L')
            elif image_path:
                img = Image.open(image_path).convert('L')
            else:
                return None

            img = img.resize(self.IMG_SIZE)
            img_array = np.array(img)
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=-1)
            
            img_array = img_array.astype(np.float32) / 255.0
            img_array = img_array * 2 - 1.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array

        except Exception as e:
            logger.error(f"Error transforming image: {e}")
            return None

    def run_inference(self, img_array: np.ndarray) -> tuple:
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        return confidence, predicted_class_idx

    def shutdown(self):
        """Clean up TensorFlow resources"""
        try:
            if self.model is not None:
                # Clear keras backend session
                tf.keras.backend.clear_session()
                del self.model
                self.model = None

            self.is_ready.clear()
            logger.info("TensorFlow classifier shutdown completed")
        except Exception as e:
            logger.error(f"Error during TensorFlow classifier shutdown: {e}")
