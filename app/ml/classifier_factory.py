from app.core.config import settings
from app.ml.base_classifier import BasePavementClassifier
from app.ml.tensorflow_classifier import TensorFlowPavementClassifier
from app.ml.pytorch_classifier import PyTorchPavementClassifier

def create_classifier() -> BasePavementClassifier:
    """Factory function to create the appropriate classifier based on config"""
    framework = settings.FRAMEWORK_IN_USE.lower()
    if framework == "tensorflow":
        return TensorFlowPavementClassifier()
    elif framework == "pytorch":
        return PyTorchPavementClassifier()
    else:
        raise ValueError(f"Unsupported framework: {framework}")