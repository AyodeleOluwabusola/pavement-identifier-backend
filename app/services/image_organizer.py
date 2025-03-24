import os
import shutil
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ImageOrganizer:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_output_dir: str = "categorized_images"):
        """
        Initialize the ImageOrganizer with a base output directory.
        Uses singleton pattern to ensure only one session directory is created.

        Args:
            base_output_dir: Base directory where categorized images will be stored
        """
        if not self._initialized:
            self.base_output_dir = base_output_dir
            self.categories = ['asphalt', 'chip-sealed', 'gravel', 'uncertain']
            self._create_category_dirs()
            self.__class__._initialized = True

    def _create_category_dirs(self) -> None:
        """Create directory structure for categorized images"""
        # Create timestamp-based subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.base_output_dir, timestamp)

        # Create directories for each category
        for category in self.categories:
            category_dir = os.path.join(self.session_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            logger.info(f"Created directory for {category}: {category_dir}")

    def organize_image(self, image_path: str, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Organize a single image based on its classification result.

        Args:
            image_path: Path to the original image
            classification_result: Dictionary containing classification details

        Returns:
            Dict containing the organization result
        """
        try:
            # Extract relevant information
            status = classification_result.get('Status', 'uncertain')
            predicted_class = classification_result.get('Predicted Class', 'uncertain')
            confidence = classification_result.get('Confidence', 0.0)

            # Determine target category
            if status == 'Success':
                target_category = predicted_class
            else:
                target_category = 'uncertain'

            # Create target path
            image_filename = os.path.basename(image_path)
            confidence_str = f"{confidence:.2f}" if confidence else "NA"
            new_filename = f"{confidence_str}_{image_filename}"
            target_path = os.path.join(self.session_dir, target_category, new_filename)

            logger.info("Target path: ", target_path)

            # Verify source image exists
            if not os.path.exists(image_path):
                logger.error(f"Source image does not exist: {image_path}")
                return {
                    'original_path': image_path,
                    'status': 'failed',
                    'error': 'Source image does not exist'
                }

            # Verify target directory exists and is writable
            target_dir = os.path.dirname(target_path)
            if not os.path.exists(target_dir):
                logger.info(f"Creating target directory: {target_dir}")
                os.makedirs(target_dir, exist_ok=True)

            # Try to copy the file with detailed error logging
            try:
                shutil.copy2(image_path, target_path)
                if not os.path.exists(target_path):
                    raise FileNotFoundError("File was not copied successfully")
                logger.info(f"Successfully copied image from {image_path} to {target_path}")
            except (PermissionError, OSError) as e:
                logger.error(f"Failed to copy file: {str(e)}")
                logger.error(f"Source path: {image_path}")
                logger.error(f"Target path: {target_path}")
                logger.error(f"Source exists: {os.path.exists(image_path)}")
                logger.error(f"Target dir exists: {os.path.exists(target_dir)}")
                logger.error(f"Source permissions: {oct(os.stat(image_path).st_mode)[-3:]}")
                return {
                    'original_path': image_path,
                    'status': 'failed',
                    'error': f'Failed to copy file: {str(e)}'
                }

            logger.info(f"Organized image {image_filename} into {target_category} category")

            return {
                'original_path': image_path,
                'new_path': target_path,
                'category': target_category,
                'confidence': confidence,
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"Error organizing image {image_path}: {e}")
            return {
                'original_path': image_path,
                'status': 'failed',
                'error': str(e)
            }

    def get_category_stats(self) -> Dict[str, int]:
        """
        Get statistics about the number of images in each category.

        Returns:
            Dict containing count of images in each category
        """
        stats = {}
        for category in self.categories:
            category_dir = os.path.join(self.session_dir, category)
            if os.path.exists(category_dir):
                stats[category] = len(os.listdir(category_dir))
            else:
                stats[category] = 0
        return stats
