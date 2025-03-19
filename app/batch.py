import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

from tqdm import tqdm

from app.services.file_service import read_image
from app.services.rabbitmq_service import publish_message

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_image(file_path: str, image_file: str) -> Dict[str, Any]:
    """Process a single image and publish to RabbitMQ"""
    try:
        # Read the image and convert it to base64
        encoded_image = read_image(file_path)

        if not encoded_image:
            logger.error(f"Failed to read image: {image_file}")
            return {"file": image_file, "status": "failed", "error": "Failed to read image"}

        # Publish the base64 encoded image to RabbitMQ
        message = {
            "image_data": encoded_image,
            "file_name": image_file,
            "image_path": file_path
        }

        if publish_message(message):
            logger.info(f"Successfully published: {image_file}")
            return {"file": image_file, "status": "success"}
        else:
            logger.error(f"Failed to publish: {image_file}")
            return {"file": image_file, "status": "failed", "error": "Failed to publish to queue"}

    except Exception as e:
        logger.error(f"Error processing {image_file}: {str(e)}")
        return {"file": image_file, "status": "failed", "error": str(e)}


def process_images_from_dir(directory_path: str) -> Dict[str, Any]:
    """Process all images in a directory concurrently"""
    logger.info(f"Processing images from directory: {directory_path}")

    try:
        # Validate directory exists
        if not os.path.exists(directory_path):
            error_msg = f"Directory does not exist: {directory_path}"
            logger.error(error_msg)
            return {"message": error_msg, "status": "failed"}

        # List all image files in the directory
        image_files = [
            f for f in os.listdir(directory_path)
            if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))
        ]

        if not image_files:
            logger.info(f"No images found in directory: {directory_path}")
            return {"message": "No images found in the directory.", "status": "completed"}

        results = []
        failed_count = 0
        success_count = 0

        # Process images concurrently with progress bar
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Create futures for all images
            future_to_file = {
                executor.submit(
                    process_single_image,
                    os.path.join(directory_path, image_file),
                    image_file
                ): image_file
                for image_file in image_files
            }

            # Process results as they complete with progress bar
            for future in tqdm(as_completed(future_to_file), total=len(image_files), desc="Processing images"):
                result = future.result()
                results.append(result)

                if result["status"] == "success":
                    success_count += 1
                else:
                    failed_count += 1

        return {
            "message": "Completed processing images from directory",
            "status": "completed",
            "total_images": len(image_files),
            "successful": success_count,
            "failed": failed_count,
            "results": results
        }

    except Exception as e:
        error_msg = f"Error processing directory: {str(e)}"
        logger.error(error_msg)
        return {
            "message": error_msg,
            "status": "failed",
            "error": str(e)
        }
