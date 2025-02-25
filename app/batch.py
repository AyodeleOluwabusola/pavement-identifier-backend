import os
from app.core.celery_config import celery
from app.services.rabbitmq_service import publish_message
from app.services.file_service import read_image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@celery.task
def process_images_from_dir(directory_path: str):
    # List all image files in the directory
    logger.info(f"Executing process_images_from_dir: {directory_path}")
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
    
    if not image_files:
        logger.info(f"No images found in the directory: {directory_path}")
        return {"message": "No images found in the directory."}
    
    results = []
    
    for image_file in image_files:
        file_path = os.path.join(directory_path, image_file)
        
        # Read the image and convert it to base64
        encoded_image = read_image(file_path)
        
        if not encoded_image:
            results.append({"file": image_file, "status": "failed"})
            continue
        
        # Publish the base64 encoded image to RabbitMQ
        message = {
            "image_data": encoded_image,
            "file_name": image_file
        }
        publish_message(message)
        results.append({"file": image_file, "status": "success"})
    
    return {"message": "Processed images from the directory", "results": results}
