import asyncio
import base64
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from app.batch import process_images_from_dir
from app.core.config import settings
from app.core.logger import setup_logging
from app.ml.pavement_classifier import PavementClassifier
from app.services.listener import start_listener
from app.services.rabbitmq_service import publish_message
from io import BytesIO
from PIL import Image

# Configure logging
logger = setup_logging()

app = FastAPI(title=settings.APP_NAME)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a ThreadPoolExecutor with a reasonable number of workers
executor = ThreadPoolExecutor(max_workers=4)

# Store background task status
task_status: Dict[str, Any] = {}

# Single instance of classifier shared across the application
classifier = PavementClassifier()

# Global variable to store the consumer manager
consumer_manager = None

# Add this class for request validation


class ImageRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    file_name: str = Field(..., description="Name of the image file")

def initialize_consumer_manager():
    """Initialize the consumer manager with shared classifier"""
    global consumer_manager
    try:
        # Initialize classifier first
        classifier.initialize()
        # Pass the initialized classifier to the consumer manager
        consumer_manager = start_listener(
            classifier=classifier,
            num_consumers=settings.RABBITMQ_NUM_CONSUMERS
        )
        logger.info("Consumer manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize consumer manager: {e}")
        consumer_manager = None


@app.get("/")
def home():
    return {"message": "Welcome to Pavement Identifier!"}


@app.get("/config")
def get_config():
    return {
        "APP_NAME": settings.APP_NAME,
        "DEBUG": settings.DEBUG,
        "DATABASE_URL": settings.DATABASE_URL,
        "RABBITMQ_URL": settings.RABBITMQ_URL
    }


async def process_startup_directory():
    """Process the startup directory if specified"""
    if not settings.BATCH_PROCESSING_STARTUP_DIRECTORY:
        return

    logger.info(
        f"Processing startup directory: {settings.BATCH_PROCESSING_STARTUP_DIRECTORY}")

    try:
        # Wait for consumer manager to be ready
        for _ in range(30):  # Wait up to 30 seconds
            if consumer_manager and consumer_manager.is_ready():
                break
            await asyncio.sleep(1)

        if not consumer_manager or not consumer_manager.is_ready():
            logger.error(
                "Consumer manager not ready after waiting. Skipping startup directory processing.")
            return

        # Process the directory
        await publish_images_from_directory(settings.BATCH_PROCESSING_STARTUP_DIRECTORY, BackgroundTasks())
        logger.info(
            f"Completed processing startup directory: {settings.BATCH_PROCESSING_STARTUP_DIRECTORY}")

    except Exception as e:
        logger.error(f"Error processing startup directory: {e}")


@app.on_event("startup")
async def startup_event():
    """Start the initialization in a separate thread"""
    threading.Thread(target=initialize_consumer_manager, daemon=True).start()
    logger.info("Started consumer manager initialization")

    # Process startup directory if specified
    if settings.BATCH_PROCESSING_STARTUP_DIRECTORY:
        logger.info(
            f"Startup directory specified: {settings.BATCH_PROCESSING_STARTUP_DIRECTORY}")
        # Create task to process directory after service is ready
        asyncio.create_task(process_startup_directory())


@app.on_event("shutdown")
async def shutdown_event():
    global consumer_manager
    if consumer_manager:
        logger.info("Shutting down RabbitMQ consumers...")
        consumer_manager.stop()


def process_directory_background(directory_path: str):
    """Background task to process directory"""
    try:
        # Update task status
        task_status[directory_path] = {
            "status": "processing", "message": "Started processing directory"}

        # Process the directory
        result = process_images_from_dir(directory_path)

        # Update task status with result
        task_status[directory_path] = {
            "status": "completed" if result.get("status") != "failed" else "failed",
            **result
        }

    except Exception as e:
        logger.error(f"Error in background task: {e}")
        task_status[directory_path] = {
            "status": "failed",
            "message": f"Error: {str(e)}"
        }


@app.post("/publish-images-from-dir/")
async def publish_images_from_directory(directory_path: str, background_tasks: BackgroundTasks):
    """
    Publish images from a directory to RabbitMQ for processing.
    Returns a task ID that can be used to check the status.
    """
    try:
        logger.info(f"Starting to process directory: {directory_path}")

        # Check if consumers are running
        global consumer_manager
        if not consumer_manager:
            logger.warning("Consumer manager not initialized")
            raise HTTPException(
                status_code=503,
                detail="Service is still initializing. Please try again in a few moments."
            )

        # Check if model is ready
        if not consumer_manager.is_ready():
            logger.warning("Model not ready")
            raise HTTPException(
                status_code=503,
                detail="Model is still loading. Please try again in a few moments."
            )

        logger.info(
            "Service checks passed, proceeding with directory processing")

        # Validate directory exists
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            raise HTTPException(
                status_code=404, detail=f"Directory not found: {directory_path}")

        # Initialize task status
        task_status[directory_path] = {
            "status": "starting", "message": "Task scheduled"}

        # Get list of image files
        image_files = [f for f in os.listdir(directory_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

        if not image_files:
            logger.warning(
                f"No image files found in directory: {directory_path}")
            raise HTTPException(
                status_code=400, detail="No image files found in directory")

        logger.info(f"Found {len(image_files)} images in directory")

        # Update task status
        task_status[directory_path].update({
            "status": "processing",
            "total_images": len(image_files),
            "processed_images": 0
        })

        # Process each image
        successful_publishes = 0
        failed_publishes = 0

        for image_file in image_files:
            image_path = os.path.join(directory_path, image_file)
            try:
                logger.info(f"Processing image: {image_file}")

                # Read the image
                with open(image_path, 'rb') as img_file:
                    image_data = base64.b64encode(
                        img_file.read()).decode('utf-8')

                # Create message
                message = {
                    "file_name": image_file,
                    "image_data": image_data,
                    "image_path": image_path
                }

                # Publish message
                logger.info(f"Publishing message for {image_file}")
                publish_message(message)
                logger.info(f"Successfully published message for {image_file}")

                # Update processed count
                task_status[directory_path]["processed_images"] += 1
                successful_publishes += 1

            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                failed_publishes += 1
                continue

        # Update final status
        status_message = (
            f"Published {successful_publishes} images successfully"
            f"{f', {failed_publishes} failed' if failed_publishes > 0 else ''}"
        )

        task_status[directory_path].update({
            "status": "completed",
            "message": status_message,
            "successful_publishes": successful_publishes,
            "failed_publishes": failed_publishes
        })

        logger.info(f"Directory processing completed. {status_message}")

        return {
            "message": "Image processing started",
            "status": "accepted",
            "directory": directory_path,
            "total_images": len(image_files),
            "successful_publishes": successful_publishes,
            "failed_publishes": failed_publishes,
            "task_id": directory_path
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error publishing images: {e}"
        logger.error(error_msg)
        task_status[directory_path] = {
            "status": "failed",
            "message": error_msg
        }
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/task-status/{directory_path:path}")
async def get_task_status(directory_path: str):
    """Get the status of a directory processing task"""
    if directory_path not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")

    return task_status[directory_path]


@app.get("/service-status")
async def get_service_status():
    """Get the current status of the service"""
    global consumer_manager

    if not consumer_manager:
        return {
            "status": "initializing",
            "details": "Service is still initializing",
            "model_status": "not_started",
            "consumers": 0
        }

    is_ready = consumer_manager.is_ready()
    active_consumers = sum(
        1 for thread in consumer_manager.consumer_threads if thread.is_alive())

    return {
        "status": "ready" if is_ready and active_consumers > 0 else "initializing",
        "details": "Service is ready" if is_ready and active_consumers > 0 else "Service is still initializing",
        "model_status": "ready" if is_ready else "loading",
        "consumers": active_consumers
    }


@app.get("/startup-status")
async def get_startup_status():
    """Get the status of startup directory processing"""
    if not settings.BATCH_PROCESSING_STARTUP_DIRECTORY:
        return {
            "status": "not_configured",
            "message": "No startup directory configured"
        }

    return {
        "status": "processing" if settings.BATCH_PROCESSING_STARTUP_DIRECTORY in task_status else "not_started",
        "directory": settings.BATCH_PROCESSING_STARTUP_DIRECTORY,
        "details": task_status.get(settings.BATCH_PROCESSING_STARTUP_DIRECTORY, {})
    }


@app.post("/classify-image/")
async def classify_single_image(request: ImageRequest):
    """
    Classify a single image from base64 data
    """
    try:
        # Check if model is ready
        if not classifier.is_ready.is_set():
            raise HTTPException(
                status_code=503,
                detail="Model is still loading. Please try again in a few moments."
            )

        # Clean base64 string by removing data URL prefix if it exists
        image_data = request.image_data
        if image_data.startswith('data:'):
            # Remove the prefix (handles any image type, not just jpeg)
            image_data = image_data.split(',', 1)[1]

        # Classify the image with cleaned base64 data
        result = classifier.classify_image(image_data=image_data)

        if result.get('Status') == 'Error':
            raise HTTPException(
                status_code=400,
                detail=result.get('Message', 'Error processing image')
            )

        # Return classification results
        return {
            'status': 'success',
            'file_name': request.file_name,
            'classification': {
                'class': result.get('Predicted Class'),
                'confidence': result.get('Confidence'),
                'status': result.get('Status')
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
