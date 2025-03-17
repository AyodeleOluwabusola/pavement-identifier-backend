from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, BackgroundTasks, HTTPException
from app.core.config import settings
from app.batch import process_images_from_dir
from app.services.file_service import read_image
from app.services.rabbitmq_service import publish_message
from app.services.listener import start_listener
import threading
import os
from typing import Dict, Any, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64
import torch
from app.ml.pavement_classifier import PavementClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME)

# Create a ThreadPoolExecutor with a reasonable number of workers
executor = ThreadPoolExecutor(max_workers=4)

# Store background task status
task_status: Dict[str, Any] = {}

# Single instance of classifier shared across the application
classifier = PavementClassifier()

# Global variable to store the consumer manager
consumer_manager = None

def initialize_consumer_manager():
    """Initialize the consumer manager with shared classifier"""
    global consumer_manager
    try:
        # Initialize classifier first
        classifier.initialize()
        # Pass the initialized classifier to the consumer manager
        consumer_manager = start_listener(
            classifier=classifier, num_consumers=3)
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


@app.on_event("startup")
async def startup_event():
    """Start the initialization in a separate thread"""
    threading.Thread(target=initialize_consumer_manager, daemon=True).start()
    logger.info("Started consumer manager initialization")


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
                    "image_data": image_data
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
