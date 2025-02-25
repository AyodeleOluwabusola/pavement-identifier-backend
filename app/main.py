from fastapi import FastAPI
from app.core.config import settings
from app.batch import process_images_from_dir
from app.services.file_service import read_image
from app.services.rabbitmq_service import publish_message

app = FastAPI(title=settings.APP_NAME)

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


@app.post("/publish-images-from-dir/")
async def publish_images_from_directory(directory_path: str):
    # Call the Celery task to process all images in the directory
    process_images_from_dir.apply_async(args=[directory_path])
    return {"message": "Batch image processing from directory started as a background job."}

def background_publish_images_from_dir(directory_path: str):
    # List all image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
    
    if not image_files:
        print(f"No images found in the directory: {directory_path}")
        return
    
    for image_file in image_files:
        file_path = os.path.join(directory_path, image_file)
        
        # Read the image and convert it to base64
        encoded_image = read_image(file_path)
        
        if not encoded_image:
            print(f"Skipping {image_file} - File not found or invalid.")
            continue
        
        # Publish the base64 encoded image to RabbitMQ
        message = {
            "image_data": encoded_image,
            "file_name": image_file
        }
        publish_message(message)
        print(f"Processed and published image: {image_file}")