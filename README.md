# Pavement Identifier Backend

This project integrates a machine learning model that classifies pavement types (asphalt, chip-sealed, gravel) based on input images. It supports both singular and batch processing, utilizing FastAPI for the backend and RabbitMQ for queuing and batch handling.

## Features

- Single image classification via API endpoint
- Batch processing of images from directories
- Asynchronous processing using RabbitMQ queues
- Results export to Excel spreadsheet
- Optional automatic organization of processed images into folders by classification
- Support for both TensorFlow and PyTorch models

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for RabbitMQ)
- MLflow server (for model management)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pavement-identifier-backend.git
cd pavement-identifier-backend
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r app/requirements.txt
```

4. Configure environment variables by creating a `.env` file in the `app` directory with the following variables:
```
APP_NAME=PavementIdentifier
DEBUG=True

# Database Settings
DATABASE_URL=sqlite:///./pavement.db

# RabbitMQ Settings
RABBITMQ_URL=amqp://guest:guest@localhost:5672/%2F
QUEUE_NAME=pavement_images
EXCHANGE_NAME=pavement_exchange
ROUTING_KEY=pavement_key
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
RABBITMQ_VHOST=/
RABBITMQ_NUM_PRODUCERS=2
RABBITMQ_NUM_CONSUMERS=3

# Model Settings
MODEL_URI_TENSORFLOW=models:/tensorflow_pavement_model/Production
MODEL_URI_PYTORCH=models:/pytorch_pavement_model/Production
MLFLOW_TRACKING_URI=http://localhost:5000
CONFIDENCE_THRESHOLD=0.7
FRAMEWORK_IN_USE=tensorflow  # or pytorch

# Directory Settings
CATEGORIZED_IMAGES_DIR=./categorized_images
RESULTS_DIR=./results
EXCEL_RESULTS_PATH=./image_processing_results.xlsx
BATCH_PROCESSING_STARTUP_DIRECTORY=  # Optional: path to process on startup

# Results Queue Settings
RESULTS_EXCHANGE_NAME=results_exchange
RESULTS_QUEUE_NAME=results_queue
RESULTS_ROUTING_KEY=results_key

# Process Management Settings
CLEANUP_GRACE_PERIOD=5
PROCESS_SHUTDOWN_TIMEOUT=10

# Logging Settings
LOG_FILE=./logs/pavement_identifier.log
LOG_LEVEL=INFO

# Image Organization Settings
ORGANIZE_IMAGES_INTO_FOLDERS=True

# AWS Settings (if using S3 for storage)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-west-2
```

## Running RabbitMQ with Docker Compose

The project includes a Docker Compose file to easily start RabbitMQ:

1. Make sure Docker and Docker Compose are installed on your system
2. Start RabbitMQ using the following command:

```bash
docker-compose up -d
```

3. Verify RabbitMQ is running by accessing the management interface at http://localhost:15672
   - Username: guest
   - Password: guest

4. To stop RabbitMQ when you're done:

```bash
docker-compose down
```

## Running the Application

Start the FastAPI application:

```bash
cd app
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## API Endpoints

- `GET /`: Welcome message
- `GET /config`: View current configuration
- `GET /service-status`: Check service status
- `POST /classify-image/`: Classify a single image
- `POST /publish-images-from-dir/`: Process all images in a directory
- `GET /task-status/{directory_path}`: Check status of a directory processing task
- `GET /startup-status`: Check status of startup directory processing

## Usage Examples

### Classify a Single Image

```python
import requests
import base64

# Read and encode image
with open("path/to/image.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Send request
response = requests.post(
    "http://localhost:8000/classify-image/",
    json={"image_data": encoded_image, "file_name": "image.jpg"}
)

print(response.json())
```

### Process a Directory of Images

```python
import requests

response = requests.post(
    "http://localhost:8000/publish-images-from-dir/",
    json={"directory_path": "/path/to/images/directory"}
)

task_id = response.json()["task_id"]

# Check status
status_response = requests.get(f"http://localhost:8000/task-status/{task_id}")
print(status_response.json())
```

## Project Structure

- `app/main.py`: FastAPI application entry point
- `app/ml/`: Machine learning model implementations
- `app/services/`: Service implementations (RabbitMQ, Excel writer, etc.)
- `app/core/`: Core functionality (config, logging)
- `app/batch.py`: Batch processing functionality

## License

[MIT License](LICENSE)
