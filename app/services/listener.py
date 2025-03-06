import base64
import os
import json
from threading import Lock, Thread
import pika
import openpyxl
from openpyxl import Workbook
from concurrent.futures import ThreadPoolExecutor
from app.core.config import settings # Add this line to import settings
import torch
from torchvision import transforms
from io import BytesIO
from PIL import Image


# Load the PyTorch model from a local file
model_path = "/Users/ayodele/Documents/PD/fastapi/pavement-identifier-backend/model.pth"
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
device = torch.device("cpu")
model.to(device)
model.eval()

# Define the classes for pavement surface classification
classes = ['asphalt', 'chip-sealed', 'gravel']

# Set image size to match training preprocessing
IMG_SIZE = (256, 256)
confidence_threshold = 0.55

# Define the inference transform
inference_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("L")),  # Force grayscale conversion
    transforms.Resize(IMG_SIZE),                      # Resize to 256x256
    transforms.ToTensor(),                            # Convert to tensor (1 channel)
    transforms.Normalize((0.5,), (0.5,))              # Normalize with mean=0.5, std=0.5
])

def transform_image(image_path=None, image_base64=None):
    """
    Preprocess an image:
      - Opens the image in grayscale.
      - Resizes to IMG_SIZE.
      - Converts to a tensor and normalizes.
      - Adds a batch dimension.
    """
    print(f"transform_image, image_path: {image_path}")
    if image_path:
        print("Using image path")
        img = Image.open(image_path)
    else:
        print("Using image data")
        try :
            image_data = base64.b64decode(image_base64)
            img = Image.open(BytesIO(image_data))
        except Exception as e:
            print(f"Error converting image: {e}")
            return None

    if img is None:
        print("Img is None")
        return None

    print("Gathering inference_transform")
    img = inference_transform(img)
    img = img.unsqueeze(0)  # Add a batch dimension
    return img

def classify_image(image_path=None, image_data=None):
    """
    Preprocess the image, run inference, and return the classification result.
    """
    print("classify_image called")
    try:
        img_tensor = transform_image(image_path, image_data)
        img_tensor = img_tensor.to(device)
        print("img_tensor: ", img_tensor)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {'Image Path': image_path, 'Message': "Error processing Image"}
    if img_tensor is None:
        print("img_tensor is None")
        return {'Image Path': image_path, 'Message': "Error processing Image"}

    with torch.no_grad():
        print("no_grad torch")
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        max_prob, pred_idx = torch.max(probs, dim=1)
        max_prob = max_prob.item()
        pred_idx = pred_idx.item()
        print(f"pred_idx: {pred_idx}")

    print("Checking max_prob")
    if max_prob >= confidence_threshold:
        result = {'Image Path': image_path, 'Message': "Success", 'Predicted Class': classes[pred_idx], 'Confidence': max_prob}
    else:
        result = {'Image Path': image_path, 'Message': "Uncertain", 'Predicted Class': "Uncertain", 'Confidence': max_prob}

    print(f"RESULT HERE IS: {result}")
    return result

def write_to_excel(file_name, status, grade):
    with lock:
        file_path = "image_processing_results.xlsx"
        if os.path.exists(file_path):
            workbook = openpyxl.load_workbook(file_path)
        else:
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Image Processing Results"
            sheet.append(["File Name", "Status", "Grade"])

        sheet = workbook.active
        sheet.append([file_name, status, grade])
        workbook.save(file_path)

lock = Lock()


# def callback(ch, method, properties, body):
#     print(" [x] Received message")
#     message = json.loads(body)
#     file_name = message.get("file_name")
#     image_data = message.get("image_data")
#     image_path = message.get("image_path")
#
#     # Process the image data and determine the grade
#     status, grade = classify_image(image_path, image_data)
#
#     # Write the result to the Excel file
#     write_to_excel(file_name, status, grade)
#
#     # Acknowledge the message
#     ch.basic_ack(delivery_tag=method.delivery_tag)
#


executor = ThreadPoolExecutor(max_workers=3)

def process_result(future, file_name):
    status, grade = future.result()
    write_to_excel(file_name, status, grade)

def callback(ch, method, properties, body):
    print(" [x] Received message")
    message = json.loads(body)
    file_name = message.get("file_name")
    image_data = message.get("image_data")
    image_path = message.get("image_path")

    # Process the image data and determine the grade in a separate thread
    future = executor.submit(classify_image, image_path, image_data)
    future.add_done_callback(lambda f: process_result(f, file_name))

    # Acknowledge the message
    ch.basic_ack(delivery_tag=method.delivery_tag)


def process_image(image_data):
    # Implement the image processing logic and determine the grade
    # For now, we'll just return a placeholder status and grade
    return "Success", "A"

def write_to_excel(file_name, status, grade):
   with lock:
        # Load or create the Excel workbook
        file_path = "image_processing_results.xlsx"
        if os.path.exists(file_path):
            workbook = openpyxl.load_workbook(file_path)
        else:
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Image Processing Results"
            sheet.append(["File Name", "Status", "Grade"])
        
        sheet = workbook.active
        sheet.append([file_name, status, grade])
        
        # Save the Excel workbook
        workbook.save(file_path)

def start_consumer():
    print("Starting consumer")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=settings.RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=settings.QUEUE_NAME, durable=True)
    
    # Set QoS to process up to 3 messages at a time
    channel.basic_qos(prefetch_count=3)

    channel.basic_consume(queue=settings.QUEUE_NAME, on_message_callback=callback)
    
    print('Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

def start_listener(num_consumers=3):
    print("Listening for messages. To exit press CTRL+C")
    threads = []
    for _ in range(num_consumers):
        thread = Thread(target=start_consumer)
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()

