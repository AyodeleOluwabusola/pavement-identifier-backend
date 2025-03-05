import os
import json
from threading import Lock
import pika
import openpyxl
from openpyxl import Workbook
from concurrent.futures import ThreadPoolExecutor
from app.core.config import settings # Add this line to import settings

lock = Lock()


def callback(ch, method, properties, body):
    message = json.loads(body)
    file_name = message.get("file_name")
    image_data = message.get("image_data")
    
    # Process the image data and determine the grade
    status, grade = process_image(image_data)
    
    # Write the result to the Excel file
    write_to_excel(file_name, status, grade)
    
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

def start_listener():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue=settings.QUEUE_NAME, durable=True)
    
    # Set QoS to process one message at a time
    channel.basic_qos(prefetch_count=3)
    
    channel.basic_consume(queue='image_queue', on_message_callback=callback)
    
    print('Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

