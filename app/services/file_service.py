import base64


def read_image(file_path: str) -> str:
    try:
        with open(file_path, "rb") as image_file:
            # Read the image and encode it to base64
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_image
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
        return ""
