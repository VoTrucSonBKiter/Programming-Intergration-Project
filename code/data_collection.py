import os
import requests
from PIL import Image
from io import BytesIO

# Define constants
API_KEY = 'YOUR_BING_SEARCH_API_KEY'
SEARCH_URL = "https://api.bing.microsoft.com/v7.0/images/search"
SEARCH_QUERY = "Vietnam damaged street"
IMAGE_DIR = 'C:/Users/ASUS/Documents/Project_241/copi_code/dataset/images'
NUM_IMAGES = 1000

def download_images():
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    params = {"q": SEARCH_QUERY, "count": NUM_IMAGES, "imageType": "Photo", "license": "Public", "safeSearch": "Strict"}

    response = requests.get(SEARCH_URL, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    for i, img in enumerate(search_results['value']):
        try:
            img_url = img['contentUrl']
            img_data = requests.get(img_url)
            img_data.raise_for_status()
            image = Image.open(BytesIO(img_data.content))
            image = image.convert('RGB')  # Ensure image is in RGB format
            image.save(os.path.join(IMAGE_DIR, f"{i:03}.jpg"))
            if i >= NUM_IMAGES - 1:
                break
        except Exception as e:
            print(f"Could not download image {i}: {e}")

if __name__ == "__main__":
    download_images()
