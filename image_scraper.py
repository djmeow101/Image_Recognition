import requests
from serpapi import GoogleSearch

# Replace with your SerpApi key
API_KEY = "YOUR_API_KEY"

params = {
    "q": "dalek",
    "tbm": "isch",  # Image search
    "api_key": API_KEY
}

search = GoogleSearch(params)
results = search.get_dict()
images = [img['original'] for img in results["images_results"]]

# Download the images
for i, img_url in enumerate(images[:100]):  # Download first 100 images
    img_data = requests.get(img_url).content
    with open(f'dalek_{i}.jpg', 'wb') as file:
        file.write(img_data)

print("Download complete!")
