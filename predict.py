import numpy as np
from PIL import Image
import requests
import torch
import torchvision
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import os
from io import BytesIO

app = Flask(__name__)

class ImageSearchEngine:
    def __init__(self, vecs_path, names_path):
        self.all_vecs = np.load(vecs_path)
        self.all_names = np.load(names_path)
        self.model = torchvision.models.resnet18(weights="DEFAULT")
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.activation = {}
        self.model.avgpool.register_forward_hook(self.get_activation("avgpool"))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def find_similar_images(self, query_vec, top_k=5):
        similarities = cosine_similarity(query_vec, self.all_vecs)
        similar_indices = similarities[0].argsort()[-top_k:][::-1]
        return [(self.all_names[i], similarities[0][i]) for i in similar_indices]

    def search_image(self, image):
        img = self.transform(image)
        img = img[None, ...]  # Add batch dimension

        with torch.no_grad():
            out = self.model(img)
            query_vec = self.activation["avgpool"].numpy().squeeze()[None, ...]

        results = self.find_similar_images(query_vec)
        return results

    def search_image_from_path(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')  # Ensure the image is RGB
            similar_images = self.search_image(image)
            return similar_images
        except Exception as e:
            return {"error": str(e)}

# Load the model and the features
search_engine = ImageSearchEngine("all_vecs.npy", "all_names.npy")


@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files and 'url' not in request.json:
        return jsonify({"error": "No image file or URL provided"}), 400

    if 'url' in request.json:
        image_url = request.json['url']
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))  
        except Exception as e:
            return jsonify({"error": f"Failed to retrieve image from URL: {str(e)}"}), 500
    else:
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        try:
            image = Image.open(image_file)  
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Now you can proceed with searching
    similar_images = search_engine.search_image(image)
    return jsonify([{"name": name, "similarity": float(similarity)} for name, similarity in similar_images])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

