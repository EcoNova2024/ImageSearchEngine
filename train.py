# train.py
import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

def extract_features(image_dir, output_vecs_path, output_names_path):
    images = os.listdir(image_dir)

    model = torchvision.models.resnet18(weights="DEFAULT")
    model.eval()

    all_names = []
    all_vecs = None

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.avgpool.register_forward_hook(get_activation("avgpool"))

    with torch.no_grad():
        for i, file in enumerate(images):
            try:
                img = Image.open(os.path.join(image_dir, file))
                img = transform(img)
                out = model(img[None, ...])
                vec = activation["avgpool"].numpy().squeeze()[None, ...]
                if all_vecs is None:
                    all_vecs = vec
                else:
                    all_vecs = np.vstack([all_vecs, vec])
                all_names.append(file)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
            if i % 100 == 0 and i != 0:
                print(i, "done")

    # Save the extracted features and names
    np.save(output_vecs_path, all_vecs)
    np.save(output_names_path, all_names)

if __name__ == "__main__":
    image_directory = "./images"  
    extract_features(image_directory, "all_vecs.npy", "all_names.npy")
