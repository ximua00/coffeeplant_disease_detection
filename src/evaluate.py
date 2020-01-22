import torchvision.transforms as transforms
import torch
from PIL import Image
import os
import io

from train import Net
from dataloader import class2idx, idx2class


def load_model(model_name="test.pt"):
    model_path = "../models/"+model_name
    
    net = Net()
    net.model.load_state_dict(torch.load(model_path))
    net.model.eval()
    return net.model

def process_image(image_bytes):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4131, 0.5085, 0.2980],
                                [0.2076, 0.2035, 0.1974])
    ])
    im = Image.open(io.BytesIO(image_bytes))
    im = im.resize((224, 224))
    im = data_transforms(im).unsqueeze(0)
    return im

def get_prediction(image_bytes):
    model = load_model()
    im = process_image(image_bytes)
    output = model(im)
    pred = torch.argmax(output).item()
    class_name = idx2class[pred]
    return class_name


if __name__ == "__main__":

    with open("../samples/C1P1E1.jpg", 'rb') as f:
        image_bytes = f.read()
        class_name = get_prediction(image_bytes)
        print(class_name)


