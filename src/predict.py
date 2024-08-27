import argparse
import json
import torch

from load_model import load_model
from preprocess import preprocess_image

with open('../model/id2label.json', 'r') as f:
    id2label = json.load(f)

def predict(image_path, model, id2label):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    image = preprocess_image(image_path)
    image = image.to(device)

    with torch.inference_mode():
        out = model(image)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        predicted_class_idx = probs.argmax().item()
        predicted_class_name = id2label[str(predicted_class_idx)]
        conf = probs[predicted_class_idx].item()

    return predicted_class_name, conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict image class")
    parser.add_argument("--image_path", type=str, required=True, help="Relative path to the input image")
    args = parser.parse_args()

    weights_path = "../model/resnext101_32x8d.pth"
    model = load_model(weights_path)
    
    predicted_class, confidence = predict(args.image_path, model, id2label)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")