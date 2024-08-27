from PIL import Image
import torchvision.transforms.v2 as T
from torch import float32

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = T.Compose([
        T.Resize((400, 400)),
        T.CenterCrop((384, 384)),
        T.Compose([T.ToImage(), T.ToDtype(float32, scale=True)]), # to supress deprecation warnings. same as ToTensor()
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return preprocess(image).unsqueeze(0)