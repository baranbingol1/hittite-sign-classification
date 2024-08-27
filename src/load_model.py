from timm import create_model
from torch import load

def load_model(wp):
    model = create_model('resnext101_32x8d', pretrained=False, num_classes=181)
    model.load_state_dict(load(wp))
    model.eval()
    return model