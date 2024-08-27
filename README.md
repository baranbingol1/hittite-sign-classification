# Hittite Sign Classification

This repository contains the implementation of a ResNeXt101-32x8d model for Hittite sign classification, as described in our paper [...]. The model is designed to classify images of Hittite signs into predefined categories.

## Repository Contents

- `src/`: Directory containing main file predict.py and other utility files
- `requirements.txt`: List of Python dependencies
- `model/`: Directory containing model weights and label mappings
  - `resnext101_32dx8_weights.pth`: Trained model weights
  - `id2label.json`: Mapping of class indices to Hittite sign labels
- `example_images/`: Directory containing example Hittite sign images for testing

## Example Usage
To classify a Hittite sign image:
Navigate into src file via
```console
cd src
```
and then run prediction script via
```console
python predict.py --image_path=../example_images/example_sign_ti.jpeg
```

## Contact
For any questions or issues, please open an issue in this repository or email 22290007@ogrenci.ankara.edu.tr
