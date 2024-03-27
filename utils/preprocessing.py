from PIL import Image
from torchvision import transforms

def preprocess_image(image_path):
    """Prétraiter une image pour la reconnaissance d'image."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    return image
