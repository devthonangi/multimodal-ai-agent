import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class VisionEncoder:
    def __init__(self, device):
        self.device = device
        print("[INFO] Loading CLIP vision encoder...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        return embedding / embedding.norm(p=2, dim=-1, keepdim=True)
