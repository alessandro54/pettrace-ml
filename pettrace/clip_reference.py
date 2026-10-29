# Mirror of apps/vision/src/vision/ml/clip.py in the PetTrace app repo: the validated PyTorch
# embedding function, used only as the parity reference when registering v1 (export group).

"""Validated CLIP ViT-B/32 embedding function (from spike testing). Do not alter."""

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"  # vision.config clip_model_name

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()


def get_embedding(image: Image.Image) -> list[float]:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        embedding = outputs.pooler_output
        embedding = model.visual_projection(embedding)
    embedding = F.normalize(embedding, p=2, dim=-1)
    return embedding[0].tolist()  # 512 floats
