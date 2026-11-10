import torch
from segment_anything import sam_model_registry, SamPredictor

def load_sam_model():
    model_type = "vit_b"
    checkpoint_path = "weights/sam_vit_b.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)

    predictor = SamPredictor(sam)

    return predictor, device