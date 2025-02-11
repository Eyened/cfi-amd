import torch
import numpy as np
from utils.mask_extraction import get_cfi_bounds

paths = {
    "disc_edge": "models/discedge_july24.pt",
    "fovea": "models/fovea_july24.pt"
}

def preprocess(image, bounds=None):
    mean = np.array([0.485, 0.456, 0.406] * 2)
    std = np.array([0.229, 0.224, 0.225] * 2)

    if bounds is None:
        bounds = get_cfi_bounds(image)
    T, bounds_cropped = bounds.crop(512)

    images = np.concatenate([
        bounds_cropped.image,
        bounds_cropped.contrast_enhanced_5], axis=2)

    images_norm = (images - mean) / std

    return T, np.transpose(images_norm, (2, 0, 1)).astype(np.float32)


def get_coordinate(heatmap):
    y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    return x + 0.5, y + 0.5


class LandmarksProcessor:
    
    def __init__(self, device):
        self.device = device
        self.models = {
            k: torch.jit.load(v).eval()
            for k, v in paths.items()
        }
        for model in self.models.values():
            model.to(device)

    def process(self, image, bounds=None):
        T, x_np = preprocess(image, bounds)
        x_torch = torch.tensor(x_np).unsqueeze(0).to(self.device)

        coordinates = {}
        for name, model in self.models.items():
            heatmaps = model(x_torch)
            heatmap = torch.mean(heatmaps, dim=0)[0, 0]
            p = get_coordinate(heatmap.cpu().detach().numpy())
            coordinates[name] = T.apply_inverse([p])[0]
        return coordinates
