from .model import UNet
import lightning as L
import torch
import numpy as np
from .utils.mask_extraction import get_cfi_bounds
import os

model_folder = 'models'


class Model(L.LightningModule):

    def __init__(self, out_channels):
        super().__init__()
        self.model = UNet(
            filters=[32, 48, 64, 96, 128, 192, 256, 384],
            bottleneck_filters=512,
            num_res_convs=1,
            in_channels=9,
            out_channels=out_channels
        )

    def forward(self, x):
        return self.model(x)

# 1 output channel


class Model1(Model):
    def __init__(self):
        super().__init__(1)

# 2 output channels


class Model2(Model):
    def __init__(self):
        super().__init__(2)


def load_models(feature, device):    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # pigment model segments both RPE degeneration and hyperpigmentation
    constructor = Model2 if feature == 'pigment' else Model1
    return [
        constructor.load_from_checkpoint(
            checkpoint_path=os.path.join(
                parent_dir, model_folder, feature, f'model_{i}.ckpt'),
            map_location=device)
        for i in range(5)]


# separate models for each feature
features = 'drusen', 'pigment', 'RPD'


class Processor:

    def __init__(self, device, mode="th_0.5"):
        '''
        args:
        device: torch device
        mode: "th_0.5" or "th_optimal"
        '''
        self.device = device
        self.mode = mode
        self.models = {
            feature: load_models(feature, device)
            for feature in features
        }

        for models in self.models.values():
            for model in models:
                model.to(device)
                model.eval()

        # optimal thresholds for each model
        # based on average dice score on validation images with reference segmentation
        self.thresholds = {
            'drusen': (0.81, 0.76, 0.44, 0.72, 0.58),
            'rpe_degeneration': (0.53, 0.8, 0.75, 0.74, 0.09),
            'hyperpigmentation': (0.44, 0.47, 0.65, 0.22, 0.28),
            'RPD': (0.19, 0.11, 0.08, 0.07, 0.39),
        }
        # optimal thresholds for each model
        # based on single dice score on full validation set
        self.thresholds_global = {
            'drusen': (0.85, 0.74, 0.47, 0.66, 0.69),
            'rpe_degeneration': (0.75, 0.76, 0.45, 0.65, 0.60),
            'hyperpigmentation': (0.59, 0.92, 0.74, 0.78, 0.37),
            'RPD': (0.85, 0.34, 0.78, 0.72, 0.70)
        }

    def combine_ensemble(self, y_preds, thresholds):
        if self.mode == "th_0.5":
            return np.mean(y_preds, axis=0)
        elif self.mode == "th_optimal":
            result = np.zeros_like(y_preds[0])
            for th, y_pred in zip(thresholds, y_preds):
                result += y_pred ** (np.log(th) / np.log(0.5))
            return result / 5

    def process(self, image, radius_fraction=1):
        bounds = get_cfi_bounds(image)
        T, bounds_cropped = bounds.crop(1024)

        bounds_cropped.radius = radius_fraction * bounds_cropped.radius

        images = np.concatenate([
            bounds_cropped.image,
            bounds_cropped.contrast_enhanced_5,
            bounds_cropped.contrast_enhanced_10
        ], axis=2)

        x = np.transpose(images, (2, 0, 1)).astype(np.float32) / 255.0
        x = torch.tensor(x).unsqueeze(0).to(self.device)

        result = {
            'bounds': bounds
        }
        for feature, models in self.models.items():
            y_preds = []
            for model in models:
                y_pred = model(x)
                y_np = torch.sigmoid(y_pred).detach().cpu().numpy()
                y_preds.append(y_np.squeeze())

            y_preds = np.array(y_preds)

            if feature == 'pigment':
                # pigment model has 2 output channels
                for i, f in enumerate(['rpe_degeneration', 'hyperpigmentation']):
                    y_pred_feature = y_preds[:, i]
                    y_pred = self.combine_ensemble(
                        y_pred_feature, self.thresholds[f])
                    y_orig = T.warp_inverse(y_pred)
                    y_orig[~bounds.mask] = 0
                    result[f] = y_orig
            else:
                y_pred = self.combine_ensemble(
                    y_preds, self.thresholds[feature])
                y_orig = T.warp_inverse(y_pred)
                y_orig[~bounds.mask] = 0
                result[feature] = y_orig
        return result
