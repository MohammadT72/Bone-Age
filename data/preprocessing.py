# Third-party library imports
import numpy as np
import cv2

# Typing imports
from typing import Hashable, Mapping, Dict

# PyTorch imports
from torch import from_numpy, unsqueeze, moveaxis, Tensor,stack

# Albumentations for data augmentation
import albumentations as A

# MONAI imports
from monai.config.type_definitions import NdarrayOrTensor
from monai import transforms
# Custom histogram normalization transform
class CustomHistNormalized(transforms.HistogramNormalized):
    def __init__(self, keys):
        super().__init__(keys=keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.max = d['max_pixel_value']
        self.min = d['min_pixel_value']
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key], d[self.mask_key]) if self.mask_key is not None else self.transform(d[key])
        return d

# Custom mask and crop transform
class MaskAndCropd(object):
    def __init__(self, keys):
        self.image = keys[0]
        self.mask = keys[1]

    def __call__(self, data):
        mask_dir = data[self.mask]
        # Convert image tensor to numpy
        img = data[self.image].squeeze(0).numpy()
        h, w = img.shape
        # Load the txt file
        with open(mask_dir, 'r') as f:
            yolo_data = list(map(float, f.read().split()[1:]))

        # Convert the data back to polygon coordinates
        coordinates = np.array(yolo_data).reshape(-1, 2)
        coordinates[:, 0] *= w  # Denormalize x
        coordinates[:, 1] *= h  # Denormalize y
        coordinates = coordinates.astype(int)

        # Create an empty mask
        mask = np.zeros_like(img)

        # Fill the polygon area in the mask with white
        cv2.fillPoly(mask, [coordinates], (255,))

        # Apply the mask to the image
        result = np.where(mask > 0, img, mask)

        # Get the min and max pixel values inside the polygon area
        data['min_pixel_value'] = int(result[mask > 0].min())
        data['max_pixel_value'] = int(result[mask > 0].max())

        # Crop the image to the bounding box of the polygon
        x_max = coordinates[:, 0].max()
        x_min = coordinates[:, 0].min()
        y_max = coordinates[:, 1].max()
        y_min = coordinates[:, 1].min()
        result = from_numpy(result[y_min:y_max, x_min:x_max])
        # Unsqueeze to add back the channel dimension
        data[self.image] = unsqueeze(result, 0)

        return data
# Normalization transform
class Normalized(object):
    def __init__(self, keys, means=(0.485,), stds=(0.229,)):
        self.norm = A.augmentations.transforms.Normalize(mean=means, std=stds, max_pixel_value=1.0, p=1.0)
        self.image = keys[0]

    def __call__(self, data):
        img = data[self.image]
        img = moveaxis(img, 0, 2).numpy()
        normalized = self.norm(image=img)['image']
        normalized = from_numpy(normalized.astype(np.float32))
        normalized = moveaxis(normalized, 2, 0)
        data[self.image] = normalized
        return data

# CLAHE (Contrast Limited Adaptive Histogram Equalization) transform
class RandomAdapthistd:
    def __init__(self, keys, p):
        clip_limit = int(np.random.randint(1, 6, 1)[0])
        self.clahe = A.augmentations.transforms.CLAHE(clip_limit=clip_limit, p=p)
        self.image = keys[0]

    def __call__(self, data):
        img = data[self.image]
        ndim = img.ndim
        if ndim > 2:
            channel_dim = img.shape[0]
            img = img[0, :, :]
        if isinstance(img, type(Tensor())):
            img = img.numpy()
        img = img.astype(np.uint8)
        img = self.clahe(image=img)['image']
        tensorized = from_numpy(img)
        if ndim > 2:
            tensorized = [tensorized] * channel_dim
            tensorized = stack(tensorized, dim=0)
        data[self.image] = tensorized
        return data
