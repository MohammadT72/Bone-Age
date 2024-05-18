# PyTorch imports
from torch import device,cuda,float32

# MONAI imports
from monai import transforms

from data.preprocessing import MaskAndCropd,CustomHistNormalized,Normalized

# Transform manager
class Transforms(object):
    def __init__(self):
        self.train_transforms = None
        self.val_transforms = None
        self.device = device("cuda" if cuda.is_available() else "cpu")

    def make_transforms(self, config, model_name, aug):
        # Base transforms
        base = [
            # Load the image and ensure the channel dimension comes first
            transforms.LoadImaged(keys=[self.image_key], ensure_channel_first=True),
            # Flip the image along the vertical axis (axis 1) for axis alignment in the PIL reader in MONAI
            transforms.Flipd(keys=[self.image_key], spatial_axis=(1)),
            # Rotate the image by -1.5708 radians (-90 degrees) to further align axes for the PIL reader in MONAI
            transforms.Rotated(keys=[self.image_key], angle=-1.5708, keep_size=False),
            # Apply masking and cropping based on polygon segmentation
            MaskAndCropd(keys=[self.image_key, self.mask_key]),
            # Custom histogram normalization based on the min and max values in the polygon segmentation area (mask)
            # This avoids normalization based on irrelevant areas like X-ray tags or a second hand in the image
            CustomHistNormalized(keys=[self.image_key]),
            # Repeat the channel 3 times to match expected input dimensions for certain models
            transforms.RepeatChanneld(keys=[self.image_key], repeats=3),
        ]

        # Main preprocessing transforms
        # suggested by torchvision library for each pretrained model
        main = self.make_list_of_preprocessing(config['models'][model_name]['preprocessing'])

        if aug:
            augmentations = self.make_augmentations()

        # End transforms
        # Convert all data to PyTorch tensors and make sure they are on the same device.
        end = [
            transforms.ToTensord(keys=[self.image_key, self.target_col_name], dtype=float32, device=self.device)
        ]

        if self.combined:
          # For combined model which accepts both image and extra features as input
          keys = [self.image_key, self.target_col_name] + self.extra_info_list
          end = [
              transforms.ToTensord(keys=keys, dtype=float32, device=self.device)
          ]


        # Combine transforms for training
        self.train_transforms = transforms.Compose(base + main + end)

        if aug:
            self.train_transforms = transforms.Compose(base + augmentations + main + end)

        # Combine transforms for validation
        self.val_transforms = transforms.Compose(base + main + end)

        # Store preprocessing steps for logging
        self.preprocessing_steps_train = [transform.__class__.__name__ for transform in self.train_transforms.transforms]
        self.preprocessing_steps_val = [transform.__class__.__name__ for transform in self.val_transforms.transforms]

    def make_list_of_preprocessing(self, dict_preprocessing: dict):
        # Create a list to store preprocessing transforms
        list_preprocess = []
        main_trans_names = ['resize', 'crop', 'scale', 'normalize']

        # Ensure only valid transform names are processed
        names = list(dict_preprocessing.keys())
        for n in main_trans_names:
            if n not in names:
                main_trans_names.remove(n)

        # Add preprocessing transforms to the list based on the configuration
        # you can change the setting for each preprocessing step in the config yaml file
        for name in main_trans_names:
            key = self.image_key
            if name == 'resize':
                size = tuple(dict_preprocessing[name]['size'])
                mode = dict_preprocessing[name]['mode']
                list_preprocess.append(transforms.Resized(keys=[key], spatial_size=size, mode=mode))
            elif name == 'crop':
                roi = tuple(dict_preprocessing[name]['roi'])
                list_preprocess.append(transforms.CenterSpatialCropd(keys=[key], roi_size=roi))
            elif name == 'normalize':
                means = tuple(dict_preprocessing[name]['means'])
                stds = tuple(dict_preprocessing[name]['stds'])
                list_preprocess.append(Normalized(keys=[key], means=means, stds=stds))
            elif name == 'scale':
                minv = dict_preprocessing[name]['minv']
                maxv = dict_preprocessing[name]['maxv']
                list_preprocess.append(transforms.ScaleIntensityd(keys=[key], minv=minv, maxv=maxv))

        return list_preprocess

    def make_augmentations(self):
      # add augmentation step into this list
        list_augs = [
            # RandomAdapthistd(keys=[self.image_key], p=0.2),
            transforms.RandRotated(keys=[self.image_key], prob=0.2),
        ]
        return list_augs