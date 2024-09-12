import torch
import numpy as np
from torchvision import transforms


LEFT_IMG_KEY = 'left_img'
RIGHT_IMG_KEY = 'right_img'


def get_transforms(size=(256, 512), mode='train'):
    """Set transformations for training images."""
    if mode == 'train':
        transformations = transforms.Compose([
            ResizeImage(size=size, mode=mode),
            RandomFlip(mode=mode),
            ToTensor(mode=mode),
            AugmentSaturation(mode=mode),
        ])
    else:
        transformations = transforms.Compose([
            ResizeImage(size=size, mode=mode),
            ToTensor(mode=mode),
        ])
    return transformations


class RandomFlip(object):
    def __init__(self, mode='train'):
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            k = np.random.uniform(0, 1, 1)
            if k > 0.5:
                left_image = sample[LEFT_IMG_KEY]
                right_image = sample[RIGHT_IMG_KEY]
                sample = {LEFT_IMG_KEY: self.transform(left_image),
                          RIGHT_IMG_KEY: self.transform(right_image)}
        return sample


class ToTensor(object):
    def __init__(self, mode='train'):
        self.transform = transforms.ToTensor()
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            left_img = sample[LEFT_IMG_KEY]
            right_img = sample[RIGHT_IMG_KEY]
            sample = {LEFT_IMG_KEY: self.transform(left_img),
                      RIGHT_IMG_KEY: self.transform(right_img)}
        else:
            sample = self.transform(sample)
        return sample


class ResizeImage(object):
    """Resize the ``sample`` pair to the given size."""
    def __init__(self, size=(256, 512), mode='train'):
        self.transform = transforms.Resize(size)
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            left_img = sample[LEFT_IMG_KEY]
            right_img = sample[RIGHT_IMG_KEY]
            sample = {LEFT_IMG_KEY: self.transform(left_img),
                      RIGHT_IMG_KEY: self.transform(right_img)}
        else:
            sample = self.transform(sample)
        return sample


class AugmentSaturation(object):
    """
    Augment the image with random saturation in given range. \n
    Default values are provided by the referenced paper.
    """
    def __init__(self, mode='train',
                 gamma_low=0.8, gamma_high=1.2,
                 brightness_low=0.5, brightness_high=2.0,
                 color_low=0.8, color_high=1.2):
        self.mode = mode
        self.gamma_low = gamma_low
        self.gamma_high = gamma_high
        self.brightness_low = brightness_low
        self.brightness_high = brightness_high
        self.color_low = color_low
        self.color_high = color_high

    def __call__(self, sample):
        left_img = sample[LEFT_IMG_KEY]
        right_img = sample[RIGHT_IMG_KEY]
        p = np.random.uniform(0, 1, 1)
        if self.mode == 'train':
            if p > 0.5:
                # randomly shift gamma
                random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                left_image_aug = left_img ** random_gamma
                right_image_aug = right_img ** random_gamma

                # randomly shift brightness
                random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
                left_image_aug = left_image_aug * random_brightness
                right_image_aug = right_image_aug * random_brightness

                # randomly shift color
                random_colors = np.random.uniform(self.color_low, self.color_high, 3)
                for i in range(3):
                    left_image_aug[i, :, :] *= random_colors[i]
                    right_image_aug[i, :, :] *= random_colors[i]

                # saturate
                left_image_aug = torch.clamp(left_image_aug, 0, 1)
                right_image_aug = torch.clamp(right_image_aug, 0, 1)

                sample = {LEFT_IMG_KEY: left_image_aug, RIGHT_IMG_KEY: right_image_aug}
        return sample
