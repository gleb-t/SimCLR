from typing import Callable

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as transforms


class TransformDataset(Dataset):

    def __init__(self, dataset: Dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int):
        dataPoint = self.dataset[index]
        if isinstance(dataPoint, tuple):
            return (self.transform(dataPoint[0]),) + dataPoint[1:]
        else:
            return self.transform(dataPoint)

    def __len__(self):
        return len(self.dataset)


class PairTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


def get_simclr_image_transform(color_scale, input_shape):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * color_scale, 0.8 * color_scale, 0.8 * color_scale, 0.2 * color_scale)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=input_shape[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * input_shape[0])),
                                          transforms.ToTensor()])

    return data_transforms


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
