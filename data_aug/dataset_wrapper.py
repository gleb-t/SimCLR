import glob
import os
from typing import *

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image

from data_aug.gaussian_blur import GaussianBlur


class CatDataset(Dataset):

    def __init__(self, imageSubdirPath: str, transform: Callable):
        self.rootPath = imageSubdirPath
        self.pathList = glob.glob(os.path.join(self.rootPath, 'cat*.jpg'))

        self.transform = transform

    def __getitem__(self, index):
        path = self.pathList[index]
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, os.path.basename(path)  # Pass image name as metadata. (Useful for export.)

    def __len__(self):
        return len(self.pathList)


class DataSetWrapper:

    def __init__(self, data_path: str, batch_size, num_workers, valid_size, input_shape, color_scale):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.color_scale = color_scale
        self.input_shape = eval(input_shape)

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        # train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
        #                                transform=SimCLRDataTransform(data_augment))
        train_dataset = CatDataset(self.data_path, transform=SimCLRDataTransform(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.color_scale, 0.8 * self.color_scale, 0.8 * self.color_scale, 0.2 * self.color_scale)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                              transforms.ToTensor()])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
