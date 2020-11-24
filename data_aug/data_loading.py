import glob
import os
from typing import *

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image


class CatDataset(Dataset):

    def __init__(self, imageSubdirPath: str, transform: Optional[Callable] = None):
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


def build_data_loaders(dataset: Dataset, batch_size, num_workers, valid_size):
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, drop_last=True, shuffle=False)

    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=num_workers, drop_last=True)

    return train_loader, valid_loader


