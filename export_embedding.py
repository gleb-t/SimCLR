import json
import os
import shutil
import sys
import time
from typing import *

import torch
import torch.functional as F
import torch.utils.data
import torchvision
import yaml

from data_aug.dataset_wrapper import CatDataset
from models.resnet_simclr import ResNetSimCLR


# Copy-paste from SimtechOne.
def parse_tuple_string(string: str, valueParser: Callable[[str], Any]=None, valueType: type = None) -> Tuple:
    if string[0] != '(' or string[-1] != ')':
        raise RuntimeError("Cannot parse a tuple from string '{}'".format(string))

    withoutParenthesis = string[1:-1]
    valueStrings = [s.strip() for s in withoutParenthesis.split(',')]

    if valueParser is not None:
        return tuple([valueParser(s) for s in valueStrings])
    elif valueType is not None:
        return tuple([valueType(s) for s in valueStrings])
    else:
        return tuple(valueStrings)


def main():
    checkpointName = 'Jun22_13-34-16_antares'

    device = torch.device('cuda')

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    model = ResNetSimCLR(**config['model']).to(device)

    checkpoints_folder = os.path.join('./runs', checkpointName, 'checkpoints')
    state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
    model.load_state_dict(state_dict)
    print("Loaded the pre-trained model.")

    imageSize = parse_tuple_string(config['dataset']['input_shape'], valueType=int)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(imageSize[:2]),
        torchvision.transforms.ToTensor()
    ])
    dataset = CatDataset(config['dataset']['data_path'],
                         transform=transform)
    sampler = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)

    resultNames = []
    resultVectors = []
    with torch.no_grad():
        it = iter(sampler)
        for batchIndex in range(100):
            images, names = next(it)  # type: torch.Tensor
            images = images.to(device)

            vectors, vectorsProj = model.forward(images)
            vectorsProjNorm = F.F.normalize(vectorsProj, dim=1)

            resultNames.extend(names)
            resultVectors.extend([v.tolist() for v in vectorsProjNorm.cpu()])

    with open(os.path.expandvars(r'${DEV_METAPHOR_DATA_PATH}/cats-10k.json'), 'w') as file:
        json.dump({
            'names': resultNames,
            'vectors': resultVectors,
            'timestamp': time.time()
        }, file)

    outImageDir = os.path.expandvars(r'${DEV_METAPHOR_DATA_PATH}/cats')
    if not os.path.exists(outImageDir):
        os.makedirs(outImageDir)
    for name in resultNames:
        imagePath = os.path.join(config['dataset']['data_path'], name)
        shutil.copy(imagePath, os.path.join(outImageDir, name))


if __name__ == '__main__':
    main()
