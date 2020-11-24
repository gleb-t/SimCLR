from data_aug.data_transform import TransformDataset, PairTransform, get_simclr_image_transform
from simclr import SimCLR
import yaml
from data_aug.data_loading import build_data_loaders, CatDataset


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    dataset = CatDataset(config['dataset']['data_path'])
    transform = PairTransform(get_simclr_image_transform(config['dataset']['color_scale'],
                                                         eval(config['dataset']['input_shape'])))
    datasetTransformed = TransformDataset(dataset, transform)

    loaderTrain, loaderVal = build_data_loaders(datasetTransformed,
                                                batch_size=config['batch_size'],
                                                num_workers=config['dataset']['num_workers'],
                                                valid_size=config['dataset']['valid_size'])

    simclr = SimCLR(loaderTrain, loaderVal, config)
    simclr.train()


if __name__ == "__main__":
    main()
