from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader

from models.resnet_simclr import ResNetSimCLR
import torch.nn.functional as F
import torch.utils.data
from loss.nt_xent import NTXentLoss
import os
import shutil
import sys

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

    def __init__(self, dataLoaderTrain: DataLoader, dataLoaderVal: DataLoader, config):
        self.config = config
        self.dataLoaderTrain = dataLoaderTrain
        self.dataLoaderVal = dataLoaderVal

        self.device = self._get_device()
        todayStr = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        self.outDirPath = os.path.join(self.config['out_dir_path'], todayStr)

        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def train(self):

        train_loader, valid_loader = self.dataLoaderTrain, self.dataLoaderVal

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), float(self.config['lr']), weight_decay=float(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.outDirPath, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            print("Epoch {}/{}".format(epoch_counter + 1, self.config['epochs']))
            for (xis, xjs), _ in train_loader:
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss, _ = self._step(model, xis, xjs)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    print(f"Batch loss: {loss}")

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                print(f"Val loss: {valid_loss}")
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()


    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)

        return loss, (ris, zis, rjs, zjs)

    def _plot_batch(self, xis, xjs):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=2, ncols=16, figsize=(12, 3))
        for i in range(16):
            axes[0, i].imshow(xis[i].cpu().numpy().transpose(1, 2, 0))
            axes[1, i].imshow(xjs[i].cpu().numpy().transpose(1, 2, 0))

        plt.show()

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss, _ = self._step(model, xis, xjs)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
