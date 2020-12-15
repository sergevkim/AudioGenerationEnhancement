import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim.optimizer import Optimizer


class AutoEncoder(nn.Module):
    def __init__(self, config):
        super(AutoEncoder, self).__init__()
        self.learning_rate = 3e-4
        self.num_features = config['num_features']
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(in_features=self.num_features, out_features=512))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(in_features=512, out_features=256))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(in_features=256, out_features=128))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(in_features=128, out_features=128))
        self.encoder.append(nn.ReLU())

        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Linear(in_features=128, out_features=128))
        self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(in_features=128, out_features=256))
        self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(in_features=256, out_features=512))
        self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(in_features=512, out_features=self.num_features))
        self.decoder.append(nn.ReLU())

        self.device = config['device']
        self.train_loss = nn.L1Loss()

    def forward(self, x):
        for l in self.encoder:
            x = l(x)
        for l in self.decoder:
            x = l(x)
        return x

    def training_step(self, batch, bacth_idx):
        batch = batch.to(self.device)
        res = self.forward(batch)
        return self.train_loss(res, batch)

    def training_step_end(self):
        pass

    def training_epoch_end(self, epoch_idx):
        pass

    def validation_step_end(self):
        pass

    def validation_epoch_end(self, epoch_idx):
        pass

    def validation_step(self, batch, bathc_idx):
        return self.training_step(batch, bathc_idx)

    def configure_optimizers(self):
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )
        
        return [optimizer], []
