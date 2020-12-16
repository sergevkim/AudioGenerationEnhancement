import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim.optimizer import Optimizer
import torchaudio


class RNNGenerator(nn.Module):
    def __init__(self, config):
        super(RNNGenerator, self).__init__()
        self.learning_rate = 3e-4
        self.mu_law_encoder = torchaudio.transforms.MuLawEncoding(256)
        self.mu_law_decoder = torchaudio.transforms.MuLawDecoding(256)
        self.lstm = nn.LSTM(
            input_size=16000,
            hidden_size=256,
            num_layers=2
        )
        self.device = config['device']
        self.train_loss = nn.CrossEntropyLoss()

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        print('x SHAPE', x.shape)
        out, (ht1, ct1) = self.lstm(x, hidden)
        # out = self.dropout(out)
        # x = self.mu_law_decoder(out)
        # return x, (ht1, ct1)
        return out, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(2, batch_size, 256, requires_grad=True).to(self.device),
               torch.zeros(2, batch_size, 256, requires_grad=True).to(self.device))

    def training_step(self, batch, bacth_idx):
        hidden = self.init_hidden(batch.shape[0])
        batch = batch.to(self.device)
        batch = self.mu_law_encoder(batch)
        gt = nn.functional.pad(batch, (1, 0))[:,:-1]
        print('x SHAPE', batch.shape)
        res, _ = self.forward(batch, hidden)
        return self.train_loss(res, gt)

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
