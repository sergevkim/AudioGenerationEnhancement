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
            input_size=1,
            hidden_size=256,
            num_layers=2
        )
        self.device = config['device']
        self.train_loss = nn.CrossEntropyLoss()

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        # out = self.dropout(out)
        # x = self.mu_law_decoder(out)
        # return x, (ht1, ct1)
        return out, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(2, batch_size, 256, requires_grad=True).to(self.device),
               torch.zeros(2, batch_size, 256, requires_grad=True).to(self.device))

    def training_step(self, batch, bacth_idx):
        hidden = self.init_hidden(batch.shape[0])
        batch = batch.to(self.device)
        batch = torch.transpose(self.mu_law_encoder(batch), 0, 1).unsqueeze(2).type(torch.float)
        gt = nn.functional.pad(batch, (0, 0, 0, 0, 1, 0))[:-1]
        print('x SHAPE', batch.shape)
        print('hidden SHAPE', hidden[0].shape)
        res, _ = self.forward(batch, hidden)
        print('res SHAPE', res.shape)
        print('gt SHAPE', gt.shape)

        # res = res.transpose(1, 2).contiguous().view((-1, 256))
        res = res.contiguous().view((-1, 256))
        gt = gt.squeeze(2).view((-1)).type(torch.long)
        print('res SHAPE', res.shape)
        print('gt SHAPE', gt.shape)

        lll = nn.L1Loss()
        l1 = lll(torch.argmax(res, dim=1, keepdim=True).type(torch.float), gt.type(torch.float))
        print('\n\nL1: ', l1.item())
        return self.train_loss(res, gt)

    def stupid_inference(self, wav):
        batch = wav.unsqueeze(0)
        hidden = self.init_hidden(batch.shape[0])
        batch = batch.to(self.device)
        batch = torch.transpose(self.mu_law_encoder(batch), 0, 1).unsqueeze(2).type(torch.float)
        gt = nn.functional.pad(batch, (0, 0, 0, 0, 1, 0))[:-1]
        print('x SHAPE', batch.shape)
        print('hidden SHAPE', hidden[0].shape)
        res, _ = self.forward(batch, hidden)
        
        #print('RESS', res[0][0])
        res = res.contiguous().view((-1, 256))
        res = torch.argmax(res, dim=1, keepdim=True)
        res = self.mu_law_decoder(res)
        #print('RESS', res[0])
        return res

    def inference(self, wav):
        batch = wav[:1].unsqueeze(0)
        hidden = self.init_hidden(batch.shape[0])
        batch = batch.to(self.device)
        batch = torch.transpose(self.mu_law_encoder(batch), 0, 1).unsqueeze(2).type(torch.float)
        answer = wav[:1].to(self.device)
        for i in range(wav.shape[0]):
            #print('batch.shape', batch.shape)
            #print('hidden.shape', hidden[0].shape)
            batch = batch.type(torch.float)
            batch, hidden = self.forward(batch, hidden)
            #if i == 1:
            #    print('RESS', batch)
            batch = torch.argmax(batch, dim=2, keepdim=True)
            #print('TMP.shape', batch.shape)
            answer = torch.cat((answer, batch[:, :, -1].squeeze(0)), dim=-1)

        res = self.mu_law_decoder(answer)
        return res

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
