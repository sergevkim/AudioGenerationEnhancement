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
        self.encoder = nn.Embedding(256, 32)
        self.mu_law_encoder = torchaudio.transforms.MuLawEncoding(256)
        self.mu_law_decoder = torchaudio.transforms.MuLawDecoding(256)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=256,
            num_layers=2
        )
        self.device = config['device']
        self.train_loss = nn.CrossEntropyLoss()

    def forward(self, x, hidden):
        x = self.encoder(x)

        #print('x SHAPE', x.shape)
        #print('hidden SHAPE', hidden[0].shape)
        out, hidden = self.lstm(x, hidden)
        # out = self.dropout(out)
        # x = self.mu_law_decoder(out)
        # return x, (ht1, ct1)
        return out, hidden

    def init_hidden(self, batch_size=1):
        return [torch.zeros(2, batch_size, 256, requires_grad=True).to(self.device),
               torch.zeros(2, batch_size, 256, requires_grad=True).to(self.device)]

    def training_step(self, batch, batch_idx, optimizer_idx):
        hidden = self.init_hidden(batch.shape[0])
        batch = batch.to(self.device)
        batch = torch.transpose(self.mu_law_encoder(batch), 0, 1)#.unsqueeze(2).type(torch.float)
        gt = nn.functional.pad(batch, (0, 0, 0, 1))[1:]

        #print('x SHAPE', batch[:10,0])
        #print('hidden SHAPE', gt[:10, 0])
        #print('x SHAPE', batch.shape)
        #print('hidden SHAPE', hidden[0].shape)
        res, _ = self.forward(batch, hidden)
        #print('res SHAPE', res.shape)
        #print('gt SHAPE', gt.shape)

        # res = res.transpose(1, 2).contiguous().view((-1, 256))
        res = res.contiguous().view((-1, 256))
        #gt = gt.squeeze(2).view((-1)).type(torch.long)
        gt = gt.view((-1)).type(torch.long)
        #print('res SHAPE', res.shape)
        #print('gt SHAPE', gt.shape)
        return self.train_loss(res, gt)

    def stupid_inference(self, wav):
        batch = wav.unsqueeze(0)
        hidden = self.init_hidden(batch.shape[0])
        batch = batch.to(self.device)
        batch = torch.transpose(self.mu_law_encoder(batch), 0, 1)#.unsqueeze(2).type(torch.float)
        gt = nn.functional.pad(batch, (0, 0, 1, 0))[:-1]
        
        res, _ = self.forward(batch, hidden)
        
        #print('RESS', res[0][0])
        res = res.contiguous().view((-1, 256))
        res = torch.argmax(res, dim=1, keepdim=True)
        res = self.mu_law_decoder(res)
        #print('RESS', res[0])
        return res

    def inference(self, wav):
        batch = wav[:1]
        hidden = self.init_hidden(batch.shape[0])
        #hidden[0] += torch.zeros_like(hidden[0]).normal_(mean=0, std=0.02).to(self.device)
        #hidden[1] += torch.zeros_like(hidden[1]).normal_(mean=0, std=0.02).to(self.device)
        
        batch = batch.to(self.device) #.type(torch.long)
        batch = self.mu_law_encoder(batch).unsqueeze(0)
        answer = wav[:1].to(self.device)
        print('RES', batch, self.mu_law_encoder(wav[0]))
        for i in range(wav.shape[0] - 10):
            if i < 8000:
                batch = self.mu_law_encoder(wav[i:i + 1]).to(self.device).unsqueeze(0)
                batch, hidden = self.forward(batch, hidden)
            else:
                batch, hidden = self.forward(batch, hidden)
            # batch, hidden = self.forward(self.mu_law_encoder(wav[i:i + 1]).to(self.device).unsqueeze(0), hidden)
            #    print('RESS', batch)
            batch = torch.argmax(batch, dim=2)
            if i < 100:
                print('RES', batch, self.mu_law_encoder(wav[i + 1]))
            tmp = batch[:, -1]
            #print('answer', answer.shape)
            #print('tmp', tmp.shape)
            answer = torch.cat((answer, tmp), dim=-1)
            hidden = [hidden[0], hidden[1]]
            #hidden[0] += torch.zeros_like(hidden[0]).normal_(mean=0, std=0.02).to(self.device)
            #hidden[1] += torch.zeros_like(hidden[1]).normal_(mean=0, std=0.02).to(self.device)

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

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, 0)

    def configure_optimizers(self):
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )
        
        return [optimizer], []
