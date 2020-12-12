from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import (
    Module,
    MSELoss,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim.optimizer import Optimizer
from torchaudio.transforms import (
    MelSpectrogram,
    MuLawDecoding,
    MuLawEncoding,
)

from agenh.models.hifi_gan_components import (
    HiFiDiscriminator,
    HiFiGenerator,
)


class HiFiGANEnhancer(Module):
    def __init__(
            self,
            device: torch.device,
            learning_rate: float,
            scheduler_step_size: int,
            scheduler_gamma: float,
            verbose: bool,
        ):
        super().__init__()
        self.learning_rate = learning_rate

        self.mel_spectrogramer = MelSpectrogram(
            sample_rate=22050,
            win_length=1024,
            hop_length=256,
            n_fft=1024,
            f_min=0,
            f_max=8000,
            n_mels=80,
            power=1.0,
        )
        self.mu_law_encoder = MuLawEncoding(quantization_channels=256)
        self.mu_law_decoder = MuLawEncoding(quantization_channels=256)
        self.criterion = MSELoss()

        self.discriminator = HiFiDiscriminator()
        self.generator = HiFiGenerator()

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.generator(x)

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        original, corrupted = batch
        original = original.to(self.device)
        corrupted = corrupted.to(self.device)

        generated_w, generated_p = self.generator(corrupted)

        loss_w = self.criterion_w(generated_w, original)
        loss_p = self.criterion_p(generated_p, original)
        loss = loss_w + loss_p

        return loss

    def training_step_end(self):
        pass

    def training_epoch_end(self):
        pass

    def validation_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        pass

    def validation_step_end(self):
        pass

    def validation_epoch_end(
            self,
            epoch_idx,
        ):
        pass

    def configure_optimizers(
            self,
        ) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = LambdaLR(
            optimizer=optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
            verbose=self.verbose,
        )

        return [optimizer], []

