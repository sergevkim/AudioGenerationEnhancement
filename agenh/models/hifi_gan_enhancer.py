from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import (
    L1Loss,
    Module,
    MSELoss,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, StepLR
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
            learning_rate: float = 3e-4,
            scheduler_step_size: int = 10,
            scheduler_gamma: float = 0.5,
            verbose: bool = True,
            device: torch.device = torch.device('cpu'),
        ):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.verbose = verbose

        self.mel_spectrogramer = MelSpectrogram(
            sample_rate=22050,
            win_length=1024,
            hop_length=256,
            n_fft=1024,
            f_min=0,
            f_max=8000,
            n_mels=80,
            power=1.0,
        ).to(device)
        self.mu_law_encoder = MuLawEncoding(quantization_channels=256)
        self.mu_law_decoder = MuLawEncoding(quantization_channels=256)
        self.l1_criterion_w = L1Loss()
        self.l1_criterion_p = L1Loss()
        self.spectrogram_criterion_w = MSELoss()
        self.spectrogram_criterion_p = MSELoss()

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
        original_waveforms, corrupted_waveforms = batch
        original_waveforms = original_waveforms.to(self.device)
        corrupted_waveforms = corrupted_waveforms.to(self.device)
        original_mel_specs = self.mel_spectrogramer(original_waveforms)

        generated_waveforms_w, generated_waveforms_p = self.generator(
            x=corrupted_waveforms,
        )
        generated_mel_specs_w = self.mel_spectrogramer(generated_waveforms_w)
        generated_mel_specs_p = self.mel_spectrogramer(generated_waveforms_p)

        l1_loss_w = self.l1_criterion_w(
            input=generated_waveforms_w,
            target=original_waveforms,
        )
        spectrogram_loss_w = self.spectrogram_criterion_w(
            input=generated_mel_specs_w,
            target=original_mel_specs,
        )

        l1_loss_p = self.l1_criterion_p(
            input=generated_waveforms_p,
            target=original_waveforms,
        )
        spectrogram_loss_p = self.spectrogram_criterion_p(
            input=generated_mel_specs_p,
            target=original_mel_specs,
        )

        loss_w = l1_loss_w + spectrogram_loss_w
        loss_p = l1_loss_p + spectrogram_loss_p
        loss = loss_w + loss_p

        return loss

    def training_step_end(
            self,
            batch_idx: int,
        ):
        pass

    def training_epoch_end(
            self,
            epoch_idx: int,
        ):
        pass

    def validation_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        pass

    def validation_step_end(
            self,
            batch_idx: int,
        ):
        pass

    def validation_epoch_end(
            self,
            epoch_idx: int,
        ):
        pass

    def configure_optimizers(
            self,
        ) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
            verbose=self.verbose,
        )

        return [optimizer], []


if __name__ == '__main__':
    enhancer = HiFiGANEnhancer()
    print(enhancer)

