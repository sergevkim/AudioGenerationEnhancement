from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import (
    BCELoss,
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
    HiFiGenerator,
    HiFiSpectrogramDiscriminator,
    HiFiWaveformDiscriminator,
)
from agenh.utils.waveform_processor import WaveformProcessor


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
            n_mels=80,
        ).to(device)
        self.waveform_processor = WaveformProcessor()
        self.mu_law_encoder = MuLawEncoding(quantization_channels=256)
        self.mu_law_decoder = MuLawEncoding(quantization_channels=256)
        self.l1_criterion_w = L1Loss()
        self.l1_criterion_p = L1Loss()
        self.spectrogram_criterion_w = MSELoss()
        self.spectrogram_criterion_p = MSELoss()
        self.adv_waveform_criterion = BCELoss()
        self.adv_spectrogram_criterion = BCELoss()

        self.generator = HiFiGenerator()
        self.waveform_discriminator = HiFiWaveformDiscriminator()
        self.spectrogram_discriminator = HiFiSpectrogramDiscriminator()

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.generator(x)

    def generator_training_step(
            self,
            generated_waveforms_w: Tensor,
            generated_mel_specs_w: Tensor,
            generated_waveforms_p: Tensor,
            generated_mel_specs_p: Tensor,
            original_waveforms: Tensor,
            original_mel_specs: Tensor,
            fake_waveform_predicts: Tensor,
            fake_mel_spec_predicts: Tensor,
        ) -> Tensor:
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
        g_fake_waveform_loss = self.adv_waveform_criterion(
            input=fake_waveform_predicts,
            target=torch.ones_like(fake_waveform_predicts),
        )
        g_fake_mel_spec_loss = self.adv_spectrogram_criterion(
            input=fake_mel_spec_predicts,
            target=torch.ones_like(fake_mel_spec_predicts),
        )
        loss = loss_w + loss_p + g_fake_waveform_loss + g_fake_mel_spec_loss

        return loss

    def discriminator_training_step(
            self,
            fake_waveform_predicts: Tensor,
            fake_mel_spec_predicts: Tensor,
            real_waveform_predicts: Tensor,
            real_mel_spec_predicts: Tensor,
        ) -> Tensor:
        d_fake_waveform_loss = self.adv_waveform_criterion(
            input=fake_waveform_predicts,
            target=torch.zeros_like(fake_waveform_predicts),
        )
        d_fake_mel_spec_loss = self.adv_spectrogram_criterion(
            input=fake_mel_spec_predicts,
            target=torch.zeros_like(fake_mel_spec_predicts),
        )
        d_real_waveform_loss = self.adv_waveform_criterion(
            input=real_waveform_predicts,
            target=torch.ones_like(real_waveform_predicts),
        )
        d_real_mel_spec_loss = self.adv_spectrogram_criterion(
            input=real_mel_spec_predicts,
            target=torch.ones_like(real_mel_spec_predicts),
        )
        loss = (
            d_fake_waveform_loss
            + d_fake_mel_spec_loss
            + d_real_waveform_loss
            + d_real_mel_spec_loss
        )

        return loss

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
            optimizer_idx: int,
        ) -> Tensor:
        original_waveforms = batch
        original_waveforms = original_waveforms.to(self.device)
        corrupted_waveforms = self.waveform_processor(original_waveforms)
        corrupted_waveforms = corrupted_waveforms.to(self.device)
        original_mel_specs = self.mel_spectrogramer(original_waveforms)

        generated_waveforms_w, generated_waveforms_p = self.generator(
            x=corrupted_waveforms,
        )
        generated_mel_specs_w = self.mel_spectrogramer(generated_waveforms_w)
        generated_mel_specs_p = self.mel_spectrogramer(generated_waveforms_p)

        fake_waveform_predicts = torch.sigmoid(self.waveform_discriminator(
            x=generated_waveforms_p,
        ))
        real_waveform_predicts = torch.sigmoid(self.waveform_discriminator(
            x=original_waveforms,
        ))
        fake_mel_spec_predicts = torch.sigmoid(self.spectrogram_discriminator(
            x=generated_waveforms_p,
        ))
        real_mel_spec_predicts = torch.sigmoid(self.spectrogram_discriminator(
            x=original_waveforms,
        ))

        if optimizer_idx % 2 == 0:
            generator_loss = self.generator_training_step(
                generated_waveforms_w=generated_waveforms_w,
                generated_mel_specs_w=generated_mel_specs_w,
                generated_waveforms_p=generated_waveforms_p,
                generated_mel_specs_p=generated_mel_specs_p,
                original_waveforms=original_waveforms,
                original_mel_specs=original_mel_specs,
                fake_waveform_predicts=fake_waveform_predicts,
                fake_mel_spec_predicts=fake_mel_spec_predicts,
            )
            return generator_loss
        else:
            discriminator_loss = self.discriminator_training_step(
                fake_waveform_predicts=fake_waveform_predicts,
                fake_mel_spec_predicts=fake_mel_spec_predicts,
                real_waveform_predicts=real_waveform_predicts,
                real_mel_spec_predicts=real_mel_spec_predicts,
            )
            return discriminator_loss

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
        loss = self.training_step(
            batch=batch,
            batch_idx=batch_idx,
            optimizer_idx=0,
        )

        return loss

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
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )

        return [optimizer], []


if __name__ == '__main__':
    enhancer = HiFiGANEnhancer()
    print(enhancer)

