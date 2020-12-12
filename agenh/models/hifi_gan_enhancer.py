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
        corrupted = corripted.to(self.device)

        generated = self.generator(corrupted)

        loss = self.criterion(generated, original)

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

    def validation_epoch_end(self):
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

