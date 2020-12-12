from collections import OrderedDict

import torch
from torch.nn import (
    Conv1d,
    Module,
    Sequential,
    Tanh,
)


class WaveNet(Module):
    def __init__(
            self,
        ):
        super().__init__()

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        pass


class PostNet(Module):
    def __init__(
            self,
            blocks_num: int = 4,
        ):
        super().__init__()

        block_ordered_dict = OrderedDict()

        for i in range(blocks_num):
            block_ordered_dict[f'block_{i}'] = Sequential(
                Conv1d(),
                Tanh(),
            )

        self.blocks = Sequential(blocks_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        x = self.blocks(x)

        return x


class HiFiGenerator(Module):
    def __init__(
            self,
        ):
        super().__init__()
        self.wavenet = WaveNet()
        self.postnet = PostNet()

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        x = self.wavenet(x)
        x = self.postnet(x)

        return x


class HiFiDiscriminator(Module):
    def __init__(
            self,
        ):
        super().__init__()

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        pass

