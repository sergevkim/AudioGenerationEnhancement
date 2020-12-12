from collections import OrderedDict
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import (
    Conv1d,
    Module,
    Sequential,
    Tanh,
)


class WaveNet(Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 5,
            hidden_dim: int = 10,
        ):
        super().__init__()
        self.conv_first = Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.conv_last = Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
            self,
            x: Tensor, #shape: (batch_size, wav_langth)
        ) -> Tensor:
        x = self.conv_1(x)
        x = self.conv_2(x)

        return x


class PostNet(Module):
    def __init__(
            self,
            blocks_num: int = 4,
            in_channels: int = 5,
            out_channels: int = 1,
            hidden_dim: int = 6,
        ):
        super().__init__()

        blocks_ordered_dict = OrderedDict()
        blocks_ordered_dict['block_0'] = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            Tanh(),
        )

        for i in range(1, blocks_num - 1):
            blocks_ordered_dict[f'block_{i}'] = Sequential(
                Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    padding=1,
                ),
                Tanh(),
            )

        blocks_ordered_dict[f'block_{blocks_num - 1}'] = Sequential(
            Conv1d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
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
        self.wavenet = WaveNet(
            in_channels=1,
            out_channels=1,
            hidden_dim=10,
        )
        self.postnet = PostNet(
            in_channels=1,
            out_channels=1,
            hidden_dim=10,
        )

    def forward(
            self,
            x: Tensor,
        ) -> Tuple[Tensor, Tensor]:
        x_w = self.wavenet(x)
        x_p = self.postnet(x_w)

        return x_w, x_p


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

