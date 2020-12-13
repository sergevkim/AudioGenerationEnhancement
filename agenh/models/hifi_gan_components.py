from collections import OrderedDict
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool1d,
    BatchNorm1d,
    Conv1d,
    LeakyReLU,
    Module,
    Sequential,
    Sigmoid,
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


class HiFiWaveformDiscriminator(Module):
    def __init__(
            self,
            strided_convs_num: int = 4,
            in_channels: int = 1,
            hidden_dim: int = 10,
        ):
        super().__init__()
        blocks_ordered_dict = OrderedDict()

        blocks_ordered_dict['conv_first'] = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            LeakyReLU(),
        )

        for i in range(strided_convs_num):
            blocks_ordered_dict[f'strided_conv_{i}'] = Sequential(
                Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    padding=1,
                ),
                LeakyReLU(),
            )

        blocks_ordered_dict[f'conv_penultimate'] = Sequential(
            Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            LeakyReLU(),
        )

        blocks_ordered_dict[f'conv_last'] = Sequential(
            Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            #Rearrange(), #do we actually need it?
            AdaptiveAvgPool1d(1),
        )

        self.blocks = Sequential(blocks_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        x = self.blocks(x)

        return x


class SpectrogramDiscriminatorBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
        ):
        super().__init__()
        blocks_ordered_dict = OrderedDict()

        self.conv_first = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            BatchNorm1d(num_features=out_channels),
        )

        self.conv_l = Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.conv_r = Sequential(
            Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            Sigmoid(),
        )

    def forward(
            self,
            x: Tensor,
        ):
        x = self.conv_first(x)
        x_l = self.conv_l(x)
        x_r = self.conv_r(x)

        return x_l + x_r


class HiFiSpectrogramDiscriminator(Module):
    def __init__(
            self,
            blocks_num: int = 4,
            in_channels: int = 1,
            hidden_dim: int = 10,
        ):
        super().__init__()
        blocks_ordered_dict = OrderedDict()

        blocks_ordered_dict[f'block_0'] = SpectrogramDiscriminatorBlock(
            in_channels=in_channels,
            out_channels=hidden_dim,
        )

        for i in range(1, blocks_num):
            blocks_ordered_dict[f'block_{i}'] = SpectrogramDiscriminatorBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
            )

        blocks_ordered_dict[f'conv_last'] = Sequential(
            Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            #Rearrange(), #do we actually need it?
            AdaptiveAvgPool1d(output_size=1),
        )

        self.blocks = Sequential(blocks_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        x = self.blocks(x)

        return x

