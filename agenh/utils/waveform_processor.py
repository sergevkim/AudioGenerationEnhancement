import torch
from torch import Tensor
from torch.nn import Module


class WaveformProcessor(Module):
    def __init__(
            self,
        ):
        super().__init__()

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        noise = torch.zeros_like(x)
        x = x + noise

        return x

