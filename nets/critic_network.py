from typing import Callable, Tuple
from torch import nn
import torch

from .graph_layers import CriticEncoder, CriticDecoder


class Critic(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ff_hidden_dim: int,
        n_heads: int,
        n_layers: int,
        normalization: str,
    ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        self.ff_hidden_dim = ff_hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.normalization = normalization
        self.encoder = nn.Sequential(
            *(
                CriticEncoder(
                    self.n_heads,
                    self.embedding_dim,
                    self.ff_hidden_dim,
                    self.normalization,
                )
                for _ in range(1)
            )
        )

        self.decoder = CriticDecoder(self.embedding_dim)

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self, h_wave: torch.Tensor, best_cost: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        y = self.encoder(h_wave.detach())
        baseline_value = self.decoder(y, best_cost)

        return baseline_value.detach().squeeze(), baseline_value.squeeze()
