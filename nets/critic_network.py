from typing import Callable, Tuple
from torch import nn
import torch

from .graph_layers import MultiHeadAttentionLayerforCritic, ValueDecoder


class Critic(nn.Module):
    def __init__(
        self,
        problem_name: str,
        embedding_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        normalization: str,
    ) -> None:

        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.normalization = normalization
        self.encoder = nn.Sequential(
            *(
                MultiHeadAttentionLayerforCritic(
                    self.n_heads,
                    self.embedding_dim,
                    self.hidden_dim,
                    self.normalization,
                )
                for _ in range(1)
            )
        )

        self.value_head = ValueDecoder(
            n_heads=self.n_heads,
            input_dim=self.embedding_dim,
            embed_dim=self.embedding_dim,
        )

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self, input: torch.Tensor, cost: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h_features = input.detach()
        h_em = self.encoder(h_features)
        baseline_value = self.value_head(h_em, cost)

        return baseline_value.detach().squeeze(), baseline_value.squeeze()
