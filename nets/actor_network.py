from typing import Callable, Dict, List, Optional, Tuple, Union
from torch import nn
import torch

from problems.problem_pdp import PDP

from .graph_layers import (
    N2SEncoder,
    N2SDecoder,
    EmbeddingNet,
    MultiHeadSelfAttentionScore,
)


class mySequential(nn.Sequential):

    __call__: Callable[..., torch.Tensor]

    def forward(
        self, *inputs: Union[Tuple[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs  # type: ignore


def get_action_sig(action_record: List[torch.Tensor]) -> torch.Tensor:
    action_record_tensor = torch.stack(action_record)
    return torch.cat(
        (
            action_record_tensor[-3:].transpose(0, 1),
            action_record_tensor.mean(0).unsqueeze(1),
        ),
        1,
    )


class Actor(nn.Module):
    def __init__(
        self,
        problem_name: str,
        embedding_dim: int,
        hidden_dim: int,
        n_heads_actor: int,
        n_layers: int,
        normalization: str,
        v_range: float,
        seq_length: int,
    ) -> None:
        super(Actor, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_layers = n_layers
        self.normalization = normalization
        self.range = v_range
        self.seq_length = seq_length
        self.clac_stacks = problem_name == 'pdtspl'
        self.node_dim = 2

        # networks
        self.embedder = EmbeddingNet(self.node_dim, self.embedding_dim, self.seq_length)

        self.encoder = mySequential(
            *(
                N2SEncoder(
                    self.n_heads_actor,
                    self.embedding_dim,
                    self.hidden_dim,
                    self.normalization,
                )
                for _ in range(self.n_layers)
            )
        )  # for NFEs

        self.pos_encoder = MultiHeadSelfAttentionScore(
            self.n_heads_actor,
            self.embedding_dim,
            self.hidden_dim,
        )  # for PFEs

        self.decoder = N2SDecoder(
            input_dim=self.embedding_dim,
            v_range=self.range,
        )  # the two propsoed decoders

        print(self.get_parameter_number())

    def get_parameter_number(self) -> Dict[str, int]:
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    __call__: Callable[
        ...,
        Union[
            torch.Tensor,
            Tuple[
                torch.Tensor,
                torch.Tensor,
                Optional[torch.Tensor],
                Optional[torch.Tensor],
            ],
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        ],
    ]

    def forward(
        self,
        problem: PDP,
        x_in: torch.Tensor,
        solution: torch.Tensor,
        exchange: torch.Tensor,
        action_record: List[torch.Tensor],
        do_sample: bool = False,
        fixed_action: bool = None,
        require_entropy: bool = False,
        to_critic: bool = False,
        only_critic: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ],
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    ]:

        # the embedded input x
        bs, gs, in_d = x_in.size()

        h_embed, h_pos, visited_time, top2 = self.embedder(
            x_in, solution, self.clac_stacks
        )

        # pass through encoder
        pos_em = self.pos_encoder(h_pos)
        h_em = self.encoder(h_embed, pos_em)[0]

        if only_critic:
            return h_em

        visited_order_map = problem.get_visited_order_map(visited_time)
        del visited_time

        # pass through decoder
        action, log_ll, entropy = self.decoder(
            problem,
            h_em,
            solution,
            x_in,
            top2,
            visited_order_map,
            exchange,
            get_action_sig(action_record).to(x_in.device),
            fixed_action,
            require_entropy=require_entropy,
        )

        if require_entropy:
            return action, log_ll.squeeze(), (h_em) if to_critic else None, entropy
        else:
            return action, log_ll.squeeze(), (h_em) if to_critic else None
