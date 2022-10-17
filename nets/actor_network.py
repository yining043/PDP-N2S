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

    __call__: Callable[..., Union[Tuple[torch.Tensor], torch.Tensor]]

    def forward(
        self, *inputs: Union[Tuple[torch.Tensor], torch.Tensor]
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs  # type: ignore


class Actor(nn.Module):
    def __init__(
        self,
        problem_name: str,
        embedding_dim: int,
        ff_hidden_dim: int,
        n_heads_actor: int,
        n_layers: int,
        normalization: str,
        v_range: float,
        seq_length: int,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.ff_hidden_dim = ff_hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_layers = n_layers
        self.normalization = normalization
        self.v_range = v_range
        self.seq_length = seq_length
        self.clac_stacks = bool(problem_name == 'pdtspl')
        self.node_dim = 2

        # networks
        self.embedder = EmbeddingNet(self.node_dim, self.embedding_dim, self.seq_length)

        self.pos_emb_encoder = MultiHeadSelfAttentionScore(
            self.n_heads_actor, self.embedding_dim
        )  # for PFEs

        self.encoder = mySequential(
            *(
                N2SEncoder(
                    self.n_heads_actor,
                    self.embedding_dim,
                    self.ff_hidden_dim,
                    self.normalization,
                )
                for _ in range(self.n_layers)
            )
        )  # for NFEs

        self.decoder = N2SDecoder(
            self.n_heads_actor, self.embedding_dim, self.v_range
        )  # the two propsoed decoders

        print(self.get_parameter_number())

    def get_parameter_number(self) -> Dict[str, int]:
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    @staticmethod
    def get_action_recent(action_record: List[torch.Tensor]) -> torch.Tensor:
        action_record_tensor = torch.stack(action_record)
        return torch.cat(
            (
                action_record_tensor[-3:].transpose(0, 1),
                action_record_tensor.mean(0).unsqueeze(1),
            ),
            1,
        )

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
        # batch_size, graph_size+1, node_dim = x_in.size()

        h_fea, g_pos, visit_index, top2 = self.embedder(
            x_in, solution, self.clac_stacks
        )

        # pass through encoder
        aux_att = self.pos_emb_encoder(g_pos)
        h_wave = self.encoder(h_fea, aux_att)[0]

        if only_critic:
            return h_wave

        # pass through decoder
        action, log_ll, entropy = self.decoder(
            problem=problem,
            h_wave=h_wave,
            solution=solution,
            x_in=x_in,
            top2=top2,
            visit_index=visit_index,
            pre_action=exchange,
            selection_recent=Actor.get_action_recent(action_record).to(x_in.device),
            fixed_action=fixed_action,
            require_entropy=require_entropy,
        )

        if require_entropy:
            return action, log_ll.squeeze(), h_wave if to_critic else None, entropy
        else:
            return action, log_ll.squeeze(), h_wave if to_critic else None
