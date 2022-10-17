from typing import Callable, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch import nn
import math

from problems.problem_pdp import PDP

TYPE_REMOVAL = 'N2S'
# TYPE_REMOVAL = 'random'
# TYPE_REMOVAL = 'greedy'

TYPE_REINSERTION = 'N2S'
# TYPE_REINSERTION = 'random'
# TYPE_REINSERTION = 'greedy'


class SkipConnection(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        in_query_dim: int,
        in_key_dim: int,
        in_val_dim: Optional[int],
        out_dim: int,
    ) -> None:
        super().__init__()

        hidden_dim = out_dim // n_heads

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.in_query_dim = in_query_dim
        self.in_key_dim = in_key_dim
        self.in_val_dim = in_val_dim

        self.norm_factor = 1 / math.sqrt(hidden_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, in_query_dim, hidden_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, in_key_dim, hidden_dim))
        if in_val_dim is not None:  # else calculate attention score
            self.W_val = nn.Parameter(torch.Tensor(n_heads, in_val_dim, hidden_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, hidden_dim, out_dim))

        self.init_parameters()

    def init_parameters(self) -> None:

        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    __call__: Callable[..., torch.Tensor]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if self.in_val_dim is None:  # calculate attention score
            assert v is None

        batch_size, n_query, in_que_dim = q.size()
        _, n_key, in_key_dim = k.size()

        if v is not None:
            in_val_dim = v.size(2)

        qflat = q.contiguous().view(
            -1, in_que_dim
        )  # (batch_size * n_query, in_que_dim)
        kflat = k.contiguous().view(-1, in_key_dim)  # (batch_size * n_key, in_key_dim)
        if v is not None:
            vflat = v.contiguous().view(-1, in_val_dim)

        shp_q = (self.n_heads, batch_size, n_query, self.hidden_dim)
        shp_kv = (self.n_heads, batch_size, n_key, self.hidden_dim)

        # Calculate queries, (n_heads, batch_size, n_query, hidden_dim)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # self.W_que: (n_heads, in_que_dim, hidden_dim)
        # Q_before_view: (n_heads, batch_size * n_query, hidden_dim)

        # Calculate keys and values (n_heads, batch_size, n_key, hidden_dim)
        K = torch.matmul(kflat, self.W_key).view(shp_kv)
        if v is not None:
            V = torch.matmul(vflat, self.W_val).view(shp_kv)

        # Calculate compatibility (n_heads, batch_size, n_query, n_key)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        if v is None:
            return compatibility

        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)  # (n_heads, batch_size, n_query, hidden_dim)

        out = torch.mm(
            heads.permute(1, 2, 0, 3)  # (batch_size, n_query, n_heads, hidden_dim)
            .contiguous()
            .view(
                -1, self.n_heads * self.hidden_dim
            ),  # (batch_size * n_query, n_heads * hidden_dim)
            self.W_out.view(-1, self.out_dim),  # (n_heads * hidden_dim, out_dim)
        ).view(batch_size, n_query, self.out_dim)

        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads: int, input_dim: int) -> None:
        super().__init__()
        self.MHA = MultiHeadAttention(
            n_heads, input_dim, input_dim, input_dim, input_dim
        )

    __call__: Callable[..., torch.Tensor]

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.MHA(q, q, q)


class MultiHeadSelfAttentionScore(nn.Module):
    def __init__(self, n_heads: int, input_dim: int) -> None:
        super().__init__()
        self.MHA = MultiHeadAttention(n_heads, input_dim, input_dim, None, input_dim)

    __call__: Callable[..., torch.Tensor]

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.MHA(q, q)


class CriticDecoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.project_graph = nn.Linear(self.input_dim, self.input_dim // 2)

        self.project_node = nn.Linear(self.input_dim, self.input_dim // 2)

        self.MLP = MLP(input_dim + 1, input_dim)

    __call__: Callable[..., torch.Tensor]

    def forward(self, h_wave: torch.Tensor, best_cost: torch.Tensor) -> torch.Tensor:

        # h_wave: (batch_size, graph_size+1, input_size)
        mean_pooling = h_wave.mean(1)  # mean Pooling (batch_size, input_size)
        graph_feature: torch.Tensor = self.project_graph(mean_pooling)[
            :, None, :
        ]  # (batch_size, 1, input_dim/2)
        node_feature: torch.Tensor = self.project_node(
            h_wave
        )  # (batch_size, graph_size+1, input_dim/2)

        # pass through value_head, get estimated value
        fusion = node_feature + graph_feature.expand_as(
            node_feature
        )  # (batch_size, graph_size+1, input_dim/2)

        fusion_feature = torch.cat(
            (
                fusion.mean(1),
                fusion.max(1)[0],  # max_pooling
                best_cost.to(h_wave.device),
            ),
            -1,
        )  # (batch_size, input_dim + 1)

        value = self.MLP(fusion_feature)

        return value


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        feed_forward_dim: int = 64,
        embedding_dim: int = 64,
        output_dim: int = 1,
        p_dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(p=p_dropout)
        self.ReLU = nn.ReLU(inplace=True)

        self.init_parameters()

    def init_parameters(self) -> None:

        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.ReLU(self.fc1(input))
        result = self.dropout(result)
        result = self.ReLU(self.fc2(result))
        result = self.fc3(result).squeeze(-1)
        return result


class NodePairRemovalDecoder(nn.Module):  # (12) (13)
    def __init__(self, n_heads: int, input_dim: int) -> None:
        super().__init__()

        # hidden_dim = input_dim // n_heads
        hidden_dim = input_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_Q = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        self.W_K = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))

        self.agg = MLP(2 * n_heads + 4, 32, 32, 1, 0)

        self.init_parameters()

    def init_parameters(self) -> None:

        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    __call__: Callable[..., torch.Tensor]

    def forward(
        self,
        h_hat: torch.Tensor,  # hidden state from encoder
        solution: torch.Tensor,  # if solution=[2,0,1], means 0->2->1->0.
        selection_recent: torch.Tensor,  # (batch_size, 4, graph_size/2)
    ) -> torch.Tensor:

        pre = solution.argsort()  # pre=[1,2,0]
        post = solution.gather(1, solution)  # post=[1,2,0]
        batch_size, graph_size_plus1, input_dim = h_hat.size()

        hflat = h_hat.contiguous().view(-1, input_dim)  #################   reshape

        shp = (self.n_heads, batch_size, graph_size_plus1, self.hidden_dim)

        # Calculate queries, (n_heads, batch_size, graph_size+1, key_size)
        hidden_Q = torch.matmul(hflat, self.W_Q).view(shp)
        hidden_K = torch.matmul(hflat, self.W_K).view(shp)

        Q_pre = hidden_Q.gather(
            2, pre.view(1, batch_size, graph_size_plus1, 1).expand_as(hidden_Q)
        )
        K_post = hidden_K.gather(
            2, post.view(1, batch_size, graph_size_plus1, 1).expand_as(hidden_Q)
        )

        compatibility = (
            (Q_pre * hidden_K).sum(-1)
            + (hidden_Q * K_post).sum(-1)
            - (Q_pre * K_post).sum(-1)
        )[
            :, :, 1:
        ]  # (n_heads, batch_size, graph_size) (12)

        compatibility_pairing = torch.cat(
            (
                compatibility[:, :, : graph_size_plus1 // 2],
                compatibility[:, :, graph_size_plus1 // 2 :],
            ),
            0,
        )  # (n_heads*2, batch_size, graph_size/2)

        compatibility_pairing = self.agg(
            torch.cat(
                (
                    compatibility_pairing.permute(1, 2, 0),
                    selection_recent.permute(0, 2, 1),
                ),
                -1,
            )
        ).squeeze()  # (batch_size, graph_size/2)

        return compatibility_pairing


class NodePairReinsertionDecoder(nn.Module):  # (14) (15)
    def __init__(self, n_heads: int, input_dim: int) -> None:
        super().__init__()

        self.n_heads = n_heads

        self.compater_insert1 = MultiHeadAttention(
            n_heads, input_dim, input_dim, None, input_dim * n_heads
        )

        self.compater_insert2 = MultiHeadAttention(
            n_heads, input_dim, input_dim, None, input_dim * n_heads
        )

        self.agg = MLP(4 * n_heads, 32, 32, 1, 0)

    def init_parameters(self) -> None:

        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    __call__: Callable[..., torch.Tensor]

    def forward(
        self,
        h_hat: torch.Tensor,
        pos_pickup: torch.Tensor,  # (batch_size)
        pos_delivery: torch.Tensor,  # (batch_size)
        solution: torch.Tensor,  # (batch, graph_size+1)
    ) -> torch.Tensor:

        batch_size, graph_size_plus1, input_dim = h_hat.size()
        shp = (batch_size, graph_size_plus1, graph_size_plus1, self.n_heads)
        shp_p = (batch_size, -1, 1, self.n_heads)
        shp_d = (batch_size, 1, -1, self.n_heads)

        arange = torch.arange(batch_size, device=h_hat.device)
        h_pickup = h_hat[arange, pos_pickup].unsqueeze(1)  # (batch_size, 1, input_dim)
        h_delivery = h_hat[arange, pos_delivery].unsqueeze(
            1
        )  # (batch_size, 1, input_dim)
        h_K_neibour = h_hat.gather(
            1, solution.view(batch_size, graph_size_plus1, 1).expand_as(h_hat)
        )  # (batch_size, graph_size+1, input_dim)

        compatibility_pickup_pre = (
            self.compater_insert1(
                h_pickup, h_hat
            )  # (n_heads, batch_size, 1, graph_size+1)
            .permute(1, 2, 3, 0)  # (batch_size, 1, graph_size+1, n_heads)
            .view(shp_p)  # (batch_size, graph_size+1, 1, n_heads)
            .expand(shp)  # (batch_size, graph_size+1, graph_size+1, n_heads)
        )
        compatibility_pickup_post = (
            self.compater_insert2(h_pickup, h_K_neibour)
            .permute(1, 2, 3, 0)
            .view(shp_p)
            .expand(shp)
        )
        compatibility_delivery_pre = (
            self.compater_insert1(
                h_delivery, h_hat
            )  # (n_heads, batch_size, 1, graph_size+1)
            .permute(1, 2, 3, 0)  # (batch_size, 1, graph_size+1, n_heads)
            .view(shp_d)  # (batch_size, 1, graph_size+1, n_heads)
            .expand(shp)  # (batch_size, graph_size+1, graph_size+1, n_heads)
        )
        compatibility_delivery_post = (
            self.compater_insert2(h_delivery, h_K_neibour)
            .permute(1, 2, 3, 0)
            .view(shp_d)
            .expand(shp)
        )

        compatibility = self.agg(
            torch.cat(
                (
                    compatibility_pickup_pre,
                    compatibility_pickup_post,
                    compatibility_delivery_pre,
                    compatibility_delivery_post,
                ),
                -1,
            )
        ).squeeze()
        return compatibility  # (batch_size, graph_size+1, graph_size+1)


class N2SDecoder(nn.Module):
    def __init__(self, n_heads: int, input_dim: int, v_range: float = 6) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.range = v_range

        if TYPE_REMOVAL == 'N2S':
            self.compater_removal = NodePairRemovalDecoder(n_heads, input_dim)
        if TYPE_REINSERTION == 'N2S':
            self.compater_reinsertion = NodePairReinsertionDecoder(n_heads, input_dim)

        self.project_graph = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.project_node = nn.Linear(self.input_dim, self.input_dim, bias=False)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]

    def forward(
        self,
        problem: PDP,
        h_wave: torch.Tensor,
        solution: torch.Tensor,
        x_in: torch.Tensor,
        top2: torch.Tensor,
        visit_index: torch.Tensor,
        pre_action: torch.Tensor,
        selection_recent: torch.Tensor,
        fixed_action: Optional[torch.Tensor] = None,
        require_entropy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        batch_size, graph_size_plus1, input_dim = h_wave.size()
        half_pos = (graph_size_plus1 - 1) // 2

        arange = torch.arange(batch_size)

        h_hat: torch.Tensor = self.project_node(h_wave) + self.project_graph(
            h_wave.max(1)[0]
        )[:, None, :].expand(
            batch_size, graph_size_plus1, input_dim
        )  # (11)

        ############# action1 removal
        if TYPE_REMOVAL == 'N2S':
            action_removal_table = (
                torch.tanh(
                    self.compater_removal(h_hat, solution, selection_recent).squeeze()
                )
                * self.range
            )
            if pre_action is not None and pre_action[0, 0] > 0:
                action_removal_table[arange, pre_action[:, 0]] = -1e20
            log_ll_removal = (
                F.log_softmax(action_removal_table, dim=-1) if self.training else None
            )  # log-likelihood
            probs_removal = F.softmax(action_removal_table, dim=-1)
        elif TYPE_REMOVAL == 'random':
            probs_removal = torch.rand(batch_size, graph_size_plus1 // 2).to(
                h_wave.device
            )
        elif TYPE_REMOVAL == 'greedy':
            # epi-greedy
            first_row = (
                torch.arange(graph_size_plus1, device=solution.device)
                .long()
                .unsqueeze(0)
                .expand(batch_size, graph_size_plus1)
            )
            d_i = x_in.gather(
                1, first_row.unsqueeze(-1).expand(batch_size, graph_size_plus1, 2)
            )
            d_i_next = x_in.gather(
                1, solution.long().unsqueeze(-1).expand(batch_size, graph_size_plus1, 2)
            )
            d_i_pre = x_in.gather(
                1,
                solution.argsort()
                .long()
                .unsqueeze(-1)
                .expand(batch_size, graph_size_plus1, 2),
            )
            cost_ = (
                (d_i_pre - d_i).norm(p=2, dim=2)
                + (d_i - d_i_next).norm(p=2, dim=2)
                - (d_i_pre - d_i_next).norm(p=2, dim=2)
            )[:, 1:]
            probs_removal = (
                cost_[:, : graph_size_plus1 // 2] + cost_[:, graph_size_plus1 // 2 :]
            )
            probs_removal_random = torch.rand(batch_size, graph_size_plus1 // 2).to(
                h_wave.device
            )
        else:
            assert False

        if fixed_action is not None:
            action_removal = fixed_action[:, :1]
        else:
            if TYPE_REMOVAL == 'greedy':
                action_removal_random = probs_removal_random.multinomial(1)
                action_removal_greedy = probs_removal.max(-1)[1].unsqueeze(1)
                action_removal = torch.where(
                    torch.rand(batch_size, 1).to(h_wave.device) < 0.1,
                    action_removal_random,
                    action_removal_greedy,
                )
            elif TYPE_REMOVAL == 'N2S' or TYPE_REMOVAL == 'random':
                action_removal = probs_removal.multinomial(1)
            else:
                assert False
        selected_log_ll_action1 = (
            log_ll_removal.gather(1, action_removal)  # type: ignore
            if self.training and TYPE_REMOVAL == 'N2S'
            else torch.tensor(0).to(h_hat.device)
        )

        ############# action2
        pos_pickup = (1 + action_removal).view(-1)
        pos_delivery = pos_pickup + half_pos
        mask_table = (
            problem.get_swap_mask(action_removal + 1, visit_index, top2)
            .expand(batch_size, graph_size_plus1, graph_size_plus1)
            .cpu()
        )
        if TYPE_REINSERTION == 'N2S':
            action_reinsertion_table = (
                torch.tanh(
                    self.compater_reinsertion(h_hat, pos_pickup, pos_delivery, solution)
                )
                * self.range
            )
        elif TYPE_REINSERTION == 'random':
            action_reinsertion_table = torch.ones(
                batch_size, graph_size_plus1, graph_size_plus1
            ).to(h_wave.device)
        elif TYPE_REMOVAL == 'greedy':
            # epi-greedy
            pos_pickup = 1 + action_removal
            pos_delivery = pos_pickup + half_pos
            rec_new = solution.clone()
            argsort = rec_new.argsort()
            pre_pairfirst = argsort.gather(1, pos_pickup)
            post_pairfirst = rec_new.gather(1, pos_pickup)
            rec_new.scatter_(1, pre_pairfirst, post_pairfirst)
            rec_new.scatter_(1, pos_pickup, pos_pickup)
            argsort = rec_new.argsort()
            pre_pairsecond = argsort.gather(1, pos_delivery)
            post_pairsecond = rec_new.gather(1, pos_delivery)
            rec_new.scatter_(1, pre_pairsecond, post_pairsecond)
            # perform calc on new rec_new
            first_row = (
                torch.arange(graph_size_plus1, device=solution.device)
                .long()
                .unsqueeze(0)
                .expand(batch_size, graph_size_plus1)
            )
            d_i = x_in.gather(
                1, first_row.unsqueeze(-1).expand(batch_size, graph_size_plus1, 2)
            )
            d_i_next = x_in.gather(
                1, rec_new.long().unsqueeze(-1).expand(batch_size, graph_size_plus1, 2)
            )
            d_pick = x_in.gather(
                1, pos_pickup.unsqueeze(1).expand(batch_size, graph_size_plus1, 2)
            )
            d_deli = x_in.gather(
                1, pos_delivery.unsqueeze(1).expand(batch_size, graph_size_plus1, 2)
            )
            cost_insert_p = (
                (d_pick - d_i).norm(p=2, dim=2)
                + (d_pick - d_i_next).norm(p=2, dim=2)
                - (d_i - d_i_next).norm(p=2, dim=2)
            )
            cost_insert_d = (
                (d_deli - d_i).norm(p=2, dim=2)
                + (d_deli - d_i_next).norm(p=2, dim=2)
                - (d_i - d_i_next).norm(p=2, dim=2)
            )
            action_reinsertion_table = -(
                cost_insert_p.view(batch_size, graph_size_plus1, 1)
                + cost_insert_d.view(batch_size, 1, graph_size_plus1)
            )
            action_reinsertion_table_random = torch.ones(
                batch_size, graph_size_plus1, graph_size_plus1
            ).to(h_wave.device)
            action_reinsertion_table_random[mask_table] = -1e20
            action_reinsertion_table_random = action_reinsertion_table_random.view(
                batch_size, -1
            )
            probs_reinsertion_random = F.softmax(
                action_reinsertion_table_random, dim=-1
            )
        else:
            assert False

        action_reinsertion_table[mask_table] = -1e20

        del visit_index, mask_table
        # reshape action_reinsertion_table
        action_reinsertion_table = action_reinsertion_table.view(batch_size, -1)
        log_ll_reinsertion = (
            F.log_softmax(action_reinsertion_table, dim=-1)
            if self.training and TYPE_REINSERTION == 'N2S'
            else None
        )
        probs_reinsertion = F.softmax(action_reinsertion_table, dim=-1)
        # fixed action
        if fixed_action is not None:
            p_selected = fixed_action[:, 1]
            d_selected = fixed_action[:, 2]
            pair_index = p_selected * graph_size_plus1 + d_selected
            pair_index = pair_index.view(-1, 1)
            action = fixed_action
        else:
            if TYPE_REINSERTION == 'greedy':
                action_reinsertion_random = probs_reinsertion_random.multinomial(1)
                action_reinsertion_greedy = probs_reinsertion.max(-1)[1].unsqueeze(1)
                pair_index = torch.where(
                    torch.rand(batch_size, 1).to(h_wave.device) < 0.1,
                    action_reinsertion_random,
                    action_reinsertion_greedy,
                )
            elif TYPE_REINSERTION == 'N2S' or TYPE_REINSERTION == 'random':
                # sample one action
                pair_index = probs_reinsertion.multinomial(1)
            else:
                assert False

            p_selected = pair_index // graph_size_plus1
            d_selected = pair_index % graph_size_plus1
            action = torch.cat(
                (action_removal.view(batch_size, -1), p_selected, d_selected), -1
            )  # batch_size, 3

        selected_log_ll_action2 = (
            log_ll_reinsertion.gather(1, pair_index)  # type: ignore
            if self.training and TYPE_REINSERTION == 'N2S'
            else torch.tensor(0).to(h_hat.device)
        )

        log_ll = selected_log_ll_action1 + selected_log_ll_action2

        if require_entropy and self.training:
            dist = Categorical(probs_reinsertion, validate_args=False)
            entropy = dist.entropy()
        else:
            entropy = None

        return action, log_ll, entropy


class Syn_Att(nn.Module):  # (6) - (10)
    def __init__(self, n_heads: int, input_dim: int) -> None:
        super().__init__()

        hidden_dim = input_dim // n_heads

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))

        self.score_aggr = nn.Sequential(
            nn.Linear(2 * n_heads, 2 * n_heads),
            nn.ReLU(inplace=True),
            nn.Linear(2 * n_heads, n_heads),
        )

        self.W_out = nn.Parameter(torch.Tensor(n_heads, hidden_dim, input_dim))

        self.init_parameters()

    def init_parameters(self) -> None:

        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    __call__: Callable[..., torch.Tensor]

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # h should be (batch_size, n_query, input_dim)
        batch_size, n_query, input_dim = h_fea.size()

        hflat = h_fea.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, n_query, self.hidden_dim)

        # Calculate queries, (n_heads, batch_size, n_query, hidden_dim)
        Q = torch.matmul(hflat, self.W_query).view(shp)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, n_key)
        compatibility = torch.cat(
            (torch.matmul(Q, K.transpose(2, 3)), aux_att_score), 0
        )

        attn_raw = compatibility.permute(
            1, 2, 3, 0
        )  # (batch_size, n_query, n_key, n_heads)
        attn = self.score_aggr(attn_raw).permute(
            3, 0, 1, 2
        )  # (n_heads, batch_size, n_query, n_key)
        heads = torch.matmul(
            F.softmax(attn, dim=-1), V
        )  # (n_heads, batch_size, n_query, hidden_dim)

        h_wave = torch.mm(
            heads.permute(1, 2, 0, 3)  # (batch_size, n_query, n_heads, hidden_dim)
            .contiguous()
            .view(
                -1, self.n_heads * self.hidden_dim
            ),  # (batch_size * n_query, n_heads * hidden_dim)
            self.W_out.view(-1, self.input_dim),  # (n_heads * hidden_dim, input_dim)
        ).view(batch_size, n_query, self.input_dim)

        return h_wave, aux_att_score


class Normalization(nn.Module):
    def __init__(self, input_dim: int, normalization: str) -> None:
        super().__init__()

        self.normalization = normalization

        if self.normalization != 'layer':
            normalizer_class = {'batch': nn.BatchNorm1d, 'instance': nn.InstanceNorm1d}[
                normalization
            ]
            self.normalizer = normalizer_class(input_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self) -> None:

        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.normalization == 'layer':
            return (input - input.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                input.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        elif self.normalization == 'batch':
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif self.normalization == 'instance':
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert False, "Unknown normalizer type"


class SynAttNormSubLayer(nn.Module):
    def __init__(self, n_heads: int, input_dim: int, normalization: str) -> None:
        super().__init__()

        self.SynAtt = Syn_Att(n_heads, input_dim)

        self.Norm = Normalization(input_dim, normalization)

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention and Residual connection
        h_wave, aux_att_score = self.SynAtt(h_fea, aux_att_score)

        # Normalization
        return self.Norm(h_wave + h_fea), aux_att_score


class FFNormSubLayer(nn.Module):
    def __init__(
        self, input_dim: int, feed_forward_hidden: int, normalization: str
    ) -> None:
        super().__init__()

        self.FF = (
            nn.Sequential(
                nn.Linear(input_dim, feed_forward_hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(feed_forward_hidden, input_dim, bias=False),
            )
            if feed_forward_hidden > 0
            else nn.Linear(input_dim, input_dim, bias=False)
        )

        self.Norm = Normalization(input_dim, normalization)

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # FF and Residual connection
        out = self.FF(input)
        # Normalization
        return self.Norm(out + input)


class N2SEncoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        feed_forward_hidden: int,
        normalization: str = 'layer',
    ) -> None:
        super().__init__()

        self.SynAttNorm_sublayer = SynAttNormSubLayer(n_heads, input_dim, normalization)

        self.FFNorm_sublayer = FFNormSubLayer(
            input_dim, feed_forward_hidden, normalization
        )

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_wave, aux_att_score = self.SynAttNorm_sublayer(h_fea, aux_att_score)
        return self.FFNorm_sublayer(h_wave), aux_att_score


class EmbeddingNet(nn.Module):
    def __init__(
        self,
        node_dim: int,
        embedding_dim: int,
        seq_length: int,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.feature_embedder = nn.Linear(node_dim, embedding_dim, bias=False)

        self.pattern = self._cyclic_position_embedding_pattern(
            seq_length, embedding_dim
        )

        self.init_parameters()

    def init_parameters(self) -> None:

        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def _base_sin(self, x: np.ndarray, omiga: float, fai: float = 0) -> np.ndarray:
        T = 2 * np.pi / omiga
        return np.sin(omiga * np.abs(np.mod(x, 2 * T) - T) + fai)

    def _base_cos(self, x: np.ndarray, omiga: float, fai: float = 0) -> np.ndarray:
        T = 2 * np.pi / omiga
        return np.cos(omiga * np.abs(np.mod(x, 2 * T) - T) + fai)

    def _cyclic_position_embedding_pattern(
        self, seq_length: int, embedding_dim: int, mean_pooling: bool = True
    ) -> torch.Tensor:

        Td_base = np.power(seq_length, 1 / (embedding_dim // 2))
        Td_set = np.linspace(Td_base, seq_length, embedding_dim // 2, dtype='int')
        g = np.zeros((seq_length, embedding_dim))

        for d in range(embedding_dim):
            Td = (
                Td_set[d // 3 * 3 + 1]
                if (d // 3 * 3 + 1) < (embedding_dim // 2)
                else Td_set[-1]
            )  # (4)

            # get z(i) in the paper (via longer_pattern)
            longer_pattern = np.arange(0, np.ceil(seq_length / Td) * Td, 0.01)

            num = len(longer_pattern)
            omiga = 2 * np.pi / Td
            fai = (
                0
                if d <= (embedding_dim // 2)
                else 2 * np.pi * ((-d + (embedding_dim // 2)) / (embedding_dim // 2))
            )

            # Eq. (2) in the paper
            if d % 2 == 1:
                g[:, d] = self._base_cos(longer_pattern, omiga, fai)[
                    np.linspace(0, num, seq_length, dtype='int', endpoint=False)
                ]
            else:
                g[:, d] = self._base_sin(longer_pattern, omiga, fai)[
                    np.linspace(0, num, seq_length, dtype='int', endpoint=False)
                ]

        pattern: torch.Tensor = torch.from_numpy(g).type(torch.FloatTensor)  # type: ignore
        pattern_sum = torch.zeros_like(pattern)

        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(seq_length)
        pooling = [0] if not mean_pooling else [-2, -1, 0, 1, 2]
        time = 0
        for d in pooling:
            time += 1
            index = (arange + d + seq_length) % seq_length
            pattern_sum += pattern.gather(0, index.view(-1, 1).expand_as(pattern))
        pattern = 1.0 / time * pattern_sum - pattern.mean(0)
        #### ----

        return pattern  # (seq_length, embedding_dim)

    def _position_embedding(
        self, solution: torch.Tensor, embedding_dim: int, clac_stacks: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = solution.size()
        half_size = seq_length // 2

        # expand for every batch
        position_emb_new = (
            self.pattern.expand(batch_size, seq_length, embedding_dim)
            .clone()
            .to(solution.device)
        )

        # get index according to the solutions
        visit_index = torch.zeros((batch_size, seq_length), device=solution.device)

        pre = torch.zeros((batch_size), device=solution.device).long()

        arange = torch.arange(batch_size)
        if clac_stacks:
            stacks = (
                torch.zeros(batch_size, half_size + 1, device=solution.device) - 0.01
            )  # fix bug: topk is not stable sorting
            top2 = torch.zeros(batch_size, seq_length, 2, device=solution.device).long()
            stacks[arange, pre] = 0  # fix bug: topk is not stable sorting

        for i in range(seq_length):
            current_nodes = solution[arange, pre]  # (batch_size,)
            visit_index[arange, current_nodes] = i + 1
            pre = current_nodes

            if clac_stacks:
                index1 = (current_nodes <= half_size) & (current_nodes > 0)
                index2 = (current_nodes > half_size) & (current_nodes > 0)
                if index1.any():
                    stacks[index1, current_nodes[index1]] = i + 1
                if index2.any():
                    stacks[
                        index2, current_nodes[index2] - half_size
                    ] = -0.01  # fix bug: topk is not stable sorting
                top2[arange, current_nodes] = stacks.topk(2)[1]
                # stack top after visit
                # node+, (current_stack_top, last_stack_top_or_0)
                # node-, (current_stack_top, last_stack_top_or_0) or (0, 1_meaningless)

        index = (
            (visit_index % seq_length)
            .long()
            .unsqueeze(-1)
            .expand(batch_size, seq_length, embedding_dim)
        )

        return (
            torch.gather(position_emb_new, 1, index),
            (visit_index % seq_length).long(),
            top2 if clac_stacks else None,
        )

    __call__: Callable[
        ..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    ]

    def forward(
        self, x: torch.Tensor, solution: torch.Tensor, clac_stacks: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pos_emb, visit_index, top2 = self._position_embedding(
            solution, self.embedding_dim, clac_stacks
        )
        fea_emb = self.feature_embedder(x)
        return fea_emb, pos_emb, visit_index, top2


class CriticEncoder(nn.Sequential):
    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        feed_forward_hidden: int,
        normalization: str = 'layer',
    ) -> None:
        super().__init__(
            SkipConnection(MultiHeadSelfAttention(n_heads, input_dim)),
            Normalization(input_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(input_dim, feed_forward_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(
                        feed_forward_hidden,
                        input_dim,
                    ),
                )
                if feed_forward_hidden > 0
                else nn.Linear(input_dim, input_dim)
            ),
            Normalization(input_dim, normalization),
        )
