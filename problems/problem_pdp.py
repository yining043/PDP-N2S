from typing import Dict, Iterable, List, Optional, Tuple
from abc import ABCMeta, abstractmethod
import os
import pickle
import torch
from torch.utils.data import Dataset


class PDP(metaclass=ABCMeta):

    name: str

    @abstractmethod
    def __init__(
        self, p_size: int, init_val_met: str = 'p2d', with_assert: bool = False
    ) -> None:
        self.size = p_size  # the number of nodes in PDTSP
        self.do_assert = with_assert
        self.init_val_met = init_val_met
        self.state = 'eval'

    @staticmethod
    def input_feature_encoding(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch['coordinates']

    @staticmethod
    def get_visited_order_map(visit_index: torch.Tensor) -> torch.Tensor:
        batch_size, graph_size = visit_index.size()
        visit_index = visit_index % graph_size

        return visit_index.view(batch_size, graph_size, 1) > visit_index.view(
            batch_size, 1, graph_size
        )

    @staticmethod
    def _insert_star(
        solution: torch.Tensor,
        pair_index: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:

        rec = solution.clone()
        bs, gs = rec.size()

        # fix connection for pairing node
        argsort = rec.argsort()
        pre_pairfirst = argsort.gather(1, pair_index)
        post_pairfirst = rec.gather(1, pair_index)
        rec.scatter_(1, pre_pairfirst, post_pairfirst)
        rec.scatter_(1, pair_index, pair_index)

        argsort = rec.argsort()

        pre_pairsecond = argsort.gather(1, pair_index + gs // 2)
        post_pairsecond = rec.gather(1, pair_index + gs // 2)

        rec.scatter_(1, pre_pairsecond, post_pairsecond)

        # fix connection for pairing node
        post_second = rec.gather(1, second)
        rec.scatter_(1, second, pair_index + gs // 2)
        rec.scatter_(1, pair_index + gs // 2, post_second)

        post_first = rec.gather(1, first)
        rec.scatter_(1, first, pair_index)
        rec.scatter_(1, pair_index, post_first)

        return rec

    @staticmethod
    def make_dataset(*args, **kwargs) -> 'PDPDataset':
        return PDPDataset(*args, **kwargs)

    @abstractmethod
    def get_initial_solutions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def get_swap_mask(
        self,
        selected_node: torch.Tensor,
        visited_order_map: torch.Tensor,
        top2: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def _check_feasibility(self, rec: torch.Tensor) -> None:
        pass

    def get_costs(
        self, batch: Dict[str, torch.Tensor], rec: torch.Tensor
    ) -> torch.Tensor:

        batch_size, size = rec.size()

        # check feasibility
        if self.do_assert:
            self._check_feasibility(rec)

        # calculate obj value
        d1 = batch['coordinates'].gather(
            1, rec.long().unsqueeze(-1).expand(batch_size, size, 2)
        )
        d2 = batch['coordinates']
        length = (d1 - d2).norm(p=2, dim=2).sum(1)

        return length

    def step(
        self,
        batch: Dict[str, torch.Tensor],
        rec: torch.Tensor,
        exchange: torch.Tensor,
        pre_bsf: torch.Tensor,
        action_record: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:

        bs, gs = rec.size()
        pre_bsf = pre_bsf.view(bs, -1)

        cur_vec = action_record.pop(0) * 0.0
        cur_vec[torch.arange(bs), exchange[:, 0]] = 1.0
        action_record.append(cur_vec)

        selected = exchange[:, 0].view(bs, 1)
        first = exchange[:, 1].view(bs, 1)
        second = exchange[:, 2].view(bs, 1)

        next_state = PDP._insert_star(rec, selected + 1, first, second)

        new_obj = self.get_costs(batch, next_state)

        now_bsf = torch.min(
            torch.cat((new_obj[:, None], pre_bsf[:, -1, None]), -1), -1
        )[0]

        reward = pre_bsf[:, -1] - now_bsf

        return (
            next_state,
            reward,
            torch.cat((new_obj[:, None], now_bsf[:, None]), -1),
            action_record,
        )


class PDPDataset(Dataset):
    def __init__(
        self,
        filename: Optional[str] = None,
        size: int = 20,
        num_samples: int = 10000,
        offset: int = 0,
    ):

        super().__init__()

        self.data: List[Dict[str, torch.Tensor]] = []
        self.size = size

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'file name error'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [
                PDPDataset._make_instance(args)
                for args in data[offset : offset + num_samples]
            ]

        else:
            self.data = [
                {
                    'loc': torch.FloatTensor(self.size, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                }
                for i in range(num_samples)
            ]

        self.N = len(self.data)

        # calculate distance matrix
        for i, instance in enumerate(self.data):
            self.data[i]['coordinates'] = torch.cat(
                (instance['depot'].reshape(1, 2), instance['loc']), dim=0
            )
            del self.data[i]['depot']
            del self.data[i]['loc']
        print(f'{self.N} instances initialized.')

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]

    @staticmethod
    def _make_instance(args: Iterable) -> Dict[str, torch.Tensor]:
        depot, loc, *args = args
        grid_size = 1
        if len(args) > 0:
            depot_types, customer_types, grid_size = args
        return {
            'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
            'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        }

    @staticmethod
    def calculate_distance(data: torch.Tensor) -> torch.Tensor:
        N_data = data.shape[0]
        dists = torch.zeros((N_data, N_data), dtype=torch.float)
        d1 = -2 * torch.mm(data, data.T)
        d2 = torch.sum(torch.pow(data, 2), dim=1)
        d3 = torch.sum(torch.pow(data, 2), dim=1).reshape(1, -1).T
        dists = d1 + d2 + d3
        dists[dists < 0] = 0
        return torch.sqrt(dists)
