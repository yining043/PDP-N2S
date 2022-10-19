from typing import Dict, Iterable, List, Optional, Tuple
from abc import ABC, abstractmethod
import os
import pickle
import torch
from torch.utils.data import Dataset


class PDP(ABC):

    name: str

    @abstractmethod
    def __init__(
        self, size: int, init_val_method: str, check_feasible: bool = False
    ) -> None:
        self.size = size  # the number of nodes in PDTSP
        self.check_feasible = check_feasible
        self.init_val_method = init_val_method
        self.state = 'eval'

    @staticmethod
    @abstractmethod
    def get_swap_mask(
        selected_node: torch.Tensor,
        visit_index: torch.Tensor,
        top2: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_initial_solutions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def _check_feasibility(self, solution: torch.Tensor) -> None:
        pass

    @staticmethod
    def input_coordinates(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch['coordinates']

    @staticmethod
    def _get_visit_order_map(visit_index: torch.Tensor) -> torch.Tensor:
        batch_size, graph_size_plus1 = visit_index.size()
        return visit_index.view(batch_size, graph_size_plus1, 1) > visit_index.view(
            batch_size, 1, graph_size_plus1
        )  # row late than column: true

    @staticmethod
    def _insert_star(
        solution: torch.Tensor,  # (batch_size, graph_size+1)
        pair_first: torch.Tensor,  # (batch_size, 1)
        first: torch.Tensor,  # (batch_size, 1)
        second: torch.Tensor,  # (batch_size, 1)
    ) -> torch.Tensor:
        solution = solution.clone()  # if solution=[2,0,1], means 0->2->1->0.
        graph_size_plus1 = solution.size(1)

        assert (
            (pair_first != first).all()
            and (pair_first != second).all()
            and ((pair_first + graph_size_plus1 // 2) != first).all()
            and ((pair_first + graph_size_plus1 // 2) != second).all()
        )

        # remove pair node
        pre = solution.argsort()  # pre=[1,2,0]
        pre_pair_first = pre.gather(1, pair_first)  # (batch_size, 1)
        post_pair_first = solution.gather(1, pair_first)  # (batch_size, 1)

        solution.scatter_(1, pre_pair_first, post_pair_first)  # remove pair first
        solution.scatter_(
            1, pair_first, pair_first
        )  # let: pair first -> pair first, for next line's pre correct

        pre = solution.argsort()

        pre_pair_second = pre.gather(1, pair_first + graph_size_plus1 // 2)
        post_pair_second = solution.gather(1, pair_first + graph_size_plus1 // 2)

        solution.scatter_(1, pre_pair_second, post_pair_second)  # remove pair second

        # insert pair node
        post_second = solution.gather(1, second)
        solution.scatter_(
            1, second, pair_first + graph_size_plus1 // 2
        )  # second -> pair_second
        solution.scatter_(1, pair_first + graph_size_plus1 // 2, post_second)

        post_first = solution.gather(1, first)
        solution.scatter_(1, first, pair_first)  # first -> pair_first
        solution.scatter_(1, pair_first, post_first)

        return solution

    @staticmethod
    def make_dataset(
        filename: Optional[str] = None,
        size: int = 20,
        num_samples: int = 10000,
        offset: int = 0,
    ) -> 'PDPDataset':
        return PDPDataset(filename, size, num_samples, offset)

    def get_costs(
        self, batch: Dict[str, torch.Tensor], solution: torch.Tensor
    ) -> torch.Tensor:

        batch_size, graph_size_plus1 = solution.size()

        # check feasibility
        if self.check_feasible:
            self._check_feasibility(solution)

        # calculate obj value
        d1 = batch['coordinates'].gather(
            1, solution.long().unsqueeze(-1).expand(batch_size, graph_size_plus1, 2)
        )
        d2 = batch['coordinates']
        length = (d1 - d2).norm(p=2, dim=2).sum(1)  # (batch_size,)

        return length

    def step(
        self,
        batch: Dict[
            str, torch.Tensor
        ],  # ['coordinates']: (batch_size, graph_size+1, 2)
        solution: torch.Tensor,  # (batch_size, graph_size+1)
        action: torch.Tensor,  # (batch_size, 3)
        pre_best_obj: torch.Tensor,  # (batch_size, 2) or (batch_size,)
        action_removal_record: List[torch.Tensor],  # len * (batch_size, graph_size/2)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:

        batch_size = solution.size(0)
        pre_best_obj = pre_best_obj.view(batch_size, -1)

        cur_vec = action_removal_record.pop(0) * 0.0
        cur_vec[torch.arange(batch_size), action[:, 0]] = 1.0
        action_removal_record.append(cur_vec)

        selected_minus1 = action[:, 0].view(batch_size, 1)
        first = action[:, 1].view(batch_size, 1)
        second = action[:, 2].view(batch_size, 1)

        next_state = PDP._insert_star(solution, selected_minus1 + 1, first, second)

        new_obj = self.get_costs(batch, next_state)

        now_best_obj = torch.min(
            torch.cat((new_obj[:, None], pre_best_obj[:, -1, None]), -1), -1
        )[0]

        reward = pre_best_obj[:, -1] - now_best_obj  # (batch_size,)

        return (
            next_state,
            reward,
            torch.cat((new_obj[:, None], now_best_obj[:, None]), -1),
            action_removal_record,
        )

    @staticmethod
    def direct_solution(solution: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = solution.size()
        arange = torch.arange(batch_size)
        visit_index = torch.zeros((batch_size, seq_length), device=solution.device)
        pre = torch.zeros((batch_size), device=solution.device).long()

        for i in range(seq_length):
            current_nodes = solution[arange, pre]  # (batch_size,)
            visit_index[arange, current_nodes] = i + 1
            pre = current_nodes

        visit_index = (visit_index % seq_length).long()
        return visit_index.argsort()


class PDPDataset(Dataset):
    def __init__(
        self, filename: Optional[str], size: int, num_samples: int, offset: int
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
                for _ in range(num_samples)
            ]

        self.N = len(self.data)

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
