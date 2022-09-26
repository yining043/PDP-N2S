from abc import ABCMeta
import torch
from torch.utils.data import Dataset
from typing import Dict, Iterable, List, Optional, Tuple


class PDP(metaclass=ABCMeta):

    NAME: str
    size: int

    def __init__(
        self, p_size: int, init_val_met: str = 'p2d', with_assert: bool = False
    ) -> None:
        pass

    def input_feature_encoding(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def get_visited_order_map(self, visited_time: torch.Tensor) -> torch.Tensor:
        pass

    def get_real_mask(
        self,
        selected_node: torch.Tensor,
        visited_order_map: torch.Tensor,
        top2: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def get_initial_solutions(
        self, batch: Dict[str, torch.Tensor], val_m: int = 1
    ) -> torch.Tensor:
        pass

    def step(
        self,
        batch: Dict[str, torch.Tensor],
        rec: torch.Tensor,
        exchange: torch.Tensor,
        pre_bsf: torch.Tensor,
        action_record: List,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]:
        pass

    def insert_star(
        self,
        solution: torch.Tensor,
        pair_index: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def check_feasibility(self, rec: torch.Tensor) -> None:
        pass

    def get_swap_mask(
        self,
        selected_node: torch.Tensor,
        visited_order_map: torch.Tensor,
        top2: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def get_costs(
        self, batch: Dict[str, torch.Tensor], rec: torch.Tensor
    ) -> torch.Tensor:
        pass

    @staticmethod
    def make_dataset(*args, **kwargs) -> 'PDPDatasetMeta':
        pass


class PDPDatasetMeta(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        filename: Optional[str] = None,
        size: int = 20,
        num_samples: int = 10000,
        offset: int = 0,
        distribution: Optional[bool] = None,
    ):
        pass

    def make_instance(self, args: Iterable) -> Dict[str, torch.Tensor]:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pass
