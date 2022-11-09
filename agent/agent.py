from typing import Dict, Tuple
from abc import ABC, abstractmethod
import torch
from tensorboard_logger import Logger as TbLogger

from nets.actor_network import Actor
from nets.critic_network import Critic
from options import Option
from problems.problem_pdp import PDP


class Agent(ABC):
    opts: Option
    actor: Actor
    critic: Critic
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR

    @abstractmethod
    def __init__(self, problem_name: str, size: int, opts: Option) -> None:
        pass

    @abstractmethod
    def load(self, load_path: str) -> None:
        pass

    @abstractmethod
    def save(self, epoch: int) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def rollout(
        self, problem: PDP, val_m: int, batch: Dict[str, torch.Tensor], show_bar: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def start_inference(
        self, problem: PDP, val_dataset: str, tb_logger: TbLogger
    ) -> None:
        pass

    @abstractmethod
    def start_training(
        self, problem: PDP, val_dataset: str, tb_logger: TbLogger
    ) -> None:
        pass
