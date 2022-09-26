from typing import Dict, Tuple
from abc import ABCMeta
import torch
from tensorboard_logger import Logger as TbLogger

from nets.actor_network import Actor
from nets.critic_network import Critic
from options import Option
from problems.problem_pdp import PDP


class Agent(metaclass=ABCMeta):
    opts: Option
    actor: Actor
    critic: Critic

    def __init__(self, problem_name: str, size: int, opts: Option) -> None:
        pass

    def load(self, load_path: str) -> None:
        pass

    def save(self, epoch: int) -> None:
        pass

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass

    def rollout(
        self,
        problem: PDP,
        val_m: int,
        batch: Dict[str, torch.Tensor],
        do_sample: bool = False,
        show_bar: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def start_inference(
        self, problem: PDP, val_dataset: str, tb_logger: TbLogger
    ) -> None:
        pass

    def start_training(
        self, problem: PDP, val_dataset: str, tb_logger: TbLogger
    ) -> None:
        pass
