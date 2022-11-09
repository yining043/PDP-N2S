from typing import Optional
import time
import torch
import os
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger

from problems.problem_pdp import PDP
from utils.logger import log_to_screen, log_to_tb_val

from .agent import Agent


def gather_tensor_and_concat(tensor: torch.Tensor) -> torch.Tensor:
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)


def validate(
    rank: int,
    problem: PDP,
    agent: Agent,
    val_dataset_str: str,
    tb_logger: TbLogger,
    distributed: bool = False,
    id_: Optional[int] = None,
) -> None:

    # Validate mode
    if rank == 0:
        print('\nValidating...', flush=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    opts = agent.opts
    agent.eval()

    val_dataset = PDP.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=val_dataset_str
    )

    if distributed and opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(
            backend='nccl', world_size=opts.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        if torch.cuda.device_count() > 1:
            agent.actor = torch.nn.parallel.DistributedDataParallel(
                agent.actor, device_ids=[rank]
            )  # type: ignore
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(
                os.path.join(
                    opts.log_dir,
                    "{}_{}".format(opts.problem, opts.graph_size),
                    opts.run_name,
                )
            )

    if distributed and opts.distributed:
        assert opts.val_batch_size % opts.world_size == 0
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )  # type: ignore
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=opts.val_batch_size // opts.world_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler,
        )
    else:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=opts.val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    s_time = time.time()
    bv_list = []
    cost_hist_list = []
    best_hist_list = []
    r_list = []
    for batch in tqdm(
        val_dataloader, desc='inference', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    ):
        bv_, cost_hist_, best_hist_, r_ = agent.rollout(
            problem, opts.val_m, batch, show_bar=rank == 0
        )
        bv_list.append(bv_)
        cost_hist_list.append(cost_hist_)
        best_hist_list.append(best_hist_)
        r_list.append(r_)
    bv = torch.cat(bv_list, 0)
    cost_hist = torch.cat(cost_hist_list, 0)
    best_hist = torch.cat(best_hist_list, 0)
    r = torch.cat(r_list, 0)

    if distributed and opts.distributed:
        dist.barrier()

    if distributed and opts.distributed:
        initial_cost = gather_tensor_and_concat(cost_hist[:, 0].contiguous())
        time_used = gather_tensor_and_concat(
            torch.tensor([time.time() - s_time]).cuda()
        )
        bv = gather_tensor_and_concat(bv.contiguous())
        costs_history = gather_tensor_and_concat(cost_hist.contiguous())
        search_history = gather_tensor_and_concat(best_hist.contiguous())
        reward = gather_tensor_and_concat(r.contiguous())

    else:
        initial_cost = cost_hist[:, 0]  # bs
        time_used = torch.tensor([time.time() - s_time])  # bs
        bv = bv
        costs_history = cost_hist
        search_history = best_hist
        reward = r
    # log to screen
    if rank == 0:
        log_to_screen(
            time_used,
            initial_cost,
            bv,
            reward,
            costs_history,
            search_history,
            batch_size=opts.val_size,
            dataset_size=len(val_dataset),
            T=opts.T_max,
        )

    # log to tb
    if (not opts.no_tb) and rank == 0:
        log_to_tb_val(
            tb_logger,
            time_used,
            initial_cost,
            bv,
            reward,
            costs_history,
            search_history,
            batch_size=opts.val_size,
            val_size=opts.val_size,
            dataset_size=len(val_dataset),
            T=opts.T_max,
            show_figs=opts.show_figs,
            epoch=id_,
        )
