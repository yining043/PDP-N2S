from typing import Dict, List, Tuple
import os
from tqdm import tqdm
import warnings
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboard_logger import Logger as TbLogger
import numpy as np
import random

from utils import clip_grad_norms, rotate_tensor
from nets.actor_network import Actor
from nets.critic_network import Critic
from utils import torch_load_cpu, get_inner_model, move_to, move_to_cuda
from utils.logger import log_to_tb_train
from problems.problem_pdp import PDP
from options import Option

from .agent import Agent
from .utils import validate


class Memory:
    def __init__(self) -> None:
        self.actions: List[torch.Tensor] = []
        self.states: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.obj: List[torch.Tensor] = []
        self.action_record: List[List[torch.Tensor]] = []

    def clear_memory(self) -> None:
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.obj[:]
        del self.action_record[:]


class PPO(Agent):
    def __init__(self, problem_name: str, size: int, opts: Option) -> None:

        # figure out the options
        self.opts = opts

        # figure out the actor
        self.actor = Actor(
            problem_name=problem_name,
            embedding_dim=opts.embedding_dim,
            ff_hidden_dim=opts.ff_hidden_dim,
            n_heads_actor=opts.actor_head_num,
            n_layers=opts.n_encode_layers,
            normalization=opts.normalization,
            v_range=opts.v_range,
            seq_length=size + 1,
        )

        if not opts.eval_only:

            # figure out the critic
            self.critic = Critic(
                embedding_dim=opts.embedding_dim,
                ff_hidden_dim=opts.ff_hidden_dim,
                n_heads=opts.critic_head_num,
                n_layers=opts.n_encode_layers,
                normalization=opts.normalization,
            )

            # figure out the optimizer
            self.optimizer = torch.optim.Adam(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}]
                + [{'params': self.critic.parameters(), 'lr': opts.lr_critic}]
            )

            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                opts.lr_decay,
                last_epoch=-1,
            )

        print(f'Distributed: {opts.distributed}')
        if opts.use_cuda and not opts.distributed:

            self.actor.to(opts.device)
            if not opts.eval_only:
                self.critic.to(opts.device)

            if torch.cuda.device_count() > 1:
                self.actor = torch.nn.DataParallel(self.actor)  # type: ignore
                if not opts.eval_only:
                    self.critic = torch.nn.DataParallel(self.critic)  # type: ignore

    def load(self, load_path: str) -> None:

        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict(
            {**model_actor.state_dict(), **load_data.get('actor', {})}
        )

        if not self.opts.eval_only:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict(
                {**model_critic.state_dict(), **load_data.get('critic', {})}
            )
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))

    def save(self, epoch: int) -> None:
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch)),
        )

    def eval(self) -> None:
        torch.set_grad_enabled(False)
        self.actor.eval()
        if not self.opts.eval_only:
            self.critic.eval()

    def train(self) -> None:
        torch.set_grad_enabled(True)
        self.actor.train()
        if not self.opts.eval_only:
            self.critic.train()

    def rollout(
        self,
        problem: PDP,
        val_m: int,
        batch: Dict[str, torch.Tensor],
        show_bar: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = move_to(batch, self.opts.device)  # batch_size, graph_size+1, 2
        batch_size, graph_size_plus1, dim = batch['coordinates'].size()
        batch['coordinates'] = batch['coordinates'].unsqueeze(1).repeat(1, val_m, 1, 1)
        augments = ['Rotate', 'Flip_x-y', 'Flip_x_cor', 'Flip_y_cor']
        if val_m > 1:
            for i in range(val_m):
                random.shuffle(augments)
                id_ = torch.rand(4)
                for aug in augments:
                    if aug == 'Rotate':
                        batch['coordinates'][:, i] = rotate_tensor(
                            batch['coordinates'][:, i], int(id_[0] * 4 + 1) * 90
                        )
                    elif aug == 'Flip_x-y':
                        if int(id_[1] * 2 + 1) == 1:
                            data = batch['coordinates'][:, i].clone()
                            batch['coordinates'][:, i, :, 0] = data[:, :, 1]
                            batch['coordinates'][:, i, :, 1] = data[:, :, 0]
                    elif aug == 'Flip_x_cor':
                        if int(id_[2] * 2 + 1) == 1:
                            batch['coordinates'][:, i, :, 0] = (
                                1 - batch['coordinates'][:, i, :, 0]
                            )
                    elif aug == 'Flip_y_cor':
                        if int(id_[3] * 2 + 1) == 1:
                            batch['coordinates'][:, i, :, 1] = (
                                1 - batch['coordinates'][:, i, :, 1]
                            )

        batch['coordinates'] = batch['coordinates'].view(-1, graph_size_plus1, dim)
        solutions = move_to(
            problem.get_initial_solutions(batch), self.opts.device
        ).long()

        obj = problem.get_costs(batch, solutions)

        obj_history = [torch.cat((obj[:, None], obj[:, None]), -1)]
        reward = []

        batch_feature = PDP.input_feature_encoding(batch)

        exchange = None
        action_record = [
            torch.zeros((batch_feature.size(0), problem.size // 2))
            for i in range(problem.size // 2)
        ]

        for t in tqdm(
            range(self.opts.T_max),
            disable=self.opts.no_progress_bar or not show_bar,
            desc='rollout',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        ):

            # pass through model
            exchange = self.actor(
                problem, batch_feature, solutions, exchange, action_record
            )[0]

            # new solution
            solutions, rewards, obj, action_record = problem.step(
                batch, solutions, exchange, obj, action_record
            )

            # record informations
            reward.append(rewards)
            obj_history.append(obj)

        out = (
            obj[:, -1].reshape(batch_size, val_m).min(1)[0],  # batch_size, 1
            torch.stack(obj_history, 1)[:, :, 0]
            .view(batch_size, val_m, -1)
            .min(1)[0],  # batch_size, T
            torch.stack(obj_history, 1)[:, :, -1]
            .view(batch_size, val_m, -1)
            .min(1)[0],  # batch_size, T
            torch.stack(reward, 1)
            .view(batch_size, val_m, -1)
            .max(1)[0],  # batch_size, T
        )

        return out

    def start_inference(
        self, problem: PDP, val_dataset: str, tb_logger: TbLogger
    ) -> None:
        if self.opts.distributed:
            mp.spawn(
                validate,
                nprocs=self.opts.world_size,
                args=(problem, self, val_dataset, tb_logger, True),
            )
        else:
            validate(0, problem, self, val_dataset, tb_logger, distributed=False)

    def start_training(
        self, problem: PDP, val_dataset: str, tb_logger: TbLogger
    ) -> None:
        if self.opts.distributed:
            mp.spawn(
                train,
                nprocs=self.opts.world_size,
                args=(problem, self, val_dataset, tb_logger),
            )
        else:
            train(0, problem, self, val_dataset, tb_logger)


def train(
    rank: int, problem: PDP, agent: Agent, val_dataset: str, tb_logger: TbLogger
) -> None:

    opts = agent.opts

    warnings.filterwarnings("ignore")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    if opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(
            backend='nccl', world_size=opts.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        agent.critic.to(device)
        for state in agent.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        if torch.cuda.device_count() > 1:
            agent.actor = torch.nn.parallel.DistributedDataParallel(
                agent.actor, device_ids=[rank]
            )  # type: ignore
            if not opts.eval_only:
                agent.critic = torch.nn.parallel.DistributedDataParallel(
                    agent.critic, device_ids=[rank]
                )  # type: ignore
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(
                os.path.join(
                    opts.log_dir,
                    "{}_{}".format(opts.problem, opts.graph_size),
                    opts.run_name,
                )
            )
    else:
        for state in agent.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    if opts.distributed:
        dist.barrier()

    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):

        agent.lr_scheduler.step(epoch)

        # Training mode
        if rank == 0:
            print('\n\n')
            print("|", format(f" Training epoch {epoch} ", "*^60"), "|")
            print(
                "Training with actor lr={:.3e} critic lr={:.3e} for run {}".format(
                    agent.optimizer.param_groups[0]['lr'],
                    agent.optimizer.param_groups[1]['lr'],
                    opts.run_name,
                ),
                flush=True,
            )
        # prepare training data
        training_dataset = PDP.make_dataset(
            size=opts.graph_size, num_samples=opts.epoch_size
        )
        if opts.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                training_dataset, shuffle=False
            )  # type: ignore
            training_dataloader = DataLoader(
                training_dataset,
                batch_size=opts.batch_size // opts.world_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=train_sampler,
            )
        else:
            training_dataloader = DataLoader(
                training_dataset,
                batch_size=opts.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

        # start training
        step = epoch * (opts.epoch_size // opts.batch_size)
        pbar = tqdm(
            total=(opts.K_epochs)
            * (opts.epoch_size // opts.batch_size)
            * (opts.T_train // opts.n_step),
            disable=opts.no_progress_bar or rank != 0,
            desc='training',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        )
        for batch_id, batch in enumerate(training_dataloader):
            train_batch(
                rank,
                problem,
                agent,
                epoch,
                step,
                batch,
                tb_logger,
                opts,
                pbar,
            )
            step += 1
        pbar.close()

        # save new model after one epoch
        if rank == 0 and not opts.distributed:
            if not opts.no_saving and (
                (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0)
                or epoch == opts.epoch_end - 1
            ):
                agent.save(epoch)
        elif opts.distributed and rank == 1:
            if not opts.no_saving and (
                (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0)
                or epoch == opts.epoch_end - 1
            ):
                agent.save(epoch)

        # validate the new model
        if rank == 0 and not opts.distributed:
            validate(rank, problem, agent, val_dataset, tb_logger, _id=epoch)
        if rank == 0 and opts.distributed:
            validate(rank, problem, agent, val_dataset, tb_logger, _id=epoch)

        # syn
        if opts.distributed:
            dist.barrier()


def train_batch(
    rank: int,
    problem: PDP,
    agent: Agent,
    epoch: int,
    step: int,
    batch: Dict[str, torch.Tensor],
    tb_logger: TbLogger,
    opts: Option,
    pbar: tqdm,
) -> None:

    # setup
    agent.train()
    memory = Memory()

    # prepare the input
    batch = (
        move_to_cuda(batch, rank) if opts.distributed else move_to(batch, opts.device)
    )  # batch_size, graph_size+1, 2
    batch_feature = (
        PDP.input_feature_encoding(batch).cuda()
        if opts.distributed
        else move_to(PDP.input_feature_encoding(batch), opts.device)
    )
    batch_size = batch_feature.size(0)
    exchange = (
        move_to_cuda(torch.tensor([-1, -1, -1]).repeat(batch_size, 1), rank)
        if opts.distributed
        else move_to(torch.tensor([-1, -1, -1]).repeat(batch_size, 1), opts.device)
    )

    action_record = [
        torch.zeros((batch_feature.size(0), problem.size // 2))
        for i in range(problem.size)
    ]
    # print(f"rank {rank}, data from {batch['id'][0]},{batch['id'][1]} , to {batch['id'][-2]},{batch['id'][-1]}")

    # initial solution
    solution = (
        move_to_cuda(problem.get_initial_solutions(batch), rank)
        if opts.distributed
        else move_to(problem.get_initial_solutions(batch), opts.device)
    )
    obj = problem.get_costs(batch, solution)

    # warm_up
    if opts.warm_up:
        agent.eval()

        for w in range(int(epoch // opts.warm_up)):

            # get model output
            exchange = agent.actor(
                problem, batch_feature, solution, exchange, action_record
            )[0]

            # state transient
            solution, rewards, obj, action_record = problem.step(
                batch, solution, exchange, obj, action_record
            )

        obj = problem.get_costs(batch, solution)

        agent.train()

    # params for training
    gamma = opts.gamma
    n_step = opts.n_step
    T = opts.T_train
    K_epochs = opts.K_epochs
    eps_clip = opts.eps_clip
    t = 0
    initial_cost = obj

    # sample trajectory
    while t < T:

        t_s = t
        memory.actions.append(exchange)

        # data array
        total_cost = torch.tensor(0)

        # for first step
        entropy_list = []
        bl_val_detached_list = []
        bl_val_list = []

        while t - t_s < n_step and not (t == T):

            memory.states.append(solution)
            memory.action_record.append(action_record.copy())

            # get model output

            exchange, log_lh, _to_critic, entro_p = agent.actor(
                problem,
                batch_feature,
                solution,
                exchange,
                action_record,
                require_entropy=True,  # take same action
                to_critic=True,
            )  # type: ignore

            memory.actions.append(exchange)
            memory.logprobs.append(log_lh)
            memory.obj.append(obj.view(obj.size(0), -1)[:, -1].unsqueeze(-1))

            entropy_list.append(entro_p.detach().cpu())  # type: ignore

            baseline_val_detached, baseline_val = agent.critic(
                _to_critic, obj.view(obj.size(0), -1)[:, -1].unsqueeze(-1)
            )

            bl_val_detached_list.append(baseline_val_detached)
            bl_val_list.append(baseline_val)

            # state transient
            solution, rewards, obj, action_record = problem.step(
                batch, solution, exchange, obj, action_record
            )
            memory.rewards.append(rewards)
            # memory.mask_true = memory.mask_true + info['swaped']

            # store info
            total_cost = total_cost + obj[:, -1]

            # next
            t = t + 1

        # store info
        t_time = t - t_s
        total_cost = total_cost / t_time

        # begin update        =======================

        # convert list to tensor
        all_actions = torch.stack(memory.actions)
        old_states = torch.stack(memory.states).detach().view(t_time, batch_size, -1)
        old_actions = all_actions[1:].view(t_time, -1, 3)
        old_logprobs = torch.stack(memory.logprobs).detach().view(-1)
        old_exchange = all_actions[:-1].view(t_time, -1, 3)
        old_action_records = memory.action_record

        old_obj = torch.stack(memory.obj)

        # Optimize ppo policy for K mini-epochs:
        old_value = None

        for _k in range(K_epochs):

            if _k == 0:
                logprobs_list = memory.logprobs

            else:
                # Evaluating old actions and values :
                logprobs_list = []
                entropy_list = []
                bl_val_detached_list = []
                bl_val_list = []

                for tt in range(t_time):
                    # get new action_prob
                    _, log_p, _to_critic, entro_p = agent.actor(
                        problem,
                        batch_feature,
                        old_states[tt],
                        old_exchange[tt],
                        old_action_records[tt],
                        fixed_action=old_actions[tt],
                        require_entropy=True,  # take same action
                        to_critic=True,
                    )  # type: ignore

                    logprobs_list.append(log_p)
                    entropy_list.append(entro_p.detach().cpu())  # type: ignore

                    baseline_val_detached, baseline_val = agent.critic(
                        _to_critic, old_obj[tt]
                    )

                    bl_val_detached_list.append(baseline_val_detached)
                    bl_val_list.append(baseline_val)

            logprobs = torch.stack(logprobs_list).view(-1)
            entropy = torch.stack(entropy_list).view(-1)
            bl_val_detached = torch.stack(bl_val_detached_list).view(-1)
            bl_val = torch.stack(bl_val_list).view(-1)

            # get traget value for critic
            Reward_list = []
            reward_reversed = memory.rewards[::-1]

            # estimate return
            R = agent.critic(
                agent.actor(
                    problem,
                    batch_feature,
                    solution,
                    exchange,
                    action_record,
                    only_critic=True,
                ),
                obj.view(obj.size(0), -1)[:, -1].unsqueeze(-1),
            )[0]
            for r in range(len(reward_reversed)):
                R = R * gamma + reward_reversed[r]
                Reward_list.append(R)

            # clip the target:
            Reward = torch.stack(Reward_list[::-1], 0)  # n_step, bs
            Reward = Reward.view(-1)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = Reward - bl_val_detached

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            reinforce_loss = -torch.min(surr1, surr2).mean()

            # define baseline loss
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2).mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(
                    bl_val - old_value, -eps_clip, eps_clip
                )
                v_max = torch.max(
                    ((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2)
                )
                baseline_loss = v_max.mean()

            # check K-L divergence
            approx_kl_divergence = (
                (0.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            )
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0

            # calculate loss
            loss = baseline_loss + reinforce_loss  # - 1e-5 * entropy.mean()

            # update gradient step
            agent.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm and get (clipped) gradient norms for logging
            current_step = int(
                step * T / n_step * K_epochs + t // n_step * K_epochs + _k
            )
            grad_norms = clip_grad_norms(
                agent.optimizer.param_groups, opts.max_grad_norm
            )

            # perform gradient descent
            agent.optimizer.step()

            # Logging to tensorboard
            if (not opts.no_tb) and rank == 0:
                if current_step % int(opts.log_step) == 0:
                    log_to_tb_train(
                        tb_logger,
                        agent,
                        Reward,
                        ratios,
                        bl_val_detached,
                        total_cost,
                        grad_norms,
                        memory.rewards,
                        entropy,
                        approx_kl_divergence,
                        reinforce_loss,
                        baseline_loss,
                        logprobs,
                        initial_cost,
                        opts.show_figs,
                        current_step,
                    )

            if rank == 0:
                pbar.update(1)

        # end update
        memory.clear_memory()
