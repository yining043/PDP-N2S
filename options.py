import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(description="Neural Neighborhood Search")

    # overall settings
    parser.add_argument('--problem', default='pdtsp', choices = ['pdtsp','pdtspl'], help="The targeted problem to solve, default 'pdp'")
    parser.add_argument('--graph_size', type=int, default=20, help="T number of customers in the targeted problem (graph size)")
    parser.add_argument('--init_val_met', choices = ['greedy', 'random'], default = 'random', help='method to generate initial solutions for inference')
    parser.add_argument('--no_cuda', action='store_true', help='disable GPUs')
    parser.add_argument('--no_tb', action='store_true', help='disable Tensorboard logging')
    parser.add_argument('--no_saving', action='store_true', help='disable saving checkpoints')
    parser.add_argument('--use_assert', action='store_true', help='enable assertion')
    parser.add_argument('--no_DDP', action='store_true', help='disable distributed parallel')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use')
    
    
    # N2S parameters
    parser.add_argument('--v_range', type=float, default=6., help='to control the entropy')
    parser.add_argument('--actor_head_num', type=int, default=4, help='head number of N2S actor')
    parser.add_argument('--critic_head_num', type=int, default=4, help='head number of N2S critic')
    parser.add_argument('--embedding_dim', type=int, default=128, help='dimension of input embeddings (NEF & PFE)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='number of stacked layers in the encoder')
    parser.add_argument('--normalization', default='layer', help="normalization type, 'layer' (default) or 'batch'")

    # Training parameters
    parser.add_argument('--RL_agent', default='ppo', choices = ['ppo'], help='RL Training algorithm')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor for future rewards')
    parser.add_argument('--K_epochs', type=int, default=3, help='mini PPO epoch')
    parser.add_argument('--eps_clip', type=float, default=0.1, help='PPO clip ratio')
    parser.add_argument('--T_train', type=int, default=250, help='number of itrations for training')
    parser.add_argument('--n_step', type=int, default=5, help='n_step for return estimation')
    parser.add_argument('--warm_up', type=float, default=2, help='hyperparameter of CL scalar $\rho^{CL}$')
    parser.add_argument('--batch_size', type=int, default=600,help='number of instances per batch during training')
    parser.add_argument('--epoch_end', type=int, default=200, help='maximum training epoch')
    parser.add_argument('--epoch_size', type=int, default=12000, help='number of instances per epoch during training')
    parser.add_argument('--lr_model', type=float, default=8e-5, help="learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=2e-5, help="learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.985, help='learning rate decay per epoch')
    parser.add_argument('--max_grad_norm', type=float, default=0.05, help='maximum L2 norm for gradient clipping')
    
    # Inference and validation parameters
    parser.add_argument('--T_max', type=int, default=1500, help='number of steps for inference')
    parser.add_argument('--eval_only', action='store_true', help='switch to inference mode')
    parser.add_argument('--val_size', type=int, default=1000, help='number of instances for validation/inference')
    parser.add_argument('--val_batch_size', type=int, default=1000, help='Number of instances per batch for validation/inference')
    parser.add_argument('--val_dataset', type=str, default = './datasets/pdp_20.pkl', help='dataset file path')
    parser.add_argument('--val_m', type=int, default=1, help='number of data augments in Algorithm 2')
    

    # resume and load models
    parser.add_argument('--load_path', default = None, help='path to load model parameters and optimizer state from')
    parser.add_argument('--resume', default = None, help='resume from previous checkpoint file')
    parser.add_argument('--epoch_start', type=int, default=0, help='start at epoch # (relevant for learning rate decay)')

    # logs/output settings
    parser.add_argument('--no_progress_bar', action='store_true', help='disable progress bar')
    parser.add_argument('--log_dir', default='logs', help='directory to write TensorBoard information to')
    parser.add_argument('--log_step', type=int, default=50, help='log info every log_step gradient steps')
    parser.add_argument('--output_dir', default='outputs', help='directory to write output models to')
    parser.add_argument('--run_name', default='run_name', help='name to identify the run')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    

    opts = parser.parse_args(args)
    
    ### figure out whether to use distributed training
    opts.world_size = torch.cuda.device_count()
    opts.distributed = (opts.world_size > 1) and (not opts.no_DDP)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4869'
    assert opts.val_m <= opts.graph_size // 2
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    ) if not opts.no_saving else None
    
    return opts
