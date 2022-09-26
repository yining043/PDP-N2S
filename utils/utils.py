import torch
import math
import numpy as np
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP


def get_rotate_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )


def rotate_tensor(x, d):
    rot_mat = get_rotate_mat(d / 360 * 2 * np.pi).to(x.device)
    return torch.matmul(x - 0.5, rot_mat) + 0.5


def torch_load_cpu(load_path):
    return torch.load(
        load_path, map_location=lambda storage, loc: storage
    )  # Load on CPU


def get_inner_model(model):
    return (
        model.module
        if isinstance(model, DataParallel) or isinstance(model, DDP)
        else model
    )


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def move_to_cuda(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.cuda(device)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm
            if max_norm > 0
            else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = (
        [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    )
    return grad_norms, grad_norms_clipped
