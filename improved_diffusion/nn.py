"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from .types_ import *


class GaussianConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "B",
        num_vars: int = 4,
        **kwargs
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = th.tensor([max_capacity], dtype=th.float32)
        self.C_stop_iter = Capacity_max_iter
        self.in_channels = in_channels
        self.num_vars = num_vars

        if hidden_dims is None:
            if self.num_vars == 4:
                hidden_dims = [16, 32, 32, 64, 64, 128]
            elif self.num_vars == 2:
                hidden_dims = [16, 32, 64, 128]
            else:
                raise ValueError(f"Unsupported num_vars={self.num_vars}")

        # Build Encoder conv trunk
        modules = []
        cur_in = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(cur_in, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            cur_in = h_dim

        self.encoder = nn.Sequential(*modules)

        # Pool to 1x1 so feature dim is stable across input sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = hidden_dims[-1]

        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_var = nn.Linear(feat_dim, latent_dim)

    def gaussian_parameters(self, h, dim=-1):
        """
        Converts generic real-valued representations into mean and variance
        parameters of a Gaussian distribution
        """
        m, h = th.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        return m, v

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns (mu, var).
        :param input: [N x C x H x W]
        :return: [mu, var] each [N x latent_dim]
        """
        h = self.encoder(input)           # [N, C, H', W']
        h = self.pool(h).flatten(1)       # [N, C]
        mu = self.fc_mu(h)
        var = F.softplus(self.fc_var(h)) + 1e-8
        return [mu, var]


class GaussianConvEncoderClf(nn.Module):
    """
    Encoder + a simple classifier head.
    - encode(x) returns (mu, var)
    - forward(x) returns classifier logit
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "B",
        num_vars: int = 4,
        **kwargs
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = th.tensor([max_capacity], dtype=th.float32)
        self.C_stop_iter = Capacity_max_iter
        self.in_channels = in_channels
        self.num_vars = num_vars

        if hidden_dims is None:
            if self.num_vars == 4:
                hidden_dims = [16, 32, 32, 64, 64, 128]
            elif self.num_vars == 2:
                hidden_dims = [16, 32, 64, 128]
            else:
                raise ValueError(f"Unsupported num_vars={self.num_vars}")

        # Build Encoder conv trunk
        modules = []
        cur_in = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(cur_in, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            cur_in = h_dim

        self.encoder = nn.Sequential(*modules)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = hidden_dims[-1]

        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_var = nn.Linear(feat_dim, latent_dim)

        # classifier head (logit)
        self.fc = nn.Linear(feat_dim, 1)

    def gaussian_parameters(self, h, dim=-1):
        m, h = th.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        return m, v

    def encode(self, input: Tensor) -> List[Tensor]:
        h = self.encoder(input)
        h = self.pool(h).flatten(1)
        mu = self.fc_mu(h)
        var = F.softplus(self.fc_var(h)) + 1e-8
        return [mu, var]

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        h = self.pool(h).flatten(1)
        out = self.fc(h)
        return out


class MLP(nn.Module):
    """a simple 2-layer MLP used per variable"""

    def __init__(self, latent_dim, num_var):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_var = num_var

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim // self.num_var, self.latent_dim),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim, self.latent_dim // self.num_var),
        )

    def forward(self, x):
        return self.net(x)


class CausalModeling(nn.Module):
    def __init__(self, latent_dim: int, num_var=None, learn: bool = False, **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_var = num_var

        if learn:
            self.A = nn.Parameter(th.zeros(self.num_var, self.num_var))
        else:
            # register as buffer so it moves with .to(device) and is saved in state_dict
            self.register_buffer("A", th.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=th.float32))

        self.nonlinearities = nn.ModuleDict()
        for i in range(self.num_var):
            self.nonlinearities[str(i)] = MLP(latent_dim=latent_dim, num_var=num_var)

    def causal_masking(self, u, A):
        u = u.reshape(-1, self.num_var, self.latent_dim // self.num_var)
        z_pre = th.matmul(A.t().to(u.device), u)
        return z_pre

    def nonlinearity_add_back_noise(self, u, z_pre):
        u = u.reshape(-1, self.num_var, self.latent_dim // self.num_var)
        z_post = th.zeros_like(u)

        for i in range(self.num_var):
            z_post[:, i, :] = self.nonlinearities[str(i)](z_pre[:, i, :]) + u[:, i, :]

        return z_post.reshape(-1, self.num_var * (self.latent_dim // self.num_var))


class MultivariateCausalFlow(nn.Module):
    """
    Simple autoregressive affine flow with causal masking.

    dim: number of variables
    k:   per-variable latent dimension (so total_dims = dim*k)

    Inputs:
      e: [B, dim*k] or [B, dim, k]
      C: adjacency mask, either [dim, dim] or [B, dim, dim]
         Convention: C[parent, child] = 1 indicates parent -> child.
    """

    def __init__(self, dim: int, k: int, nh: int = 100):
        super().__init__()
        self.dim = dim
        self.k = k
        self.total_dims = dim * k

        self.s_cond = nn.Sequential(
            nn.Linear(self.total_dims, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, self.k),
        )
        self.t_cond = nn.Sequential(
            nn.Linear(self.total_dims, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, self.k),
        )

    def _ensure_C(self, C: th.Tensor, batch_size: int, device: th.device) -> th.Tensor:
        # Accept C as [dim, dim] or [B, dim, dim]
        if C.dim() == 2:
            Cb = C.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        elif C.dim() == 3:
            Cb = C.to(device)
            if Cb.size(0) != batch_size:
                # If user passes a single C with batch dim 1
                if Cb.size(0) == 1:
                    Cb = Cb.expand(batch_size, -1, -1)
                else:
                    raise ValueError(f"C batch size {Cb.size(0)} != e batch size {batch_size}")
        else:
            raise ValueError(f"Unexpected C shape: {tuple(C.shape)}")
        return Cb

    def flow(self, e: th.Tensor, C: th.Tensor):
        # e -> z
        if e.dim() == 2:
            e = e.reshape(-1, self.dim, self.k)
        elif e.dim() == 3:
            assert e.size(1) == self.dim and e.size(2) == self.k
        else:
            raise ValueError(f"Unexpected e shape: {tuple(e.shape)}")

        B = e.size(0)
        device = e.device
        Cb = self._ensure_C(C, B, device)

        z = th.zeros_like(e)
        log_det = th.zeros(B, device=device)

        for i in range(self.dim):
            # parents mask: which variables are parents of i?
            # C[parent, child]=1 => parents of i are C[:, i]
            parents = Cb[:, :, i]  # [B, dim]
            # mask applied to current partial z (flattened)
            mask = parents.repeat_interleave(self.k, dim=1)  # [B, dim*k]

            context = z.reshape(B, self.total_dims) * mask
            s = 0.1 * th.tanh(self.s_cond(context))          # [B, k] stabilized
            t = self.t_cond(context)                         # [B, k]

            z[:, i, :] = th.exp(s) * e[:, i, :] + t
            log_det += s.sum(dim=1)

        return [z.reshape(B, self.total_dims), log_det]

    def reverse(self, z: th.Tensor, C: th.Tensor):
        # z -> e, and prior log prob under standard normal
        if z.dim() == 2:
            z = z.reshape(-1, self.dim, self.k)
        elif z.dim() == 3:
            assert z.size(1) == self.dim and z.size(2) == self.k
        else:
            raise ValueError(f"Unexpected z shape: {tuple(z.shape)}")

        B = z.size(0)
        device = z.device
        Cb = self._ensure_C(C, B, device)

        e = th.zeros_like(z)
        log_det = th.zeros(B, device=device)

        for i in range(self.dim):
            parents = Cb[:, :, i]  # [B, dim]
            mask = parents.repeat_interleave(self.k, dim=1)  # [B, dim*k]

            context = z.reshape(B, self.total_dims) * mask
            s = 0.1 * th.tanh(self.s_cond(context))
            t = self.t_cond(context)

            e[:, i, :] = th.exp(-s) * (z[:, i, :] - t)
            log_det -= s.sum(dim=1)

        # Standard normal log-prob (independent)
        p_log_prob = -0.5 * (e.reshape(B, self.total_dims) ** 2 + math.log(2 * math.pi)).sum(dim=1)
        return [log_det, p_log_prob]

# --- rest of your utilities unchanged ---

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def kl_normal(qm, qv, pm, pv):
    """
    KL(q||p) for diagonal Gaussians where qv/pv are variances (not log-variances).
    """
    element_wise = 0.5 * (th.log(pv) - th.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    return element_wise.sum(-1)


def reparameterize(m, v):
    """
    Reparameterization Trick where v is variance.
    """
    eps = th.randn_like(m)
    return m + th.sqrt(v) * eps


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32)).to(
        device=timesteps.device
    ) / half
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads