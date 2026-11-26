import math
import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import numpy as np
from . import loss

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, field_fn, channels=3, loss_type='l1', conditional=True,
                 schedule_opt=None, loss_lambda=1, loss_opt=None, lddmm_opt=None):
        super().__init__()
        self.loss_opt = loss_opt
        self.lddmm_opt = lddmm_opt
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.field_fn = field_fn
        self.conditional = conditional
        self.loss_type = loss_type
        self.lambda_L = loss_lambda
        if schedule_opt is not None:
            pass
    

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(device)
        
        k = self.loss_opt['ncc_kernel']
        self.loss_ncc = loss.CrossCorrelation3D(1, kernel=(k, k, k)).to(device)
        self.loss_reg = loss.SmoothnessLoss().to(device)
        self.loss_reg_v = loss.LDDMMRegularizerLoss(alpha=self.lddmm_opt['alpha'], s=self.lddmm_opt['s']).to(device)
        self.loss_jdet = loss.JacobianDeterminantLoss(eps=self.loss_opt['jac_eps']).to(device)


    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


    # @torch.no_grad()
    def p_sample_loop(self, x_in, nsample, continuous=False):
        device = self.betas.device
        x = x_in
        x_m = x_in[:, :1]
        x_f = x_in[:, 1:]
        b, c, d, h, w = x_m.shape

        with torch.no_grad():
            t = torch.full((b,), 0, device=device, dtype=torch.long)
            score = self.denoise_fn(torch.cat([x, x_f], dim=1), t)

            gamma = np.linspace(0, 1, nsample)
            b, c, d, h, w = x_f.shape
            flow_stack = torch.zeros([1, 3, d, h, w], device=device)
            code_stack = score
            defm_stack = x_m

            for i in (gamma):
                #print('-------- Deform with eta=%.3f' % i)
                code = score * (i)
                deform, flow, velocity = self.field_fn(torch.cat([x_m, code], dim=1))
                code_stack = torch.cat([code_stack, code], dim=0)
                defm_stack = torch.cat([defm_stack, deform], dim=0)
                flow_stack = torch.cat([flow_stack, flow], dim=0)

        if continuous:
            return deform, flow, defm_stack[1:], flow_stack[1:]
        else:
            return deform, flow, defm_stack[-1], flow_stack[-1]

    @torch.no_grad()
    def registration(self, x_in, nsample=7, continuous=False):
        return self.p_sample_loop(x_in, nsample, continuous)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # fix gamma
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['F']
        [b, c, d, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoise_fn(torch.cat([x_in['M'], x_in['F'], x_noisy], dim=1), t)
        l_pix = self.loss_func(noise, x_recon)

        output, flow, velocity = self.field_fn(torch.cat([x_in['M'], x_recon], dim=1))

        l_sim = self.loss_ncc(output, x_in['F']) * self.lambda_L
        l_smt = self.loss_reg(flow) * self.loss_opt['w_smt']
        l_lddmm = self.loss_reg_v(velocity) * self.loss_opt['w_lddmm']
        l_jac = self.loss_jdet(flow) * self.loss_opt['w_jac']

        loss = l_pix + l_sim + l_smt + l_lddmm+ l_jac
        return [x_recon, output, flow], [l_jac, l_lddmm, l_pix, l_sim, l_smt, loss]

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
