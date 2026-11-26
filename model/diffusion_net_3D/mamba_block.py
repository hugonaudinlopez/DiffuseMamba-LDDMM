import torch
import torch.nn as nn

from mamba_ssm import Mamba2

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLNModulation(nn.Module):
    def __init__(self, cond_dim, dim, num_params):
        super().__init__()
        self.act = nn.SiLU()
        self.num_params = num_params

        self.linear = nn.Linear(cond_dim, num_params * dim)
        
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c):
        return self.linear(self.act(c)).chunk(self.num_params, dim=1)
    



class DiTMambaBlock3D(nn.Module):
    def __init__(self, dim, d_state, d_conv, expand, head_dim, cond_dim, grid_shape):
        super().__init__()
        self.dim = dim
        self.grid_shape = grid_shape
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # a mamba per dimension
        self.mamba_d = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=head_dim)
        self.mamba_h = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=head_dim)
        self.mamba_w = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=head_dim)

        # independent modules
        self.mamba_d_bwd = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=head_dim)
        self.mamba_h_bwd = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=head_dim)
        self.mamba_w_bwd = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=head_dim)

        self.proj = nn.Linear(dim * 3, dim)
        
        # shifft scale and gate
        self.adaLN = AdaLNModulation(cond_dim, dim, num_params=3)


    def forward(self, x, c,  grid_shape=None):
        B, L, dim = x.shape
        D, H, W = grid_shape if grid_shape is not None else self.grid_shape
        shift, scale, gate = self.adaLN(c)
        
        # modulation
        x_norm = modulate(self.norm(x), shift, scale)
        
        # bidirectional
        x_3d = x_norm.view(B, D, H, W, dim)
        
        # d axis
        out_d_fwd = self.mamba_d(x_norm)

        # Sep
        out_d_bwd = torch.flip(self.mamba_d_bwd(torch.flip(x_norm, dims=[1])), dims=[1])
        out_d = (out_d_fwd + out_d_bwd) / 2.0
        
        # h axis
        x_h = x_3d.permute(0, 2, 1, 3, 4).contiguous().view(B, L, dim)
        out_h_fwd = self.mamba_h(x_h)
    
        # Sep
        out_h_bwd = torch.flip(self.mamba_h_bwd(torch.flip(x_h, dims=[1])), dims=[1])

        out_h = (out_h_fwd + out_h_bwd) / 2.0
        out_h = out_h.view(B, H, D, W, dim).permute(0, 2, 1, 3, 4).contiguous().view(B, L, dim)
        
        # w axis
        x_w = x_3d.permute(0, 3, 1, 2, 4).contiguous().view(B, L, dim)
        out_w_fwd = self.mamba_w(x_w)
        
        # Sep
        out_w_bwd = torch.flip(self.mamba_w_bwd(torch.flip(x_w, dims=[1])), dims=[1])
        out_w = (out_w_fwd + out_w_bwd) / 2.0
        out_w = out_w.view(B, W, D, H, dim).permute(0, 2, 3, 1, 4).contiguous().view(B, L, dim)
        
        out = torch.cat([out_d, out_h, out_w], dim=-1) 
        out = self.proj(out)     
        
        return x + gate.unsqueeze(1) * out
 
