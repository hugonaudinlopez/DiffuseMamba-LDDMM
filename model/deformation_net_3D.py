import torch
import torch.nn as nn
from .field_transformer import FieldTransformer

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Networks
##############################################################################


class Cblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Cblock, self).__init__()
        self.block = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)

    def forward(self, x):
        return self.block(x)

class CRblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(CRblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.LeakyReLU(0.2, True))

    def forward(self, x):
        return self.block(x)

class inblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super(inblock, self).__init__()
        self.block = CRblock(in_ch, out_ch, stride=stride)

    def forward(self, x):
        return self.block(x)

class outblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, output_padding=1):
        super(outblock, self).__init__()
        self.block = nn.Conv3d(in_ch, out_ch, 3, padding=1, stride=stride)

    def forward(self, x):
        x = self.block(x)
        return x

class downblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downblock, self).__init__()
        self.block = CRblock(in_ch, out_ch, stride=2)

    def forward(self, x):
        return self.block(x)

class upblock(nn.Module):
    def __init__(self, in_ch, CR_ch, out_ch):
        super(upblock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_ch, in_ch, 3, padding=1, stride=2, output_padding=1)
        self.block = CRblock(CR_ch, out_ch)

    def forward(self, x1, x2):
        upconved = self.upconv(x1)
        x = torch.cat([x2, upconved], dim=1)
        return self.block(x)


class registUnetBlock(nn.Module):
    def __init__(self, input_nc, encoder_nc, decoder_nc, steps=7, alpha=0.1, s=3):        
        super(registUnetBlock, self).__init__()
        self.alpha = alpha
        self.s = s
        self.inconv = inblock(input_nc, encoder_nc[0], stride=1)
        self.downconv1 = downblock(encoder_nc[0], encoder_nc[1])
        self.downconv2 = downblock(encoder_nc[1], encoder_nc[2])
        self.downconv3 = downblock(encoder_nc[2], encoder_nc[3])
        self.downconv4 = downblock(encoder_nc[3], encoder_nc[4])
        self.upconv1 = upblock(encoder_nc[4], encoder_nc[4]+encoder_nc[3], decoder_nc[0])
        self.upconv2 = upblock(decoder_nc[0], decoder_nc[0]+encoder_nc[2], decoder_nc[1])
        self.upconv3 = upblock(decoder_nc[1], decoder_nc[1]+encoder_nc[1], decoder_nc[2])
        self.keepblock = CRblock(decoder_nc[2], decoder_nc[3])
        self.upconv4 = upblock(decoder_nc[3], decoder_nc[3]+encoder_nc[0], decoder_nc[4])
        self.outconv = outblock(decoder_nc[4], decoder_nc[5], stride=1)
        self.stn = FieldTransformer(steps  = steps)

    def apply_K(self, m):
        alpha, s = self.alpha, self.s
        B, C, D, H, W = m.shape
        grid_x = torch.fft.fftfreq(D, device=m.device).view(-1, 1, 1)
        grid_y = torch.fft.fftfreq(H, device=m.device).view(1, -1, 1)
        grid_z = torch.fft.fftfreq(W, device=m.device).view(1, 1, -1)
        
        momentum_freq = 2 * (3 - torch.cos(2 * torch.pi * grid_x) 
                               - torch.cos(2 * torch.pi * grid_y) 
                               - torch.cos(2 * torch.pi * grid_z))
        
        L_op = (1.0 + alpha * momentum_freq) ** s
        L_dagger_L_inv = 1.0 / (L_op ** 2)
        
        v = torch.zeros_like(m)
        for i in range(C):
            m_fft = torch.fft.fftn(m[:, i, ...])
            v[:, i, ...] = torch.fft.ifftn(m_fft * L_dagger_L_inv).real
        return v

    def forward(self, input):
        x1 = self.inconv(input)
        x2 = self.downconv1(x1)
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)
        x5 = self.downconv4(x4)
        x = self.upconv1(x5, x4)
        x = self.upconv2(x, x3)
        x = self.upconv3(x, x2)
        x = self.keepblock(x)
        x = self.upconv4(x, x1)
        momentum = self.outconv(x)

        # velocity after applying operator
        velocity = self.apply_K(momentum)
        mov = (input[:, :1] + 1) / 2.0

        # integrate and apply field
        out, flow = self.stn(velocity=velocity, src=mov)
        
        return out, flow, velocity
