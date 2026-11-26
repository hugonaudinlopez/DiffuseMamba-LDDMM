import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossCorrelation3D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9, 9), gamma=1):
        super(CrossCorrelation3D, self).__init__()
        self.in_ch = in_ch
        self.kernel = kernel
        self.gamma = gamma

    def forward(self, input, target):
        filt = torch.ones([1, self.in_ch, self.kernel[0], self.kernel[1], self.kernel[2]], device=input.device)

        I = input
        J = target

        I2 = I * I
        J2 = J * J
        IJ = I * J

        pad = self.kernel[0] // 2

        I_sum = F.conv3d(I, filt, padding=pad)
        J_sum = F.conv3d(J, filt, padding=pad)
        I2_sum = F.conv3d(I2, filt, padding=pad)
        J2_sum = F.conv3d(J2, filt, padding=pad)
        IJ_sum = F.conv3d(IJ, filt, padding=pad)

        win_size = self.kernel[0] * self.kernel[1] * self.kernel[2]
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-6)
        return -1.0 * torch.mean(cc)

class JacobianDeterminantLoss(nn.Module):
    def __init__(self, eps=0.5):
        super(JacobianDeterminantLoss, self).__init__()
        self.eps = eps

    def forward(self, u):
        dx = u[:, :, 1:, :-1, :-1] - u[:, :, :-1, :-1, :-1]
        dy = u[:, :, :-1, 1:, :-1] - u[:, :, :-1, :-1, :-1]
        dz = u[:, :, :-1, :-1, 1:] - u[:, :, :-1, :-1, :-1]
        J11 = 1 + dx[:, 0, ...]
        J12 = dy[:, 0, ...]
        J13 = dz[:, 0, ...]

        J21 = dx[:, 1, ...]
        J22 = 1 + dy[:, 1, ...]
        J23 = dz[:, 1, ...]

        J31 = dx[:, 2, ...]
        J32 = dy[:, 2, ...]
        J33 = 1 + dz[:, 2, ...]

        Jdet = J11 * (J22 * J33 - J23 * J32) - J12 * (J21 * J33 - J23 * J31) + J13 * (J21 * J32 - J22 * J31)
        loss = F.relu(-Jdet + self.eps)
        return torch.mean(loss**2)
    
class LDDMMRegularizerLoss(nn.Module):
    def __init__(self, alpha=0.001, s=2):
        super(LDDMMRegularizerLoss, self).__init__()
        self.alpha = alpha
        self.s = s

    def forward(self, v):
        batch_size, channels, D, H, W = v.shape
        
        grid_x = torch.fft.fftfreq(D, device=v.device).view(-1, 1, 1)
        grid_y = torch.fft.fftfreq(H, device=v.device).view(1, -1, 1)
        grid_z = torch.fft.fftfreq(W, device=v.device).view(1, 1, -1)
        
        momentum_freq = 2 * (3 - torch.cos(2 * torch.pi * grid_x) 
                               - torch.cos(2 * torch.pi * grid_y) 
                               - torch.cos(2 * torch.pi * grid_z))
        
        L_op = (1.0 + self.alpha * momentum_freq) ** (2*self.s)
        
        loss = 0.0
        for i in range(channels):
            v_fft = torch.fft.fftn(v[:, i, ...])
            v_filtered = torch.fft.ifftn(v_fft * L_op).real
            loss += torch.mean(v[:, i, ...] * v_filtered)
            
        return loss
    
class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, u):
        loss = (((u[:, :, 1:, :, :] - u[:, :, :-1, :, :]) ** 2).mean() + 
                ((u[:, :, :, 1:, :] - u[:, :, :, :-1, :]) ** 2).mean() + 
                ((u[:, :, :, :, 1:] - u[:, :, :, :, :-1]) ** 2).mean())
        return loss
