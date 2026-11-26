import numpy as np
import torch

def transform_grid(dDepth, dHeight, dWidth):
    batchSize = dDepth.shape[0]
    dpt = dDepth.shape[1]
    hgt = dDepth.shape[2]
    wdt = dDepth.shape[3]

    D_mesh = torch.linspace(0.0, dpt - 1.0, dpt).unsqueeze_(1).unsqueeze_(1).expand(dpt, hgt, wdt)
    h_t = torch.matmul(torch.linspace(0.0, hgt - 1.0, hgt).unsqueeze_(1), torch.ones((1, wdt)))
    H_mesh = h_t.unsqueeze_(0).expand(dpt, hgt, wdt)
    w_t = torch.matmul(torch.ones((hgt, 1)), torch.linspace(0.0, wdt - 1.0, wdt).unsqueeze_(1).transpose(1, 0))
    W_mesh = w_t.unsqueeze_(0).expand(dpt, hgt, wdt)

    D_mesh = D_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
    H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
    W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
    D_upmesh = dDepth.float() + D_mesh
    H_upmesh = dHeight.float() + H_mesh
    W_upmesh = dWidth.float() + W_mesh
    return torch.stack([D_upmesh, H_upmesh, W_upmesh], dim=1)


def mse_metric(img_pred, img_target):
    return np.mean((img_pred - img_target) ** 2)

def jacobian_determinant_3d(displacement):
    """
    % determinants < 0, minimum and maximum
    """
    grad_x = np.gradient(displacement[0], axis=0)
    grad_y = np.gradient(displacement[1], axis=1)
    grad_z = np.gradient(displacement[2], axis=2)

    grad_xy = np.gradient(displacement[0], axis=1)
    grad_xz = np.gradient(displacement[0], axis=2)
    grad_yx = np.gradient(displacement[1], axis=0)
    grad_yz = np.gradient(displacement[1], axis=2)
    grad_zx = np.gradient(displacement[2], axis=0)
    grad_zy = np.gradient(displacement[2], axis=1)

    #det of id + grad
    jac = (1 + grad_x) * ((1 + grad_y) * (1 + grad_z) - grad_yz * grad_zy) \
        - grad_xy * (grad_yx * (1 + grad_z) - grad_yz * grad_zx) \
        + grad_xz * (grad_yx * grad_zy - (1 + grad_y) * grad_zx)

    # percentage
    jac_folds = (np.sum(jac <= 0) / np.prod(jac.shape)) * 100.0
    
    # min and max
    min_jac = np.min(jac)
    max_jac = np.max(jac)

    return jac_folds, min_jac, max_jac


def get_jacobian_determinant(flow):
    e1 = torch.reshape(torch.Tensor([1, 0, 0]), (1, 3, 1, 1, 1)).type(torch.float32).cuda()
    e2 = torch.reshape(torch.Tensor([0, 1, 0]), (1, 3, 1, 1, 1)).type(torch.float32).cuda()
    e3 = torch.reshape(torch.Tensor([0, 0, 1]), (1, 3, 1, 1, 1)).type(torch.float32).cuda()

    stacks = torch.stack([
        flow[..., 1:, :-1, :-1] - flow[..., :-1, :-1, :-1] + e1,
        flow[..., :-1, 1:, :-1] - flow[..., :-1, :-1, :-1] + e2,
        flow[..., :-1, :-1, 1:] - flow[..., :-1, :-1, :-1] + e3
    ], dim=2)
    stacks = torch.permute(stacks, [0, 3, 4, 5, 1, 2])
    determinant = torch.linalg.det(stacks)
    return determinant


def jacobian_det_var(flow):
    determinant = get_jacobian_determinant(flow)
    variance = torch.var(determinant, dim=[1, 2, 3])
    return torch.sqrt(variance)


def neg_jacobian_det(flow):
    determinant = get_jacobian_determinant(flow)
    neg = (determinant <= 0).sum(dim=[1, 2, 3]) / torch.prod(torch.tensor(determinant.size()[1:]))
    return neg * 100

def dice_OASIS(img_gt, img_pred, num_classes=36):
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("Las matrices deben tener la misma dimensión")

    dices = []
    for c in range(num_classes):
        gt_c = (img_gt == c)
        pred_c = (img_pred == c)

        top = 2 * np.sum(np.logical_and(gt_c, pred_c))
        bottom = np.sum(gt_c) + np.sum(pred_c)

        if bottom == 0:
            dices.append(1.0 if top == 0 else 0.0)
        else:
            dices.append(top / bottom)

    return dices
