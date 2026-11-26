import torch 
import torch.nn as nn
import torch.nn.functional as F

class FieldTransformer(nn.Module):
    def __init__(self, steps=7):
        super(FieldTransformer, self).__init__()
        self.steps = steps 
        self.grid = None  

    def _spatial_transform(self, src, flow, mode="bilinear"):
        shape = flow.shape[2:]

        # id si no hay grid o si la forma del grid no coincide con la forma actual del campo
        if self.grid is None or self.grid.shape[2:] != shape:
            vectors = [torch.arange(0, s) for s in shape]
            grids = torch.meshgrid(vectors, indexing="ij")
            self.grid = torch.stack(grids).unsqueeze(0).to(src.device, dtype=torch.float32)

        # grid + displacement
        new_locs = self.grid + flow

        # nornalize
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=mode)

    def forward(self, velocity, src=None, mode="bilinear"):
        # velocity -> field prediction
        # src -> moving image
        flow = velocity / (2 ** self.steps)
        
        # phi_t = phi_{t/2} ◦ phi_{t/2}
        for _ in range(self.steps):
            flow = flow + self._spatial_transform(flow, flow, mode="bilinear")
            
        # flow is the field
        if src is not None:
            warped_src = self._spatial_transform(src, flow, mode=mode)
            return warped_src, flow
            
        return flow