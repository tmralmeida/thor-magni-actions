import torch
from torchmetrics import Metric


class AverageDisplacementError(Metric):
    """ADE metric for torchmetrics API"""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.shape != target.shape:
            raise ValueError("Problem on FDE shapes")
        loss = torch.mean(torch.norm(preds - target, dim=-1), dim=-1)
        self.dist += loss.sum()
        self.total += target.shape[0]  # evaluating bs

    def compute(self):
        return self.dist.float() / self.total


class FinalDisplacementError(Metric):
    """FDE metric for torchmetrics API"""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.shape != target.shape:
            raise ValueError("Problem on FDE shapes")
        last_gt, last = target[:, -1, :], preds[:, -1, :]
        loss = torch.norm(last_gt - last, 2, 1)
        self.dist += loss.sum()
        self.total += target.shape[0]  # evaluating bs

    def compute(self):
        return self.dist.float() / self.total