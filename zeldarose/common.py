from typing import Optional, Tuple, Union
import pydantic
import torch
import torchmetrics


class TrainConfig(pydantic.BaseModel):
    batch_size: int = 64
    betas: Tuple[float, float] = (0.9, 0.98)
    epsilon: float = 1e-8
    gradient_clipping: Optional[Union[float, int]] = None
    learning_rate: float = 1e-4
    lr_decay_steps: Optional[int] = None
    warmup_steps: int = 0
    weight_decay: Optional[float] = None


class MaskedAccuracy(torchmetrics.Metric):
    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(compute_on_step=False, dist_sync_on_step=dist_sync_on_step)

        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        mask = target.ne(self.ignore_index)
        if mask.any():
            self.correct += preds.eq(target).logical_and(mask).int().sum()
            self.total += mask.sum()

    def compute(self):
        return self.correct.true_divide(self.total)
