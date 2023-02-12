from abc import ABC, abstractmethod
import pathlib
from typing import Optional, Tuple, Union

import pydantic
import torch
import torchmetrics
import transformers

import pytorch_lightning as pl


class TrainingModule(pl.LightningModule, ABC):
    @abstractmethod
    def get_data_module(
        self,
        loader_batch_size: int,
        num_workers: int,
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        tokenizer_name: str,
        train_path: Union[str, pathlib.Path],
        data_dir: Optional[pathlib.Path] = None,
        val_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> pl.LightningDataModule:
        raise NotImplementedError()

    @abstractmethod
    def save_transformer(
        self,
        save_dir: pathlib.Path,
        tokenizer: Optional[
            Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]
        ] = None,
    ):
        raise NotImplementedError()


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
    full_state_update: bool = False
    higher_is_better: Optional[bool] = True
    is_differentiable = False

    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(compute_on_step=False, dist_sync_on_step=dist_sync_on_step)

        self.ignore_index = ignore_index
        self.correct: torch.Tensor
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.total: torch.Tensor
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):  # type: ignore[override]
        assert preds.shape == target.shape
        mask = target.ne(self.ignore_index)
        if mask.any():
            self.correct += preds.eq(target).logical_and(mask).int().sum()
            self.total += mask.sum()

    def compute(self):
        return self.correct.true_divide(self.total)
