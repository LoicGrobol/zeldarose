from abc import ABC, abstractmethod
import pathlib
from typing import Optional, Tuple, Union

import pydantic
import torch
import torchmetrics
import transformers

import pytorch_lightning as pl

from loguru import logger


class TrainConfig(pydantic.BaseModel):
    batch_size: int = 64
    betas: Tuple[float, float] = (0.9, 0.98)
    epsilon: float = 1e-8
    gradient_clipping: Optional[Union[float, int]] = None
    learning_rate: float = 1e-4
    lr_decay_steps: Optional[int] = None
    max_epochs: Optional[int] = None
    max_input_length: Optional[int] = None
    max_steps: Optional[int] = None
    warmup_steps: int = 0
    weight_decay: Optional[float] = None


class TrainingModule(pl.LightningModule, ABC):
    model: transformers.PreTrainedModel
    training_config: TrainConfig

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

    def configure_optimizers(self):
        if self.training_config.weight_decay is not None:
            no_decay = ["bias", "LayerNorm.weight"]
            decay_rate = self.training_config.weight_decay
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": decay_rate,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            decay_rate = 0.0
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters()],
                    "weight_decay": 0.0,
                },
            ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=self.training_config.betas,
            fused=True,
            lr=self.training_config.learning_rate,
            eps=self.training_config.epsilon,
            weight_decay=decay_rate,
        )
        if self.training_config.lr_decay_steps:
            if self.training_config.lr_decay_steps == -1:
                num_training_steps = (
                    self.trainer.estimated_stepping_batches - self.training_config.warmup_steps
                )
                logger.info(
                    f"Number of lr decay steps set at {num_training_steps} since -1 was asked"
                )
            else:
                if (
                    self.training_config.lr_decay_steps
                    > 2 * self.trainer.estimated_stepping_batches
                ):
                    logger.warning(
                        f"Asked for {self.training_config.lr_decay_steps} LR decay steps but the"
                        f" model will only be trained for {self.trainer.estimated_stepping_batches}"
                        " steps, this might be an oversight."
                    )
                num_training_steps = self.training_config.lr_decay_steps

            schedule = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.training_config.warmup_steps,
                num_training_steps=num_training_steps + self.training_config.warmup_steps,
            )
            schedulers = [{"scheduler": schedule, "interval": "step"}]
        elif self.training_config.warmup_steps > 0:
            schedule = transformers.get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.training_config.warmup_steps,
            )
            schedulers = [{"scheduler": schedule, "interval": "step"}]
        else:
            schedulers = []

        return [optimizer], schedulers


class MaskedAccuracy(torchmetrics.Metric):
    full_state_update: bool = False
    higher_is_better: Optional[bool] = True
    is_differentiable = False

    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

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
            self.total += mask.int().sum()

    def compute(self):
        return self.correct.true_divide(self.total)
