from typing import NamedTuple, Optional, Tuple

import pydantic
import pytorch_lightning as pl
import torch
import torch.jit
import torch.utils.data
import torchmetrics
import transformers

from loguru import logger

import zeldarose.data


class MaskedTokens(NamedTuple):
    inputs: torch.Tensor
    labels: torch.Tensor


# TODO: How to do whole-word masking?
@torch.jit.script
def mask_tokens(
    inputs: torch.Tensor,
    input_mask_index: int,
    vocabulary_size: int,
    change_ratio: float,
    mask_ratio: float,
    switch_ratio: float,
    keep_mask: Optional[torch.Tensor] = None,
    label_mask_indice: int = -100,
) -> MaskedTokens:
    """Prepare masked tokens inputs/labels for masked language modeling

    Notes
    -----

    - This modifies `inputs` in place, which is not very pure but avoids a (useless in practice)
      copy operation.
    - hf transformers use `-100` for label mask because it's the default ignore index of
      `torch.nn.CrossEntropy`
    """

    labels = inputs.clone()
    # Tells us what to do with each token according to its value `v` in `what_to_do`
    # - `v <= change_ratio` change the token and use it in the loss
    #   - `v <= change_ratio * mask_ratio`: replace with [MASK] ('s id)
    #   - `change_ratio * mask_ratio < v <= change_ratio * (mask_ratio + switch_ratio)`: replace with a random word
    #   - `change_ratio * (mask_ratio + switch_ratio) < v <= change_ratio`: keep as is
    # - `change_ratio < v`: keep as is and don't use in the loss
    # FIXME: layout and device should be inferred here, file an issue
    what_to_do = torch.rand_like(
        labels, layout=torch.strided, dtype=torch.float, device=labels.device
    )
    # This ensures that the internal tokens and the padding is not changed and not used in the loss
    if keep_mask is not None:
        what_to_do.masked_fill_(keep_mask, 1.0)
    preserved_tokens = what_to_do.gt(change_ratio)
    # We only compute loss on masked tokens
    labels.masked_fill_(preserved_tokens, label_mask_indice)

    # replace some input tokens with tokenizer.mask_token ([MASK])
    masked_tokens = what_to_do.le(change_ratio * mask_ratio)
    inputs.masked_fill_(masked_tokens, input_mask_index)

    # Replace masked input tokens with random word
    switched_tokens = what_to_do.le(
        change_ratio * (mask_ratio + switch_ratio)
    ).logical_and(masked_tokens.logical_not())
    random_words = torch.randint_like(labels, vocabulary_size)
    # FIXME: probably still an unnecessary copy here
    inputs[switched_tokens] = random_words[switched_tokens]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return MaskedTokens(inputs, labels)


class MaskedAccuracy(torchmetrics.Metric):
    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

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


class MLMTaskConfig(pydantic.BaseModel):
    change_ratio: float = 0.15
    mask_ratio: float = 0.8
    switch_ratio: float = 0.1


class MLMFinetunerConfig(pydantic.BaseModel):
    batch_size: int = 64
    betas: Tuple[float, float] = (0.9, 0.98)
    epsilon: float = 1e-8
    gradient_clipping = 0
    learning_rate: float = 1e-4
    lr_decay_steps: Optional[int] = None
    warmup_steps: int = 0
    weight_decay: Optional[float] = None


class MLMFinetuner(pl.LightningModule):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        mask_token_index: int,
        vocabulary_size: int,
        config: Optional[MLMFinetunerConfig] = None,
        task_config: Optional[MLMTaskConfig] = None,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = MLMFinetunerConfig()
        if task_config is not None:
            self.task_config = task_config
        else:
            self.task_config = MLMTaskConfig()
        logger.info(f"MLM trainer config: {self.config}")
        logger.info(f"MLM task config: {self.task_config}")
        self.mask_token_index = mask_token_index
        self.vocabulary_size = vocabulary_size

        self.accuracy = MaskedAccuracy()
        self.model = model

        self.save_hyperparameters("config", "task_config")

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        mlm_labels: torch.Tensor,
    ) -> "transformers.modeling_outputs.MaskedLMOutput":
        output = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=mlm_labels,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        return output

    def training_step(
        self, batch: zeldarose.data.TextBatch, batch_idx: int
    ) -> torch.Tensor:
        tokens, attention_mask, internal_tokens_mask, token_type_ids = batch
        with torch.no_grad():
            masked = mask_tokens(
                inputs=tokens,
                change_ratio=self.task_config.change_ratio,
                keep_mask=internal_tokens_mask,
                mask_ratio=self.task_config.mask_ratio,
                input_mask_index=self.mask_token_index,
                switch_ratio=self.task_config.switch_ratio,
                vocabulary_size=self.vocabulary_size,
            )

        outputs = self(
            tokens=masked.inputs,
            attention_mask=attention_mask,
            mlm_labels=masked.labels,
            token_type_ids=token_type_ids,
        )

        loss = outputs.loss

        preds = torch.argmax(outputs.logits, dim=-1)
        perplexity = torch.exp(loss)
        self.accuracy(preds, masked.labels)

        self.log(
            "train/loss",
            loss,
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/perplexity",
            perplexity,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/accuracy",
            self.accuracy,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: zeldarose.data.TextBatch, batch_idx: int):
        tokens, attention_mask, internal_tokens_mask, token_type_ids = batch
        with torch.no_grad():
            masked = mask_tokens(
                inputs=tokens,
                change_ratio=self.task_config.change_ratio,
                keep_mask=internal_tokens_mask,
                mask_ratio=self.task_config.mask_ratio,
                input_mask_index=self.mask_token_index,
                switch_ratio=self.task_config.switch_ratio,
                vocabulary_size=self.vocabulary_size,
            )

        outputs = self(
            tokens=masked.inputs,
            attention_mask=attention_mask,
            mlm_labels=masked.labels,
            token_type_ids=token_type_ids,
        )

        loss = outputs.loss
        perplexity = torch.exp(loss)

        preds = torch.argmax(outputs.logits, dim=-1)
        self.accuracy(preds, masked.labels)

        self.log("validation/loss", loss, sync_dist=True)
        self.log(
            "validation/perplexity",
            perplexity,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "validation/accuracy",
            self.accuracy,
            on_epoch=True,
        )

    def configure_optimizers(self):
        if self.config.weight_decay is not None:
            no_decay = ["bias", "LayerNorm.weight"]
            decay_rate = self.config.weight_decay
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
            betas=self.config.betas,
            lr=self.config.learning_rate,
            eps=self.config.epsilon,
            weight_decay=decay_rate,
        )
        if self.config.lr_decay_steps:
            if self.config.lr_decay_steps == -1:
                num_training_steps = self.trainer.max_steps
            else:
                num_training_steps = self.config.lr_decay_steps

            schedule = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps,
            )
            schedulers = [{"scheduler": schedule, "interval": "step"}]
        elif self.config.warmup_steps > 0:
            schedule = transformers.get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
            )
            schedulers = [{"scheduler": schedule, "interval": "step"}]
        else:
            schedulers = []

        return [optimizer], schedulers
