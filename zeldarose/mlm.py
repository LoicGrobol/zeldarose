from typing import NamedTuple, Optional, Tuple

import pydantic
import pytorch_lightning as pl
import torch
import torch.jit
import torch.utils.data
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

    This modifies `inputs` in place, which is not very pure but avoids a (useless in practice) copy
    operation.
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
    if keep_mask is not None:
        what_to_do.masked_fill_(keep_mask, 1.0)
    preserved_tokens = what_to_do.gt(change_ratio)
    # We only compute loss on masked tokens
    labels.masked_fill_(preserved_tokens, label_mask_indice)

    # replace some input tokens with tokenizer.mask_token ([MASK])
    masked_tokens = what_to_do.le(change_ratio * mask_ratio)
    inputs.masked_fill_(masked_tokens, input_mask_index)

    # Replace masked input tokens with random word
    switched_tokens = (
        what_to_do.le(change_ratio * (mask_ratio + switch_ratio))
        & masked_tokens.logical_not()
    )
    random_words = torch.randint_like(labels, vocabulary_size)
    # FIXME: probably still an unnecessary copy here
    inputs[switched_tokens] = random_words[switched_tokens]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return MaskedTokens(inputs, labels)


# TODO: add validation
class MLMTaskConfig(pydantic.BaseModel):
    change_ratio: float = 0.15
    mask_ratio: float = 0.8
    switch_ratio: float = 0.1


# TODO: add validation
class MLMFinetunerConfig(pydantic.BaseModel):
    batch_size: int = 64
    betas: Tuple[float, float] = (0.9, 0.98)
    epsilon: float = 1e-8
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
        self.model = model
        self.mask_token_index = mask_token_index
        self.vocabulary_size = vocabulary_size

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        mlm_labels: torch.Tensor,
    ):

        output = self.model(
            input_ids=tokens,
            masked_lm_labels=mlm_labels,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        return output

    def training_step(self, batch: zeldarose.data.TextBatch, batch_idx: int):
        # FIXME: this because lightning doesn't preserve namedtuples
        tokens, attention_mask, internal_tokens_mask, token_type_ids = batch
        with torch.no_grad():
            masked = mask_tokens(
                inputs=tokens,
                input_mask_index=self.mask_token_index,
                vocabulary_size=self.vocabulary_size,
                change_ratio=self.task_config.change_ratio,
                mask_ratio=self.task_config.mask_ratio,
                switch_ratio=self.task_config.switch_ratio,
                keep_mask=internal_tokens_mask,
            )

        outputs = self.forward(
            tokens=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            mlm_labels=tokens,
        )

        loss = outputs[0]
        perplexity = torch.exp(loss)

        tensorboard_logs = {"train/train_loss": loss, "train/perplexity": perplexity}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        perplexity = torch.exp(avg_loss)

        results = {"avg_train_loss": avg_loss, "train_perplexity": perplexity}
        return results

    # def validation_step(self, batch: MLMBatch, batch_idx: int):
    #     outputs = self.forward(
    #         tokens=batch.tokens,
    #         attention_mask=batch.attention_mask,
    #         token_type_ids=batch.token_type_ids,
    #         mlm_labels=batch.mlm_labels,
    #     )
    #     loss = outputs[0]

    #     preds = torch.argmax(outputs[1], dim=-1)
    #     correct_preds = preds.eq(batch.mlm_labels) & batch.mlm_labels.ne(-100)
    #     accuracy = correct_preds.mean()

    #     return {"val_loss": loss, "val_accuracy": accuracy}

    # def validation_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

    #     perplexity = torch.exp(avg_loss)

    #     tensorboard_logs = {
    #         "validation/loss": avg_loss,
    #         "validation/accuracy": avg_acc,
    #         "validation/perplexity": perplexity,
    #     }
    #     return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        if self.config.weight_decay is not None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.config.weight_decay,
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
        )
        if self.config.lr_decay_steps:
            schedule = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.num_steps,
            )

            schedulers = [{"scheduler": schedule, "interval": "step"}]
        else:
            schedulers = []

        return [optimizer], schedulers
