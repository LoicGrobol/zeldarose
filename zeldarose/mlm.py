from typing import NamedTuple, Optional, Sequence

import pydantic
import pytorch_lightning as pl
import torch
import torch.jit
import torch.utils.data
import transformers

from torch.nn.utils.rnn import pad_sequence

import zeldarose.data


class MaskedTokens(NamedTuple):
    inputs: torch.Tensor
    labels: torch.Tensor


# FUTURE: use `torch.logical_xxx` functions in torch >= 1.5.0
# TODO: How to do whole-word masking?
@torch.jit.script
def mask_tokens(
    inputs: torch.Tensor,
    input_mask_indice: int,
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
    what_to_do = torch.rand_like(labels)
    if keep_mask is not None:
        what_to_do.masked_fill_(keep_mask, 1.0)
    preserved_tokens = what_to_do.gt(change_ratio)
    # We only compute loss on masked tokens
    labels.masked_fill_(preserved_tokens, label_mask_indice)

    # replace some input tokens with tokenizer.mask_token ([MASK])
    masked_tokens = what_to_do.le(change_ratio * mask_ratio)
    inputs.masked_fill_(masked_tokens, input_mask_indice)

    # Replace masked input tokens with random word
    switched_tokens = (
        what_to_do.le(change_ratio * (mask_ratio + switch_ratio))
        & masked_tokens.logical_not()
    )
    random_words = torch.randint_like(labels, vocabulary_size)
    inputs[switched_tokens] = random_words[switched_tokens]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return MaskedTokens(inputs, labels)


# TODO: add validation
class MLMTaskConfig(pydantic.BaseModel):
    change_ratio: float = 0.15
    mask_ratio: float = 0.8
    switch_ratio: float = 0.1


class MLMBatch(NamedTuple):
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    mlm_labels: torch.Tensor


class MLMLoader(torch.utils.data.Dataloader):
    def __init__(
        self,
        dataset: zeldarose.data.TextDataset,
        task_config: MLMTaskConfig,
        *args,
        **kwargs
    ):
        super().__init__(*args, collate_fn=self.collate, **kwargs)
        self.task_config = task_config
        mask_token_index = getattr(self.dataset.tokenizer, "mask_token_id")
        if mask_token_index is None:
            mask_token_index = self.dataset.tokenizer.convert_tokens_to_ids(
                self.dataset.tokenizer.mask_token
            )
        self.mask_token_index = mask_token_index
        padding_value = getattr(self.dataset.tokenizer, "pad_token_id", 0)
        if padding_value is None:
            padding_value = self.dataset.tokenizer.convert_tokens_to_ids(
                self.dataset.tokenizer.padding_value
            )

    def collate(self, batch: Sequence[torch.Tensor]) -> MLMBatch:
        padded_batch = pad_sequence(
            batch, batch_first=True, padding_value=self.padding_value
        )
        padding_mask = padded_batch.eq(self.padding_value)
        # We only deal with single sequences here
        token_type_ids = torch.zeros_like(padded_batch)
        attention_mask = padding_mask.logical_not()

        special_tokens_mask = torch.tensor(
            [
                self.dataset.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in batch
            ],
            dtype=torch.bool,
        )
        keep_mask = special_tokens_mask | padding_mask
        masked = mask_tokens(
            inputs=padded_batch,
            input_mask_index=self.mask_token_id,
            vocabulary_size=len(self.dataset.tokenizer),
            change_ratio=self.task_config.change_ratio,
            mask_ratio=self.task_config.mask_ratio,
            switch_ratio=self.task_config.switch_ratio,
            keep_mask=keep_mask,
        )
        return MLMBatch(
            tokens=masked.inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            mlm_labels=masked.labels,
        )


# TODO: add validation
class MLMFinetunerConfig(pydantic.BaseModel):
    batch_size: int
    epsilon: float
    learning_rate: float
    num_steps: int
    warmup_steps: int
    weight_decay: Optional[float]


class MLMFinetuner(pl.LightningModule):
    def __init__(
        self, model: transformers.PreTrainedModel, config: MLMFinetunerConfig,
    ):
        super().__init__()
        self.config = config
        self.model = model

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        mlm_labels: torch.Tensor,
    ):

        output = self.model(
            input_ids=tokens,
            mlm_labels=mlm_labels,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        return output

    def training_step(self, batch: MLMBatch, batch_idx: int):
        outputs = self.forward(
            tokens=batch.tokens,
            attention_mask=batch.attention_mask,
            token_type_ids=batch.token_type_ids,
            mlm_labels=batch.mlm_labels,
        )

        loss = outputs[0]
        perplexity = torch.exp(loss)

        tensorboard_logs = {"train/train_loss": loss, "train/perplexity": perplexity}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch: MLMBatch, batch_idx: int):
        outputs = self.forward(
            tokens=batch.tokens,
            attention_mask=batch.attention_mask,
            token_type_ids=batch.token_type_ids,
            mlm_labels=batch.mlm_labels,
        )
        loss = outputs[0]

        preds = torch.argmax(outputs[1], dim=-1)
        correct_preds = preds.eq(batch.mlm_labels) & batch.mlm_labels.ne(-100)
        accuracy = correct_preds.mean()

        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        perplexity = torch.exp(avg_loss)

        tensorboard_logs = {
            "validation/loss": avg_loss,
            "validation/accuracy": avg_acc,
            "validation/perplexity": perplexity,
        }
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

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

        t_total = self.config.num_steps

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.epsilon,
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=t_total,
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler_config]
