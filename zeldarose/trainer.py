from typing import NamedTuple, Optional

import pytorch_lightning as pl
import torch
import torch.jit
import transformers


def get_keep_mask(
    inputs: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Takes a batch of MLM inputs and return a mask for the tokens that should no be
    changed: special tokens and possibly padding.
    """
    special_tokens_mask = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in inputs.tolist()
        ],
        dtype=torch.bool,
    )
    keep_mask = special_tokens_mask
    if padding_mask is not None:
        keep_mask = keep_mask | padding_mask
    elif tokenizer._pad_token is not None:
        keep_mask = keep_mask | inputs.eq(tokenizer.pad_token_id)
    return keep_mask


class MaskedBatch(NamedTuple):
    inputs: torch.Tensor
    labels: torch.Tensor


# FUTURE: use `torch.logical_xxx` functions in torch >= 1.5.0
# TODO: How to do whole-word masking?
@torch.jit.script
def mask_tokens(
    inputs: torch.Tensor,
    input_mask_indice: int,
    vocabulary_size: int,
    change_ratio: float = 0.15,
    mask_ratio: float = 0.8,
    switch_ratio: float = 0.1,
    keep_mask: Optional[torch.Tensor] = None,
    label_mask_indice: int = -100,
) -> MaskedBatch:
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
    return MaskedBatch(inputs, labels)


class MLMFinetuner(pl.LightningModule):
    def __init__(self, model: transformers.PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        mlm_labels: torch.Tensor,
    ):

        output = self.model(
            input_ids=inputs,
            mlm_labels=mlm_labels,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        return output

    def training_step(self, batch, batch_idx):
        inputs, attention_mask, token_type_ids, mlm_labels = batch
        outputs = self.forward(
            inputs=inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            mlm_labels=mlm_labels,
        )

        loss = outputs[0]
        perplexity = torch.exp(loss)

        self._log_lr()
        tensorboard_logs = {"train/train_loss": loss, "train/perplexity": perplexity}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, attention_mask, token_type_ids, mlm_labels = batch
        outputs = self.forward(
            inputs=inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            mlm_labels=mlm_labels,
        )
        loss = outputs[0]

        preds = torch.argmax(outputs[1], dim=-1)
        correct_preds = preds.eq(mlm_labels).logical_and(mlm_labels.ne(-100))
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

        t_total = self.config.num_steps

        optimizer = torch.optim.Adam(
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

    def _log_lr(self):
        """Logs learning rate to tensorboard.
        """
        # get LR schedulers from the pytorch-lightning trainer object.
        scheduler = self.trainer.lr_schedulers[0]["scheduler"]

        # tie LR stepping to global step.
        for i, lr in enumerate(scheduler.get_lr()):
            # add the scalar to the Experiment object.
            self.logger.experiment.add_scalar(f"lr_{i}", lr, self.global_step)
