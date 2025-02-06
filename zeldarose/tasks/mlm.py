import pathlib
from typing import Any, cast, Dict, NamedTuple, Optional, TYPE_CHECKING, Union

import pydantic
import torch
import torch.jit
import torch.utils.data
import transformers
from loguru import logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import zeldarose.datasets.transform
from zeldarose.common import MaskedAccuracy, TrainConfig, TrainingModule


if TYPE_CHECKING:
    import transformers.modeling_outputs


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
    #   - `change_ratio * mask_ratio < v <= change_ratio * (mask_ratio + switch_ratio)`: replace
    #     with a random word
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
    switched_tokens = what_to_do.le(change_ratio * (mask_ratio + switch_ratio)).logical_and(
        masked_tokens.logical_not()
    )
    random_words = torch.randint_like(labels, vocabulary_size)
    # FIXME: probably still an unnecessary copy here
    inputs[switched_tokens] = random_words[switched_tokens]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return MaskedTokens(inputs, labels)


class MLMTaskConfig(pydantic.BaseModel):
    change_ratio: float = 0.15
    mask_ratio: float = 0.8
    switch_ratio: float = 0.1


class MLMTrainingModel(TrainingModule):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        mask_token_index: int,
        vocabulary_size: int,
        training_config: Optional[TrainConfig] = None,
        task_config: Optional[MLMTaskConfig] = None,
    ):
        super().__init__()
        if training_config is not None:
            self.training_config = training_config
        else:
            self.training_config = TrainConfig()
        if task_config is not None:
            self.task_config = task_config
        else:
            self.task_config = MLMTaskConfig()
        logger.info(f"MLM trainer config: {self.training_config}")
        logger.info(f"MLM task config: {self.task_config}")
        self.mask_token_index = mask_token_index
        self.vocabulary_size = vocabulary_size

        self.train_accuracy = MaskedAccuracy()
        self.val_accuracy = MaskedAccuracy()
        self.model = model

        self.save_hyperparameters("training_config", "task_config")

    def forward(  # type: ignore[override]
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

    def training_step(  # type: ignore[override]
        self, batch: zeldarose.datasets.transform.TextBatch, batch_idx: int
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

        with torch.no_grad():
            preds = torch.argmax(outputs.logits, dim=-1)
            perplexity = torch.exp(loss)
            self.train_accuracy.update(preds, masked.labels)

            self.log(
                "train/loss",
                loss,
                batch_size=tokens.shape[0],
                reduce_fx=torch.mean,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/perplexity",
                perplexity,
                batch_size=tokens.shape[0],
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/accuracy",
                self.train_accuracy,
                batch_size=tokens.shape[0],
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch: zeldarose.datasets.transform.TextBatch, batch_idx: int):  # type: ignore[override]
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
        self.val_accuracy.update(preds, masked.labels)

        self.log("validation/loss", loss, batch_size=tokens.shape[0], sync_dist=True)
        self.log(
            "validation/perplexity",
            perplexity,
            batch_size=tokens.shape[0],
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "validation/accuracy",
            self.val_accuracy,
            batch_size=tokens.shape[0],
            on_epoch=True,
            sync_dist=True,
        )

    def get_data_module(
        self,
        loader_batch_size: int,
        num_workers: int,
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        tokenizer_name: str,
        train_path: Union[str, pathlib.Path],
        data_dir: Optional[pathlib.Path] = None,
        val_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> zeldarose.datasets.transform.TextDataModule:
        if (max_length := getattr(self.model.config, "max_position_embeddings", None)) is None:
            max_length = tokenizer.max_len_single_sentence
        else:
            # FIXME: we shouldn't need num_special_tokens_to_add here
            max_length = min(
                tokenizer.max_len_single_sentence,
                max_length - tokenizer.num_special_tokens_to_add(pair=False),
            )
        if self.training_config.max_input_length is not None:
            max_length = min(max_length, self.training_config.max_input_length)

        return zeldarose.datasets.transform.TextDataModule(
            loader_batch_size=loader_batch_size,
            num_workers=num_workers,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            train_path=train_path,
            data_dir=data_dir,
            max_length=max_length,
            val_path=val_path,
        )

    @rank_zero_only
    def save_transformer(
        self,
        save_dir: pathlib.Path,
        tokenizer: Optional[
            Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]
        ] = None,
    ):
        """Save the wrapped transformer model."""
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {save_dir}")
        self.model.save_pretrained(str(save_dir))
        if tokenizer is not None:
            logger.info(f"Saving tokenizer to {save_dir}")
            tokenizer.save_pretrained(str(save_dir), legacy_format=not tokenizer.is_fast)


def get_training_model(
    model_config: Optional[Union[str, pathlib.Path]],
    pretrained_model: Optional[Union[str, pathlib.Path]],
    task_config: Optional[Dict[str, Any]],
    tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    training_config: TrainConfig,
) -> MLMTrainingModel:
    if task_config is not None:
        _task_config = MLMTaskConfig.model_validate(task_config)
    else:
        _task_config = MLMTaskConfig()

    if (
        mask_token_index := cast(Union[int, None], getattr(tokenizer, "mask_token_id", None))
    ) is None:
        mask_token_index = cast(int, tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
    vocabulary_size = tokenizer.vocab_size

    if pretrained_model is not None:
        logger.info(f"Loading pretrained model {pretrained_model!r}")
        model = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_model)
    elif model_config is not None:
        logger.info(f"Loading pretrained config {model_config!r}")
        _model_config = transformers.AutoConfig.from_pretrained(model_config)
        logger.info("Generating model from config")
        # TODO: check the other parameters?
        if vocabulary_size is not None and _model_config.vocab_size != vocabulary_size:
            logger.warning(
                f"Vocabulary size mismatch between model config ({_model_config.vocab_size})"
                f" and pretrained tokenizer ({vocabulary_size}), using {vocabulary_size}."
            )
            _model_config.vocab_size = vocabulary_size
        model = transformers.AutoModelForMaskedLM.from_config(_model_config)
    else:
        raise ValueError("You must provide either a pretrained model or a model config")

    logger.info("Creating MLM training model")

    logger.debug(f"Mask token index: {mask_token_index}")
    training_model = MLMTrainingModel(
        model=model,
        mask_token_index=mask_token_index,
        vocabulary_size=vocabulary_size,
        task_config=_task_config,
        training_config=training_config,
    )

    return training_model
