import pathlib
from typing import TYPE_CHECKING, Any

import torch
import torch.utils.data
import transformers
from loguru import logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import zeldarose.datasets.transform
from zeldarose.common import MaskedAccuracy, TrainConfig, TrainingModule

if TYPE_CHECKING:
    import transformers.modeling_outputs


class NTPTrainingModel(TrainingModule):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        vocabulary_size: int,
        training_config: TrainConfig | None = None,
    ):
        super().__init__()
        if training_config is not None:
            self.training_config = training_config
        else:
            self.training_config = TrainConfig()

        logger.info(f"NTP trainer config: {self.training_config}")
        self.vocabulary_size = vocabulary_size

        # We won't be masking anything but this still works better
        self.train_accuracy = MaskedAccuracy()
        self.val_accuracy = MaskedAccuracy()
        self.model = model

        self.save_hyperparameters("training_config")

    def forward(  # type: ignore[override]
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> "transformers.modeling_outputs.MaskedLMOutput":
        output = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=tokens,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        return output

    def training_step(  # type: ignore[override]
        self, batch: zeldarose.datasets.transform.TextBatch, batch_idx: int
    ) -> torch.Tensor:
        tokens, attention_mask, internal_tokens_mask, token_type_ids = batch

        outputs = self(
            tokens=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = outputs.loss

        with torch.no_grad():
            preds = torch.argmax(outputs.logits, dim=-1)
            perplexity = torch.exp(loss)
            self.train_accuracy.update(preds[..., :-1], tokens[..., 1:])

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

        outputs = self(
            tokens=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = outputs.loss
        perplexity = torch.exp(loss)

        preds = torch.argmax(outputs.logits, dim=-1)
        self.val_accuracy.update(preds[..., :-1], tokens[..., 1:])

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
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
        tokenizer_name: str,
        train_path: str | pathlib.Path,
        data_dir: pathlib.Path | None = None,
        val_path: str | pathlib.Path | None = None,
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
        tokenizer: None
        | (transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast) = None,
    ):
        """Save the wrapped transformer model."""
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {save_dir}")
        self.model.save_pretrained(str(save_dir))
        if tokenizer is not None:
            logger.info(f"Saving tokenizer to {save_dir}")
            tokenizer.save_pretrained(str(save_dir), legacy_format=not tokenizer.is_fast)


def get_training_model(
    model_config: str | pathlib.Path | None,
    pretrained_model: str | pathlib.Path | None,
    task_config: dict[str, Any] | None,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    training_config: TrainConfig,
) -> NTPTrainingModel:
    if task_config is not None:
        raise ValueError("NTP Models don't have a task config!")

    vocabulary_size = tokenizer.vocab_size

    if pretrained_model is not None:
        logger.info(f"Loading pretrained model {pretrained_model!r}")
        model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model)
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
        model = transformers.AutoModelForCausalLM.from_config(_model_config)
    else:
        raise ValueError("You must provide either a pretrained model or a model config")

    logger.info("Creating NTP training model")

    training_model = NTPTrainingModel(
        model=model,
        vocabulary_size=vocabulary_size,
        training_config=training_config,
    )

    return training_model
