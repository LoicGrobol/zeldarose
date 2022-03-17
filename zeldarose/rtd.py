import pathlib
from typing import Any, Dict, List, Literal, NamedTuple, Optional, cast

import pydantic
import pytorch_lightning as pl
import torch
import torch.jit
import torch.utils.data
import transformers
import transformers.modeling_outputs

from loguru import logger
from pytorch_lightning.utilities import rank_zero_only

import zeldarose.data

from zeldarose.common import MaskedAccuracy, TrainConfig
from zeldarose.utils import (
    OneWayShareTransformersEmbeddingsCallback,
    ShareTransformersEmbeddingsCallback,
)


class MaskedTokens(NamedTuple):
    inputs: torch.Tensor
    labels: torch.Tensor


class RTDOutput(NamedTuple):
    discriminator_output: transformers.modeling_outputs.TokenClassifierOutput
    generator_predictions: torch.Tensor
    discriminator_predictions: torch.Tensor
    generator_output: transformers.modeling_outputs.MaskedLMOutput
    rtd_labels: torch.Tensor


@torch.jit.script
def mask_tokens(
    tokens: torch.Tensor,
    input_mask_index: int,
    mask_ratio: float,
    keep_mask: Optional[torch.Tensor] = None,
    label_mask_index: int = -100,
) -> MaskedTokens:
    """Prepare masked tokens inputs/labels for masked language modeling

    ## Notes

    - The default for label mask is `-100` because it's the default ignore index of
      `torch.nn.CrossEntropy`
    """

    inputs = tokens.clone()
    labels = tokens.clone()
    # Tells us what to do with each token according to its value `v` in `what_to_do`
    # - `v <= mask_ratio` mask the token and use it in the loss
    # - `mask_ratio < v`: keep as is and don't use in the loss
    # FIXME: layout and device should be inferred here, file an issue
    what_to_do = torch.rand_like(
        inputs, layout=torch.strided, dtype=torch.float, device=labels.device
    )
    # This ensures that the internal tokens and the padding is not changed and not used in the loss
    if keep_mask is not None:
        what_to_do.masked_fill_(keep_mask, 1.0)
    preserved_tokens = what_to_do.gt(mask_ratio)
    # We only compute loss on masked tokens
    labels.masked_fill_(preserved_tokens, label_mask_index)

    # replace some input tokens with tokenizer.mask_token ([MASK])
    masked_tokens = what_to_do.le(mask_ratio)
    inputs.masked_fill_(masked_tokens, input_mask_index)

    return MaskedTokens(inputs, labels)


class RTDTaskConfig(pydantic.BaseModel):
    discriminator_loss_weight: float = 1.0
    embeddings_sharing: Optional[Literal["deberta", "electra"]] = None
    mask_ratio: float = 0.15


class RTDTrainingModel(pl.LightningModule):
    def __init__(
        self,
        discriminator: transformers.PreTrainedModel,
        generator: transformers.PreTrainedModel,
        mask_token_index: int,
        vocabulary_size: int,
        training_config: Optional[TrainConfig] = None,
        task_config: Optional[RTDTaskConfig] = None,
    ):
        super().__init__()
        if training_config is not None:
            self.training_config = training_config
        else:
            self.training_config = TrainConfig()
        if task_config is not None:
            self.task_config = task_config
        else:
            self.task_config = RTDTaskConfig()
        logger.info(f"RTD trainer config: {self.training_config}")
        logger.info(f"RTD task config: {self.task_config}")
        self.mask_token_index = mask_token_index
        self.vocabulary_size = vocabulary_size

        self.generator_accuracy = MaskedAccuracy()
        self.discriminator_accuracy = MaskedAccuracy()
        self.generator = generator
        self.discriminator = discriminator
        self.max_len = min(
            getattr(generator.config, "max_position_embeddings", float("inf")),
            getattr(discriminator.config, "max_position_embeddings", float("inf")),
        )

        self.save_hyperparameters("training_config", "task_config")

    # type: ignore[override]
    def forward(
        self,
        attention_mask: torch.Tensor,
        mlm_labels: torch.Tensor,
        internal_tokens_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> RTDOutput:
        generator_output = cast(
            transformers.modeling_outputs.MaskedLMOutput,
            self.generator(
                input_ids=tokens,
                attention_mask=attention_mask,
                labels=mlm_labels,
                token_type_ids=token_type_ids,
                return_dict=True,
            ),
        )

        generator_predictions = torch.argmax(generator_output.logits, dim=-1).detach()
        rtd_labels = generator_predictions.ne(tokens).to(torch.long)
        rtd_labels.masked_fill(internal_tokens_mask, -100)
        rtd_labels.masked_fill(attention_mask, -100)

        discriminator_output = cast(
            transformers.modeling_outputs.TokenClassifierOutput,
            self.discriminator(
                input_ids=tokens,
                attention_mask=attention_mask,
                labels=rtd_labels,
                token_type_ids=token_type_ids,
                return_dict=True,
            ),
        )
        discriminator_predictions = discriminator_output.logits.argmax(dim=-1)

        return RTDOutput(
            discriminator_output=discriminator_output,
            discriminator_predictions=discriminator_predictions,
            generator_output=generator_output,
            generator_predictions=generator_predictions,
            rtd_labels=rtd_labels,
        )

    # type: ignore[override]
    def training_step(
        self, batch: zeldarose.data.TextBatch, batch_idx: int
    ) -> torch.Tensor:
        tokens, attention_mask, internal_tokens_mask, token_type_ids = batch
        with torch.no_grad():
            masked = mask_tokens(
                tokens=tokens,
                keep_mask=internal_tokens_mask,
                label_mask_index=-100,
                mask_ratio=self.task_config.mask_ratio,
                input_mask_index=self.mask_token_index,
            )

        outputs: RTDOutput = self(
            tokens=masked.inputs,
            attention_mask=attention_mask,
            mlm_labels=masked.labels,
            internal_tokens_mask=internal_tokens_mask,
            token_type_ids=token_type_ids,
        )

        combined_loss = cast(
            torch.Tensor, outputs.generator_output.loss
        ) + self.task_config.discriminator_loss_weight * cast(
            torch.Tensor, outputs.discriminator_output.loss
        )

        with torch.no_grad():
            generator_perplexity = torch.exp(
                cast(torch.Tensor, outputs.generator_output.loss)
            )
            self.generator_accuracy(outputs.generator_predictions, masked.labels)

            self.log(
                "train/generator_loss",
                cast(torch.Tensor, outputs.generator_output.loss),
                reduce_fx=torch.mean,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/generator_perplexity",
                generator_perplexity,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/generator_accuracy",
                self.generator_accuracy,
                on_epoch=True,
            )

            self.discriminator_accuracy(
                outputs.discriminator_predictions, outputs.rtd_labels
            )
            self.log(
                "train/discriminator_loss",
                cast(torch.Tensor, outputs.discriminator_output.loss),
                reduce_fx=torch.mean,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/discriminator_accuracy",
                self.discriminator_accuracy,
                on_epoch=True,
            )

            self.log(
                "train/combined_loss",
                combined_loss,
                reduce_fx=torch.mean,
                on_epoch=True,
                sync_dist=True,
            )

        return combined_loss

    def validation_step(self, batch: zeldarose.data.TextBatch, batch_idx: int):
        tokens, attention_mask, internal_tokens_mask, token_type_ids = batch
        with torch.no_grad():
            masked = mask_tokens(
                tokens=tokens,
                keep_mask=internal_tokens_mask,
                label_mask_index=-100,
                mask_ratio=self.task_config.mask_ratio,
                input_mask_index=self.mask_token_index,
            )

        outputs: RTDOutput = self(
            tokens=masked.inputs,
            attention_mask=attention_mask,
            mlm_labels=masked.labels,
            internal_tokens_mask=internal_tokens_mask,
            token_type_ids=token_type_ids,
        )
        generator_perplexity = torch.exp(
            cast(torch.Tensor, outputs.generator_output.loss)
        )
        self.generator_accuracy(outputs.generator_predictions, masked.labels)

        self.log(
            "validation/generator_loss",
            cast(torch.Tensor, outputs.generator_output.loss),
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "validation/generator_perplexity",
            generator_perplexity,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "validation/generator_accuracy",
            self.generator_accuracy,
            on_epoch=True,
        )

        self.discriminator_accuracy(
            outputs.discriminator_predictions, outputs.rtd_labels
        )
        self.log(
            "validation/discriminator_loss",
            cast(torch.Tensor, outputs.discriminator_output.loss),
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "validation/discriminator_accuracy",
            self.discriminator_accuracy,
            on_epoch=True,
        )

    def configure_callbacks(self):
        callbacks: List[pl.Callback] = []
        if self.task_config.embeddings_sharing == "electra":
            callbacks.append(
                ShareTransformersEmbeddingsCallback(
                    leader=self.generator, follower=self.discriminator
                )
            )
        elif self.task_config.embeddings_sharing == "deberta":
            callbacks.append(
                OneWayShareTransformersEmbeddingsCallback(
                    leader=self.generator, follower=self.discriminator
                )
            )
        return callbacks

    def configure_optimizers(self):
        named_parameters = [
            (n, p)
            for model in [self.generator, self.discriminator]
            for n, p in model.named_parameters()
        ]
        if self.training_config.weight_decay is not None:
            no_decay = ["bias", "LayerNorm.weight"]
            decay_rate = self.training_config.weight_decay
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in named_parameters
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": decay_rate,
                },
                {
                    "params": [
                        p
                        for n, p in named_parameters
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            decay_rate = 0.0
            optimizer_grouped_parameters = [
                {
                    "params": [p for _, p in named_parameters],
                    "weight_decay": 0.0,
                },
            ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=self.training_config.betas,
            lr=self.training_config.learning_rate,
            eps=self.training_config.epsilon,
            weight_decay=decay_rate,
        )
        if self.training_config.lr_decay_steps:
            if self.training_config.lr_decay_steps > 2 * self.trainer.max_steps:
                logger.warning(
                    f"Asked for {self.training_config.lr_decay_steps} LR decay steps"
                    f" but the model will only be trained for {self.trainer.max_steps} steps"
                    ", this might be an oversight."
                )
            if self.training_config.lr_decay_steps == -1:
                num_training_steps = (
                    self.trainer.max_steps - self.training_config.warmup_steps
                )
                logger.info(
                    f"Number of lr decay steps set at {num_training_steps} since -1 was asked"
                )
            else:
                num_training_steps = self.training_config.lr_decay_steps

            schedule = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.training_config.warmup_steps,
                num_training_steps=num_training_steps
                + self.training_config.warmup_steps,
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

    @rank_zero_only
    def save_transformer(
        self,
        save_dir: pathlib.Path,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
    ):
        """Save the wrapped transformer model."""
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving models to {save_dir}")
        for model_name, model in [
            ("discriminator", self.discriminator),
            ("generator", self.generator),
        ]:
            save_subdir = save_dir / model_name
            model.save_pretrained(str(save_subdir))
            if tokenizer is not None:
                logger.info(f"Saving tokenizer to {save_subdir}")
                tokenizer.save_pretrained(
                    str(save_subdir), legacy_format=not tokenizer.is_fast
                )


def get_training_model(
    model_config_path: Optional[str],
    pretrained_model: Optional[str],
    task_config_dict: Optional[Dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    training_config: TrainConfig,
) -> RTDTrainingModel:
    if task_config_dict is not None:
        task_config = RTDTaskConfig.parse_obj(task_config_dict)
    else:
        task_config = RTDTaskConfig()

    if (mask_token_index := getattr(tokenizer, "mask_token_id", None)) is None:
        mask_token_index = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    vocabulary_size = tokenizer.vocab_size

    if pretrained_model is not None:
        pretrained_discriminator, pretrained_generator = pretrained_model.split(",")
        logger.info(f"Loading pretrained discriminator {pretrained_discriminator!r}")
        discriminator = transformers.AutoModelForTokenClassification.from_pretrained(
            pretrained_discriminator
        )
        logger.info(f"Loading pretrained generator {pretrained_generator!r}")
        generator = transformers.AutoModelForMaskedLM.from_pretrained(
            pretrained_generator
        )
    elif model_config_path is not None:
        (
            discriminator_config_path,
            generator_config_path,
        ) = model_config_path.split(",")
        logger.info(f"Loading discriminator config {discriminator_config_path!r}")
        discriminator_config = transformers.AutoConfig.from_pretrained(
            discriminator_config_path
        )
        if (
            vocabulary_size is not None
            and discriminator_config.vocab_size != vocabulary_size
        ):
            logger.warning(
                f"Vocabulary size mismatch between discriminator config ({discriminator_config.vocab_size})"
                f" and pretrained tokenizer ({vocabulary_size}), using {vocabulary_size}."
            )
            discriminator_config.vocab_size = vocabulary_size
        logger.info(f"Loading generator config {generator_config_path,!r}")
        generator_config = transformers.AutoConfig.from_pretrained(
            generator_config_path
        )
        if (
            vocabulary_size is not None
            and generator_config.vocab_size != vocabulary_size
        ):
            logger.warning(
                f"Vocabulary size mismatch between generator config ({generator_config.vocab_size})"
                f" and pretrained tokenizer ({vocabulary_size}), using {vocabulary_size}."
            )
            generator_config.vocab_size = vocabulary_size
        logger.info("Generating discriminator from config")
        discriminator = transformers.AutoModelForMaskedLM.from_config(
            discriminator_config
        )
        logger.info("Generating generator from config")
        generator = transformers.AutoModelForMaskedLM.from_config(generator_config)
    else:
        raise ValueError("You must provide either pretrained models or model configs")
    # 2 is the default
    # (<https://github.com/huggingface/transformers/blob/e68c3756fea7c811d02b8470539ae17ec3ec0e71/src/transformers/configuration_utils.py#L302>)
    # but it could have been overriden
    if not discriminator.config.num_labels == 2:
        raise ValueError(
            f"RTD discriminator must have exactly 2 tokens classes, found {discriminator.config.num_labels}"
        )
    if discriminator.config.vocab_size != generator_config.vocab_size:
        raise ValueError(
            "Vocabulary size mismatch between discriminator and generator:"
            f" {discriminator.config.vocab_size} vs {generator_config.vocab_size}"
        )

    logger.info("Creating RTD training model")

    logger.debug(f"Mask token index: {mask_token_index}")
    training_model = RTDTrainingModel(
        discriminator=discriminator,
        generator=generator,
        mask_token_index=mask_token_index,
        vocabulary_size=vocabulary_size,
        task_config=task_config,
        training_config=training_config,
    )

    return training_model
