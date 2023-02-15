import math
import pathlib
from typing import Any, cast, Dict, NamedTuple, Optional, TYPE_CHECKING, Union

import pydantic
import torch
import torch.jit
import torch.utils.data
import transformers
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import zeldarose.datasets.mbart
from zeldarose.common import MaskedAccuracy, TrainConfig, TrainingModule


if TYPE_CHECKING:
    import transformers.modeling_outputs


class InfilledSent(NamedTuple):
    input_ids: torch.Tensor
    mask: torch.Tensor


# NOTE(2023-02-14): There's a good chance that this is far from optimal and it should be revisited
# but remember that it is far from trivial and in particular, I don't see a way to avoid loops.
# FIXME(2023-02-15): we need to transform the special tokens mask too è_é
def infill_noise(
    change_ratio: float,
    input_ids: torch.Tensor,
    input_mask_id: int,
    poisson_lambda: float,
    padding_id: int = 0,
    keep_mask: Optional[torch.Tensor] = None,
) -> InfilledSent:
    """BART-like span masking with Poisson jumps

    
    ## Notes

    This is probably somewhat different from the original BART implementation, e.g. here we can have
    several consecutive <MASK> when two masked spans are contiguous. This does not, however,
    significantly change the nature of the task.
    """
    # Vector precomputations for what is easily vectorized
    lengths = input_ids.ne(input_mask_id).sum(dim=-1)
    replace_mask = torch.bernoulli(
        torch.full_like(input_ids, fill_value=change_ratio, dtype=torch.float)
    ).to(torch.bool)
    if keep_mask is not None:
        replace_mask = replace_mask.logical_and(keep_mask.logical_not())
    else:
        keep_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    all_jumps = torch.poisson(
        torch.full_like(input_ids, fill_value=poisson_lambda, dtype=torch.float)
    ).to(torch.long)
    res = []
    # TODO(2023-02-14): This is embarassingly parallel: the sentences can be treated independently
    # BIG LOOP OF HELL
    # You might believe this makes unnecessary copies but it actually Does Not!
    for sent, mask, jump, l, keep in zip(input_ids, replace_mask, all_jumps, lengths, keep_mask):
        current = []
        pos = 0
        while pos < l:
            if mask[pos]:
                current.append(input_mask_id)
                # If it was a 0-length jump, we have to advance to the next token normally
                if jump[pos] == 0:
                    current.append(sent[pos].item())
                    pos += 1
                else:
                    offset = 0
                    # NOTE(2023-02-14): This probably causes our jump distribution to be sub-Poisson (octopus?)
                    while offset < jump[pos]:
                        if keep[pos + offset]:
                            break
                        offset += 1
                    pos += offset
            else:
                current.append(sent[pos].item())
                pos += 1
        res.append(input_ids.new_tensor(current))
    lengths = input_ids.new_tensor([len(t) for t in res])
    mask = torch.arange(lengths.max().item()).unsqueeze(0).lt(lengths.unsqueeze(1))
    return InfilledSent(
        input_ids=pad_sequence(res, batch_first=True, padding_value=padding_id),
        mask=mask,
    )


class MBartTaskConfig(pydantic.BaseModel):
    change_ratio: float = 0.3
    poisson_lambda: float = 3.0


class MBartTrainingModel(TrainingModule):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        mask_token_index: int,
        padding_token_index: int,
        vocabulary_size: int,
        training_config: Optional[TrainConfig] = None,
        task_config: Optional[MBartTaskConfig] = None,
    ):
        super().__init__()
        if training_config is not None:
            self.training_config = training_config
        else:
            self.training_config = TrainConfig()
        if task_config is not None:
            self.task_config = task_config
        else:
            self.task_config = MBartTaskConfig()
        logger.info(f"mBART trainer config: {self.training_config}")
        logger.info(f"mBART task config: {self.task_config}")
        self.mask_token_index = mask_token_index
        self.pad_token_index = padding_token_index
        self.vocabulary_size = vocabulary_size

        self.accuracy = MaskedAccuracy()
        self.model = model

        self.save_hyperparameters("training_config", "task_config")

    def forward(  # type: ignore[override]
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        mlm_labels: torch.Tensor,
    ) -> "transformers.modeling_outputs.Seq2SeqLMOutput":
        output = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=mlm_labels,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        return output

    def training_step(  # type: ignore[override]
        self, batch: zeldarose.datasets.mbart.TwoMBartBatches, batch_idx: int
    ) -> torch.Tensor:
        denoise, translate = batch

        if denoise is not None:
            with torch.no_grad():
                noisy = infill_noise(
                    change_ratio=self.task_config.change_ratio,
                    input_ids=denoise.input_ids,
                    input_mask_id=self.mask_token_index,
                    keep_mask=denoise.special_tokens_mask,
                    padding_id=self.pad_token_index,
                    poisson_lambda=self.task_config.poisson_lambda,
                )

            denoise_outputs = self(
                tokens=noisy.inputs,
                attention_mask=attention_mask,
                mlm_labels=masked.labels,
                token_type_ids=token_type_ids,
            )

        loss = outputs.loss

        with torch.no_grad():
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
            sync_dist=True,
        )

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
                num_training_steps = self.trainer.max_steps - self.training_config.warmup_steps
                logger.info(
                    f"Number of lr decay steps set at {num_training_steps} since -1 was asked"
                )
            else:
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
        if (max_length := getattr(self.model.config, "max_position_embeddings")) is None:
            max_length = tokenizer.max_len_single_sentence
        else:
            # FIXME: we shouldn't need num_special_tokens_to_add here
            max_length = min(
                tokenizer.max_len_single_sentence,
                max_length - tokenizer.num_special_tokens_to_add(pair=False),
            )

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
        _task_config = MLMTaskConfig.parse_obj(task_config)
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