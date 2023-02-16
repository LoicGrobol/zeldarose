import pathlib
from typing import Any, List, cast, Dict, NamedTuple, Optional, TYPE_CHECKING, Union

import pydantic
import torch
import torch.jit
import torch.utils.data
import transformers
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from lightning_utilities.core.rank_zero import rank_zero_only
from torchmetrics import SacreBLEUScore

import zeldarose.datasets.mbart
from zeldarose.common import TrainConfig, TrainingModule


if TYPE_CHECKING:
    import transformers.modeling_outputs


class InfilledSent(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


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
                    while pos + offset < len(sent) and offset < jump[pos]:
                        if keep[pos + offset]:
                            break
                        offset += 1
                    pos += offset
            else:
                current.append(sent[pos].item())
                pos += 1
        res.append(input_ids.new_tensor(current))
    lengths = input_ids.new_tensor([len(t) for t in res])
    # Not optimal, but concise
    attention_mask = torch.arange(lengths.max().item()).unsqueeze(0).lt(lengths.unsqueeze(1))
    return InfilledSent(
        input_ids=pad_sequence(res, batch_first=True, padding_value=padding_id),
        attention_mask=attention_mask,
    )


class MBartTaskConfig(pydantic.BaseModel):
    change_ratio: float = 0.3
    denoise_langs: Optional[List[str]]
    poisson_lambda: float = 3.0
    source_langs: Optional[List[str]]
    target_langs: Optional[List[str]]


class MBartTrainingModel(TrainingModule):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        mask_token_index: int,
        padding_token_index: int,
        vocabulary_size: int,
        task_config: MBartTaskConfig,
        tokenizer: transformers.PreTrainedTokenizerFast,
        training_config: Optional[TrainConfig] = None,
    ):
        super().__init__()
        if training_config is not None:
            self.training_config = training_config
        else:
            self.training_config = TrainConfig()
        self.task_config = task_config
        logger.info(f"mBART trainer config: {self.training_config}")
        logger.info(f"mBART task config: {self.task_config}")
        self.mask_token_index = mask_token_index
        self.pad_token_index = padding_token_index
        self.sacrebleu_score = SacreBLEUScore()
        self.vocabulary_size = vocabulary_size

        self.model = model
        self.tokenizers = tokenizer

        self.save_hyperparameters("training_config", "task_config")

    def forward(  # type: ignore[override]
        self,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        tokens: torch.Tensor,
        labels: torch.Tensor,
    ) -> "transformers.modeling_outputs.Seq2SeqLMOutput":
        output = self.model(
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            input_ids=tokens,
            labels=labels,
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
                tokens=noisy.input_ids,
                decoder_input_ids=denoise.decoder_input_ids,
                labels=denoise.labels,
                attention_mask=noisy.attention_mask,
            )

            denoise_loss = denoise_outputs.loss
            denoise_batch_size = denoise.input_ids.shape[0]

        else:
            denoise_loss = torch.zeros(1, device=self.device)
            denoise_batch_size = 0

        self.log(
            "train/denoise_loss",
            denoise_loss,
            batch_size=denoise_batch_size,
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )

        if translate is not None:
            translate_outputs = self(
                tokens=translate.input_ids,
                decoder_input_ids=translate.decoder_input_ids,
                labels=translate.labels,
                attention_mask=translate.attention_mask,
            )

            translate_loss = translate_outputs.loss
            translate_batch_size = translate.input_ids.shape[0]
        else:
            translate_loss = torch.zeros(1, device=self.device)
            translate_batch_size = 0

        self.log(
            "train/translate_loss",
            translate_loss,
            batch_size=translate_batch_size,
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )

        loss = translate_loss + denoise_loss

        batch_size = denoise_batch_size + translate_batch_size
        self.log(
            "train/loss",
            loss,
            batch_size=batch_size,
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: zeldarose.datasets.mbart.TwoMBartBatches, batch_idx: int):  # type: ignore[override]
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
                tokens=noisy.input_ids,
                decoder_input_ids=denoise.decoder_input_ids,
                labels=denoise.labels,
                attention_mask=noisy.attention_mask,
            )

            denoise_loss = denoise_outputs.loss
            denoise_batch_size = denoise.input_ids.shape[0]

        else:
            denoise_loss = torch.zeros(1, device=self.device)
            denoise_batch_size = 0

        self.log(
            "validation/denoise_loss",
            denoise_loss,
            batch_size=denoise_batch_size,
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )

        if translate is not None:
            translate_outputs = self(
                tokens=translate.input_ids,
                decoder_input_ids=translate.decoder_input_ids,
                labels=translate.labels,
                attention_mask=translate.attention_mask,
            )

            translate_loss = translate_outputs.loss
            translate_batch_size = translate.input_ids.shape[0]

            generated_id: List[torch.Tensor] = []
            for input_ids, decoder_input_ids in zip(
                translate.input_ids, translate.decoder_input_ids
            ):
                generated_id.append(
                    self.model.generate(
                        input_ids=input_ids.unsqueeze(0), forced_bos_token_id=decoder_input_ids[0], num_beams=4
                    )[0]
                )
            generated_txt = self.tokenizers.batch_decode(generated_id, skip_special_tokens=True)
            self.sacrebleu_score(generated_txt, translate.tgt_text)

        else:
            translate_loss = torch.zeros(1, device=self.device)
            translate_batch_size = 0

        self.log(
            "validation/translate_loss",
            denoise_loss,
            batch_size=translate_batch_size,
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("validation/sacrebleu", self.sacrebleu_score, on_epoch=True, on_step=False)

        loss = translate_loss + denoise_loss
        batch_size = denoise_batch_size + translate_batch_size
        self.log(
            "validation/loss",
            loss,
            batch_size=batch_size,
            reduce_fx=torch.mean,
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
    ) -> zeldarose.datasets.mbart.MBartDataModule:
        if (max_length := getattr(self.model.config, "max_position_embeddings")) is None:
            max_length = tokenizer.max_len_single_sentence
        else:
            # FIXME: we shouldn't need num_special_tokens_to_add here
            max_length = min(
                tokenizer.max_len_single_sentence,
                max_length - tokenizer.num_special_tokens_to_add(pair=False),
            )

        return zeldarose.datasets.mbart.MBartDataModule(
            data_dir=data_dir,
            denoise_langs=(
                self.task_config.denoise_langs if self.task_config.denoise_langs is not None else []
            ),
            loader_batch_size=loader_batch_size,
            max_length=max_length,
            num_workers=num_workers,
            source_langs=(
                self.task_config.source_langs if self.task_config.source_langs is not None else []
            ),
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            train_path=train_path,
            target_langs=(
                self.task_config.target_langs if self.task_config.target_langs is not None else []
            ),
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
    task_config: Dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizerFast,
    training_config: TrainConfig,
) -> MBartTrainingModel:
    _task_config = MBartTaskConfig.parse_obj(task_config)

    if (
        mask_token_index := cast(Union[int, None], getattr(tokenizer, "mask_token_id", None))
    ) is None:
        mask_token_index = cast(int, tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
    vocabulary_size = tokenizer.vocab_size

    if pretrained_model is not None:
        logger.info(f"Loading pretrained model {pretrained_model!r}")
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
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
        model = transformers.AutoModelForSeq2SeqLM.from_config(_model_config)
    else:
        raise ValueError("You must provide either a pretrained model or a model config")

    logger.info("Creating MLM training model")

    logger.debug(f"Mask token index: {mask_token_index}")
    training_model = MBartTrainingModel(
        model=model,
        mask_token_index=mask_token_index,
        padding_token_index=cast(int, tokenizer.pad_token_id),
        vocabulary_size=vocabulary_size,
        task_config=_task_config,
        tokenizer=tokenizer,
        training_config=training_config,
    )

    return training_model
