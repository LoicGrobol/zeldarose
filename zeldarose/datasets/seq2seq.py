import os
import pathlib

from typing import Any, Dict, Mapping, Union, cast, List, NamedTuple, Optional, Sequence, TypedDict

import datasets
import pytorch_lightning as pl
import torch
import torch.utils.data
import transformers

from datasets.fingerprint import Hasher
from loguru import logger
from torch.nn.utils.rnn import pad_sequence


# NOTE: this also caches the raw and encoded dataset in HF_DATASETS_CACHE, which is different from OUR cache
# end users can still manually set HF_DATASETS_CACHE if e.g. their home has a small quota
def encode_dataset(
    save_path: pathlib.Path,
    source_column: str,
    target_column: str,
    text_path: Union[pathlib.Path, str],
    tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    tokenizer_name: str,
    max_length: Optional[int] = None,
    decoder_start_token_id: Optional[int] = None,
):
    if decoder_start_token_id is None:
        logger.info("No decoder start token id provided, the BOS token will be used instead.")
        decoder_start_token_id = tokenizer.bos_token_id
        assert decoder_start_token_id is not None
    logger.info(f"Loading data from {text_path}")
    try:
        raw_dataset = datasets.load_dataset("jsonl", data_files=str(text_path), split="train")
    except FileNotFoundError as e:
        if isinstance(text_path, str):
            dataset_name, dataset_config, dataset_split = text_path.split(":")
            raw_dataset = datasets.load_dataset(
                dataset_name,
                name=dataset_config if dataset_config else None,
                split=dataset_split if dataset_split else None,
            )
        else:
            raise e

    def preprocess(
        batch: Mapping[str, Sequence[Any]]
    ) -> transformers.tokenization_utils.BatchEncoding:
        tokenized = tokenizer(
            cast(List[str], batch[source_column]),
            text_target=cast(List[str], batch[target_column]),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            truncation=True,
        )
        # FIXME: probably what we do in mBART would also be appropriate here
        # NOTE(2023-02-12): This is NOT what 🤗 transformers's
        # `prepare_decoder_input_ids_from_labels`/[`shift_tokens_right`](https://github.com/huggingface/transformers/blob/c836f77266be9ace47bff472f63caf71c0d11333/src/transformers/models/mbart/modeling_mbart.py#L62)
        # does for mBART, but it seems more correct. See also 🤗 transformers issue
        # [#19500](https://github.com/huggingface/transformers/issues/19500).
        tokenized["decoder_input_ids"] = [
            [decoder_start_token_id, *labels[:-1]]
            for labels in cast(List[List[int]], tokenized["labels"])
        ]
        return tokenized

    logger.info("Preprocessing dataset")
    raw_fingerprint = raw_dataset._fingerprint  # type: ignore
    encoded_dataset = cast(datasets.Dataset, raw_dataset).map(
        preprocess,
        batched=True,
        desc="Tokenizing",
        remove_columns=[source_column, target_column],
        new_fingerprint=Hasher.hash(
            f"{raw_fingerprint}-{tokenizer_name}-{decoder_start_token_id}-{max_length}"
        ),
    )
    logger.info(f"Saving dataset to {save_path}")
    # FIXME: this causes an obscure crash when two instances want to access the same --cache-dir
    encoded_dataset.save_to_disk(str(save_path))


class Seq2SeqBatch(NamedTuple):
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    decoder_input_ids: torch.Tensor
    labels: torch.Tensor


class EncodedSample(TypedDict):
    attention_mask: List[int]
    input_ids: List[int]
    decoder_input_ids: List[int]
    labels: List[int]


class Seq2SeqLoader(torch.utils.data.DataLoader[EncodedSample]):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset[EncodedSample],
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        *args,
        **kwargs,
    ):
        self.dataset: torch.utils.data.Dataset[EncodedSample]
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = self.collate
        super().__init__(dataset, *args, **kwargs)
        self.tokenizer = tokenizer
        padding_value = self.tokenizer.pad_token_id
        if padding_value is None:
            raise ValueError("Tokenizers without a padding id are not supported")
        self._padding_value = padding_value

    # TODO(2023-02-12): this could be made more generic and put in a toolbox
    def collate(self, batch: Sequence[EncodedSample]) -> Seq2SeqBatch:
        # NOTE(2021-08-12): we have to pad/batch manually instead of deferring to 🤗, since the fast
        # tokenizers can't take pre-encoded inputs (yet?)
        padded_input_ids = pad_sequence(
            [torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch],
            batch_first=True,
            padding_value=self._padding_value,
        )
        padded_decoder_input_ids = pad_sequence(
            [torch.tensor(sample["decoder_input_ids"], dtype=torch.long) for sample in batch],
            batch_first=True,
            padding_value=self._padding_value,
        )
        padded_labels = pad_sequence(
            [torch.tensor(sample["labels"], dtype=torch.long) for sample in batch],
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = padded_input_ids.ne(self._padding_value)

        return Seq2SeqBatch(
            attention_mask=attention_mask,
            input_ids=padded_input_ids,
            decoder_input_ids=padded_decoder_input_ids,
            labels=padded_labels,
        )


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        loader_batch_size: int,
        num_workers: int,
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        tokenizer_name: str,
        train_path: Union[str, pathlib.Path],
        data_dir: Optional[pathlib.Path] = None,
        max_length: Optional[int] = None,
        source_column: str = "source",
        target_column: str = "target",
        val_path: Optional[Union[str, pathlib.Path]] = None,
    ):
        super().__init__()
        self.loader_batch_size = loader_batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.source_column = source_column
        self.target_column = target_column
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.train_path = train_path
        self.val_path = val_path

        if data_dir is None:
            self.data_dir = pathlib.Path(self.train_path).parent
            if not self.data_dir.exists():
                raise ValueError(
                    "You must provide a cache path if you are loading a dataset from an url."
                )
        else:
            self.data_dir = data_dir

        self.train_dataset_path = self.data_dir / "train_set"
        self.train_dataset_path.mkdir(exist_ok=True, parents=True)

        self.val_dataset_path: Optional[pathlib.Path]
        if self.val_path is not None:
            self.val_dataset_path = self.data_dir / "val_set"
            self.val_dataset_path.mkdir(exist_ok=True, parents=True)
        else:
            self.val_dataset_path = None

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # NOTE (2021-08-12): This should'nt be needed since this method should only be called on rank 0, but since it
        # is called in every process AND before DDP init (at least in SLURM) we have to enforce it
        # ourselves
        # TODO (2023-01-07): see if the note above is still true
        if os.environ.get("SLURM_PROCID", "0") == "0":
            encode_dataset(
                max_length=self.max_length,
                save_path=self.train_dataset_path,
                source_column=self.source_column,
                target_column=self.target_column,
                text_path=self.train_path,
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
            )
            if self.val_path is not None:
                assert self.val_dataset_path is not None
                encode_dataset(
                    max_length=self.max_length,
                    save_path=self.val_dataset_path,
                    source_column=self.source_column,
                    target_column=self.target_column,
                    text_path=self.val_path,
                    tokenizer=self.tokenizer,
                    tokenizer_name=self.tokenizer_name,
                )

    def setup(self, stage=None):
        self.train_dataset = cast(
            datasets.Dataset, datasets.load_from_disk(str(self.train_dataset_path))
        )
        if self.val_dataset_path is not None:
            self.val_dataset = cast(
                datasets.Dataset, datasets.load_from_disk(str(self.val_dataset_path))
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        # FIXME(2023-02-07): that cast hereunder is wrong, self.train_dataset is **not** a torch Dataset
        return Seq2SeqLoader(
            cast(torch.utils.data.Dataset[EncodedSample], self.train_dataset),
            batch_size=self.loader_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            tokenizer=self.tokenizer,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        # FIXME(2023-02-07): that cast hereunder is wrong, self.val_dataset is **not** a torch Dataset
        return Seq2SeqLoader(
            cast(torch.utils.data.Dataset[EncodedSample], self.val_dataset),
            batch_size=self.loader_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            tokenizer=self.tokenizer,
        )
