import os
import pathlib

from typing import Union, cast, List, NamedTuple, Optional, Sequence, TypedDict

import datasets
import pytorch_lightning as pl
import torch
import torch.utils.data
import transformers

from loguru import logger
from torch.nn.utils.rnn import pad_sequence


# NOTE: this also caches the raw and encoded dataset in HF_DATASETS_CACHE, which is different from OUR cache
# end users can still manually set HF_DATASETS_CACHE if e.g. their home has a small quota
def encode_dataset(
    save_path: pathlib.Path,
    text_path: Union[pathlib.Path, str],
    tokenizer: transformers.PreTrainedTokenizer,
    tokenizer_name: str,
    max_length: Optional[int] = None,
):
    logger.info(f"Loading data from {text_path}")
    try:
        full_dataset = datasets.load_dataset("text", data_files=str(text_path), split="train")
    except FileNotFoundError as e:
        if isinstance(text_path, str):
            dataset_name, dataset_config, dataset_split = text_path.split(":")
            full_dataset = datasets.load_dataset(
                dataset_name,
                name=dataset_config if dataset_config else None,
                split=dataset_split if dataset_split else None,
            )
        else:
            raise e

    def dataset_filter(example: str) -> bool:
        return len(example) > 0 and not example.isspace()

    raw_dataset = cast(datasets.Dataset, full_dataset,).filter(
        dataset_filter,
        input_columns="text",
    )

    def tokenizer_transform(examples: Sequence[str]):
        return tokenizer(
            examples,
            add_special_tokens=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            truncation=True,
        )

    logger.info("Tokenizing")
    encoded_dataset = raw_dataset.map(
        function=tokenizer_transform,
        batched=True,
        desc="Tokenizing",
        input_columns="text",
    )
    logger.info(f"Saving dataset to {save_path}")
    # FIXME: this causes an obscure crash when two instance want to access the same --cache-dir
    encoded_dataset.save_to_disk(save_path)


class TextBatch(NamedTuple):
    """A batch of text for self-supervised tasks.

    Attributes
    ==========

    :tokens: A batch of encoded (with special tokens) and padded tokens.
    :attention_mask: A boolean mask, `True` for content and special tokens, `False` for padding.
    :internal_tokens_mask: A boolean mask, `True` for content tokens, `False` for padding and
      special tokens.
    :token_type_ids: The `token_type_ids` tensor needed internally for hugginface transformers
      implementations.
    """

    tokens: torch.Tensor
    attention_mask: torch.Tensor
    internal_tokens_mask: torch.Tensor
    token_type_ids: torch.Tensor


class EncodedSample(TypedDict):
    attention_mask: List[int]
    input_ids: List[int]
    special_tokens_mask: List[int]
    text: str


class TextLoader(torch.utils.data.DataLoader[TextBatch]):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset[TextBatch],
        tokenizer: transformers.PreTrainedTokenizer,
        *args,
        **kwargs,
    ):
        self.dataset: torch.utils.data.Dataset[TextBatch]
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = self.collate
        super().__init__(dataset, *args, **kwargs)
        self.tokenizer = tokenizer
        padding_value = self.tokenizer.pad_token_id
        if padding_value is None:
            raise ValueError("Tokenizers without a padding id are not supported")
        self._padding_value = padding_value

    def collate(self, batch: Sequence[EncodedSample]) -> TextBatch:
        # NOTE: we have to pad/batch manually instead of deferring to huggingface, since the fast
        # tokenizers can't take pre-encoded inputs (yet?)
        padded_batch = pad_sequence(
            [torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch],
            batch_first=True,
            padding_value=self._padding_value,
        )
        padding_mask = padded_batch.eq(self._padding_value)
        # FIXME: Is the next line general enough?
        token_type_ids = torch.zeros_like(padded_batch)
        # We only deal with single sequences here
        attention_mask = padding_mask.logical_not()

        special_tokens_mask = pad_sequence(
            [torch.tensor(sample["special_tokens_mask"], dtype=torch.bool) for sample in batch],
            batch_first=True,
            padding_value=False,
        )
        internal_tokens_mask = special_tokens_mask.logical_or(padding_mask)
        return TextBatch(
            tokens=padded_batch,
            attention_mask=attention_mask,
            internal_tokens_mask=internal_tokens_mask,
            token_type_ids=token_type_ids,
        )


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        loader_batch_size: int,
        num_workers: int,
        tokenizer: transformers.PreTrainedTokenizerBase,
        tokenizer_name: str,
        train_text: Union[str, pathlib.Path],
        data_dir: Optional[pathlib.Path] = None,
        max_length: Optional[int] = None,
        val_text: Optional[Union[str, pathlib.Path]] = None,
    ):
        super().__init__()
        self.loader_batch_size = loader_batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.train_text = train_text
        self.val_text = val_text

        if data_dir is None:
            self.data_dir = pathlib.Path(self.train_text).parent
            if not self.data_dir.exists():
                raise ValueError(
                    "You must provide a cache path if you are loading a dataset from an url."
                )
        else:
            self.data_dir = data_dir

        self.train_dataset_path = self.data_dir / "train_set"
        self.train_dataset_path.mkdir(exist_ok=True, parents=True)

        self.val_dataset_path: Optional[pathlib.Path]
        if self.val_text is not None:
            self.val_dataset_path = self.data_dir / "val_set"
            self.val_dataset_path.mkdir(exist_ok=True, parents=True)
        else:
            self.val_dataset_path = None

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # This should'nt be needed since this method should only be called on rank 0, but since it
        # is called in every process AND before DDP init (at least in SLURM) we have to enforce it
        # ourselves
        if os.environ.get("SLURM_PROCID", "0") == "0":
            encode_dataset(
                max_length=self.max_length,
                save_path=self.train_dataset_path,
                text_path=self.train_text,
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
            )
            if self.val_dataset_path is not None:
                encode_dataset(
                    max_length=self.max_length,
                    save_path=self.val_dataset_path,
                    text_path=self.val_text,
                    tokenizer=self.tokenizer,
                    tokenizer_name=self.tokenizer_name,
                )

    def setup(self, stage=None):
        self.train_dataset = datasets.load_from_disk(self.train_dataset_path)
        if self.val_dataset_path is not None:
            self.val_dataset = datasets.load_from_disk(self.val_dataset_path)

    def train_dataloader(self):
        return TextLoader(
            self.train_dataset,
            batch_size=self.loader_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            tokenizer=self.tokenizer,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return TextLoader(
            self.val_dataset,
            batch_size=self.loader_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            tokenizer=self.tokenizer,
        )
