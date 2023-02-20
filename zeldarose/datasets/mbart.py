import os
import pathlib

from typing import (
    Collection,
    Generator,
    Mapping,
    Union,
    cast,
    List,
    NamedTuple,
    Optional,
    TypedDict,
)

import datasets
import jsonlines
import pytorch_lightning as pl
import torch
import torch.utils.data
import transformers

from datasets.fingerprint import Hasher
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

# Nouvo plan : un lecteur de jsonlines custom qui prédécoupe en source/target avec attribut src et
# tgt, on charge ça dans dataset pour le système de cache, puis dans le dataloader on sample et
# quelque part dans le trainmodule on ajoute le bruit utiliser
# <https://huggingface.co/docs/datasets/loading#python-generator> comme ça on peut streamer l'entrée


class DataLine(TypedDict):
    source: str
    target: str
    src_lang: str
    tgt_lang: str


def extract_from_jsonline(
    example: Union[Mapping[str, str], Mapping[str, Mapping[str, str]]],
    denoise_langs: Collection[str],
    source_langs: Collection[str],
    target_langs: Collection[str],
) -> Generator[DataLine, None, None]:
    # We deal with both top-level tranlatikons and 🤗's conventional format for this task
    example = cast(Mapping[str, str], example.get("translation", example))
    for dns_lang in denoise_langs:
        if (dns_str := example.get(dns_lang)) is None:
            continue
        yield {"source": dns_str, "target": dns_str, "src_lang": dns_lang, "tgt_lang": dns_lang}
    for src_lang in source_langs:
        if (src_str := example.get(src_lang)) is None:
            continue
        for tgt_lang in target_langs:
            if (tgt_str := example.get(src_lang)) is None:
                continue
            yield {"source": src_str, "target": tgt_str, "src_lang": src_lang, "tgt_lang": tgt_lang}


class EncodedSample(TypedDict):
    attention_mask: List[int]
    decoder_input_ids: List[int]
    input_ids: List[int]
    labels: List[int]
    src_lang: str
    src_text: str
    special_tokens_mask: List[int]
    tgt_lang: str
    tgt_text: str


# NOTE: this also caches the raw and encoded dataset in HF_DATASETS_CACHE, which is different from OUR cache
# end users can still manually set HF_DATASETS_CACHE if e.g. their home has a small quota
def encode_dataset(
    denoise_langs: Collection[str],
    save_path: pathlib.Path,
    source_langs: Collection[str],
    target_langs: Collection[str],
    text_path: Union[pathlib.Path, str],
    tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    tokenizer_name: str,
    max_length: Optional[int] = None,
):
    if not hasattr(tokenizer, "src_lang") or not hasattr(tokenizer, "tgt_lang"):
        raise ValueError(
            "The tokenizer used in mBART training must have `src_lang` and `tgt_lang` attributes."
        )

    logger.info(f"Loading data from {text_path}")

    with jsonlines.open(text_path) as in_stream:

        def gen():
            for example in in_stream:
                yield from extract_from_jsonline(
                    example=example,
                    denoise_langs=denoise_langs,
                    source_langs=source_langs,
                    target_langs=target_langs,
                )

        raw_dataset = datasets.Dataset.from_generator(gen)

    logger.info("Preprocessing dataset")

    def preprocess(example: DataLine) -> EncodedSample:
        tokenizer.src_lang = example["src_lang"]  # type: ignore
        tokenizer.tgt_lang = example["tgt_lang"]  # type: ignore
        tokenized = tokenizer(
            example["source"],
            text_target=example["target"],
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            truncation=True,
        )
        # NOTE(2023-02-12): This stuff with decoder_input_ids is what 🤗 transformers's
        # `prepare_decoder_input_ids_from_labels`/[`shift_tokens_right`](https://github.com/huggingface/transformers/blob/c836f77266be9ace47bff472f63caf71c0d11333/src/transformers/models/mbart/modeling_mbart.py#L62)
        # does for mBART, but it seems a bit weird See also 🤗 transformers issue
        # [#19500](https://github.com/huggingface/transformers/issues/19500).
        labels = cast(List[int], tokenized["labels"])
        decoder_input_ids = [labels[-1], *labels[:-1]]
        # No loss for the langid token
        labels[0] = -100
        return EncodedSample(
            attention_mask=tokenized.attention_mask,
            decoder_input_ids=decoder_input_ids,
            input_ids=tokenized.input_ids,
            labels=labels,
            src_lang=example["src_lang"],
            src_text=example["source"],
            special_tokens_mask=cast(List[int], tokenized["special_tokens_mask"]),
            tgt_lang=example["tgt_lang"],
            tgt_text=example["target"],
        )

    raw_fingerprint = raw_dataset._fingerprint  # type: ignore
    encoded_dataset = cast(datasets.Dataset, raw_dataset).map(
        preprocess,
        batched=False,
        desc="Tokenizing",
        remove_columns=["source", "target"],
        new_fingerprint=Hasher.hash(
            f"{raw_fingerprint}-{tokenizer_name}-{source_langs}×{target_langs}×{denoise_langs}-{max_length}"
        ),
    )
    logger.info(f"Saving dataset to {save_path}")
    # FIXME: this causes an obscure crash when two instances want to access the same --cache-dir
    encoded_dataset.save_to_disk(str(save_path))


class MBARTBatch(NamedTuple):
    attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor
    src_lang: List[str]
    src_text: List[str]
    special_tokens_mask: torch.Tensor
    tgt_lang: List[str]
    tgt_text: List[str]


class TwoMBartBatches(NamedTuple):
    denoise: Optional[MBARTBatch]
    translate: Optional[MBARTBatch]


class MBartLoader(torch.utils.data.DataLoader[EncodedSample]):
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

    def collate(self, batch: List[EncodedSample]) -> TwoMBartBatches:
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
        special_tokens_mask = pad_sequence(
            [torch.tensor(sample["special_tokens_mask"], dtype=torch.long) for sample in batch],
            batch_first=True,
            padding_value=0,
        )
        attention_mask = padded_input_ids.ne(self._padding_value)

        # NOTE(2023-02-15): doing it this way probably results in overpadding at some point
        denoise_indices = [i for i, s in enumerate(batch) if s["src_lang"] == s["tgt_lang"]]
        if denoise_indices:
            denoise_batch = MBARTBatch(
                attention_mask=attention_mask[denoise_indices],
                decoder_input_ids=padded_decoder_input_ids[denoise_indices],
                input_ids=padded_input_ids[denoise_indices],
                labels=padded_labels[denoise_indices],
                src_lang=[batch[i]["src_lang"] for i in denoise_indices],
                src_text=[batch[i]["src_text"] for i in denoise_indices],
                special_tokens_mask=special_tokens_mask[denoise_indices],
                tgt_lang=[batch[i]["tgt_lang"] for i in denoise_indices],
                tgt_text=[batch[i]["tgt_text"] for i in denoise_indices],
            )
        else:
            denoise_batch = None

        translate_indices = [i for i, s in enumerate(batch) if s["src_lang"] != s["tgt_lang"]]
        if translate_indices:
            translate_batch = MBARTBatch(
                attention_mask=attention_mask[translate_indices],
                decoder_input_ids=padded_decoder_input_ids[translate_indices],
                input_ids=padded_input_ids[translate_indices],
                labels=padded_labels[translate_indices],
                src_lang=[batch[i]["src_lang"] for i in translate_indices],
                src_text=[batch[i]["src_text"] for i in translate_indices],
                special_tokens_mask=special_tokens_mask[translate_indices],
                tgt_lang=[batch[i]["tgt_lang"] for i in translate_indices],
                tgt_text=[batch[i]["tgt_text"] for i in translate_indices],
            )
        else:
            translate_batch = None
        return TwoMBartBatches(denoise=denoise_batch, translate=translate_batch)


class MBartDataModule(pl.LightningDataModule):
    def __init__(
        self,
        denoise_langs: Collection[str],
        loader_batch_size: int,
        num_workers: int,
        source_langs: Collection[str],
        target_langs: Collection[str],
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        tokenizer_name: str,
        train_path: Union[str, pathlib.Path],
        data_dir: Optional[pathlib.Path] = None,
        max_length: Optional[int] = None,
        val_path: Optional[Union[str, pathlib.Path]] = None,
    ):
        super().__init__()
        self.denoise_langs = sorted(set(denoise_langs))
        self.loader_batch_size = loader_batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.source_langs = sorted(set(source_langs))
        self.target_langs = sorted(set(target_langs))
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
                denoise_langs=self.denoise_langs,
                max_length=self.max_length,
                save_path=self.train_dataset_path,
                source_langs=self.source_langs,
                target_langs=self.target_langs,
                text_path=self.train_path,
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
            )
            if self.val_path is not None:
                assert self.val_dataset_path is not None
                encode_dataset(
                    denoise_langs=self.denoise_langs,
                    max_length=self.max_length,
                    save_path=self.val_dataset_path,
                    source_langs=self.source_langs,
                    target_langs=self.target_langs,
                    text_path=self.train_path,
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
        return MBartLoader(
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
        return MBartLoader(
            cast(torch.utils.data.Dataset[EncodedSample], self.val_dataset),
            batch_size=self.loader_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            tokenizer=self.tokenizer,
        )
