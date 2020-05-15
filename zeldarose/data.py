import functools
import pathlib
import pickle  # nosec

from typing import List, NamedTuple, Sequence

import torch
import torch.utils.data
import tqdm
import transformers

from loguru import logger
from torch.nn.utils.rnn import pad_sequence


# TODO: this is in-memory, if need be we can change it for a memmap (possibly lmdb) backed dataset
class TextDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` that stores a tokenized and encoded raw text as examples of
    `block_size` tokens.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        text_path: pathlib.Path,
        block_size: int = 512,
        model_name: str = "lm",
        overwrite_cache: bool = False,
    ):
        if not text_path.is_file():
            raise ValueError(f"{text_path} is not a valid text file.")
        self.tokenizer = tokenizer

        try:
            self.block_size = block_size - tokenizer.num_special_tokens_to_add(
                pair=False
            )
        except AttributeError:
            # Non-fast tokenizers
            self.block_size = block_size - (
                tokenizer.max_len - tokenizer.max_len_single_sentence
            )

        cached_features_file = (
            text_path.parent / f"{model_name}_{block_size}_{text_path.stem}_cache.pt"
        )
        if cached_features_file.exists() and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, "rb") as handle:
                self.examples = torch.load(handle)
        else:
            logger.info(f"Creating features from dataset file at {text_path}")

            self.examples = self.load_file(text_path)

            logger.info(f"Saving features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                torch.save(
                    self.examples, handle, pickle_protocol=pickle.HIGHEST_PROTOCOL,
                )

    def load_file(
        self, text_path: pathlib.Path, read_size_hint: int = 2 ** 20
    ) -> List[torch.Tensor]:
        """Tokenize, encode and split a raw text in blocks.

        **Note**: This reads files by chunks of about 1MiB which seemed to work best for the roberta
        fast tokenizer on my setup it might benefit from fine-tuning. The tradeof is: using larger
        chunks will reduce the number of calls to `transformers.encode` but pass it larger texts, so
        what is most efficient depends on the implementation of this method. Of course a larger
        chunk size will also be heavier on the RAM (and significantly so for some tokenizers).
        """
        examples: List[torch.Tensor] = []
        #  Try to avoid list resize
        buffer = [0] * self.block_size
        offset = 0
        with open(text_path, "rb") as in_stream:
            pbar = tqdm.tqdm(
                desc="Loading input data",
                unit="iB",
                total=text_path.stat().st_size,
                leave=False,
                unit_divisor=1024,
                unit_scale=True,
                mininterval=1,
            )

            # Process by chunks instead of loading everything in RAM
            # We still read by line to ensure we don't end up in the middle of a unicode
            # characterload_file(text_path)
            for raw_lines in iter(
                functools.partial(in_stream.readlines, read_size_hint), []
            ):
                chunk = b"".join(raw_lines)
                decoded = chunk.decode(encoding="utf-8")
                # This is the best choice for fast tokenizers
                encoded = self.tokenizer.encode(decoded, add_special_tokens=False)
                new_offset = len(encoded) + offset

                # Top up
                if new_offset >= self.block_size:
                    n_top_up = self.block_size - offset
                    buffer[offset:] = encoded[:n_top_up]
                    examples.append(
                        torch.tensor(
                            self.tokenizer.build_inputs_with_special_tokens(buffer),
                            dtype=torch.long,
                        ),
                    )
                    # Flush whole blocks
                    remaining = len(encoded) - n_top_up
                    while remaining >= self.block_size:
                        next_remaining = remaining - self.block_size
                        # We write it this way because of the stupid `next_remaning==0` case
                        block = encoded[
                            len(encoded) - remaining : len(encoded) - next_remaining
                        ]
                        examples.append(
                            self.tokenizer.build_inputs_with_special_tokens(block),
                        )
                        remaining = next_remaining
                    buffer[:remaining] = encoded[len(encoded) - remaining :]
                    offset = remaining
                else:
                    buffer[offset:new_offset] = encoded
                    offset = new_offset
                pbar.n = in_stream.tell()
                pbar.update(0)
            pbar.close()
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


class LineByLineTextDataset(TextDataset):
    """A `TextDataset` that store text blocks given line-by-line instead of a single blob.

    Lines that are too long are simply truncated, so it is *your* responsibility to ensure that
    there not many of these or use a `TextDataset` instead)
    """

    def load_file(
        self, text_path: pathlib.Path, read_size_hint: int = 2 ** 20
    ) -> List[torch.Tensor]:
        examples: List[torch.Tensor] = []
        with open(text_path, "rb") as in_stream:
            pbar = tqdm.tqdm(
                desc="Loading input data",
                unit="iB",
                total=text_path.stat().st_size,
                leave=False,
                unit_divisor=1024,
                unit_scale=True,
                mininterval=1,
            )
            for raw_lines in iter(
                functools.partial(in_stream.readlines, read_size_hint), []
            ):
                decoded = [l.decode("utf-8") for l in raw_lines]
                encoded = self.tokenizer.batch_encode_plus(
                    decoded,
                    add_special_tokens=True,
                    max_length=self.tokenizer.max_len_single_sentence,
                )["input_ids"]
                examples.extend(encoded)
                pbar.n = in_stream.tell()
                pbar.update(0)
            pbar.close()
            return examples


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


class TextLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: TextDataset, *args, **kwargs):
        self.dataset: TextDataset
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = self.collate
        super().__init__(dataset, *args, **kwargs)
        padding_value = getattr(self.dataset.tokenizer, "pad_token_id")
        if padding_value is None:
            padding_value = self.dataset.tokenizer.convert_tokens_to_ids(
                self.dataset.tokenizer.padding_value
            )
        self._padding_value = padding_value

    def collate(self, batch: Sequence[torch.Tensor]) -> TextBatch:
        padded_batch = pad_sequence(
            batch, batch_first=True, padding_value=self._padding_value
        )
        padding_mask = padded_batch.eq(self._padding_value)
        # FIXME: Is the next line general enough?
        # We only deal with single sequences here
        token_type_ids = torch.zeros_like(padded_batch)
        attention_mask = padding_mask.logical_not()

        special_tokens_mask = pad_sequence(
            [
                torch.tensor(
                    self.dataset.tokenizer.get_special_tokens_mask(
                        val, already_has_special_tokens=True
                    ),
                    dtype=torch.bool,
                )
                for val in batch
            ],
            batch_first=True,
            padding_value=0,
        )
        internal_tokens_mask = special_tokens_mask | padding_mask
        return TextBatch(
            tokens=padded_batch,
            attention_mask=attention_mask,
            internal_tokens_mask=internal_tokens_mask,
            token_type_ids=token_type_ids,
        )
