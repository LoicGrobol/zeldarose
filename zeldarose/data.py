import pathlib
import pickle  # nosec

from typing import List, NamedTuple, Optional

import torch
import torch.jit
import torch.utils.data
import tqdm
import transformers

from loguru import logger


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
        overwrite_cache: bool = False,
    ):
        if not text_path.is_file():
            raise ValueError(f"{text_path} is not a valid text file.")
        self.tokenizer = tokenizer

        self.block_size = block_size - (
            tokenizer.max_len - tokenizer.max_len_single_sentence
        )

        cached_features_file = (
            text_path.parent
            / f"lm_{block_size}_{text_path.stem}_cache.pt"
        )
        if cached_features_file.exists() and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, "rb") as handle:
                self.examples = torch.load(handle)
        else:
            logger.info(f"Creating features from dataset file at {text_path.parent}")

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
        fast tokenizer on my setup it might benefit from fine-tuning. The trade-off is: using larger
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

            # Process by chunks loading everything in RAM
            # We still read by line to ensure we don't end up in the middle of a unicode
            # characterload_file(text_path)
            raw_lines = in_stream.readlines(read_size_hint)  # Oh a := would be so handy
            while raw_lines:
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
                            torch.tensor(
                                self.tokenizer.build_inputs_with_special_tokens(block),
                                dtype=torch.long,
                            )
                        )
                        remaining = next_remaining
                    buffer[:remaining] = encoded[len(encoded) - remaining :]
                    offset = remaining
                else:
                    buffer[offset:new_offset] = encoded
                    offset = new_offset
                pbar.n = in_stream.tell()
                pbar.update(0)
                raw_lines = in_stream.readlines(read_size_hint)
            pbar.close()
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


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
            raw_lines = in_stream.readlines(read_size_hint)  # Oh a := would be so handy
            while raw_lines:
                decoded = [l.decode("utf-8") for l in raw_lines]
                encoded = self.tokenizer.batch_encode_plus(
                    decoded, max_length=self.block_size, return_tensors="pt"
                )["input_ids"]
                examples.extend(encoded)
                pbar.n = in_stream.tell()
                pbar.update(0)
                raw_lines = in_stream.readlines(read_size_hint)
            pbar.close()
            return examples


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