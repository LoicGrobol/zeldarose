import pathlib
import pickle  # nosec

from typing import List

import torch
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
        model_type: str,
        block_size: int = 512,
        overwrite_cache: bool = False,
    ):
        if not text_path.is_file():
            raise ValueError(f"{text_path} is not a valid text file.")
        self.tokenizer = tokenizer
        self.block_size = block_size

        block_size = block_size - (
            tokenizer.max_len - tokenizer.max_len_single_sentence
        )

        cached_features_file = (
            text_path.parent / f"{model_type}_cached_lm_{block_size}_{text_path.stem}"
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
                    self.examples, handle, pickle_protocol=pickle.HIGHEST_PROTOCOL
                )

    def load_file(
        self, text_path: pathlib.Path, read_size_hint: int = 2 ** 20
    ) -> List[torch.Tensor]:
        """Tokenize, encode and split a raw text in blocks.

        **Note**: This reads files by chunks of about 4MiB which seemed to work best for the roberta
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
                mininterval=2,
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
                    # No easy way to avoid a slice copy here
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
                pbar.update(len(chunk))
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
        self, text_path: pathlib.Path, read_size_hint: int = 2 ** 19
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
                mininterval=2,
            )
            # FUTURE: This reads one line at a time since there is no uniform API for encoding
            # batches that works for bost slow and fast tokenizers, but it should be changed as soon
            # as we get one.
            for raw_line in in_stream:
                decoded = raw_line.decode("utf-8")
                encoded = self.tokenizer.encode(decoded, add_special_tokens=False)[
                    : self.block_size
                ]
                examples.append(
                    torch.tensor(
                        self.tokenizer.build_inputs_with_special_tokens(encoded),
                        dtype=torch.long,
                    ),
                )
                pbar.update(len(raw_line))
            pbar.close()
            return examples
