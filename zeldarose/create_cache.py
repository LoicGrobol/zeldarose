from __future__ import annotations

import pathlib

from typing import Optional, Type

import click
import click_pathlib

import transformers

from loguru import logger

from zeldarose import data


@click.command()
@click.argument(
    "tokenizer_name",
    type=str,
    metavar="NAME_OR_PATH",
)
@click.argument(
    "raw_text",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--cache-dir",
    type=click_pathlib.Path(resolve_path=True, exists=True, file_okay=False),
    help="Where to create the cache (defaults to the raw text directory)",
)
def main(
    cache_dir: Optional[pathlib.Path],
    raw_text: pathlib.Path,
    tokenizer_name: str,
):
    logger.info(f"Loading pretrained tokenizer {tokenizer_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True
    )

    logger.info(f"Loading train dataset from {raw_text}")
    _ = data.load_dataset(
        cache_path=cache_dir,
        model_name=tokenizer_name.replace("/", "_"),
        overwrite_cache=True,
        text_path=raw_text,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
