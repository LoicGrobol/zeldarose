import logging
import pathlib

from typing import Type

import click
import click_pathlib
import transformers

import data


logging.getLogger(None).setLevel(logging.ERROR)


@click.command()
@click.argument(
    "raw_text", type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--model-type", type=str, default="roberta", help="Which muppet to train"
)
@click.option(
    "--tokenizer-name",
    type=str,
    default="roberta-large",
    help="A pretrained tokenizer model to use",
)
@click.option(
    "--model-name", type=str, default=None, help="A name to give to the model"
)
@click.option("--line-by-line", is_flag=True)
@click.option("--overwrite-cache", is_flag=True)
def main(
    raw_text: pathlib.Path,
    model_type: str,
    tokenizer_name: str,
    model_name: str,
    line_by_line: bool,
    overwrite_cache: bool,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True
    )
    dataset_type: Type[data.TextDataset]
    if line_by_line:
        dataset_type = data.LineByLineTextDataset
    else:
        dataset_type = data.TextDataset

    dataset_type(
        tokenizer=tokenizer,
        text_path=raw_text,
        model_type=model_type,
        overwrite_cache=overwrite_cache,
    )


if __name__ == "__main__":
    main()
