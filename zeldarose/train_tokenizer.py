import pathlib
import typing as ty

import click
import click_pathlib
import tokenizers

from loguru import logger


@click.command()
@click.argument(
    "raw_texts",
    nargs=-1,
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--out-path",
    default=".",
    type=click_pathlib.Path(resolve_path=True, file_okay=False, allow_dash=True),
    help="Where to save the trained model",
)
@click.option(
    "--vocab-size", type=int, default=50_000, help="Size of the trained vocabulary"
)
@click.option("--model-name", type=str, default=None, help="A name to give to the model")
def main(
    raw_texts: ty.Collection[pathlib.Path],
    out_path: pathlib.Path,
    vocab_size: int,
    model_name: str,
):
    tokenizer = tokenizers.SentencePieceBPETokenizer()
    tokenizer.train([str(t) for t in raw_texts], vocab_size=vocab_size)
    out = tokenizer.save(str(out_path), model_name)
    logger.info(f"Saved files: {', '.join(out)}")


if __name__ == "__main__":
    main()
