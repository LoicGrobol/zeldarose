import pathlib
import typing as ty

import click
import click_pathlib
import tokenizers
import transformers

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
@click.option(
    "--model-name", type=str, default=None, help="A name to give to the model"
)
def main(
    raw_texts: ty.Collection[pathlib.Path],
    out_path: pathlib.Path,
    vocab_size: int,
    model_name: str,
):
    tokenizer = tokenizers.implementations.ByteLevelBPETokenizer()
    tokenizer.train(
        [str(t) for t in raw_texts],
        vocab_size=vocab_size,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
    )
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    model_path = out_path / model_name
    model_path.mkdir(exist_ok=True, parents=True)
    tokenizer.save_model(str(model_path))
    tranformers_tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
        str(model_path), max_len=512
    )
    tranformers_tokenizer.save_pretrained(str(model_path))
    logger.info(f"Saved tokenizer in {model_path}")


if __name__ == "__main__":
    main()
