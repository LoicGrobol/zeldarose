import pathlib
import typing as ty

import click
import click_pathlib
import tokenizers
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.processors
import tokenizers.trainers
import transformers

from loguru import logger


@click.command()
@click.argument(
    "raw_texts",
    nargs=-1,
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--max-len",
    type=int,
    default=512,
    help="The maximum number of tokens in a sequence",
)
@click.option(
    "--model-name", type=str, default=None, help="A name to give to the model"
)
@click.option(
    "--out-path",
    default=".",
    type=click_pathlib.Path(resolve_path=True, file_okay=False, allow_dash=True),
    help="Where to save the trained model",
)
@click.option(
    "--vocab-size", type=int, default=4096, help="Size of the trained vocabulary"
)
def main(
    raw_texts: ty.Collection[pathlib.Path],
    max_len: int,
    model_name: str,
    out_path: pathlib.Path,
    vocab_size: int,
):
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    # Special tokens hardcoded from RoBERTa's default, see `__init__` in
    # <https://huggingface.co/transformers/_modules/transformers/tokenization_roberta_fast.html#RobertaTokenizerFast>
    # and do not forget to adapt this if we allow other tokenizer configs here
    trainer = tokenizers.trainers.BpeTrainer(
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
        vocab_size=vocab_size,
    )
    tokenizer.train(
        [str(t) for t in raw_texts],
        trainer=trainer,
    )
    tokenizer.post_processor = tokenizers.processors.RobertaProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=max_len)
    model_path = out_path / model_name
    model_path.mkdir(exist_ok=True, parents=True)
    model_file = model_path / "tokenizer.json"
    tokenizer.save(str(model_file))
    # TODO: set special tokens instead of hardcoding the default values of RoBERTa (see
    # <https://huggingface.co/transformers/_modules/transformers/tokenization_utils_base.html#SpecialTokensMixin>)
    tranformers_tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
        str(model_path),
        max_len=max_len,
    )
    tranformers_tokenizer.save_pretrained(str(model_path))
    # Useless in principle since we don't specify a model here but needed in practice for
    # compatibility with `AutoTokenizer`, see
    # <https://github.com/huggingface/transformers/issues/6368> See also
    # <https://github.com/huggingface/transformers/blob/38f6739cd6c1725ecd75a40d5371483f738097c2/src/transformers/tokenization_utils_base.py#L1670>
    # for some hope that this will be improved some day
    config = transformers.RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_len,
        type_vocab_size=1,
    )
    config.to_json_file(str(model_path / "config.json"))
    logger.info(f"Saved tokenizer in {model_path}")


if __name__ == "__main__":
    main()
