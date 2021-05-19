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
    required=True,
)
@click.option(
    "--max-len",
    type=int,
    default=512,
    help="The maximum number of tokens in a sequence",
    show_default=True,
)
@click.option(
    "--model-name",
    type=str,
    default="muppet",
    help="A name to give to the model",
    show_default=True,
)
@click.option(
    "--out-path",
    default=".",
    type=click_pathlib.Path(resolve_path=True, file_okay=False),
    help="Where to save the trained model",
)
@click.option(
    "--vocab-size",
    type=int,
    default=4096,
    help="Size of the trained vocabulary",
    show_default=True,
)
def main(
    raw_texts: ty.Collection[pathlib.Path],
    max_len: int,
    model_name: str,
    out_path: pathlib.Path,
    vocab_size: int,
):
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()
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
    logger.info("Training")
    tokenizer.train(
        [str(t) for t in raw_texts],
        trainer=trainer,
    )
    logger.info("Finalizing")
    tokenizer.post_processor = tokenizers.processors.RobertaProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=max_len)
    logger.info("Saving model")
    model_dir = out_path / model_name
    model_dir.mkdir(exist_ok=True, parents=True)
    model_file = model_dir / "tokenizer.json"
    tokenizer.save(str(model_file))
    # TODO: set special tokens instead of hardcoding the default values of RoBERTa (see
    # <https://huggingface.co/transformers/_modules/transformers/tokenization_utils_base.html#SpecialTokensMixin>)
    tranformers_tokenizer = transformers.RobertaTokenizerFast(
        tokenizer_file=str(model_file),
        vocab_file=None,
        merges_file=None,
    )
    tranformers_tokenizer.save_pretrained(str(model_dir), legacy_format=False)
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
    config.to_json_file(str(model_dir / "config.json"))
    logger.info(f"Saved tokenizer in {model_dir}")


if __name__ == "__main__":
    main()
