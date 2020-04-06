import logging
import pathlib

from typing import Optional, Type

import click
import click_pathlib
import pytorch_lightning as pl
import transformers

from loguru import logger

from zeldarose import data
from zeldarose import mlm


# logging.getLogger(None).setLevel(logging.ERROR)

# TODO: allow reading all these from a config file (except perhaps for paths)
# TODO: refactor the api to have a single `zeldarose` entrypoint with subcommands.
@click.command()
@click.argument(
    "raw_text", type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--distributed-backend",
    type=str,
    help="The lightning distributed backed to use (see lightning doc)",
)
@click.option(
    "--line-by-line",
    is_flag=True,
    help="Assume that the dataset is pre-segmented in sentences",
)
@click.option(
    "--model-config",
    "model_config_path",
    type=str,
    default="roberta-large",
    show_default=True,
    metavar="NAME_OR_PATH",
    help="A config name or path to create a muppet from scratch",
)
@click.option(
    "--model-name",
    type=str,
    default="generic_muppet",
    metavar="NAME",
    help="A name to give to the model",
)
@click.option(
    "--n-gpus", type=int, help="How many GPUs to train on",
)
@click.option(
    "--out-dir",
    default=".",
    type=click_pathlib.Path(resolve_path=True, file_okay=False, allow_dash=True),
    help="Where to save the trained model",
)
@click.option(
    "--overwrite-cache",
    is_flag=True,
    help="Do not load the dataset from a pre-existing cache",
)
@click.option(
    "--pretrained-model",
    type=str,
    help="A pretrained model to fine-tune",
    metavar="NAME_OR_PATH",
)
@click.option(
    "--task-config",
    "task_config_path",
    type=click_pathlib.Path(resolve_path=True, dir_okay=False, exists=True),
    help="A task config file",
)
@click.option(
    "--tokenizer-name",
    type=str,
    default="roberta-large",
    show_default=True,
    metavar="NAME_OR_PATH",
    help="A pretrained tokenizer model to use",
)
@click.option(
    "--tuning-config",
    "tuning_config_path",
    type=click_pathlib.Path(resolve_path=True, dir_okay=False, exists=True,),
    help="A fine-tuning config file",
)
def main(
    distributed_backend: Optional[str],
    line_by_line: bool,
    model_config_path: Optional[str],
    model_name: str,
    out_dir: pathlib.Path,
    overwrite_cache: bool,
    n_gpus: Optional[int],
    pretrained_model: Optional[str],
    raw_text: pathlib.Path,
    task_config_path: Optional[pathlib.Path],
    tokenizer_name: str,
    tuning_config_path: Optional[pathlib.Path],
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True
    )

    if pretrained_model is not None:
        model = transformers.AutoModelWithLMHead.from_pretrained(pretrained_model)
    elif model_config_path is not None:
        model_config = transformers.AutoConfig.from_pretrained(model_config_path)
        model = transformers.AutoModelWithLMHead.from_config(model_config)

    dataset_type: Type[data.TextDataset]
    if line_by_line:
        dataset_type = data.LineByLineTextDataset
    else:
        dataset_type = data.TextDataset
    train_set = dataset_type(
        tokenizer=tokenizer, text_path=raw_text, overwrite_cache=overwrite_cache,
    )

    if task_config_path is not None:
        task_config = mlm.MLMTaskConfig.parse_file(task_config_path)
    else:
        task_config = mlm.MLMTaskConfig()
    train_loader = mlm.MLMLoader(train_set, task_config=task_config)

    if tuning_config_path is not None:
        tuning_config = mlm.MLMFinetunerConfig.parse_file(tuning_config_path)
    else:
        tuning_config = mlm.MLMFinetunerConfig()
    finetuning_model = mlm.MLMFinetuner(model, config=tuning_config)
    trainer = pl.Trainer(
        distributed_backend=distributed_backend,
        default_save_path=out_dir,
        gpus=n_gpus,
    )

    logging.info("Training the model")
    trainer.fit(finetuning_model, train_dataloader=train_loader)

    logger.info(f"Saving model to {out_dir}")
    model.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
