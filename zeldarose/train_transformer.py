import logging
import pathlib
import sys

from typing import Optional, Type

import click
import click_pathlib
import pytorch_lightning as pl
import toml
import transformers

from loguru import logger

from zeldarose import data
from zeldarose import mlm


def setup_logging(verbose: bool, logfile: Optional[pathlib.Path]):
    logger.remove(0)  # Remove the default logger
    if verbose:
        log_level = "DEBUG"
        log_fmt = (
            "[uuparser] "
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
            "<level>{message}</level>"
        )
    else:
        log_level = "INFO"
        log_fmt = (
            "[uuparser] "
            "<green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
            "<level>{message}</level>"
        )

    logger.add(
        sys.stderr, level=log_level, format=log_fmt, colorize=True,
    )

    if logfile:
        logger.add(
            logfile, level="DEBUG", colorize=False,
        )


# logging.getLogger(None).setLevel(logging.ERROR)

# TODO: allow reading all these from a config file (except perhaps for paths)
# TODO: refactor the api to have a single `zeldarose` entrypoint with subcommands.
@click.command()
@click.argument(
    "raw_text", type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--batch-split",
    type=int,
    default=1,
    help="Number of pieces in which to split batches during training (trades time for memory)",
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
    default="roberta-base",
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
    help="A task config file (in TOML format)",
)
@click.option(
    "--tokenizer", "tokenizer_name",
    type=str,
    default="roberta-base",
    show_default=True,
    metavar="NAME_OR_PATH",
    help="A pretrained tokenizer model to use",
)
@click.option(
    "--tuning-config",
    "tuning_config_path",
    type=click_pathlib.Path(resolve_path=True, dir_okay=False, exists=True,),
    help="A fine-tuning config file (in TOML format)",
)
@click.option("--verbose", is_flag=True, help="More detailed logs")
def main(
    batch_split: int,
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
    verbose: bool,
):
    setup_logging(verbose, out_dir / "train.log")
    if task_config_path is not None:
        task_config = mlm.MLMTaskConfig.parse_obj(
            toml.loads(task_config_path.read_text())
        )
    else:
        task_config = mlm.MLMTaskConfig()
    if tuning_config_path is not None:
        tuning_config = mlm.MLMFinetunerConfig.parse_obj(
            toml.loads(tuning_config_path.read_text())
        )
    else:
        tuning_config = mlm.MLMFinetunerConfig()

    logger.info(f"Loading pretrained tokenizer {tokenizer_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True
    )

    if pretrained_model is not None:
        logger.info(f"Loading pretrained model {pretrained_model!r}")
        model = transformers.AutoModelWithLMHead.from_pretrained(pretrained_model)
    elif model_config_path is not None:
        logger.info(f"Loading pretrained config {model_config_path!r}")
        model_config = transformers.AutoConfig.from_pretrained(model_config_path)
        logger.info(f"Generating model from config")
        model = transformers.AutoModelWithLMHead.from_config(model_config)
    # TODO: try to automate this by estimating the memory used by the model
    # (dummy_input) can probably help for that
    if tuning_config.batch_size % batch_split:
        logger.warning(
            f"Batch size ({tuning_config.batch_size}) is not a muliple of batch split ({batch_split})"
        )

    dataset_type: Type[data.TextDataset]
    if line_by_line:
        dataset_type = data.LineByLineTextDataset
    else:
        dataset_type = data.TextDataset
    logger.info(f"Loading raw text dataset from {raw_text}")
    train_set = dataset_type(
        tokenizer=tokenizer, text_path=raw_text, overwrite_cache=overwrite_cache,
    )

    logger.info(f"Creating dataloader")
    train_loader = mlm.MLMLoader(
        train_set,
        task_config=task_config,
        batch_size=tuning_config.batch_size // batch_split,
    )

    logger.info(f"Creating MLM Finetuner")
    finetuning_model = mlm.MLMFinetuner(model, config=tuning_config)
    logger.info(f"Creating trainer")
    trainer = pl.Trainer(
        accumulate_grad_batches=batch_split,
        distributed_backend=distributed_backend,
        default_save_path=out_dir,
        gpus=n_gpus,
        reload_dataloaders_every_epoch=True,
    )

    logging.info("Training the model")
    trainer.fit(finetuning_model, train_dataloader=train_loader)

    logger.info(f"Saving model to {out_dir}")
    model.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
