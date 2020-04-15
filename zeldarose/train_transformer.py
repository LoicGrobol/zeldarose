import logging
import math
import pathlib
import sys
import tempfile

from typing import Optional, Type, Union

import click
import click_pathlib
import pytorch_lightning as pl
import toml
import torch
import torch.cuda
import transformers

from loguru import logger

from zeldarose import data
from zeldarose import mlm


def setup_logging(verbose: bool, logfile: Optional[pathlib.Path]):
    logger.remove(0)  # Remove the default logger
    if verbose:
        log_level = "DEBUG"
        log_fmt = (
            "[zeldarose] "
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
            "<level>{message}</level>"
        )
    else:
        log_level = "INFO"
        log_fmt = (
            "[zeldarose] "
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


def max_gpu_batch_size(
    dataset: data.TextDataset,
    finetuner: pl.LightningModule,
    task_config: mlm.MLMTaskConfig,
    guess_batch_size: int = 4,
    n_samples: int = 100,
    device: Union[torch.device, int] = 0,
) -> int:
    """Tries to find a maximal batch size for a device, assumes that it can fit at least a batch
    size of 2 and that memory usage is an affine function of batch size.

    The estimate is rather conservative, so hopefully with this batch size, no crash should occur.
    """
    device = torch.device(device)  # type: ignore
    assert 2 <= guess_batch_size

    def test_run(batch_size):
        with tempfile.TemporaryDirectory(prefix="zeldarose-profile") as temp_dir:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            loader = mlm.MLMLoader(
                dataset, task_config=task_config, batch_size=batch_size,
            )
            trainer = pl.Trainer(
                default_save_path=temp_dir,
                overfit_pct=n_samples / len(loader),
                gpus=[device.index],
                max_epochs=2,
            )
            trainer.fit(finetuner, train_dataloader=loader)
        return torch.cuda.max_memory_allocated(device)

    usage_with_guess = test_run(guess_batch_size)
    logger.debug(
        f"Memory usage with batch size {guess_batch_size}: {usage_with_guess} B"
    )
    usage_with_half_guess = test_run(guess_batch_size // 2)
    logger.debug(
        f"Memory usage with batch size {guess_batch_size // 2}: {usage_with_half_guess} B"
    )
    mem_per_sample = math.ceil(
        2 * (usage_with_guess - usage_with_half_guess) / guess_batch_size
    )
    logger.debug(f"Inferred memory usage per sample: {mem_per_sample} B")
    fixed_mem = math.ceil(usage_with_guess - guess_batch_size * mem_per_sample)
    logger.debug(f"Inferred fixed memory usage: {fixed_mem} B")
    device_max_mem = torch.cuda.get_device_properties(device.index).total_memory
    logger.debug(f"Device total memory: {device_max_mem} B")
    res = math.floor((device_max_mem - fixed_mem) / mem_per_sample)
    assert guess_batch_size <= res
    logger.debug(f"Inferred max batch size: {res}, making a test run")
    try:
        usage_with_max_size = test_run(res)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning(f"Non-affine or unstable memory usage: assuming linear")
            return math.floor(guess_batch_size * device_max_mem / usage_with_guess)
        else:
            raise e
    logger.debug(f"Mem usage with inferred batch size: {usage_with_max_size} B")
    return res


# logging.getLogger(None).setLevel(logging.ERROR)

# TODO: allow reading all these from a config file (except perhaps for paths)
# TODO: refactor the api to have a single `zeldarose` entrypoint with subcommands.
@click.command()
@click.argument(
    "raw_text", type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--device-batch-size",
    type=int,
    help=(
        "Number of samples in a processing batch"
        " (must be a divisor of training bath size, defaults to training batch size)"
    ),
)
@click.option(
    "--guess-batch-size",
    is_flag=True,
    help=(
        "Try to find the max device batch size automatically"
        " (ignored if --device-batch-size is provided)"
    ),
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
@click.option("--profile", is_flag=True, help="Run in profiling mode")
@click.option(
    "--task-config",
    "task_config_path",
    type=click_pathlib.Path(resolve_path=True, dir_okay=False, exists=True),
    help="A task config file (in TOML format)",
)
@click.option(
    "--tokenizer",
    "tokenizer_name",
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
    device_batch_size: Optional[int],
    distributed_backend: Optional[str],
    guess_batch_size: bool,
    line_by_line: bool,
    model_config_path: Optional[str],
    model_name: str,
    out_dir: pathlib.Path,
    overwrite_cache: bool,
    n_gpus: Optional[int],
    pretrained_model: Optional[str],
    profile: bool,
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

    dataset_type: Type[data.TextDataset]
    if line_by_line:
        dataset_type = data.LineByLineTextDataset
    else:
        dataset_type = data.TextDataset
    logger.info(f"Loading raw text dataset from {raw_text}")
    train_set = dataset_type(
        tokenizer=tokenizer, text_path=raw_text, overwrite_cache=overwrite_cache,
    )

    logger.info(f"Creating MLM Finetuner")
    finetuning_model = mlm.MLMFinetuner(model, config=tuning_config)

    # TODO: try to automate this by estimating the memory used by the model
    # (dummy_input) can probably help for that
    if device_batch_size is None:
        if guess_batch_size:
            logger.info("Running a quick profile to find out the best batch size")
            device_batch_size = max_gpu_batch_size(
                dataset=train_set, finetuner=finetuning_model, task_config=task_config,
            )
            logger.info(f"Inferred max batch size: {device_batch_size}")
        else:
            device_batch_size = tuning_config.batch_size
    elif tuning_config.batch_size % device_batch_size:
        logger.warning(
            f"Batch size ({tuning_config.batch_size}) is not a muliple"
            f" of device batch size({device_batch_size})"
        )

    logger.info(f"Creating dataloader")
    train_loader = mlm.MLMLoader(
        train_set, task_config=task_config, batch_size=device_batch_size,
    )

    logger.info(f"Creating trainer")
    if profile:
        logger.info("Running in profile mode")
        profiler = pl.profiler.AdvancedProfiler(output_filename=out_dir / "profile.txt")
        profile_kwargs = {
            "profiler": profiler,
            "overfit_pct": 1000 / len(train_loader),
            "max_epochs": 2,
        }
    else:
        profile_kwargs = dict()
    trainer = pl.Trainer(
        accumulate_grad_batches=tuning_config.batch_size // device_batch_size,
        distributed_backend=distributed_backend,
        default_save_path=out_dir,
        gpus=n_gpus,
        reload_dataloaders_every_epoch=True,
        **profile_kwargs,
    )

    if n_gpus:
        logging.info(f"Training the model on {n_gpus} GPUs")
    else:
        logging.info(f"Training the model on CPU")
    trainer.fit(finetuning_model, train_dataloader=train_loader)

    logger.info(f"Saving model to {out_dir}")
    model.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
