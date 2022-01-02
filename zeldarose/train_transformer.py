from __future__ import annotations

import logging
import os
import pathlib
import sys
import warnings

from typing import Any, Dict, List, Optional, cast

import click
import click_pathlib
import pytorch_lightning as pl
import toml
import transformers

from loguru import logger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only
from zeldarose import data, rtd
from zeldarose import mlm
from zeldarose.common import TrainConfig


def setup_logging(
    verbose: bool, logfile: Optional[pathlib.Path] = None, replace_warnings: bool = True
):
    logger.remove(0)  # Remove the default logger
    if "SLURM_JOB_ID" in os.environ:
        appname = f"zeldarose ({os.environ.get('SLURM_PROCID', 'somerank')} [{os.environ.get('SLURM_LOCALID', 'someproc')}@{os.environ.get('SLURMD_NODENAME', 'somenode')}])"
    else:
        appname = "zeldarose"

    if verbose:
        log_level = "DEBUG"
        log_fmt = (
            f"[{appname}]"
            " <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
            " <level>{message}</level>"
        )
    else:
        logging.getLogger(None).setLevel(logging.CRITICAL)
        log_level = "INFO"
        log_fmt = (
            f"[{appname}]"
            " <green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
            " <level>{message}</level>"
        )

    logger.add(
        sys.stderr,
        level=log_level,
        format=log_fmt,
        colorize=True,
    )

    if logfile:
        logger.add(
            logfile,
            level="DEBUG",
            format=(
                f"[{appname}]"
                " {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} |"
                " {message}"
            ),
            colorize=False,
        )

    # Deal with stdlib.logging

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    transformers_logger = logging.getLogger("transformers")
    # FIXME: ugly, but is there a better way?
    transformers_logger.handlers.pop()
    transformers_logger.addHandler(InterceptHandler())

    pl_logger = logging.getLogger("pytorch_lightning")
    # FIXME: ugly, but is there a better way?
    pl_logger.handlers.pop()
    pl_logger.addHandler(InterceptHandler())

    # Deal with stdlib.warnings

    def showwarning(message, category, filename, lineno, file=None, line=None):
        logger.warning(
            warnings.formatwarning(message, category, filename, lineno, None).strip()
        )

    if replace_warnings:
        warnings.showwarning = showwarning


class SavePretrainedModelCallback(pl.callbacks.Callback):
    def __init__(
        self,
        save_dir: pathlib.Path,
        tokenizer: transformers.PreTrainedTokenizerBase,
        period: int = 1,
    ):
        self.period = period
        self.save_dir = save_dir
        self.tokenizer = tokenizer

    @rank_zero_only
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: mlm.MLMTrainingModel):
        if not trainer.current_epoch % self.period:
            epoch_save_dir = self.save_dir / f"epoch_{trainer.current_epoch}"
            logger.info(f"Saving intermediate model to {epoch_save_dir}")
            pl_module.save_transformer(epoch_save_dir, self.tokenizer)


# TODO: allow reading all these from a config file (except perhaps for paths)
# TODO: refactor the api to have a single `zeldarose` entrypoint with subcommands.
@click.command()
@click.argument(
    "raw_text",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--accelerator",
    type=str,
    help="The lightning accelerator to use (see lightning doc)",
)
@click.option(
    "--cache-dir",
    type=click_pathlib.Path(resolve_path=True, file_okay=False),
    help="Where to cache the input data",
)
@click.option(
    "--checkpoint",
    "checkpoint",
    type=click_pathlib.Path(resolve_path=True, dir_okay=False, exists=True),
    help="A checkpoint to restore the training state from",
)
@click.option(
    "--config",
    "config_path",
    type=click_pathlib.Path(resolve_path=True, dir_okay=False, exists=True),
    help="A config file (in TOML format)",
)
@click.option(
    "--device-batch-size",
    type=click.IntRange(0),
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
    "--max-epochs",
    type=click.IntRange(0),
    help="How many epochs to train for",
)
@click.option(
    "--max-steps",
    type=click.IntRange(0),
    help="How many steps to train for",
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
    default="muppet",
    metavar="NAME",
    help="A name to give to the model",
)
@click.option(
    "--n-gpus",
    default=0,
    type=click.IntRange(0),
    help="How many GPUs to train on. In ddp_cpu mode, this is the number of processes",
)
@click.option(
    "--n-nodes",
    type=click.IntRange(0),
    default=os.environ.get("SLURM_JOB_NUM_NODES", 1),
    help="How many nodes to train on (for clusters), defaults to $SLURM_JOB_NUM_NODES if on SLURM and 1 otherwise",
)
@click.option(
    "--n-workers",
    type=click.IntRange(0),
    default=0,
    help="How many data loading workers to use",
)
@click.option(
    "--out-dir",
    default=".",
    type=click_pathlib.Path(resolve_path=True, file_okay=False),
    help="Where to save the trained model",
)
@click.option(
    "--pretrained-model",
    type=str,
    help="A pretrained model to fine-tune",
    metavar="NAME_OR_PATH",
)
@click.option(
    "--save-period",
    type=click.IntRange(0),
    help="The number of epoch between intermediate model saving",
    default=0,
)
@click.option(
    "--sharded-ddp",
    is_flag=True,
    help="Activate to use the sharded DDP mode (requires fairscale)",
)
@click.option("--profile", is_flag=True, help="Run in profiling mode")
@click.option(
    "--tokenizer",
    "tokenizer_name",
    type=str,
    show_default=True,
    metavar="NAME_OR_PATH",
    help=(
        "A pretrained tokenizer model to use"
        " (default to the pretrained transformer model if there is one)"
    ),
)
@click.option(
    "--use-fp16",
    is_flag=True,
    help="Activate half-preicions mode (only on GPUs)",
)
@click.option(
    "--val-text",
    "val_path",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
    help="A raw corpus for validation",
)
@click.option("--verbose", is_flag=True, help="More detailed logs")
def main(
    accelerator: Optional[str],
    cache_dir: Optional[pathlib.Path],
    checkpoint: Optional[pathlib.Path],
    config_path: Optional[pathlib.Path],
    device_batch_size: Optional[int],
    guess_batch_size: bool,
    max_epochs: Optional[int],
    max_steps: Optional[int],
    model_config_path: Optional[str],
    model_name: str,
    n_gpus: int,
    n_nodes: int,
    n_workers: int,
    out_dir: pathlib.Path,
    pretrained_model: Optional[str],
    profile: bool,
    raw_text: pathlib.Path,
    save_period: int,
    sharded_ddp: bool,
    tokenizer_name: Optional[str],
    use_fp16: bool,
    val_path: Optional[pathlib.Path],
    verbose: bool,
):
    if (slurm_procid := os.environ.get("SLURM_PROCID")) is not None:
        log_file = out_dir / "logs" / f"train{slurm_procid}.log"
    else:
        log_file = out_dir / "train.log"
    setup_logging(verbose, log_file)
    logger.debug(f"Current environment: {os.environ}")

    # NOTE: this is likely duplicated somewhere in pl codebase but we need it now unless pl rolls
    # out something like `optim_batch_size` that takes into account the number of tasks and the
    # number of samples per gpu
    if (num_slurm_tasks := os.environ.get("SLURM_NTASKS")) is not None:
        n_devices = int(num_slurm_tasks)
    elif n_gpus:
        n_devices = n_nodes * n_gpus
    else:
        n_devices = 1
    logger.info(f"Training on {n_devices} devices.")

    if tokenizer_name is None:
        if pretrained_model is not None:
            tokenizer_name = pretrained_model
        else:
            raise ValueError("Missing both pretrained tokenizer and pretrained model")
    logger.info(f"Loading pretrained tokenizer {tokenizer_name}")
    tokenizer: transformers.PreTrainedTokenizerBase = (
        transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    )

    if config_path is not None:
        with open(config_path) as in_stream:
            config = toml.load(in_stream)
    else:
        config = dict()
    tuning_config = TrainConfig.parse_obj(config.get("tuning", dict()))

    task_type = config.get("type", "mlm")
    if task_type == "mlm":
        training_model = mlm.get_training_model(
            model_config_path=model_config_path,
            pretrained_model=pretrained_model,
            task_config_dict=config.get("task"),
            tokenizer=tokenizer,
            training_config=tuning_config,
        )
    elif task_type == "rtd":
        training_model = rtd.get_training_model(
            model_config_path=model_config_path,
            pretrained_model=pretrained_model,
            task_config_dict=config.get("task"),
            tokenizer=tokenizer,
            training_config=tuning_config,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type!r}")
    training_model.train()

    if device_batch_size is None:
        device_batch_size = tuning_config.batch_size
    elif tuning_config.batch_size < device_batch_size * n_devices:
        raise ValueError(
            f"Batch size ({tuning_config.batch_size}) is smaller than"
            f" loader batch size({device_batch_size} samples per device × {n_devices} devices)"
            " try using fewer devices"
        )
    elif tuning_config.batch_size % (device_batch_size * n_devices):
        remainder = tuning_config.batch_size % device_batch_size * n_devices
        logger.warning(
            f"Batch size ({tuning_config.batch_size}) is not a muliple"
            f" of loader batch size({device_batch_size} samples per device × {n_devices} devices)"
            f" the actual tuning batch size used will be {tuning_config.batch_size-remainder}."
        )

    # A pl Trainer batch is in fact one batch per device, so if we use multiple devices
    accumulate_grad_batches = tuning_config.batch_size // (
        device_batch_size * n_devices
    )
    logger.info(f"Using {accumulate_grad_batches} steps gradient accumulation.")

    # In DP mode, every batch is split between the devices
    if accelerator == "dp":
        loader_batch_size = device_batch_size * n_devices
    else:
        loader_batch_size = device_batch_size

    # FIXME: we shouldn't need num_special_tokens_to_add here
    max_length = min(
        tokenizer.max_len_single_sentence,
        training_model.max_len - tokenizer.num_special_tokens_to_add(pair=False),
    )
    logger.info(f"Training with a maximum sequence length of {max_length} tokens")
    logger.info("Creating data modules")
    datamodule = data.TextDataModule(
        data_dir=cache_dir,
        loader_batch_size=loader_batch_size,
        max_length=max_length,
        num_workers=n_workers,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name.replace("/", "_"),
        train_text=raw_text,
        val_text=val_path,
    )
    datamodule.prepare_data_per_node = False

    logger.info("Creating trainer")
    additional_kwargs: Dict[str, Any] = dict()
    if profile:
        logger.info("Running in profile mode")
        profiler = pl.profiler.AdvancedProfiler(
            output_filename=str(out_dir / "profile.txt")
        )
        additional_kwargs.update({"profiler": profiler, "overfit_batches": 1024})

    if guess_batch_size:
        logger.info("Automatic batch size selection")
        additional_kwargs.update({"auto_scale_batch_size": "binsearch"})

    if accelerator is not None and "ddp" in accelerator:
        cast(List, additional_kwargs.setdefault("plugins", [])).append(
            DDPPlugin(find_unused_parameters=profile),
        )

    if accelerator == "ddp_cpu":
        # FIXME: works but seems like bad practice
        additional_kwargs["num_processes"] = n_gpus
        n_gpus = 0

    if sharded_ddp:
        if accelerator == "ddp":
            logger.info("Using sharded DDP")
            cast(List[str], additional_kwargs.setdefault("plugins", [])).append(
                "ddp_sharded"
            )
        elif accelerator == "ddp_spawn":
            logger.info("Using sharded spawn DDP")
            cast(List[str], additional_kwargs.setdefault("plugins", [])).append(
                "ddp_sharded_spawn"
            )
        else:
            logger.warning(
                "--sharded-ddp only makes sense when using ddp accelerators. Ignoring the flag."
            )

    if n_gpus:
        if use_fp16:
            logger.info(f"Training the model on {n_gpus} GPUs with half precision")
            additional_kwargs["precision"] = 16
    elif accelerator == "ddp_cpu":
        logger.info(
            f"Training the model on CPU in {additional_kwargs['num_processes']} processes"
        )
    else:
        logger.info("Training the model on CPU")

    callbacks: List[pl.callbacks.Callback] = [
        pl.callbacks.RichProgressBar(),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    if save_period:
        training_model.save_transformer(
            out_dir / "partway_models" / "initial", tokenizer
        )
        callbacks.append(
            SavePretrainedModelCallback(
                out_dir / "partway_models",
                tokenizer,
                save_period,
            )
        )
    if profile and n_gpus and accelerator is not None and "cpu" not in accelerator:
        callbacks.append(pl.callbacks.GPUStatsMonitor())

    if checkpoint is not None:
        additional_kwargs["resume_from_checkpoint"] = checkpoint

    if max_steps is None:
        if tuning_config.lr_decay_steps is not None:
            max_steps = tuning_config.lr_decay_steps + tuning_config.warmup_steps
            logger.info(
                f"Setting the max number of steps at {max_steps} according to the tuning config"
            )
        else:
            max_steps = -1

    trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        auto_select_gpus=n_gpus > 0,
        callbacks=callbacks,
        default_root_dir=out_dir,
        accelerator=accelerator,
        gpus=n_gpus,
        gradient_clip_val=tuning_config.gradient_clipping,
        limit_val_batches=1.0 if val_path is not None else 0,
        max_epochs=max_epochs,
        max_steps=max_steps,
        num_nodes=n_nodes,
        **additional_kwargs,
    )

    logger.info("Start training")

    trainer.fit(training_model, datamodule=datamodule)

    save_dir = out_dir / model_name
    training_model.save_transformer(save_dir, tokenizer)


if __name__ == "__main__":
    main()
