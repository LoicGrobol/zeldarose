import logging
import os
import pathlib
from types import ModuleType
import warnings

from typing import Any, Dict, List, Literal, Optional, Union

import click
import pytorch_lightning as pl
from pytorch_lightning import (
    callbacks as pl_callbacks,
    profilers as pl_profilers,
    strategies as pl_strategies,
)
from pytorch_lightning.utilities import types as pl_types
import rich
import tomli
import torch
import transformers

from lightning_utilities.core.rank_zero import rank_zero_only
from loguru import logger
from zeldarose.common import TrainConfig, TrainingModule
from zeldarose.tasks import mbart, mlm, rtd


def setup_logging(
    console: rich.console.Console,
    verbose: bool,
    log_file: Optional[pathlib.Path] = None,
    replace_warnings: bool = True,
):
    logger.remove(0)  # Remove the default logger
    if "SLURM_JOB_ID" in os.environ:
        local_id = os.environ.get("SLURM_LOCALID", "someproc")
        node_name = os.environ.get("SLURMD_NODENAME", "somenode")
        appname = (
            f"zeldarose ({os.environ.get('SLURM_PROCID', 'somerank')} [{local_id}@{node_name}])"
        )
    else:
        appname = "zeldarose"

    if verbose:
        log_level = "DEBUG"
        log_fmt = (
            f"\\[{appname}]"
            " [green]{time:YYYY-MM-DD HH:mm:ss.SSS}[/green] | [blue]{level: <8}[/blue] |"
            " {message}"
        )
    else:
        logging.getLogger(None).setLevel(logging.CRITICAL)
        log_level = "INFO"
        log_fmt = (
            f"\\[{appname}]"
            " [green]{time:YYYY-MM-DD}T{time:HH:mm:ss}[/green] {level} "
            " {message}"
        )

    logger.add(
        lambda m: console.print(m, end=""),
        colorize=True,
        format=log_fmt,
        level=log_level,
    )

    if log_file:
        logger.add(
            log_file,
            colorize=False,
            format=(f"[{appname}] {{time:YYYY-MM-DD HH:mm:ss.SSS}} | {{level: <8}} | {{message}}"),
            level="DEBUG",
        )

    # Deal with stdlib.logging

    class InterceptHandler(logging.Handler):
        def __init__(self, wrapped_name: Optional[str] = None, *args, **kwargs):
            self.wrapped_name = wrapped_name
            super().__init__(*args, **kwargs)

        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame is not None and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            if frame is None:
                warnings.warn(
                    "Catching calls to logging is impossible in stackless environment,"
                    " logging from external libraries might be lost.",
                    stacklevel=2,
                )
            else:
                if self.wrapped_name is not None:
                    logger.opt(depth=depth, exception=record.exc_info).log(
                        level, f"[bold]{self.wrapped_name} says:[/bold] {record.getMessage()}"
                    )
                else:
                    logger.opt(depth=depth, exception=record.exc_info).log(
                        level, record.getMessage()
                    )

    for libname in (
        "datasets",
        "huggingface_hub",
        "lightning_fabric",
        "pytorch_lightning",
        "torch",
        "torchmetrics",
        "transformers",
    ):
        lib_logger = logging.getLogger(libname)
        # FIXME: ugly, but is there a better way? What if they rely on having other handlers?
        if lib_logger.handlers:
            lib_logger.handlers = []
        lib_logger.addHandler(InterceptHandler(libname))
        logger.info(f"Intercepting logging from {libname}")

    # Deal with stdlib.warnings
    def showwarning(message, category, filename, lineno, file=None, line=None):
        logger.warning(warnings.formatwarning(message, category, filename, lineno, None).strip())

    if replace_warnings:
        warnings.showwarning = showwarning


class SavePretrainedModelCallback(pl.Callback):
    def __init__(
        self,
        save_dir: pathlib.Path,
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        epoch_period: Optional[int] = 1,
        step_period: Optional[int] = None,
    ):
        self.epoch_period = epoch_period
        self.step_period = step_period
        self.save_dir = save_dir
        self.tokenizer = tokenizer

    @rank_zero_only
    def on_before_zero_grad(
        self,
        trainer: pl.Trainer,
        pl_module: TrainingModule,
        optimizer: torch.optim.Optimizer,
    ):
        step = trainer.global_step
        if self.step_period is not None and step % self.step_period == 0 and step != 0:
            step_save_dir = self.save_dir / f"step_{step}"
            logger.info(f"Saving intermediate model to {step_save_dir}")
            pl_module.save_transformer(step_save_dir, self.tokenizer)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: TrainingModule):
        epoch = trainer.current_epoch
        if self.epoch_period is not None and epoch % self.epoch_period == 0:
            epoch_save_dir = self.save_dir / f"epoch_{trainer.current_epoch}"
            logger.info(f"Saving intermediate model to {epoch_save_dir}")
            pl_module.save_transformer(epoch_save_dir, self.tokenizer)


# TODO: allow reading all these from a config file (except perhaps for paths)
@click.command("transformer", help="Train a transformer model.")
@click.argument("train_data")
@click.option(
    "--accelerator",
    default="auto",
    metavar="NAME",
    show_default=True,
    type=str,
    help="The lightning accelerator to use (see lightning doc).",
)
@click.option(
    "--cache-dir",
    type=click.Path(resolve_path=True, file_okay=False, path_type=pathlib.Path),
    help="Where to cache the input data",
)
@click.option(
    "--checkpoint",
    "checkpoint",
    type=click.Path(resolve_path=True, dir_okay=False, exists=True, path_type=pathlib.Path),
    help="A checkpoint to restore the training state from",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(resolve_path=True, dir_okay=False, exists=True, path_type=pathlib.Path),
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
    help="DEPRECATED: set this as a tuning config option instead.",
)
@click.option(
    "--max-steps",
    type=click.IntRange(0),
    help="DEPRECATED: set this as a tuning config option instead.",
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
    "--num-devices",
    default=os.environ.get("SLURM_NTASKS", 1),
    type=click.IntRange(1),
    help="How many devices to train on. If `accelerator` is `cpu`, this is the number of processes",
)
@click.option(
    "--num-nodes",
    type=click.IntRange(0),
    default=os.environ.get("SLURM_JOB_NUM_NODES", 1),
    help=(
        "How many nodes to train on (for clusters), defaults"
        " to $SLURM_JOB_NUM_NODES if on SLURM and 1 otherwise"
    ),
)
@click.option(
    "--num-workers",
    type=click.IntRange(0),
    default=0,
    help="How many data loading workers to use",
)
@click.option(
    "--out-dir",
    default=".",
    type=click.Path(resolve_path=True, file_okay=False, path_type=pathlib.Path),
    help="Where to save the trained model. (defaults to cwd)",
)
@click.option(
    "--precision",
    type=click.Choice(["64", "32", "16", "bf16"]),
    help="The precision of float for training",
)
@click.option(
    "--pretrained-model",
    type=str,
    help="A pretrained model to fine-tune",
    metavar="NAME_OR_PATH",
)
@click.option(
    "--save-period",
    "--epoch-save-period",
    "epoch_save_period",
    type=click.IntRange(0),
    help="The number of epoch between intermediate model saving",
)
@click.option("--seed", "seed", type=int, default=2713, help="Random seed to set.")
@click.option(
    "--step-save-period",
    type=click.IntRange(0),
    help="The number of steps between intermediate model saving",
)
@click.option(
    "--strategy",
    default="auto",
    help="The lightning strategy to use (see lightning doc)",
    show_default=True,
    type=click.Choice(["auto", *pl_strategies.StrategyRegistry.available_strategies()]),
)
@click.option("--profile", is_flag=True, help="Run in profiling mode")
@click.option(
    "--tf32-mode",
    type=click.Choice(["highest", "high", "medium"]),
    help="Ampere matmul optimisation mode (for supported GPUs)",
)
@click.option(
    "--tokenizer",
    "tokenizer_name",
    type=str,
    metavar="NAME_OR_PATH",
    help=(
        "A pretrained tokenizer model to use"
        " (default to the pretrained transformer model if there is one)"
    ),
)
@click.option(
    "--val-check-period",
    type=click.IntRange(0),
    help="The number of steps between validation runs (useful for very long epochs)",
)
@click.option(
    "--val-data",
    "--val-text",
    "val_path",
    help=(
        "A raw corpus for validation."
        " Either as a path or as a `handle:config:split` identifier for 🤗 hub."
        " (handle can be a url), e.g. `lgrobol/openminuscule:default:train`"
    ),
)
@click.option("--verbose", is_flag=True, help="More detailed logs")
def main(
    accelerator: str,
    cache_dir: Optional[pathlib.Path],
    checkpoint: Optional[pathlib.Path],
    config_path: Optional[pathlib.Path],
    device_batch_size: Optional[int],
    epoch_save_period: Optional[int],
    guess_batch_size: bool,
    max_epochs: Optional[int],
    max_steps: Optional[int],
    model_config_path: Optional[str],
    model_name: str,
    num_nodes: int,
    num_workers: int,
    num_devices: int,
    out_dir: pathlib.Path,
    precision: Optional[Literal["64", "32", "16", "bf16"]],
    pretrained_model: Optional[str],
    profile: bool,
    train_data: str,
    seed: int,
    step_save_period: Optional[int],
    strategy: str,
    tf32_mode: Optional[Literal["highest", "high", "medium"]],
    tokenizer_name: Optional[str],
    val_check_period: Optional[int],
    val_path: Optional[str],
    verbose: bool,
):
    """Train a Transformer model.

    The training dataset should be given with `raw_text` as either a path to a local file or or as a
    `handle:config:split` identifier for 🤗 hub (handle can be a url).
    """

    # ## Logging setup
    console = rich.get_console()
    console.stderr = True
    if (slurm_procid := os.environ.get("SLURM_PROCID")) is not None:
        log_file = out_dir / "logs" / f"train{slurm_procid}.log"
    else:
        log_file = out_dir / "train.log"
    setup_logging(
        console=console,
        log_file=log_file,
        verbose=verbose,
    )
    logger.debug(f"Current environment: {os.environ}")

    logger.info(f"Using random seed {seed}")
    pl.seed_everything(seed, workers=True)
    transformers.set_seed(seed)

    logger.debug(f"Loading config from {config_path}")
    if config_path is not None:
        with open(config_path, "rb") as in_stream:
            config = tomli.load(in_stream)
    else:
        logger.warning(
            "No config given, we'll use default values but this is probably not what you want."
        )
        config = dict()

    logger.debug("Finding out batch size depending on environment and tuning config.")
    # NOTE: this is likely duplicated somewhere in pl codebase but we need it now unless pl rolls
    # out something like `optim_batch_size` that takes into account the number of tasks and the
    # number of samples per gpu
    if (num_slurm_tasks := os.environ.get("SLURM_NTASKS")) is not None:
        total_devices = int(num_slurm_tasks)
    else:
        total_devices = num_nodes * num_devices
    logger.info(f"Training on a total of {total_devices} devices.")

    tuning_config = TrainConfig.model_validate(config.get("tuning", dict()))

    if tuning_config.max_steps is not None:
        max_steps = tuning_config.max_steps

    if tuning_config.max_epochs is not None:
        max_epochs = tuning_config.max_epochs

    if device_batch_size is None:
        device_batch_size = tuning_config.batch_size
    elif tuning_config.batch_size < device_batch_size * total_devices:
        raise ValueError(
            f"Batch size ({tuning_config.batch_size}) is smaller than"
            f" loader batch size({device_batch_size} samples per device × {total_devices} devices)"
            " try using fewer devices"
        )
    elif tuning_config.batch_size % (device_batch_size * total_devices) != 0:
        remainder = tuning_config.batch_size % device_batch_size * total_devices
        logger.warning(
            f"Batch size ({tuning_config.batch_size}) is not a multiple of loader batch size"
            f" ({device_batch_size} samples per device × {total_devices} devices)"
            f" the actual tuning batch size used will be {tuning_config.batch_size-remainder}."
        )

    # A pl Trainer batch is in fact one batch per device, so if we use multiple devices
    accumulate_grad_batches = tuning_config.batch_size // (device_batch_size * total_devices)
    logger.info(f"Using {accumulate_grad_batches} steps gradient accumulation.")

    # In DP mode, every batch is split between the devices
    if accelerator == "dp":
        loader_batch_size = device_batch_size * total_devices
    else:
        loader_batch_size = device_batch_size

    logger.debug("Loading tokenizer, model and dataset.")
    task_type = config.get("type", "mlm")

    # TODO: make `task` here an object that will hold the config and be responsible for the rest
    task: ModuleType
    if task_type == "mlm":
        task = mlm
    elif task_type == "rtd":
        task = rtd
    elif task_type == "mbart":
        task = mbart
    else:
        raise ValueError(f"Unknown task type: {task_type!r}")

    if tokenizer_name is None:
        if pretrained_model is not None:
            tokenizer_name = pretrained_model
        else:
            raise ValueError("Missing both pretrained tokenizer and pretrained model")
    logger.info(f"Loading pretrained tokenizer {tokenizer_name}")
    tokenizer: Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ] = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    training_model: TrainingModule = task.get_training_model(
        model_config=model_config_path,
        pretrained_model=pretrained_model,
        task_config=config.get("task"),
        tokenizer=tokenizer,
        training_config=tuning_config,
    )
    training_model.train()

    logger.info("Creating data modules")
    datamodule = training_model.get_data_module(
        data_dir=cache_dir,
        loader_batch_size=loader_batch_size,
        num_workers=num_workers,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name.replace("/", "_"),
        train_path=train_data,
        val_path=val_path,
    )
    datamodule.prepare_data_per_node = False

    logger.info("Creating trainer")
    additional_kwargs: Dict[str, Any] = dict()
    if profile:
        logger.info("Running in profile mode")
        profiler = pl_profilers.AdvancedProfiler(dirpath=out_dir, filename="profile.txt")
        additional_kwargs.update({"profiler": profiler, "overfit_batches": 128})

    if guess_batch_size:
        logger.info("Automatic batch size selection")
        additional_kwargs.update({"auto_scale_batch_size": "binsearch"})

    if precision is not None:
        logger.info(f"Training the model using {precision} precision")
        additional_kwargs["precision"] = precision
    if accelerator == "gpu":
        if tf32_mode is not None:
            logger.info(f"Using Ampere matmul optimisations level {tf32_mode}")
            torch.set_float32_matmul_precision(tf32_mode)
    elif accelerator == "cpu":
        logger.info(f"Training the model on CPU in {num_devices} processes")
    else:
        logger.info(f"Training the model on {num_devices} devices")

    callbacks: List[pl.Callback] = [
        pl_callbacks.RichProgressBar(console_kwargs={"stderr": True}),
        pl_callbacks.LearningRateMonitor("step"),
    ]
    if epoch_save_period is not None or step_save_period is not None:
        training_model.save_transformer(out_dir / "partway_models" / "initial", tokenizer)
        callbacks.append(
            SavePretrainedModelCallback(
                out_dir / "partway_models",
                tokenizer,
                epoch_period=epoch_save_period,
                step_period=step_save_period,
            )
        )

    if checkpoint is not None:
        logger.info(f"Restarting training from the checkpoint at {checkpoint}")

    if tuning_config.lr_decay_steps is not None and tuning_config.lr_decay_steps != -1:
        max_steps = tuning_config.lr_decay_steps + tuning_config.warmup_steps
        logger.info(
            f"Setting the max number of steps at {max_steps} according to the tuning config"
        )
    else:
        max_steps = -1

    if val_check_period is not None:
        additional_kwargs["val_check_interval"] = val_check_period

    trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        accelerator=accelerator,
        callbacks=callbacks,
        default_root_dir=str(out_dir),
        devices=num_devices,
        gradient_clip_val=tuning_config.gradient_clipping,
        limit_val_batches=1.0 if val_path is not None else 0,
        max_epochs=max_epochs,
        max_steps=max_steps,
        num_nodes=num_nodes,
        strategy=strategy,
        **additional_kwargs,
    )

    logger.info("Start training")

    trainer.fit(training_model, datamodule=datamodule, ckpt_path=checkpoint)

    save_dir = out_dir / model_name
    logger.info(f"Saving the final model in {save_dir}")
    training_model.save_transformer(save_dir, tokenizer)


if __name__ == "__main__":
    main()
