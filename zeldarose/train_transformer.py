from __future__ import annotations

import logging
import os
import pathlib
import sys
import warnings

from typing import List, Optional, Type

import click
import click_pathlib
import pytorch_lightning as pl
import toml
import torch
import torch.nn
import torch.cuda
import transformers

from loguru import logger
from pytorch_lightning.utilities import rank_zero_only

from zeldarose import data
from zeldarose import mlm


def setup_logging(
    verbose: bool, logfile: Optional[pathlib.Path] = None, replace_warnings: bool = True
):
    logger.remove(0)  # Remove the default logger
    if verbose:
        log_level = "DEBUG"
        log_fmt = (
            "[zeldarose] "
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
            "<level>{message}</level>"
        )
    else:
        logging.getLogger(None).setLevel(logging.CRITICAL)
        log_level = "INFO"
        log_fmt = (
            "[zeldarose] "
            "<green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
            "<level>{message}</level>"
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
            colorize=False,
        )

    # Deal with stdlib.logging

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Retrieve context where the logging call occurred, this happens to be in the 6th frame upward
            logger_opt = logger.opt(depth=6, exception=record.exc_info)
            logger_opt.log(record.levelno, record.getMessage())

    transformers_logger = logging.getLogger("transformers")
    # FIXME: ugly, but is there a better way?
    transformers_logger.handlers.pop()
    transformers_logger.addHandler(InterceptHandler())

    # Deal with stdlib.warnings

    def showwarning(message, *args, **kwargs):
        logger.warning(message)

    if replace_warnings:
        warnings.showwarning = showwarning


def reset_transformer_vocab(model: transformers.PreTrainedModel):
    logger.info("Reinitializing model embeddings")
    # There is no consensus in hf transformers as to how the underlying transformer of a MLM
    # model is called
    transformer_model = next(
        tr_model
        for transformer_name in ("bert", "roberta", "transformer")
        for tr_model in [getattr(model, transformer_name, None)]
        if tr_model is not None
    )
    if isinstance(transformer_model.embeddings, torch.nn.Embedding):
        transformer_model.embeddings.reset_parameters()
    # Assume a custom huggingface embedding class
    else:
        transformer_model.embeddings = type(transformer_model.embeddings)(
            transformer_model.config
        )
    logger.info("Reinitializing LM head")
    # There is no consensus in hf transformers as to how the LM head of a MLM model is
    # called so we have to do an ugly song and dance here
    lm_head_name = next(
        layer_name
        for layer_name in ("lm_head", "cls", "pred_layer")
        if hasattr(model, layer_name)
    )
    setattr(model, lm_head_name, type(getattr(model, lm_head_name))(model.config))


class SavePretrainedModelCallback(pl.callbacks.Callback):
    def __init__(
        self,
        save_dir: pathlib.Path,
        tokenizer: transformers.PreTrainedTokenizer,
        period: int = 1,
    ):
        self.period = period
        self.save_dir = save_dir
        self.tokenizer = tokenizer

    @rank_zero_only
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: mlm.MLMFinetuner):
        if not trainer.current_epoch % self.period:
            transformer_model = pl_module.model
            epoch_save_dir = self.save_dir / f"epoch_{trainer.current_epoch}"
            logger.info(f"Saving intermediate model to {epoch_save_dir}")
            save_model(transformer_model, epoch_save_dir, self.tokenizer)


# logging.getLogger(None).setLevel(logging.ERROR)

# TODO: allow reading all these from a config file (except perhaps for paths)
# TODO: refactor the api to have a single `zeldarose` entrypoint with subcommands.
# TODO: allow restarting from checkpoint
@click.command()
@click.argument(
    "raw_text",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--config",
    "config_path",
    type=click_pathlib.Path(resolve_path=True, dir_okay=False, exists=True),
    help="A config file (in TOML format)",
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
    "--accelerator",
    type=str,
    help="The lightning accelerator to use (see lightning doc)",
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
    "--line-by-line",
    is_flag=True,
    help="Assume that the dataset is pre-segmented in sentences",
)
@click.option(
    "--max-epochs",
    type=int,
    default=2,
    help="How many steps to train for",
)
@click.option(
    "--max-steps",
    type=int,
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
    type=int,
    help="How many GPUs to train on. In ddp_cpu mode, this is the number of processes",
)
@click.option(
    "--n-nodes",
    type=int,
    default=os.environ.get("SLURM_JOB_NUM_NODES", 1),
    help="How many nodes to train on (for clusters), defaults to $SLURM_JOB_NUM_NODES if on SLURM and 1 otherwise",
)
@click.option(
    "--n-workers",
    type=int,
    default=0,
    help="How many data loading workers to use",
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
    help="Ignore pre-existing dataset cache (problematic in multiprocessing context)",
)
@click.option(
    "--pretrained-model",
    type=str,
    help="A pretrained model to fine-tune",
    metavar="NAME_OR_PATH",
)
@click.option(
    "--reset-vocab",
    is_flag=True,
    help="Re-init the pretrained model embeddings and LM head layers (to train using a different tokenizer)",
)
@click.option(
    "--save-period",
    type=int,
    help="The number of epoch between intermediate model saving",
    default=0,
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
    "--val-text",
    "val_path",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
    help="A raw corpus for validation",
)
@click.option("--verbose", is_flag=True, help="More detailed logs")
def main(
    config_path: Optional[pathlib.Path],
    device_batch_size: Optional[int],
    accelerator: Optional[str],
    guess_batch_size: bool,
    line_by_line: bool,
    max_epochs: int,
    max_steps: Optional[int],
    model_config_path: Optional[str],
    model_name: str,
    n_gpus: int,
    n_nodes: int,
    n_workers: int,
    out_dir: pathlib.Path,
    overwrite_cache: bool,
    pretrained_model: Optional[str],
    profile: bool,
    raw_text: pathlib.Path,
    reset_vocab: bool,
    save_period: int,
    tokenizer_name: Optional[str],
    val_path: Optional[pathlib.Path],
    verbose: bool,
):
    setup_logging(verbose, out_dir / "train.log")
    if config_path is not None:
        config = toml.loads(config_path.read_text())
        task_config = mlm.MLMTaskConfig.parse_obj(config.get("task", dict()))
        tuning_config = mlm.MLMFinetunerConfig.parse_obj(config.get("tuning", dict()))
    else:
        task_config = mlm.MLMTaskConfig()
        tuning_config = mlm.MLMFinetunerConfig()

    # NOTE: this is likely duplicated somewhere in pl codebase but we need it now unless pl rolls
    # out something like `optim_batch_size` that takes into account the number of tasks and the
    # number of samples per gpu
    num_slurm_tasks = os.environ.get("SLURM_NTASKS")
    if num_slurm_tasks is not None:
        n_devices = int(num_slurm_tasks)
    elif n_gpus:
        n_devices = n_nodes * n_gpus
    else:
        n_devices = 1
    logger.info(f"Training on {n_devices} devices.")

    if pretrained_model is not None:
        logger.info(f"Loading pretrained model {pretrained_model!r}")
        model = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_model)
        if reset_vocab:
            reset_transformer_vocab(model)
    elif model_config_path is not None:
        logger.info(f"Loading pretrained config {model_config_path!r}")
        model_config = transformers.AutoConfig.from_pretrained(model_config_path)
        logger.info("Generating model from config")
        model = transformers.AutoModelForMaskedLM.from_config(model_config)
    else:
        raise ValueError("You must provide either a pretrained model or a model config")
    model.train()

    if tokenizer_name is None:
        if pretrained_model is not None:
            tokenizer_name = pretrained_model
        else:
            raise ValueError("Missing both pretrained tokenizer and pretrained model")
    logger.info(f"Loading pretrained tokenizer {tokenizer_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True
    )

    dataset_type: Type[data.TextDataset]
    if line_by_line:
        dataset_type = data.LineByLineTextDataset
    else:
        dataset_type = data.TextDataset
    logger.info(f"Loading train dataset from {raw_text}")
    train_set = dataset_type(
        tokenizer=tokenizer,
        text_path=raw_text,
        model_name=tokenizer_name.replace("/", "_"),
        overwrite_cache=overwrite_cache,
    )
    val_set: Optional[data.TextDataset]
    if val_path is not None:
        val_set = dataset_type(
            tokenizer=tokenizer,
            text_path=val_path,
            model_name=tokenizer_name.replace("/", "_"),
            overwrite_cache=overwrite_cache,
        )
    else:
        val_set = None

    logger.info("Creating MLM Finetuner")
    mask_token_index = getattr(tokenizer, "mask_token_id", None)
    if mask_token_index is None:
        mask_token_index = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    logger.debug(f"Mask token index: {mask_token_index}")
    finetuning_model = mlm.MLMFinetuner(
        model,
        mask_token_index=mask_token_index,
        vocabulary_size=len(tokenizer),
        config=tuning_config,
        task_config=task_config,
    )

    if device_batch_size is None:
        device_batch_size = tuning_config.batch_size
    elif tuning_config.batch_size % device_batch_size * n_devices:
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

    # In DP mode, every batch is split between the devices
    if accelerator == "dp":
        loader_batch_size = device_batch_size * n_devices
    else:
        loader_batch_size = device_batch_size

    logger.info("Creating dataloaders")
    train_loader = data.TextLoader(
        train_set,
        batch_size=loader_batch_size,
        num_workers=n_workers,
        shuffle=True,
    )

    val_loaders: Optional[List[data.TextLoader]]
    if val_set is not None:
        val_loaders = [
            data.TextLoader(
                val_set,
                batch_size=loader_batch_size,
                num_workers=n_workers,
                shuffle=False,
            )
        ]
    else:
        val_loaders = None

    logger.info("Creating trainer")
    additional_kwargs = dict()
    if profile:
        logger.info("Running in profile mode")
        profiler = pl.profiler.AdvancedProfiler(output_filename=out_dir / "profile.txt")
        additional_kwargs.update({"profiler": profiler, "overfit_batches": 1024})

    if guess_batch_size:
        logger.info("Automatic batch size selection")
        additional_kwargs.update({"auto_scale_batch_size": "binsearch"})

    if accelerator == "ddp_cpu":
        # FIXME: works but seems like bad practice
        additional_kwargs["num_processes"] = n_gpus
        n_gpus = 0

    callbacks: List[pl.callbacks.Callback] = [
        pl.callbacks.ProgressBar(),
    ]
    if save_period:
        callbacks.append(
            SavePretrainedModelCallback(
                out_dir / "partway_models",
                tokenizer,
                save_period,
            )
        )

    if n_gpus:
        logger.info(f"Training the model on {n_gpus} GPUs")
        callbacks.append(pl.callbacks.GPUStatsMonitor())
    elif accelerator == "ddp_cpu":
        logger.info(
            f"Training the model on CPU in {additional_kwargs['num_processes']} processes"
        )
    else:
        logger.info("Training the model on CPU")

    trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        default_root_dir=out_dir,
        accelerator=accelerator,
        gpus=n_gpus,
        num_nodes=n_nodes,
        max_epochs=max_epochs,
        max_steps=max_steps,
        limit_val_batches=1.0 if val_loaders is not None else 0,
        **additional_kwargs,
    )

    # Hotfix for https://github.com/PyTorchLightning/pytorch-lightning/issues/2100
    if val_loaders is None:
        finetuning_model.validation_step = pl.LightningModule.validation_step  # type: ignore
        finetuning_model.validation_end = pl.LightningModule.validation_end  # type: ignore

    trainer.fit(
        finetuning_model, train_dataloader=train_loader, val_dataloaders=val_loaders
    )

    save_dir = out_dir / model_name
    save_model(model, save_dir, tokenizer)


@rank_zero_only
def save_model(
    model: transformers.PreTrainedModel,
    save_dir: pathlib.Path,
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
):
    """Save a transformer model."""
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {save_dir}")
    model.save_pretrained(save_dir)
    if tokenizer is not None:
        logger.info(f"Saving tokenizer to {save_dir}")
        tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
