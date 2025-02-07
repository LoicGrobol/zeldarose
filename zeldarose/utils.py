import logging
import os
import pathlib
import subprocess
import sys
import warnings
from types import FrameType
from typing import Callable, Dict, Optional, TextIO, Union

import loguru
import pytorch_lightning as pl
import rich
import torch
import transformers
from loguru import logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


# FIXME: don't assume pip is available
def dump_environment(output_dir: pathlib.Path):
    logger.info(f"Saving environement info in {output_dir}.")
    with open(output_dir / "frozen_requirements.txt", "w") as out_stream:
        try:
            subprocess.run(["python", "-m", "pip", "freeze"], stdout=out_stream, check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Running pip exited with code {e.returncode}")
    with open(output_dir / "command.txt", "w") as out_stream:
        out_stream.write(str(sys.argv))
    with open(output_dir / "platform.txt", "w") as out_stream:
        # out_stream.write(str(sysinfo.sysInfo))
        # out_stream.write("\n")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                out_stream.write(f"gpu #{i}: {torch.cuda.get_device_name(i)}")
                out_stream.write("\n")
        else:
            out_stream.write("No GPU available\n")
    with open(output_dir / "env.txt", "w") as out_stream:
        out_stream.write(str(os.environ))


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
        frame: Optional[FrameType] = logging.currentframe()
        depth = 2
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
                    level, f"({self.wrapped_name}) {record.getMessage()}"
                )
            else:
                logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    appname: str = "zeldarose",
    console: Optional[rich.console.Console] = None,
    log_file: Optional[pathlib.Path] = None,
    replace_warnings: bool = True,
    sink: Union[
        Callable[["loguru.Message"], None],
        logging.Handler,
        "loguru.PathLikeStr",
        str,
        TextIO,
        "loguru.Writable",
    ]
    | None = None,
    verbose: bool = False,
) -> list[int]:
    res: list[int] = []
    if console is None:
        console = rich.get_console()
    if sink is None:
        sink = lambda m: console.print(m, end="")  # noqa: E731
    logger.remove()  # Remove the default logger
    if "SLURM_JOB_ID" in os.environ:
        local_id = os.environ.get("SLURM_LOCALID", "someproc")
        node_name = os.environ.get("SLURMD_NODENAME", "somenode")
        appname = (
            f"{appname} ({os.environ.get('SLURM_PROCID', 'somerank')} [{local_id}@{node_name}])"
        )

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

    res.append(
        logger.add(
            sink,
            colorize=True,
            enqueue=True,
            format=log_fmt,
            level=log_level,
        )
    )

    if log_file:
        res.append(
            logger.add(
                log_file,
                colorize=False,
                enqueue=True,
                format=(
                    f"[{appname}] {{time:YYYY-MM-DD HH:mm:ss.SSS}} | {{level: <8}} | {{message}}"
                ),
                level="DEBUG",
            )
        )

    # Deal with stdlib.logging
    # Yes, listing them all is annoying
    for libname in (
        "datasets",
        "huggingface/tokenizers",
        "huggingface_hub",
        "lightning",
        "lightning.pytorch",
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
        logger.debug(f"Intercepting logging from {libname}")

    # Deal with stdlib.warnings
    def showwarning(message, category, filename, lineno, file=None, line=None):
        logger.warning(warnings.formatwarning(message, category, filename, lineno, None).strip())

    if replace_warnings:
        warnings.showwarning = showwarning

    return res


def get_internal_transformer_model(
    model: transformers.PreTrainedModel,
) -> torch.nn.Module:
    # There is no consensus in hf transformers as to how the underlying transformer of a MLM
    # model is called
    transformer_model = next(
        tr_model
        for transformer_name in ("bert", "electra", "roberta", "transformer")
        for tr_model in [getattr(model, transformer_name, None)]
        if tr_model is not None
    )
    return transformer_model


def reset_transformer_vocab(model: transformers.PreTrainedModel):
    logger.info("Reinitializing model embeddings")
    transformer_model = get_internal_transformer_model(model)
    if isinstance(transformer_model.embeddings, torch.nn.Embedding):
        transformer_model.embeddings.reset_parameters()
    # Assume a custom huggingface embedding class
    else:
        transformer_model.embeddings = type(transformer_model.embeddings)(transformer_model.config)
    logger.info("Reinitializing LM head")
    # There is no consensus in hf transformers as to how the LM head of a MLM model is
    # called so we have to do an ugly song and dance here
    lm_head_name = next(
        layer_name for layer_name in ("lm_head", "cls", "pred_layer") if hasattr(model, layer_name)
    )
    setattr(model, lm_head_name, type(getattr(model, lm_head_name))(model.config))


@rank_zero_only
def save_model(
    model: transformers.PreTrainedModel,
    save_dir: pathlib.Path,
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
):
    """Save a transformer model."""
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {save_dir}")
    model.save_pretrained(str(save_dir))
    if tokenizer is not None:
        logger.info(f"Saving tokenizer to {save_dir}")
        tokenizer.save_pretrained(str(save_dir), legacy_format=not tokenizer.is_fast)


class ShareTransformersEmbeddingsCallback(pl.Callback):
    def __init__(
        self,
        leader: transformers.PreTrainedModel,
        follower: transformers.PreTrainedModel,
    ):
        self.leader = leader
        self.follower = follower
        self.leader_transformer = get_internal_transformer_model(leader)
        self.follower_transformer = get_internal_transformer_model(follower)
        if not hasattr(self.leader_transformer, "embeddings") or not isinstance(
            self.leader_transformer, torch.nn.Module
        ):
            raise ValueError(
                f"Unsupported transformer: {self.leader_transformer} has no embedding submodule"
            )
        if not hasattr(self.follower_transformer, "embeddings") or not isinstance(
            self.follower_transformer, torch.nn.Module
        ):
            raise ValueError(
                f"Unsupported transformer: {self.follower_transformer} has no embedding submodule"
            )
        if not isinstance(
            self.leader_transformer.embeddings,
            type(self.follower_transformer.embeddings),
        ):
            logger.warning(
                "Sharing embeddings between different model types might not work well"
                f" {type(self.follower_transformer.embeddings)} vs {type(self.leader_transformer.embeddings)}"
            )

    def on_train_start(self, trainer, pl_module):
        self.follower_transformer.embeddings = self.leader_transformer.embeddings
        # Tying weights if needed
        # This works because in 🤗, embeddings tying makes output embeddings follow input embeddings
        # so here everyone will transitively follow the leader's embeddings
        if hasattr(self.follower, "tie_weights"):
            self.follower.tie_weights()

    def on_train_end(self, trainer, pl_module):
        self.follower_transformer.embeddings = type(self.follower_transformer.embeddings)(
            self.follower_transformer.config
        )
        self.follower_transformer.embeddings.load_state_dict(
            self.leader_transformer.embeddings.state_dict()
        )
        # Retying weights, here again, we rely on the fact that this uses
        # the input embeddings as ground truth
        if hasattr(self.follower, "tie_weights"):
            self.follower.tie_weights()


class CombiningLayer(torch.nn.Module):
    def __init__(self, leader: torch.nn.Module, follower: torch.nn.Module):
        super().__init__()
        self.leader = leader
        self.follower = follower

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            leader_out = self.leader(*args, **kwargs)
        follower_out = self.follower(*args, **kwargs)

        return leader_out + follower_out


class OneWayShareTransformersEmbeddingsCallback(pl.Callback):
    def __init__(
        self,
        leader: transformers.PreTrainedModel,
        follower: transformers.PreTrainedModel,
    ):
        self.leader = leader
        self.follower = follower
        self.leader_transformer = get_internal_transformer_model(leader)
        self.follower_transformer = get_internal_transformer_model(follower)
        if not hasattr(self.leader_transformer, "embeddings") or not isinstance(
            self.leader_transformer, torch.nn.Module
        ):
            raise ValueError(
                f"Unsupported transformer: {self.leader_transformer} has no embedding submodule"
            )
        if not hasattr(self.follower_transformer, "embeddings") or not isinstance(
            self.follower_transformer, torch.nn.Module
        ):
            raise ValueError(
                f"Unsupported transformer: {self.follower_transformer} has no embedding submodule"
            )
        if not isinstance(
            self.leader_transformer.embeddings,
            type(self.follower_transformer.embeddings),
        ):
            logger.warning(
                "Sharing embeddings between different model types might not work well:"
                f" {type(self.follower_transformer.embeddings)} vs {type(self.leader_transformer.embeddings)}"
            )

        self.replaced_layer: Dict[str, CombiningLayer] = dict()

    def on_train_start(self, trainer, pl_module):
        for layer_name, layer in self.follower.embeddings.named_children():
            if isinstance(layer, torch.nn.Embedding):
                logger.debug(f"One-way sharing of {layer_name}.")
                torch.nn.init.zeros_(layer)
                leader_equiv = getattr(self.leader.embeddings, layer_name)
                combiner = CombiningLayer(leader=leader_equiv, follower=layer)
                setattr(self.follower.embeddings, layer_name, combiner)
                self.replaced_layer[layer_name] = combiner

        if not self.replaced_layer:
            logger.warning("No embeddings actually shared, you should probably check your config.")

    def on_train_end(self, trainer, pl_module):
        for layer_name, combiner in self.replaced_layer.items():
            combiner.follower.weight += combiner.leader.weight
            setattr(self.follower.embeddings, layer_name, combiner.follower)
        # Retying weights, here again, we rely on the fact that this uses
        # the input embeddings as ground truth
        if hasattr(self.follower, "tie_weights"):
            self.follower.tie_weights()
        self.replaced_layer = dict()


class TieEmbeddingsCallback(pl.Callback):
    def __init__(
        self,
        leader_embeddings: torch.nn.Embedding,
        follower_embeddings: torch.nn.Embedding,
    ):
        if leader_embeddings.weight.shape != follower_embeddings.weight.shape:
            raise ValueError(
                "Embeddings must have the same shape to be tied:"
                " {leader_embeddings.weight.shape} vs {follower_embeddings.weight.shape}"
            )
        self.leader_embeddings = leader_embeddings
        self.follower_embeddings = follower_embeddings

    def on_train_start(self, trainer, pl_module):
        self.follower_embeddings.weight = self.leader_embeddings.weight

    def on_train_end(self, trainer, pl_module):
        self.follower_embeddings.weight = self.follower_embeddings.weight.detach().clone()
