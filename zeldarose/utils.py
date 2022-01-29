import pathlib
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import transformers

from loguru import logger

import transformers.models.auto.configuration_auto as autoconf
import transformers.models.auto.modeling_auto as automod

config_type_to_config_name = {t: n for n, t in autoconf.CONFIG_MAPPING.items()}
model_type_to_model_name = {
    m: config_type_to_config_name[c]
    for c, m in automod.MODEL_FOR_MASKED_LM_MAPPING.items()
}


def get_internal_transformer_model(
    model: transformers.PreTrainedModel,
) -> torch.nn.Module:
    # There is no consensus in hf transformers as to how the underlying transformer of a MLM
    # model is called
    transformer_model = next(
        tr_model
        for transformer_name in (
            model_type_to_model_name[type(model)],
            "bert",
            "roberta",
            "deberta",
            "transformer",
            "model",
        )
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
        if not hasattr(self.leader_transformer, "embeddings"):
            raise ValueError(
                f"Unsupported transformer: {self.leader_transformer} has no embedding submodule"
            )
        if not hasattr(self.follower_transformer, "embeddings"):
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
        self.follower_transformer.embeddings = type(
            self.follower_transformer.embeddings
        )(self.follower_transformer.config)
        self.follower_transformer.embeddings.load_state_dict(
            self.leader_transformer.embeddings.state_dict()
        )
        # Retying weights, here again, we rely on the fact that this uses
        # the input embeddings as ground truth
        if hasattr(self.follower, "tie_weights"):
            self.follower.tie_weights()


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
        self.follower_embeddings.weight = (
            self.follower_embeddings.weight.detach().clone()
        )
