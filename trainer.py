from typing import NamedTuple, Optional

import torch
import torch.jit
import transformers


def get_keep_mask(
    inputs: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer
) -> torch.Tensor:
    """Takes a batch of MLM inputs and return a mask for the tokens that should no be
    changed: special tokens and possibly padding.
    """
    special_tokens_mask = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in inputs.tolist()
        ],
        dtype=torch.bool,
    )
    if tokenizer._pad_token is not None:
        padding_mask = inputs.eq(tokenizer.pad_token_id)
        keep_mask = special_tokens_mask | padding_mask
    else:
        keep_mask = special_tokens_mask
    return keep_mask


class MLMBatch(NamedTuple):
    inputs: torch.Tensor
    labels: torch.Tensor


# FUTURE: use `torch.logical_xxx` functions in torch >= 1.5.0
@torch.jit.script
def mask_tokens(
    inputs: torch.Tensor,
    mask_indice: int,
    vocabulary_size: int,
    change_ratio: float = 0.15,
    mask_ratio: float = 0.8,
    switch_ratio: float = 0.1,
    keep_mask: Optional[torch.Tensor] = None,
) -> MLMBatch:
    """Prepare masked tokens inputs/labels for masked language modeling
    
    This modifies `inputs` in place, which is not very pure but avoids a (useless in practice) copy
    operation.
    """

    labels = inputs.clone()
    # Tells us what to do with each token according to its value `v` in `what_to_do`
    # - `v <= change_ratio` change the token and use it in the loss
    #   - `v <= change_ratio * mask_ratio`: replace with [MASK] ('s id)
    #   - `change_ratio * mask_ratio < v <= change_ratio * (mask_ratio + switch_ratio)`: replace with a random word
    #   - `change_ratio * (mask_ratio + switch_ratio) < v <= change_ratio`: keep as is
    # - `change_ratio < v`: keep as is and don't use in the loss
    what_to_do = torch.rand_like(labels)
    if keep_mask is not None:
        preserved_tokens = keep_mask | what_to_do.gt(change_ratio)
    else:
        preserved_tokens = what_to_do.gt(change_ratio)
    labels.masked_fill_(preserved_tokens, -100)  # We only compute loss on masked tokens

    # replace some input tokens with tokenizer.mask_token ([MASK])
    masked_tokens = what_to_do.le(change_ratio * mask_ratio)
    inputs.masked_fill_(masked_tokens, mask_indice)

    # Replace masked input tokens with random word
    switched_tokens = (
        what_to_do.le(change_ratio * (mask_ratio + switch_ratio))
        & masked_tokens.logical_not()
    )
    random_words = torch.randint_like(labels, vocabulary_size)
    inputs[switched_tokens] = random_words[switched_tokens]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return MLMBatch(inputs, labels)
