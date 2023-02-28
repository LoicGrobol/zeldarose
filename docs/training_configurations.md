Tuning configurations
=======================

(content:references:tuning-parameters)=
## Parameters

### Batch size

- `batch_size: int = 64`.

This is the number of sample in a forward-backward pass. If you use several devices and/or have
device batches of a size bigger than $1$, this **must** be a multiple of `device_batch_size*
total_devices`

### Adam parameters

- `betas: Tuple[float, float] = (0.9, 0.98)`
- `epsilon: float = 1e-8`
- `learning_rate: float = 1e-4`
- `weight_decay: Optional[float] = None`

These are respectively the $β$ and $ε$ parameters and the base learning rate for the Adam optimizer
{cite}`kingma2014AdamMethodStochastic` and the weight decay rate. See the [Pytorch
documentation](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) for more details.


### Gradient clipping

- `gradient_clipping: Optional[Union[float, int]] = None`

If non-`None`, this is the maximum allowed gradient norm. Longer gradients will be clipped to this
length, preserving their direction. See the [Pytorch
documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) for
implementation details.

### Learning rate schedule

- `lr_decay_steps: Optional[int] = None`
- `warmup_steps: int = 0`

These are the number of step in the slanted triangular learning rate schedule
{cite}`howard2018UniversalLanguageModel`: the base learning rate is made to follow an upward linear
slope for `warmup_steps` steps up to `learning_rate`, then decayed linearly to $0$ in
`lr_decay_steps`.

## Bibliography

```{bibliography}
:filter: docname in docnames
:style: alpha
```