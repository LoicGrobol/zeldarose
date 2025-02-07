Get started
===========

## Installation

Install with [uv](https://docs.astral.sh/uv) if you only need the command-line client

```bash
uv tool install zeldarose
```

If you plan to use Zelda Rose together with other python programs, install it in your
project's environment

```bash
uv pip install zeldarose
```

If the model you want to use relies on [`sentencepiece`](https://github.com/google/sentencepiece),
also install that.

## Train a tokenizer

```bash
TOKENIZERS_PARALLELISM=true zeldarose tokenizer --vocab-size 4096 --out-path /tokenizer/out/path  --model-name "my-muppet" /path/to/a/raw/text/file
```

The input format is raw text files, with one sample (e.g. sentence) per line.

## Train a MLM model 

```bash
zeldarose transformer --tokenizer /tokenizer/path --pretrained-model flaubert/flaubert_small_cased --out-dir local/muppet --val-text tests/fixtures/raw.txt tests/fixtures/raw.txt
```

- The input format here is raw text files as well. 
- The arguments to `--tokenizer` and `--pretrained-model` can be either paths to local directories, or [🤗 hub](https://huggingface.co/models) model identifiers.

## Next steps

Now that you know how to train a default model on a default task and with default settings, the next step is to configure all these.

- Parameters that have to do with Zelda Rose behaviour towards the host machine (train on CPU or
  GPU, the batch size per device…) are set by command line arguments and can be accessed from
  `zeldarose transformer --help`
- Other settings are passed via a *config file*, which the next section will tell you about.

### Transformers config

If you want to train a model on another task, or if you want to configure hyperparameters, you will need to pass a configuration to `zeldarose transformer` via its `--config` option. It takes a path to a local [TOML](https://toml.io) file that provides a [tuning configuration](content:references:tuning-parameters) (optimizer hyperparameters such as batch size, learning rate, etc.) and a task configuration (masked language modelling, replaced tokens detection, etc.) with its hyperparameters.

They look like this 

```toml
type = "mlm"  # The name of the task.

[task]
change_ratio = 0.15  # The proportion of tokens to modify
mask_ratio = 0.8     # The proportion of modified tokens to mask
switch_ratio = 0.1   # The proportion of modified tokens to change to a random token

[tuning]
batch_size = 64
betas = [0.9, 0.98]
epsilon = 1e-8
learning_rate = 1e-4
```

There are example configurations for every task in the
[`examples`](https://github.com/LoicGrobol/zeldarose/tree/main/examples) directory in Zelda Rose
development repository. The options in the `task` section are documented in their respective page in “Tasks” and the tuning options in [“Tuning configurations”](content:references:tuning-parameters).

