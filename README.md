Zelda Rose
==========

[![Latest PyPI version](https://img.shields.io/pypi/v/zeldarose.svg)](https://pypi.org/project/zeldarose)
[![Build Status](https://github.com/LoicGrobol/zeldarose/actions/workflows/ci.yml/badge.svg)](https://github.com/LoicGrobol/zeldarose/actions?query=workflow%3ACI)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A straightforward trainer for transformer-based models.

## Installation

Simply install with pip

```console
pip install zeldarose
```

## Train MLM models

Here is a short example of training first a tokenizer, then a transformer MLM model:

```console
TOKENIZERS_PARALLELISM=true zeldarose tokenizer --vocab-size 4096 --out-path local/tokenizer  --model-name "my-muppet" tests/fixtures/raw.txt
zeldarose 
transformer --tokenizer local/tokenizer --pretrained-model flaubert/flaubert_small_cased --out-dir local/muppet --val-text tests/fixtures/raw.txt tests/fixtures/raw.txt
```

The `.txt` files are meant to be raw text files, with one sample (e.g. sentence) per line.

There are other parameters (see `zeldarose transformer --help` for a comprehensive list), the one
you are probably mostly interested in is `--config`, giving the path to a training config (for which
we have [`examples/`](examples)).

The parameters `--pretrained-models`, `--tokenizer` and `--model-config` are all fed directly to
[Huggingface's `transformers`](https://huggingface.co/transformers) and can be [pretrained
models](https://huggingface.co/transformers/pretrained_models.html) names or local path.


## Distributed training

This is somewhat tricky, you have several options

- If you are running in a SLURM cluster use `--strategy ddp` and invoke via `srun`
  - You might want to preprocess your data first outside of the main compute allocation. The
    `--profile` option might be abused for that purpose, since it won't run a full training, but
    will run any data preprocessing you ask for. It might also be beneficial at this step to load a
    placeholder model such as
    [RoBERTa-minuscule](https://huggingface.co/lgrobol/roberta-minuscule/tree/main) to avoid runnin
    out of memory, since the only thing that matter for this preprocessing is the tokenizer.
- Otherwise you have two options

  - Run with `--strategy ddp_spawn`, which uses `multiprocessing.spawn` to start the process
    swarm (tested, but possibly slower and more limited, see `pytorch-lightning` doc)
  - Run with `--strategy ddp` and start with `torch.distributed.launch` with `--use_env` and
    `--no_python` (untested)

## Other hints

- Data management relies on 🤗 datasets and use their cache management system. To run in a clear
  environment, you might have to check the cache directory pointed to by the`HF_DATASETS_CACHE`
  environment variable.

## Inspirations

- <https://github.com/shoarora/lmtuners>
- <https://github.com/huggingface/transformers/blob/243e687be6cd701722cce050005a2181e78a08a8/examples/run_language_modeling.py>
