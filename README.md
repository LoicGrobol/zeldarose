Zelda Rose
==========

[![Latest PyPI version](https://img.shields.io/pypi/v/zeldarose.svg)](https://pypi.org/project/zeldarose)
[![Build Status](https://github.com/LoicGrobol/zeldarose/actions/workflows/ci.yml/badge.svg)](https://github.com/LoicGrobol/zeldarose/actions?query=workflow%3ACI)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A trainer for transformer-based models.

## Installation

Simply install with pip (preferably in a virtual env, you know the drill)

```console
pip install zeldarose
```

## Train a model

Here is a short example:

```console
TOKENIZERS_PARALLELISM=true zeldarose-tokenizer --vocab-size 4096 --out-path local/tokenizer  --model-name "my-muppet" tests/fixtures/raw.txt
zeldarose-transformer --tokenizer local/tokenizer --pretrained-model flaubert/flaubert_small_cased --out-dir local/muppet --val-text tests/fixtures/raw.txt tests/fixtures/raw.txt
```

There are other parameters (see `zeldarose-transformer --help` for a comprehensive list), the one
you are probably mostly interested in is `--config` (for which there is an example target in
[`examples/`](examples)).

The parameters `--pretrained-models`, `--tokenizer` and `--model-config` are all fed directly to
[Huggingface's `transformers`](https://huggingface.co/transformers) and can be [pretrained
models](https://huggingface.co/transformers/pretrained_models.html) names or local path.

## Distributed training

This is somewhat tricky, you have several options

- If you are running in a SLURM cluster use `--strategy ddp` and invoke via `srun`
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
