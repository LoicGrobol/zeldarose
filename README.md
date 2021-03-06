# Zelda Rose

A trainer for transformer-based models.

## Installation

Simply install with pip (preferably in a virtual env, you know the drill)

```console
pip install git+https://github.com/LoicGrobol/zeldarose.git
```

## Train a model

Here is a short example:

```console
zeldarose-transformer --tokenizer roberta-base --pretrained-model roberta-large --out-dir my/experiment/dir my_raw_corpus.txt
```

There are other parameters (see `zeldarose-transformer --help` for a comprehensive list), the one you are probably mostly interested in is `--config` (for which there is an example target in [`examples/`](examples)).

The parameters `--pretrained-models`, `--tokenizer` and `--model-config` are all fed directly to [Huggingface's `transformers`](https://huggingface.co/transformers) and can be [pretrained models](https://huggingface.co/transformers/pretrained_models.html) names or local path.

## Distributed training

This is somewhat tricky, you have several options

- If you are running in a SLURM cluster use `--distributed-backend ddp` and invoke via `srun`
- Otherwise you have two options

  - Run with `--distributed-backend ddp_spawn`, which uses `multiprocessing.spawn` to start the process swarm (tested, but possibly slower and more limited, see `pytorch-lightning` doc)
  - Run with `--distributed-backend ddp` and start with `torch.distributed.launch` with `--use_env` and `--no_python` (untested)

Whatever you do, for now it's safer to run once without distributed training in order to preprocess
the raw texts in a predictable environment.

## Inspirations

- <https://github.com/shoarora/lmtuners>
- <https://github.com/huggingface/transformers/blob/243e687be6cd701722cce050005a2181e78a08a8/examples/run_language_modeling.py>
