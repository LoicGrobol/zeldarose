Zelda Rose
==========

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

There are other parameters (see `zeldarose-transformer --help` for a comprehensive list), the ones
you are probably mostly interested in are `--task-config` and `--tuning-config` (for which there are
examples target in [`examples/`](examples)).

The parameters `--pretrained-models`, `--tokenizer` and `--model-config` are all fed directly to
[Huggingface's `transformers`](https://huggingface.co/transformers) and can be [pretrained
models](https://huggingface.co/transformers/pretrained_models.html) names or local path.

## Inspirations

- https://github.com/shoarora/lmtuners
- https://github.com/huggingface/transformers/blob/243e687be6cd701722cce050005a2181e78a08a8/examples/run_language_modeling.py