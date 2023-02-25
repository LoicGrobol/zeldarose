Get started
===========

## Installation

Install with [pipx](https://pypa.github.io/pipx/) if you only need the command-line client

```bash
pipx install zeldarose
```

If you plan to use Zelda Rose together with other python programs, install it with Pip in your
project's environment

```bash
pip install zeldarose
```

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

There are other parameters (see `zeldarose transformer --help` for a comprehensive list), the one
you are probably mostly interested in is `--config`, giving the path to a training config (for which
we have [`examples`](https://github.com/LoicGrobol/zeldarose/tree/main/examples)).

