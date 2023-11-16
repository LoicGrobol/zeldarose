Zelda Rose
==========

[![Latest PyPI version](https://img.shields.io/pypi/v/zeldarose.svg)](https://pypi.org/project/zeldarose)
[![Build Status](https://github.com/LoicGrobol/zeldarose/actions/workflows/ci.yml/badge.svg)](https://github.com/LoicGrobol/zeldarose/actions?query=workflow%3ACI)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/zeldarose/badge/?version=stable)](https://zeldarose.readthedocs.io/en/latest/?badge=stable)

A straightforward trainer for transformer-based models. In others words [a teacher of
muppets](https://muppet.fandom.com/wiki/Zelda_Rose).

> Zelda Rose is a command line interface for pretraining transformer-based models. Its purpose is to
> enable an easy start for users interested in training these ubiquitous models, but unable or
> unwilling to engage with more comprehensive --- but more complex --- frameworks and the complex
> interactions between libraries for managing models, datasets and computations. Training a model
> requires no code on the user's part and produce models directly compatible with the 🤗 ecosystem,
> allowing quick and easy distribution and reuse. A particular care is given to lowering the cost of
> maintainability and future-proofing, by making the code as modular as possible and taking
> advantage of third-party libraries to limit ad-hoc code to the strict minimum. (Grobol, [2023](https://hal.science/hal-04262806))

## Getting Started

Check out [Get started](get_started.md).

## Citation

```bibtex
@inproceedings{grobol:hal-04262806,
    TITLE = {{Zelda Rose: a tool for hassle-free training of transformer models}},
    AUTHOR = {Grobol, Lo{\"i}c},
    URL = {https://hal.science/hal-04262806},
    BOOKTITLE = {{3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS)}},
    ADDRESS = {Singapore, Indonesia},
    YEAR = {2023},
    MONTH = Dec,
    PDF = {https://hal.science/hal-04262806/file/Zeldarose_OSS_EMNLP23.pdf},
    HAL_ID = {hal-04262806},
    HAL_VERSION = {v1},
}
```
