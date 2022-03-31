Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project adheres to
[Semantic Versioning](http://semver.org/).

## [Unreleased]

[Unreleased]: https://github.com/LoicGrobol/zeldarose/compare/v0.5.0...HEAD

## [0.5.0] — 2022-03-31

[0.4.0]: https://github.com/LoicGrobol/zeldarose/compare/v0.4.0...v0.5.0

### Added

- `lint` extra that install linting tools and plugins
- Config for [flakeheaven](https://github.com/flakeheaven/flakeheaven)
- Support for [`pytorch-lightning
  1.6`](https://github.com/PyTorchLightning/pytorch-lightning/releases/tag/1.6.0)

### Changed

- Move packaging config to `pyproject.toml` and require `setuptools>=61`.
- `click_pathlib` is no longer a dependency and `click` has a minimal version of `8.0.3`

## [0.4.0] — 2022-03-18

[0.4.0]: https://github.com/LoicGrobol/zeldarose/compare/v0.3.4...v0.4.0

### Added

- Replaced Token Detection ([ELECTRA](https://arxiv.org/abs/2003.10555)-like) pretraining
  - Some of the API is still provisional, the priority was to get it out, a nicer interface will
    hopefully come later.
- `--val-check-period` and `--step-save-period` allowing to evaluate and save a model decoupled
  from epochs. This should be useful for training with very long epochs.
- The datasets path in `zeldarose-transformer` can now be 🤗 hub handles. See `--help`.

### Changed

- The command line options have been changed to reflect change in Lightning
  - `--accelerator` is now used for devices, tested values are `"cpu"` and `"gpu"`
  - `--strategy` now specifies how to train, tested values are `None` (missing), `"ddp"`,
    `"ddp_sharded"` `"ddp_spawn"` and`"ddp_sharded_spawn"`.
  - No more option to select sharded training, use the strategy alias for that
  - `--n-gpus` has been renamed to `--num-devices`.
  - `--n-workers` and `--n-nodes` have been respectively renamed to `--num-workers` and
    `--num-nodes`.
- Training task configs now have a `type` config key to specify the task type
- Lightning progress bars are now provided by [Rich](https://rich.readthedocs.io)
- Now supports Pytorch 1.11 and Python 3.10

### Internal

- Tests now run in [Pytest](https://pytest.org) using the [console-scripts
  plugin](https://github.com/kvas-it/pytest-console-scripts) for smoke tests.
- Smoke tests now include `ddp_spawn` tests and tests on gpu devices if available.
- Some refactoring for better factorization of the common utilities for MLM and RTD.

## [0.3.4] — 2021-12-21

[0.3.4]: https://github.com/LoicGrobol/zeldarose/compare/v0.3.3...v0.3.4

## Changed

- Bump lightning to 1.5.x

## [0.3.3] — 2021-11-01

[0.3.3]: https://github.com/LoicGrobol/zeldarose/compare/v0.3.2...v0.3.3

### Changed

- `max_steps` is automatically inferred from the tuning config if a number of lr decay steps is
  given
- `max_epochs` is now optional (if both `max_steps` and `max_epochs` are unset and no lr schedule is
  provided, Lightning's default will be used)
- `find_unused_parameters` is now disabled in DDP mode, unless in profile mode
- Bumped lightning to 1.4.x

### Fixed

- Linear decay now properly takes the warmup period into account

## [0.3.2] — 2021-05-31

[0.3.2]: https://github.com/LoicGrobol/zeldarose/compare/v0.3.1...v0.3.2

### Fixed

- Accuracy should stop NaN-ing
- Empty lines in datasets are now ignored

## [0.3.1] — 2021-05-19

[0.3.1]: https://github.com/LoicGrobol/zeldarose/compare/v0.3.0...v0.3.1

### Fixed

- Stop saving tokenizers in legacy format *also* when training transformers
- The RoBERTa tokenizers now correctly use ByteLevel processing, to make it consistent with 🤗
  transformers
- Add back automatic truncation of inputs in training transformers

### Removed

- The `--overwrite-cache` option, which was a no-op since 0.2.0 has been removed. Resetting the
  cache should be done manually if needed (but usually shouldn't be needed).

## [0.3.0] — 2021-04-23

[0.3.0]: https://github.com/LoicGrobol/zeldarose/compare/v0.3.0...v0.2.0

### Changed

- Stop saving tokenizers in legacy format
- Create data dir if they don't exist

## [0.2.0] — 2021-04-23

[0.2.0]: https://github.com/LoicGrobol/zeldarose/compare/v0.2.0...v0.1.1

### Added

- `--checkpoint` option to load an existing lightning checkpoint
- DDP sharding is now also possible with `ddp_spawn`

### Changed

- Text datasets are now loaded line-by-line by default and the block mode has been removed.
- We now use [🤗 datasets](https://github.com/huggingface/datasets) as backend, so the datasets are
  implemented as memory-mapped files with dynamic loaders instead of being held in RAM. This
  significantly decrease RAM consumption for a very decent speed cost and allows us to train on much
  larger datasets.
- GPU usage is now logged in `--profile` mode when relevant.
- LR is now logged.

### Removed

- The `--line-by-line` flag has been removed, since this is now the default behaviour.
- The `zeldarose-create-cache` has been removed, since dataset processing now works correctly in
  ddp.
- The `data` module has been completely rewritten and the Dataset classes are no more.
- `mlm.masked_accuracy` since it was not used anywhere.

### Fixed

- Logging has been improved for internal pytorch warnings and pytorch-lightning and 🤗 transformers.

## [0.1.1] — 2021-04-06

### Fixed

- Updated some obsolete doc

[0.1.1]: https://github.com/LoicGrobol/zeldarose/compare/v0.1.0...v0.1.1

## [0.1.0] — 2021-04-06

Initial release

[0.1.0]: https://github.com/LoicGrobol/zeldarose/tree/v0.1.0
