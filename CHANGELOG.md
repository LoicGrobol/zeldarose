Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project adheres to
[Semantic Versioning](http://semver.org/).

## [Unreleased] 

[Unreleased]: https://github.com/LoicGrobol/zeldarose/compare/v0.1.0...HEAD

### Added

- `--checkpoint` option to load an existing lightning checkpoint
- DDP sharding is now also possible with `ddp_spawn`

### Changed

- Text datasets are now loaded line-by-line by default and the block mode has been removed
- We now use [🤗 datasets](https://github.com/huggingface/datasets) as backend, so the datasets are
  implemented as memory-mapped files with dynamic loaders instead of being held in RAM. This
  significantly decrease RAM consumption for a very decent speed cost and allows us to train on much
  larger datasets.

### Removed

- The `--line-by-line` flag has been removed, since this is now the default behaviour.
- The `zeldarose-create-cache` has been removed, since dataset processing now works correctly in ddp
- The `data` module has been completely rewritten and the Dataset classes are no more
- `mlm.masked_accuracy` since it was not used anywhere

### Fixed

- Logging has been improved for internal pytorch warnings and pytorch-lightning and 🤗 transformers.

## [0.1.0] - 2021-04-06

### Fixed

- Updated some obsolete doc

[0.0.0]: https://github.com/LoicGrobol/zeldarose/compare/v0.1.0...v0.1.1

## [0.1.0] - 2021-04-06

Initial release

[0.0.0]: https://github.com/LoicGrobol/zeldarose/tree/v0.1.0
