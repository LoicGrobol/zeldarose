import pathlib
from typing import Union

import torch.cuda

import pytest
import pytest_console_scripts

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")


def test_train_tokenizer(
    raw_text_path: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
):
    ret = script_runner.run(
        "zeldarose-tokenizer",
        "--vocab-size",
        "4096",
        "--out-path",
        str(tmp_path / "tokenizer"),
        "--model-name",
        "my-muppet",
        str(raw_text_path),
    )
    assert ret.success


def test_train_mlm(
    mlm_model_config: Union[pathlib.Path, str],
    mlm_task_config: pathlib.Path,
    raw_text_path: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
    tokenizer_name_or_path: Union[pathlib.Path, str],
):
    ret = script_runner.run(
        "zeldarose-transformer",
        "--config",
        str(mlm_task_config),
        "--tokenizer",
        str(tokenizer_name_or_path),
        "--model-config",
        str(mlm_model_config),
        "--device-batch-size",
        "8",
        "--out-dir",
        str(tmp_path / "train-out"),
        "--cache-dir",
        str(tmp_path / "tokenizer-cache"),
        "--val-text",
        str(raw_text_path),
        str(raw_text_path),
        "--max-epochs",
        "2",
    )
    assert ret.success


def test_train_rtd(
    rtd_model_config: Union[pathlib.Path, str],
    rtd_task_config: pathlib.Path,
    raw_text_path: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
    tokenizer_name_or_path: Union[pathlib.Path, str],
):
    ret = script_runner.run(
        "zeldarose-transformer",
        "--config",
        str(rtd_task_config),
        "--tokenizer",
        str(tokenizer_name_or_path),
        "--model-config",
        str(rtd_model_config),
        "--device-batch-size",
        "8",
        "--out-dir",
        str(tmp_path / "train-out"),
        "--cache-dir",
        str(tmp_path / "tokenizer-cache"),
        "--val-text",
        str(raw_text_path),
        str(raw_text_path),
        "--max-epochs",
        "2",
    )
    assert ret.success
