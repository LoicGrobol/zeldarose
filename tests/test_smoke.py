import pathlib
from typing import List, Optional, Tuple, Union

import torch.cuda

import pytest
import pytest_console_scripts


accelerators_strategies_devices = [
    ("cpu", None, None),
    ("cpu", "ddp_spawn", 2),
]
if torch.cuda.is_available():
    accelerators_strategies_devices.append(("gpu", None, None))
    if torch.cuda.device_count() > 1:
        accelerators_strategies_devices.extend(
            [("gpu", "ddp_spawn", 2), ("gpu", "ddp_sharded_spawn", 2)]
        )


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


@pytest.mark.parametrize(
    "accelerators_strategies_devices",
    [
        pytest.param(v, id="+".join(map(str, v)))
        for v in accelerators_strategies_devices
    ],
)
def test_train_mlm(
    accelerators_strategies_devices: Tuple[str, Optional[str], Optional[int]],
    mlm_model_config: Union[pathlib.Path, str],
    mlm_task_config: pathlib.Path,
    raw_text_path: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
    tokenizer_name_or_path: Union[pathlib.Path, str],
):
    accelerator, strategy, devices = accelerators_strategies_devices
    extra_args: List[str] = []
    if strategy is not None:
        extra_args.extend(["--strategy", strategy])
    if devices is not None:
        extra_args.extend(["--num-devices", str(devices)])

    ret = script_runner.run(
        "zeldarose-transformer",
        "--accelerator",
        accelerator,
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
        *extra_args,
    )
    assert ret.success


@pytest.mark.parametrize(
    "accelerators_strategies_devices",
    [
        pytest.param(v, id="+".join(map(str, v)))
        for v in accelerators_strategies_devices
    ],
)
def test_train_rtd(
    accelerators_strategies_devices: Tuple[str, Optional[str], Optional[int]],
    rtd_model_config: Union[pathlib.Path, str],
    rtd_task_config: pathlib.Path,
    raw_text_path: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
    tokenizer_name_or_path: Union[pathlib.Path, str],
):
    accelerator, strategy, num_devices = accelerators_strategies_devices

    extra_args: List[str] = []
    if strategy is not None:
        extra_args.extend(["--strategy", strategy])
    if num_devices is not None:
        extra_args.extend(["--num-devices", str(num_devices)])
    ret = script_runner.run(
        "zeldarose-transformer",
        "--accelerator",
        accelerator,
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
        *extra_args,
    )
    assert ret.success
