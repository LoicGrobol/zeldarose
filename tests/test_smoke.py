import os
import pathlib
from typing import List, Optional, Tuple, Union

import pytest
import pytest_console_scripts
import torch.cuda
from pytorch_lightning.overrides.fairscale import _FAIRSCALE_AVAILABLE

accelerators_strategies_devices = [
    ("cpu", None, None),
    ("cpu", "ddp_spawn", 2),
]
if torch.cuda.is_available():
    accelerators_strategies_devices.append(("gpu", None, None))
    if torch.cuda.device_count() > 1:
        accelerators_strategies_devices.append(("gpu", "ddp_spawn", 2))
        # accelerators_strategies_devices.append(("gpu", "fsdp_native", 2))
        if _FAIRSCALE_AVAILABLE:
            accelerators_strategies_devices.append(("gpu", "ddp_sharded_spawn", 2))


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
    [pytest.param(v, id="+".join(map(str, v))) for v in accelerators_strategies_devices],
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
        env={"TORCH_DISTRIBUTED_DEBUG": "DETAIL", **os.environ},
    )
    assert ret.success


@pytest.mark.parametrize(
    "accelerators_strategies_devices",
    [pytest.param(v, id="+".join(map(str, v))) for v in accelerators_strategies_devices],
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
        env={"TORCH_DISTRIBUTED_DEBUG": "DETAIL", **os.environ},
    )
    assert ret.success


def test_train_mlm_with_remote_dataset(
    mlm_task_config: pathlib.Path,
    remote_raw_text: str,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
):
    ret = script_runner.run(
        "zeldarose-transformer",
        "--strategy",
        "ddp_spawn",
        "--num-devices",
        "2",
        "--config",
        str(mlm_task_config),
        "--tokenizer",
        "lgrobol/roberta-minuscule",
        "--model-config",
        "lgrobol/roberta-minuscule",
        "--device-batch-size",
        "8",
        "--out-dir",
        str(tmp_path / "train-out"),
        "--cache-dir",
        str(tmp_path / "tokenizer-cache"),
        "--val-text",
        remote_raw_text,
        remote_raw_text,
        "--max-epochs",
        "2",
    )
    assert ret.success
