import os
import pathlib

import pytest
import pytest_console_scripts
import torch.cuda

accelerators_strategies_devices = [
    ("cpu", None, None),
    ("cpu", "ddp_find_unused_parameters_true", 2),
    ("cpu", "ddp_spawn_find_unused_parameters_false", 2),
]
if torch.cuda.is_available():
    accelerators_strategies_devices.append(("gpu", None, None))
    if torch.cuda.device_count() > 1:
        accelerators_strategies_devices.append(("gpu", "ddp_find_unused_parameters_true", 2))
        accelerators_strategies_devices.append(("gpu", "ddp_spawn_find_unused_parameters_false", 2))
        # accelerators_strategies_devices.append(("gpu", "fsdp_native", 2))


def test_train_tokenizer(
    raw_text_path: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
):
    ret = script_runner.run(
        [
            "zeldarose",
            "tokenizer",
            "--vocab-size",
            "4096",
            "--out-path",
            str(tmp_path / "tokenizer"),
            "--model-name",
            "my-muppet",
            str(raw_text_path),
        ]
    )
    assert ret.success


@pytest.mark.parametrize(
    "accelerators_strategies_devices",
    [pytest.param(v, id="+".join(map(str, v))) for v in accelerators_strategies_devices],
)
def test_train_mbart(
    accelerators_strategies_devices: tuple[str, str | None, int | None],
    mbart_model_config: pathlib.Path | str,
    mbart_task_config: pathlib.Path,
    translation_dataset_path: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
):
    accelerator, strategy, devices = accelerators_strategies_devices
    extra_args: list[str] = []
    if strategy is not None:
        extra_args.extend(["--strategy", strategy])
    if devices is not None:
        extra_args.extend(["--num-devices", str(devices)])

    ret = script_runner.run(
        [
            "zeldarose",
            "transformer",
            "--accelerator",
            accelerator,
            "--config",
            str(mbart_task_config),
            "--tokenizer",
            str(mbart_model_config),
            "--model-config",
            str(mbart_model_config),
            "--device-batch-size",
            "2",
            "--out-dir",
            str(tmp_path / "train-out"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--val-text",
            str(translation_dataset_path),
            str(translation_dataset_path),
            *extra_args,
        ],
        env={"TORCH_DISTRIBUTED_DEBUG": "INFO", **os.environ},
    )
    assert ret.success


@pytest.mark.parametrize(
    "accelerators_strategies_devices",
    [pytest.param(v, id="+".join(map(str, v))) for v in accelerators_strategies_devices],
)
def test_train_mlm(
    accelerators_strategies_devices: tuple[str, str | None, int | None],
    mlm_model_config: pathlib.Path | str,
    mlm_task_config: pathlib.Path,
    raw_text_path: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
    tokenizer_name_or_path: pathlib.Path | str,
):
    accelerator, strategy, devices = accelerators_strategies_devices
    extra_args: list[str] = []
    if strategy is not None:
        extra_args.extend(["--strategy", strategy])
    if devices is not None:
        extra_args.extend(["--num-devices", str(devices)])

    ret = script_runner.run(
        [
            "zeldarose",
            "transformer",
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
            *extra_args,
        ],
        env={"TORCH_DISTRIBUTED_DEBUG": "INFO", **os.environ},
    )
    assert ret.success


@pytest.mark.parametrize(
    "accelerators_strategies_devices",
    [pytest.param(v, id="+".join(map(str, v))) for v in accelerators_strategies_devices],
)
def test_finetune_ntp(
    accelerators_strategies_devices: tuple[str, str | None, int | None],
    ntp_model_name_or_path: pathlib.Path | str,
    ntp_task_config: pathlib.Path,
    raw_text_path: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
):
    accelerator, strategy, devices = accelerators_strategies_devices
    extra_args: list[str] = []
    if strategy is not None:
        extra_args.extend(["--strategy", strategy])
    if devices is not None:
        extra_args.extend(["--num-devices", str(devices)])

    ret = script_runner.run(
        [
            "zeldarose",
            "transformer",
            "--accelerator",
            accelerator,
            "--config",
            str(ntp_task_config),
            "--pretrained-model",
            str(ntp_model_name_or_path),
            "--device-batch-size",
            "8",
            "--out-dir",
            str(tmp_path / "train-out"),
            "--cache-dir",
            str(tmp_path / "tokenizer-cache"),
            "--val-text",
            str(raw_text_path),
            str(raw_text_path),
            *extra_args,
        ],
        env={"TORCH_DISTRIBUTED_DEBUG": "INFO", **os.environ},
    )
    print(ret.__dict__)
    assert ret.success


@pytest.mark.parametrize(
    "accelerators_strategies_devices",
    [
        pytest.param(v, id=runtime_id)
        for v in accelerators_strategies_devices
        if "find_unused_parameters_false" not in (runtime_id := "+".join(map(str, v)))
    ],
)
def test_train_rtd(
    accelerators_strategies_devices: tuple[str, str | None, int | None],
    rtd_model_config: pathlib.Path | str,
    rtd_task_config: pathlib.Path,
    raw_text_path: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
    tokenizer_name_or_path: pathlib.Path | str,
):
    accelerator, strategy, num_devices = accelerators_strategies_devices

    extra_args: list[str] = []
    if strategy is not None:
        extra_args.extend(["--strategy", strategy])
    if num_devices is not None:
        extra_args.extend(["--num-devices", str(num_devices)])
    ret = script_runner.run(
        [
            "zeldarose",
            "transformer",
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
            *extra_args,
        ],
        env={"TORCH_DISTRIBUTED_DEBUG": "INFO", **os.environ},
    )
    assert ret.success


def test_train_mlm_with_remote_dataset(
    mlm_task_config: pathlib.Path,
    remote_raw_text: str,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
):
    ret = script_runner.run(
        [
            "zeldarose",
            "transformer",
            "--accelerator",
            "cpu",
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
        ]
    )
    assert ret.success
