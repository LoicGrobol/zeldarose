import pathlib

import pytest

fixtures_dir = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_data_dir() -> pathlib.Path:
    return fixtures_dir


@pytest.fixture(scope="session")
def raw_text_path(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / "raw.txt"


@pytest.fixture(scope="session")
def remote_raw_text() -> str:
    return "lgrobol/openminuscule:default:train"

@pytest.fixture(scope="session")
def translation_dataset_path(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / test_data_dir /"translation.jsonl"

@pytest.fixture(
    params=[
        "lgrobol/flaubert-minuscule",
        "lgrobol/roberta-minuscule",
        pytest.param(fixtures_dir / "roberta-minuscule", id="roberta-minuscule-local"),
    ],
    scope="session",
)
def tokenizer_name_or_path(request) -> str:
    return request.param


@pytest.fixture(
    params=[
        "lgrobol/mbart-minuscule",
    ],
    scope="session",
)
def mbart_model_config(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def mbart_task_config(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / "mbart-config.toml"


@pytest.fixture(
    params=[
        "lgrobol/flaubert-minuscule",
        "lgrobol/roberta-minuscule",
        pytest.param(fixtures_dir / "roberta-minuscule", id="roberta-minuscule-local"),
    ],
    scope="session",
)
def mlm_model_config(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def mlm_task_config(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / "mlm-config.toml"


@pytest.fixture(
    params=[
        pytest.param(
            "lgrobol/electra-minuscule-discriminator,lgrobol/electra-minuscule-generator",
            id="lgrobol/electra-minuscule",
        )
    ],
    scope="session",
)
def rtd_model_config(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def rtd_task_config(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / "rtd-config.toml"
