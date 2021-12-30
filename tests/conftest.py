import pathlib

import pytest


fixtures_dir = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_data_dir() -> pathlib.Path:
    return fixtures_dir


@pytest.fixture(scope="session")
def raw_text_path(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / "raw.txt"


@pytest.fixture(
    params=[
        "lgrobol/flaubert-minuscule",
        "lgrobol/roberta-minuscule",
        fixtures_dir / "roberta-minuscule",
    ],
    scope="session",
)
def tokenizer_name_or_path(request) -> str:
    return request.param


@pytest.fixture(
    params=[
        "lgrobol/flaubert-minuscule",
        "lgrobol/roberta-minuscule",
        fixtures_dir / "roberta-minuscule",
    ],
    scope="session",
)
def mlm_model_config(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def mlm_task_config(test_data_dir: pathlib.Path) -> str:
    return test_data_dir / "mlm-config.toml"


@pytest.fixture(
    params=["google/electra-small-discriminator,google/electra-small-generator"],
    scope="session",
)
def rtd_model_config(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def rtd_task_config(test_data_dir: pathlib.Path) -> str:
    return test_data_dir / "rtd-config.toml"
