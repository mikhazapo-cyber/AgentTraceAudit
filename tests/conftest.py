from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def dev_dataset() -> Path:
    return ROOT / "data" / "dev"


@pytest.fixture
def dev_labels() -> Path:
    return ROOT / "data" / "labels" / "dev"


@pytest.fixture
def synthetic_dataset() -> Path:
    return ROOT / "data" / "synthetic"


@pytest.fixture
def synthetic_labels() -> Path:
    return ROOT / "data" / "synthetic" / "labels"
