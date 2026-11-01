from __future__ import annotations

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_wheel_contents.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_wheel_contents", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_wheel(path: Path, names: list[str]) -> None:
    with ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, "test")


def test_validate_wheel_accepts_only_ambulatory_resources(tmp_path: Path) -> None:
    module = _load_module()
    wheel = tmp_path / "profees_skills-0.0.11184-py3-none-any.whl"
    _write_wheel(wheel, [str(module.EXPECTED_SKILL), str(module.EXPECTED_SCHEMA)])

    module.validate_wheel(wheel)


def test_validate_wheel_rejects_legacy_skills_package(tmp_path: Path) -> None:
    module = _load_module()
    wheel = tmp_path / "profees_skills-0.0.11184-py3-none-any.whl"
    _write_wheel(
        wheel,
        [
            str(module.EXPECTED_SKILL),
            str(module.EXPECTED_SCHEMA),
            "skills_package/skills/profees-coding/SKILL.md",
        ],
    )

    with pytest.raises(ValueError, match="excluded skills_package namespace"):
        module.validate_wheel(wheel)
