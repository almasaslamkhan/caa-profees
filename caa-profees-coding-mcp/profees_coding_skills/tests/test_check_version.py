from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_version.py"


def _load_check_version_module():
    spec = importlib.util.spec_from_file_location("check_version", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_project_version_updates_project_version(tmp_path) -> None:
    module = _load_check_version_module()
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        "\n".join(
            [
                "[project]",
                'name = "profees-skills"',
                'version = "0.1.8"',
                "",
                "[tool.example]",
                'version = "keep-this"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    module._write_project_version(pyproject_path, "0.0.10763")

    assert pyproject_path.read_text(encoding="utf-8") == "\n".join(
        [
            "[project]",
            'name = "profees-skills"',
            'version = "0.0.10763"',
            "",
            "[tool.example]",
            'version = "keep-this"',
            "",
        ]
    )


def test_write_project_version_handles_crlf_project_section(tmp_path) -> None:
    module = _load_check_version_module()
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        "\r\n".join(
            [
                "[build-system]",
                'requires = ["setuptools>=64", "wheel"]',
                "",
                "[project]",
                'name = "profees-skills"',
                'version = "0.0.10763"',
                "",
                "[tool.example]",
                'version = "keep-this"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    module._write_project_version(pyproject_path, "0.0.10764")

    updated = pyproject_path.read_text(encoding="utf-8")
    assert 'version = "0.0.10764"' in updated
    assert 'version = "keep-this"' in updated


def test_oci_build_version_pattern_only_matches_build_number_versions() -> None:
    module = _load_check_version_module()

    assert module.OCI_BUILD_VERSION_PATTERN.fullmatch("0.0.10763")
    assert not module.OCI_BUILD_VERSION_PATTERN.fullmatch("0.1.8")
    assert not module.OCI_BUILD_VERSION_PATTERN.fullmatch("0.0.dev")
