from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

VERSION_PATTERN = re.compile(r'^(version\s*=\s*")[^"]+("[ \t]*)$')
OCI_BUILD_VERSION_PATTERN = re.compile(r"^0\.0\.\d+$")


def _write_project_version(pyproject_path: Path, version: str) -> None:
    text = pyproject_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    in_project_section = False
    updated_lines: list[str] = []
    updated_version = False

    for line in lines:
        content = line
        line_ending = ""
        for suffix in ("\r\n", "\n", "\r"):
            if content.endswith(suffix):
                content = content[: -len(suffix)]
                line_ending = suffix
                break

        stripped = content.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project_section = stripped == "[project]"

        if in_project_section:
            replaced = VERSION_PATTERN.sub(rf"\g<1>{version}\2", content, count=1)
            if replaced != content:
                updated_lines.append(replaced + line_ending)
                updated_version = True
                continue

        updated_lines.append(line)

    if not updated_version:
        print("Could not update project.version in pyproject.toml", file=sys.stderr)
        raise SystemExit(2)

    pyproject_path.write_text("".join(updated_lines), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: check_version.py <expected-version>", file=sys.stderr)
        return 2

    expected_version = sys.argv[1]
    project_root = Path(__file__).resolve().parents[1]
    pyproject_path = project_root / "pyproject.toml"
    metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    actual_version = metadata["project"]["version"]

    if actual_version != expected_version:
        if not OCI_BUILD_VERSION_PATTERN.fullmatch(expected_version):
            print(
                "profees-skills version mismatch: " f"pyproject.toml has {actual_version}, expected {expected_version}",
                file=sys.stderr,
            )
            return 1
        _write_project_version(pyproject_path, expected_version)
        print(f"profees-skills version: updated {actual_version} -> {expected_version}")
        return 0

    print(f"profees-skills version: {actual_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
