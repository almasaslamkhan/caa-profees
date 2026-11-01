from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

EXPECTED_SKILL = PurePosixPath(
    "profees_ambulatory_coding_skills/skills/ambulatory-coding/SKILL.md"
)
EXPECTED_SCHEMA = PurePosixPath(
    "profees_ambulatory_coding_skills/skills/ambulatory-coding/schemas/ambulatory_coding_canvas.json"
)


def validate_wheel(wheel: Path) -> None:
    with ZipFile(wheel) as archive:
        names = {PurePosixPath(name) for name in archive.namelist()}

    legacy_entries = sorted(str(name) for name in names if "skills_package" in name.parts)
    if legacy_entries:
        raise ValueError(
            f"{wheel.name} contains the excluded skills_package namespace: "
            + ", ".join(legacy_entries)
        )

    missing = sorted(str(path) for path in (EXPECTED_SKILL, EXPECTED_SCHEMA) if path not in names)
    if missing:
        raise ValueError(f"{wheel.name} is missing required ambulatory resources: " + ", ".join(missing))


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: check_wheel_contents.py <wheel-or-dist-directory>", file=sys.stderr)
        return 2

    target = Path(sys.argv[1])
    wheels = sorted(target.glob("*.whl")) if target.is_dir() else [target]
    if not wheels:
        print(f"No wheel found at {target}", file=sys.stderr)
        return 2

    try:
        for wheel in wheels:
            validate_wheel(wheel)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"Validated {len(wheels)} profees-skills wheel(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
