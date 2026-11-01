#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import request

API_URL = os.environ.get("API_URL", "http://0.0.0.0:8002/predict")
TEST_CASE_DIR = Path(os.environ.get("TEST_CASE_DIR", "cdrc/profees_coding/deploy/api/test_cases"))
TIMEOUT_SECONDS = float(os.environ.get("API_TIMEOUT_SECONDS", "60"))


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def code_fields(value: Any, path: str = "") -> list[tuple[str, Any]]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            next_path = f"{path}.{key}" if path else key
            if key == "code":
                found.append((next_path, child))
            found.extend(code_fields(child, next_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(code_fields(child, f"{path}[{index}]"))
    return found


def post_payload(payload_path: Path) -> Any:
    payload_bytes = payload_path.read_bytes()
    req = request.Request(
        API_URL,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode())


def expected_response_path(payload_path: Path) -> Path:
    return payload_path.with_name(payload_path.name.replace("payload_", "response_", 1))


def compare_code_fields(actual: Any, expected: Any) -> bool:
    return code_fields(actual) == code_fields(expected)


def display_code_field_diff(payload_name: str, actual_codes: list[tuple[str, Any]], expected_codes: list[tuple[str, Any]]) -> None:
    actual_by_path = dict(actual_codes)
    expected_by_path = dict(expected_codes)

    changed = [
        (path, expected_value, actual_by_path[path])
        for path, expected_value in expected_codes
        if path in actual_by_path and actual_by_path[path] != expected_value
    ]
    missing = [(path, expected_value) for path, expected_value in expected_codes if path not in actual_by_path]
    unexpected = [(path, actual_value) for path, actual_value in actual_codes if path not in expected_by_path]

    print(f"DIFF {payload_name}")
    print(f"Summary: {len(changed)} changed, {len(missing)} missing expected, {len(unexpected)} unexpected actual")
    if changed:
        print("Changed code fields:")
        for path, expected_value, actual_value in changed:
            print(f"  - {path}: expected {expected_value}, actual {actual_value}")
    if missing:
        print("Missing expected code fields:")
        for path, expected_value in missing:
            print(f"  - {path}: expected {expected_value}")
    if unexpected:
        print("Unexpected actual code fields:")
        for path, actual_value in unexpected:
            print(f"  - {path}: actual {actual_value}")


def invoke_case(payload_path: Path) -> bool:
    response_path = expected_response_path(payload_path)
    if not response_path.exists():
        print(f"Missing expected response for {payload_path}: {response_path}", file=sys.stderr)
        return False

    print(f"Invoking {payload_path}")
    actual = post_payload(payload_path)
    # print(f"Received response for {payload_path}: {json.dumps(actual, indent=2)}")
    expected = load_json(response_path)

    actual_codes = code_fields(actual)
    expected_codes = code_fields(expected)
    if actual_codes != expected_codes:
        display_code_field_diff(payload_path.name, actual_codes, expected_codes)
        return False

    print(f"PASS {payload_path.name}")
    return True


def main() -> int:
    payload_paths = sorted(TEST_CASE_DIR.glob("payload_ambulatory_*.json"))
    if not payload_paths:
        print(f"No payload files found in {TEST_CASE_DIR}", file=sys.stderr)
        return 1

    for payload_path in payload_paths:
        invoke_case(payload_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
