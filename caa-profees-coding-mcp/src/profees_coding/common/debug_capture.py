"""Opt-in local capture of sensitive diagnostic payloads, never their headers."""

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def capture_payload(name: str, payload) -> Path | None:
    directory = os.getenv("PROFEES_DEBUG_CAPTURE_DIR")
    if not directory:
        return None
    path = None
    try:
        root = Path(directory).expanduser()
        if not root.is_absolute():
            raise ValueError("Capture directory must be absolute")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
            raise ValueError("Capture directory must be owned by this user and private (mode 700)")
        serialized = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=f"{name}-", suffix=".json", dir=root, delete=False) as output:
            path = Path(output.name)
            output.write(serialized + "\n")
        destination = root / f"{name}.json"
        os.replace(path, destination)
        path = None
        logger.info("[profees-mcp][debug_capture] %s saved to %s; contains sensitive clinical data", name, destination)
        return destination
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        # Diagnostic failure must not affect prediction or billing persistence.
        logger.warning("[profees-mcp][debug_capture] %s capture failed; payload omitted", name)
        return None
