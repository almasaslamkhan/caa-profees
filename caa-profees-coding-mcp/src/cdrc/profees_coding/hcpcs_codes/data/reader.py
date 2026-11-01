from __future__ import annotations

import json
from typing import Callable, Optional, Tuple

try:  # pragma: no cover - optional dependency for environments without loguru
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from cdrc.common.data.reader import ConcurrentDataReader


class ConcurrentHCPCSReader(ConcurrentDataReader):
    """Expand inpatient HCPCS payloads into per-service-date and per-specialty slices."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_specialty_slices", True)
        super().__init__(*args, **kwargs)
