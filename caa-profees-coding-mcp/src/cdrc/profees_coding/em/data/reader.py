import json
from typing import Callable, Optional, Tuple

try:  # pragma: no cover - optional dependency for environments without loguru
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from cdrc.common.data.reader import ConcurrentDataReader


class ConcurrentEMDataReader(ConcurrentDataReader):
    """Expand inpatient encounters into per-service-date and per-specialty slices."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_specialty_slices", True)
        super().__init__(*args, **kwargs)
