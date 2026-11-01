from __future__ import annotations

import json
from typing import Callable, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from cdrc.common.data.reader import ConcurrentDataReader


class ConcurrentModifiersReader(ConcurrentDataReader):
    """Expand inpatient encounters into per service-date and per-specialty slices for modifier coding."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_specialty_slices", True)
        kwargs.setdefault("postprocess_slice", self._postprocess_slice)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _postprocess_slice(day_input, output_model):
        if output_model is None:
            return
        if not day_input.em:
            day_input.em = [em.model_copy(deep=True) for em in output_model.em]
        if not day_input.procedure:
            day_input.procedure = [proc.model_copy(deep=True) for proc in output_model.procedure]
