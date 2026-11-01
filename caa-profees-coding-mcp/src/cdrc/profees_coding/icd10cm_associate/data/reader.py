from __future__ import annotations

import json
from typing import Callable, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from cdrc.common.data.reader import ConcurrentDataReader


class ConcurrentICD10CMAssociateReader(ConcurrentDataReader):
    """Expand inpatient encounters into per-service-date and per-specialty slices for association coding."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_specialty_slices", True)
        kwargs.setdefault("postprocess_slice", self._postprocess_slice)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _postprocess_slice(day_input, output_model):
        if output_model is None:
            return
        if output_model.em:
            day_input.em = list(day_input.em or []) + [em.model_copy(deep=True) for em in output_model.em]
        if output_model.procedure:
            day_input.procedure = list(day_input.procedure or []) + [proc.model_copy(deep=True) for proc in output_model.procedure]
        if output_model.diagnosis:
            day_input.diagnosis = list(day_input.diagnosis or []) + [dx.model_copy(deep=True) for dx in output_model.diagnosis]
