from typing import Any, Dict, List

from cdrc.common.data.filters import AVAILABLE_FILTERS as COMMON_FILTERS

# Dictionary to store CPT-specific filters
AVAILABLE_FILTERS = {}

# Merge common filters - any common filter can be overridden by redefining it above
AVAILABLE_FILTERS.update({k: v for k, v in COMMON_FILTERS.items() if k not in AVAILABLE_FILTERS})
