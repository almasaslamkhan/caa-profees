from typing import Any, Dict, List

from cdrc.common.data.filters import AVAILABLE_FILTERS as COMMON_FILTERS

# Dictionary to store CPT-specific filters
AVAILABLE_FILTERS = {}

# def linked_diagnosis_filter(data: List[Dict[str, Any]], encounter_types: List[str] = None) -> List[Dict[str, Any]]:
#     if not encounter_types:
#         return data

#     def get_encounter_types(item):
#         em_list = item.get("gold", {}).get("em") or item.get("em_items") or []
#         if not isinstance(em_list, list):
#             em_list = [em_list]
#         types = []
#         for em in em_list:
#             et = em.get("encounter_type") if isinstance(em, dict) else getattr(em, "encounter_type", None)
#             if isinstance(et, dict):
#                 et = et.get("encounter_type")
#             if isinstance(et, str):
#                 types.append(et)
#         return types

#     return [item for item in data if any(et in encounter_types for et in get_encounter_types(item))]

# # Merge common filters - any common filter can be overridden by redefining it above
AVAILABLE_FILTERS.update({k: v for k, v in COMMON_FILTERS.items() if k not in AVAILABLE_FILTERS})
