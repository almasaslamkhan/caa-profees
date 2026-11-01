import copy
from itertools import chain
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, ValidationError, model_validator

from cdrc.common.data.schema import (
    ICD10CM,
    EMItem,
    InputSchema,
    LinkedDiagnosis,
    Modifier,
    Order,
    OrderDisplay,
    Problem,
    Procedure,
    format_orders,
    format_soap_note,
)
from cdrc.common.utils import StringEnum

MEDICARE_PAYER_NAMES = {
    "MEDICARE",
    "MEDICARE FFS",
    "CMS",
    "MEDICARE PART B",
    "MEDICARE PART A",
    # companies that offers medicare plans
    "BCBS",
    "UHC",
    "UMR",
    "HEALTHPARTNERS",
    "WPS",
    "MVA",
    "AETNA",
}


class Problem(Problem):
    pass


class OrderDisplay(OrderDisplay):
    pass


class Order(Order):
    pass


class InputSchema(InputSchema):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")
    payer_name: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of payer names or payer dicts",
    )

    @model_validator(mode="before")
    def remove_linked_diagnosis_modifiers(cls, values: Dict[str, Any]):
        """
        - Ensure procedure/em are lists of dict-like items
        - Strip `linked_diagnosis`and 'modifiers' from each procedure/em entry to prevent label leakage
        """
        if not isinstance(values, dict):
            return values

        v = copy.deepcopy(values)

        def to_dict(obj: Any) -> Dict[str, Any]:
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) for x in obj):
                return dict(obj)  # tuples-of-pairs → dict
            if hasattr(obj, "model_dump"):  # pydantic v2 models
                return obj.model_dump()
            if hasattr(obj, "__dict__"):  # plain objects
                return {k: getattr(obj, k) for k in vars(obj) if not k.startswith("_")}
            return {}

        def normalize_list(x: Any) -> List[Dict[str, Any]]:
            if not x:
                return []
            if isinstance(x, list):
                return [to_dict(i) for i in x]
            return [to_dict(x)]

        proc = normalize_list(v.get("procedure"))
        for p in proc:
            p.pop("linked_diagnosis", None)
            p.pop("modifiers", None)
        v["procedure"] = proc

        em_items = normalize_list(v.get("em"))
        for e in em_items:
            e.pop("linked_diagnosis", None)
            e.pop("modifiers", None)
        v["em"] = em_items

        diag = v.get("diagnosis")
        if diag is not None and not isinstance(diag, list):
            v["diagnosis"] = [diag]

        return v

    @model_validator(mode="before")
    def normalize_payer_name(cls, values: Dict[str, Any]):
        if not isinstance(values, dict):
            return values

        v = copy.deepcopy(values)
        payer_value = v.get("payer_name")

        def _payer_is_medicare(name: str) -> bool:
            normalized = " ".join(str(name).split()).upper()
            return normalized in MEDICARE_PAYER_NAMES

        def _payer_to_dict(item: Any) -> Dict[str, Any]:
            if isinstance(item, dict):
                out = copy.deepcopy(item)
                payer_name = str(out.get("payer_name", out.get("name", ""))).strip()
                out["payer_name"] = payer_name
                out["is_medicare"] = _payer_is_medicare(payer_name) if payer_name else None
                return out

            if hasattr(item, "model_dump"):
                out = item.model_dump()
                payer_name = str(out.get("payer_name", out.get("name", ""))).strip()
                out["payer_name"] = payer_name
                out["is_medicare"] = _payer_is_medicare(payer_name) if payer_name else None
                return out

            if hasattr(item, "__dict__"):
                out = {k: getattr(item, k) for k in vars(item) if not k.startswith("_")}
                payer_name = str(out.get("payer_name", out.get("name", ""))).strip()
                out["payer_name"] = payer_name
                out["is_medicare"] = _payer_is_medicare(payer_name) if payer_name else None
                return out

            payer_name = str(item).strip()
            return {
                "payer_name": payer_name,
                "is_medicare": _payer_is_medicare(payer_name) if payer_name else None,
            }

        if not payer_value:
            v["payer_name"] = []
            return v

        if isinstance(payer_value, list):
            v["payer_name"] = [_payer_to_dict(item) for item in payer_value if (isinstance(item, dict) or str(item).strip())]
            return v

        v["payer_name"] = [_payer_to_dict(payer_value)]
        return v

    # # TBD: change to include HCPCS too when we support it
    # @model_validator(mode="before")
    # def _filter_cpt_procedures(cls, v: Dict[str, Any]):
    #     """
    #     Keeps only items where system == 'CPT'.
    #     Automatically converts single items or object lists to dicts.
    #     """
    #     if not isinstance(v, dict):
    #         return v

    #     procedures = v.get("procedure")
    #     if not procedures:
    #         return v

    #     # Normalize to list of dicts
    #     if isinstance(procedures, dict):
    #         procedures = [procedures]
    #     elif not isinstance(procedures, list):
    #         return v

    #     normalized = []
    #     for item in procedures:
    #         if hasattr(item, "model_dump"):
    #             item = item.model_dump()
    #         elif hasattr(item, "__dict__"):
    #             item = vars(item)
    #         if not isinstance(item, dict):
    #             continue

    #         if str(item.get("system", "")).strip().upper() == "CPT":
    #             normalized.append(item)

    #     v["procedure"] = normalized or None
    #     return v


class OutputSchema(BaseModel):
    example_id: Optional[Union[int, str]] = Field(None, description="The unique identifier for the example")
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    additional_output: Optional[Dict[Any, Any]] = Field(default_factory=dict, description="Additional output fields")

    @model_validator(mode="before")
    def move_unknown_fields_to_additional(cls, values):
        if not isinstance(values, dict):
            return values
        known = set(cls.model_fields.keys())
        extra = {k: values[k] for k in list(values.keys()) if k not in known}
        for k in extra:
            values.pop(k, None)
        if extra:
            values["additional_output"] = (values.get("additional_output") or {}) | extra
        return values

    @classmethod
    def _normalize_linked_dx(cls, d: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize `linked_diagnosis` into a list of ICD dicts:
          - Accepts dict / list / tuple-of-pairs / pydantic
          - Fills default system "ICD-10-CM" if missing
          - Coerces is_primary to bool
        """
        raw = d.get("linked_diagnosis")
        if not raw:
            return []
        items = raw if isinstance(raw, list) else [raw]
        out: List[Dict[str, Any]] = []
        for it in items:
            dx = cls._to_dict(it)
            code = (dx.get("code") or "").strip().upper()
            if not code:
                continue
            sys = dx.get("system") or "ICD-10-CM"
            is_primary = dx.get("is_primary")
            if isinstance(is_primary, str):
                is_primary = is_primary.strip().lower() in {"1", "true", "t", "yes", "y"}
            elif isinstance(is_primary, (int, float)):
                is_primary = bool(is_primary)
            elif is_primary is None:
                is_primary = False
            out.append(
                {
                    "code": code,
                    "system": sys,
                    "is_primary": is_primary,
                    "description": dx.get("description"),
                    "justification": dx.get("justification"),
                }
            )
        return out

    # # TBD: change to include HCPCS too when we support it
    # @model_validator(mode="before")
    # def _filter_cpt_procedures(cls, v: Dict[str, Any]):
    #     """
    #     Keeps only items where system == 'CPT'.
    #     Automatically converts single items or object lists to dicts.
    #     """
    #     if not isinstance(v, dict):
    #         return v

    #     procedures = v.get("procedure")
    #     if not procedures:
    #         return v

    #     # Normalize to list of dicts
    #     if isinstance(procedures, dict):
    #         procedures = [procedures]
    #     elif not isinstance(procedures, list):
    #         return v

    #     normalized = []
    #     for item in procedures:
    #         if hasattr(item, "model_dump"):
    #             item = item.model_dump()
    #         elif hasattr(item, "__dict__"):
    #             item = vars(item)
    #         if not isinstance(item, dict):
    #             continue

    #         if str(item.get("system", "")).strip().upper() == "CPT":
    #             normalized.append(item)

    #     v["procedure"] = normalized or None
    #     return v

    @model_validator(mode="before")
    def _coerce_em_items(cls, v: Dict[str, Any]):
        if not isinstance(v, dict):
            return v

        em_val = v.get("em")
        if em_val is None:
            return v

        # List cases
        if isinstance(em_val, list):
            fixed: List[Dict[str, Any]] = []
            # case: list of (k,v) pairs representing ONE EM
            if em_val and all(isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) for x in em_val):
                v["em"] = [dict(em_val)]
                return v

            for item in em_val:
                if isinstance(item, dict):
                    fixed.append(item)
                elif (
                    isinstance(item, (list, tuple))
                    and item
                    and all(isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) for x in item)
                ):
                    fixed.append(dict(item))  # list/tuple of pairs → dict
                elif hasattr(item, "model_dump"):
                    fixed.append(item.model_dump())  # pydantic v2 model instance
                elif hasattr(item, "__dict__"):
                    fixed.append({k: getattr(item, k) for k in vars(item).keys() if not k.startswith("_")})
                else:
                    raise ValueError(f"Unsupported EM item shape: {type(item)}")
            v["em"] = fixed
            return v

        # Anything else → leave; pydantic will error (by design)
        return v

    @model_validator(mode="before")
    def _coerce_icd_items(cls, v: Dict[str, Any]):
        if not isinstance(v, dict):
            return v

        icd_val = v.get("diagnosis")
        if icd_val is None:
            return v

        def to_dict(obj) -> Dict[str, Any]:
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) for x in obj):
                return dict(obj)  # tuple/list of pairs
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return {k: getattr(obj, k) for k in vars(obj) if not k.startswith("_")}
            return {}

        items = icd_val if isinstance(icd_val, list) else [icd_val]

        norm: List[Dict[str, Any]] = []
        seen = set()
        for it in items:
            d = to_dict(it)
            code = (d.get("code") or "").strip().upper()
            if not code:
                continue

            # normalize fields
            system = d.get("system") or "ICD-10-CM"
            is_primary = d.get("is_primary")
            if isinstance(is_primary, str):
                is_primary = is_primary.strip().lower() in {"1", "true", "t", "yes", "y"}
            elif isinstance(is_primary, (int, float)):
                is_primary = bool(is_primary)
            elif is_primary is None:
                is_primary = False

            key = (code, is_primary)
            if key in seen:
                continue
            seen.add(key)

            norm.append(
                {
                    "code": code,
                    "system": system,
                    "is_primary": is_primary,
                    "description": d.get("description"),
                    "justification": d.get("justification"),
                }
            )

        v["diagnosis"] = norm or None
        return v

    class Config:
        use_enum_values = True  # Automatically use enum values as strings
