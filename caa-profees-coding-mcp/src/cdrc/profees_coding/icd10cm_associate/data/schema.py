import copy
from itertools import chain
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, ValidationError, model_validator

from cdrc.common.data.schema import ICD10CM, EMItem, InputSchema, LinkedDiagnosis, Order, OrderDisplay, Problem, Procedure
from cdrc.common.utils import StringEnum


class Problem(Problem):
    pass


class OrderDisplay(OrderDisplay):
    pass


class Order(Order):
    pass


class MergedModelInputSchema(InputSchema):
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")

    @model_validator(mode="before")
    def remove_linked_diagnosis(cls, values: Dict[str, Any]):
        """
        - Ensure procedure/em are lists of dict-like items
        - Strip `linked_diagnosis` from each procedure/em entry to prevent label leakage
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
        v["procedure"] = proc

        em_items = normalize_list(v.get("em"))
        for e in em_items:
            e.pop("linked_diagnosis", None)
        v["em"] = em_items

        diag = v.get("diagnosis")
        if diag is not None and not isinstance(diag, list):
            v["diagnosis"] = [diag]

        return v

    @model_validator(mode="before")
    def filter_hcpcs_to_g2211(cls, values: Dict[str, Any]):
        """
        Remove all HCPCS procedures except for code G2211 (case-insensitive).
        """
        if not isinstance(values, dict):
            return values

        procs = values.get("procedure")
        if procs is None:
            return values

        def to_dict(obj):
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return {k: getattr(obj, k) for k in vars(obj) if not k.startswith("_")}
            return {}

        def passes_filter(p):
            system = (p.get("system") or "").strip().upper()
            code = (p.get("code") or "").strip().upper()
            if system == "HCPCS":
                return code == "G2211"
            return True

        if isinstance(procs, list):
            filtered = [to_dict(p) for p in procs if passes_filter(to_dict(p))]
        else:
            p = to_dict(procs)
            filtered = [p] if passes_filter(p) else []
        values["procedure"] = filtered
        return values


class InputSchema(MergedModelInputSchema):
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")


class OutputSchema(BaseModel):
    example_id: Optional[Union[int, str]] = Field(None, description="The unique identifier for the example")
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT/HCPCS procedures")
    em: List[EMItem] = Field(default_factory=list, description="List of E/M items")
    diagnosis: List[ICD10CM] = Field(default_factory=list, description="List of diagnoses")
    problems: Optional[List[Problem]] = Field(None, description="The problems the encounter is about.")
    intermediate_raw_associations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Raw association snapshot before post-processing.",
    )
    intermediate_enhanced_associations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Association snapshot immediately after the optional enhancer stage.",
    )
    intermediate_combo_rule_associations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Association snapshot after deterministic ICD-10-CM combo-code rules.",
    )
    intermediate_validated_associations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Association snapshot after code validation.",
    )
    intermediate_cpt_ncd_associations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Association snapshot after CPT NCD filtering.",
    )
    intermediate_hcpcs_ncd_associations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Association snapshot after HCPCS NCD filtering.",
    )
    intermediate_regenerated_associations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Association snapshot after diagnosis regeneration.",
    )
    intermediate_regeneration_recommendations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Rule-based recommendation candidates used by diagnosis regeneration.",
    )
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

    @model_validator(mode="before")
    def filter_hcpcs_to_g2211(cls, values: Dict[str, Any]):
        """
        Remove all HCPCS procedures except for code G2211 (case-insensitive).
        """
        if not isinstance(values, dict):
            return values

        procs = values.get("procedure")
        if procs is None:
            return values

        def to_dict(obj):
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return {k: getattr(obj, k) for k in vars(obj) if not k.startswith("_")}
            return {}

        def passes_filter(p):
            system = (p.get("system") or "").strip().upper()
            code = (p.get("code") or "").strip().upper()
            # return code == "G2211"
            if system == "HCPCS":
                return code == "G2211"  # False  #
            return True

        if isinstance(procs, list):
            filtered = [to_dict(p) for p in procs if passes_filter(to_dict(p))]
        else:
            p = to_dict(procs)
            filtered = [p] if passes_filter(p) else []
        values["procedure"] = filtered
        return values

    """
    @model_validator(mode="after")
    def _validate_atleast_one_primary_per_service(self) -> "OutputSchema":
        def _check(items, kind: str):
            for i, it in enumerate(items or []):
                linked = getattr(it, "linked_diagnosis", None) or []
                prim_count = sum(bool(getattr(dx, "is_primary", False)) for dx in linked)
                if prim_count < 1:
                    raise ValueError(f"{kind}[{i}] must have at least one primary diagnosis; found {prim_count}.")

        _check(self.em, "em")
        _check(self.procedure, "procedure")
        return self
    """

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

        # v["diagnosis"] = norm or None
        v["diagnosis"] = norm if norm else []

        return v

    class Config:
        use_enum_values = True  # Automatically use enum values as strings


def format_soap_note(soap):
    """Dynamic formatting of soap note"""

    soap_str = ""

    for k, v in soap.items():
        k = k.capitalize().replace("_", " ")
        if isinstance(v, str):
            if v:
                soap_str += f"{k}:\n{''.join(['-'] * len(k))}\n{v}\n"
        elif isinstance(v, dict):
            soap_str += f"{k}:\n{''.join(['-'] * len(k))}\n"
            for kk, vv in v.items():
                if kk == "cc":
                    kk = "Chief Complaint"
                elif kk == "hpi":
                    kk = "History of Present Illness"
                kk = kk.capitalize().replace("_", " ")
                if isinstance(vv, list) and all(isinstance(elem, str) for elem in vv):
                    vv = "\n".join(vv)
                elif isinstance(vv, list) and all(
                    isinstance(elem, dict) for elem in vv
                ):  # to unpack diagnoses which is sometimes in assessment_and_plan section
                    try:
                        vv = "\n" + "\n\n".join(["\n".join(elem.values()) for elem in vv])
                    except Exception:
                        pass

                if isinstance(vv, list) and all(isinstance(elem, str) for elem in vv):
                    vv = "\n".join(vv)
                elif isinstance(vv, list) and all(
                    isinstance(elem, dict) for elem in vv
                ):  # to unpack diagnoses which is sometimes in assessment_and_plan section
                    try:
                        vv = "\n" + "\n\n".join(["\n".join(elem.values()) for elem in vv])
                    except Exception:
                        pass

                if vv:
                    soap_str += f"{kk}: {vv}\n"
        else:
            raise ValueError(f"Unknown type in SOAP note dictionary: {v} of type {type(v)}")
        soap_str += "\n"
        soap_str = soap_str.replace("<br/>", " ")
        soap_str = soap_str.replace("<br/>", " ")
    return soap_str


def format_orders(orders: List[Order]) -> str:
    if not orders:
        return ""

    seen_mnemonics: Set[str] = set()

    # Separate into non-medication and medication orders
    non_med_orders = [order for order in orders if order.order_type == "Non Medication Order"]
    med_orders = [order for order in orders if order.order_type == "Medication Order"]
    other_orders = [order for order in orders if order.order_type not in ["Non Medication Order", "Medication Order"]]
    other_orders = [order for order in orders if order.order_type not in ["Non Medication Order", "Medication Order"]]

    output = []

    # Add non-medication orders first
    if non_med_orders:
        output.append("Non-Medication Orders:")
        for order in non_med_orders:
            if order.display.hna_mnemonic not in seen_mnemonics:
                output.append(f"- {order.catalog_type}: {order.display.hna_mnemonic} (order status: {order.status})")
                seen_mnemonics.add(order.display.hna_mnemonic)

    # Add medication orders
    if med_orders:
        if output:
            output.append("")
        output.append("Medication Orders:")
        for order in med_orders:
            if order.display.hna_mnemonic not in seen_mnemonics:
                output.append(
                    f"- {order.catalog_type}: {order.display.hna_mnemonic} (order status: {order.status}) (ordered as: {order.display.ordered_as})"
                )
                seen_mnemonics.add(order.display.hna_mnemonic)

    # Add other orders
    if other_orders:
        if output:
            output.append("")
        output.append("Other Orders:")
        for order in other_orders:
            if order.display.hna_mnemonic not in seen_mnemonics:
                output.append(
                    f"- {order.catalog_type}: {order.display.hna_mnemonic} (order status: {order.status}) (ordered as: {order.display.ordered_as})"
                )
                seen_mnemonics.add(order.display.hna_mnemonic)

    return "\n".join(output)
