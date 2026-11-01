from itertools import chain
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, ValidationError, model_validator

from cdrc.common.data.schema import ICD10CM, InputSchema, Order, OrderDisplay, Problem
from cdrc.common.utils import StringEnum


class Problem(Problem):
    pass


class OrderDisplay(OrderDisplay):
    pass


class Order(Order):
    pass


class InputSchema(InputSchema):
    pass


class OutputSchema(BaseModel):
    example_id: Optional[Union[int, str]] = Field(None, description="The unique identifier for the example")
    diagnosis: List[ICD10CM] = Field(None, description="List of ICD-10-CM codes")
    problems: Optional[List[Problem]] = Field(None, description="The problems the encounter is about.")
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

        v["diagnosis"] = norm or []
        return v

    # @model_validator(mode="after")
    # def _validate_atleast_one_primary_per_service(self) -> "OutputSchema":
    #     prim_count = sum(bool(getattr(dx, "is_primary", False)) for dx in self.diagnosis)
    #     if prim_count < 1:
    #         raise ValueError(f"Each encounter must have at least one primary diagnosis; found {prim_count}.")

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
