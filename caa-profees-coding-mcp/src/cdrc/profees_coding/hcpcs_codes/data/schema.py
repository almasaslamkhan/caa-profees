from itertools import chain
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, ValidationError, model_validator

from cdrc.common.data.schema import AdditionalInput, AWVHistory, HealthPlan, InputSchema, Order, OrderDisplay, Problem, Procedure
from cdrc.common.utils import StringEnum


class Problem(Problem):
    pass


class OrderDisplay(OrderDisplay):
    pass


class Order(Order):
    pass


class HealthPlan(HealthPlan):
    pass


class InputSchema(InputSchema):
    customer_name: Optional[str] = Field(None, description="The name of the customer")

    @model_validator(mode="before")
    def populate_customer_name_and_awv_history(cls, values):
        if not isinstance(values, dict):
            return values

        additional_input = values.get("additional_input")
        if isinstance(additional_input, BaseModel):
            additional_input = additional_input.model_dump()

        if isinstance(additional_input, dict):
            customer_name = additional_input.get("customer_name")
            if customer_name and not values.get("customer_name"):
                values["customer_name"] = customer_name

            awv_history = additional_input.get("awv_history") or additional_input.get("hcpcs_history")
            if awv_history and not values.get("awv_history"):
                values["awv_history"] = awv_history

        return values


class OutputSchema(BaseModel):
    example_id: Optional[Union[int, str]] = Field(None, description="The unique identifier for the example")
    procedure: List[Procedure] = Field(None, description="List of HCPCS procedure codes")
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
    def _filter_hcpcs_procedures(cls, v: Dict[str, Any]):
        """
        Keeps only items where system == 'HCPCS'.
        Automatically converts single items or object lists to dicts.
        """
        if not isinstance(v, dict):
            return v

        procedures = v.get("procedure")
        if not procedures:
            return v

        # Normalize to list of dicts
        if isinstance(procedures, dict):
            procedures = [procedures]
        elif not isinstance(procedures, list):
            return v

        normalized = []
        for item in procedures:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "__dict__"):
                item = vars(item)
            if not isinstance(item, dict):
                continue

            if str(item.get("system", "")).strip().upper() == "HCPCS":
                normalized.append(item)

        v["procedure"] = normalized or []
        return v

    class Config:
        use_enum_values = True


def format_soap_note(soap):
    """Dynamic formatting of soap note"""

    soap_str = ""

    for k, v in soap.items():
        k = k.capitalize().replace("_", " ")
        if isinstance(v, str):
            if v:
                soap_str += f"{k}:\n{''.join(['-']*len(k))}\n{v}\n"
        elif isinstance(v, dict):
            soap_str += f"{k}:\n{''.join(['-']*len(k))}\n"
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
