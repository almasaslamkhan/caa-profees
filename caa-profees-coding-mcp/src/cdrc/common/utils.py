import argparse
import ast
import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, EnumMeta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, get_args, get_origin

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel, Field, create_model

if TYPE_CHECKING:
    from cdrc.common.data.schema import EncounterDocument, EncounterMetaData, InputSchema, Order


INPATIENT_PRIMARY_DOCUMENT_TYPES: Set[str] = {
    "h&p",
    "anesthesia sheets",
    "progress notes",
    "consultations",
    "operative reports",
    "procedure notes",
    "discharge summary",
    "pathology",
    "er notes",
    "addendums",
}

EXCLUDED_DOCUMENT_TYPES: Set[str] = {
    "echocardiogram",
}


class __ContainsEnumMeta(EnumMeta):
    """
    Enum metaclass that enables presence checks with values of the enum member type
    """

    def __contains__(cls, member: Any) -> bool:
        if isinstance(member, cls._member_type_):
            try:
                cls(member)
            except ValueError:
                return False
            return True
        return super().__contains__(member)


class StringEnum(str, Enum):
    def __new__(cls, value, type_cast=None):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.type_cast = type_cast
        return obj

    def __call__(self):
        if self.type_cast:
            return self.type_cast(self._value_)
        return self._value_


def convert_enums_to_strings(data: Any) -> Any:
    """
    Recursively convert all Enum instances in the data to their string representations.
    Works with nested structures such as dicts, lists, and models.
    """
    if isinstance(data, (Enum, StringEnum)):  # If the value is an Enum, convert it to its string representation
        return data.value

    elif isinstance(data, dict):  # Recursively handle dictionaries
        return {key: convert_enums_to_strings(value) for key, value in data.items()}

    elif isinstance(data, list):  # Recursively handle lists
        return [convert_enums_to_strings(item) for item in data]

    elif isinstance(data, BaseModel):  # Recursively handle Pydantic models
        return {key: convert_enums_to_strings(value) for key, value in data.dict().items()}

    # Return the value if it isn't an Enum, dict, list, or Pydantic model
    return data


def _normalize_document_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    return normalized or None


def _document_matches_inpatient_anchor(document: "EncounterDocument") -> bool:
    normalized = getattr(document, "document_type_normalized", None)
    if normalized is None:
        normalized = _normalize_document_type(getattr(document, "document_type", None))
    return normalized in INPATIENT_PRIMARY_DOCUMENT_TYPES


def _is_inpatient_encounter(metadata: Optional["EncounterMetaData"]) -> bool:
    if metadata is None:
        return False
    encounter_type = getattr(metadata, "type", None)
    if not encounter_type:
        return False
    return "inpatient" in encounter_type.strip().lower()


def _concatenate_document_texts(documents: Sequence["EncounterDocument"]) -> str:
    segments: List[str] = []
    for document in documents or []:
        text = getattr(document, "text", None)
        if not text:
            continue
        stripped = text.strip()
        if stripped:
            segments.append(stripped)
    return "\n\n".join(segments)


def _filter_orders_by_inpatient_window(orders: Sequence["Order"], metadata: Optional["EncounterMetaData"]) -> List["Order"]:
    if metadata is None:
        return list(orders or [])
    admit_date = _normalize_service_date(getattr(metadata, "admit_time", None))
    discharge_date = _normalize_service_date(getattr(metadata, "discharge_time", None))
    if admit_date is None and discharge_date is None:
        return list(orders or [])

    def _in_window(order: "Order") -> bool:
        timestamp = getattr(order, "time_created", None) or getattr(order, "time_last_updated", None)
        date_value = _normalize_service_date(timestamp)
        if date_value is None:
            return True
        if admit_date and date_value < admit_date:
            return False
        if discharge_date and date_value > discharge_date:
            return False
        return True

    return [order for order in orders or [] if _in_window(order)]


def _filter_documents_by_inpatient_window(
    documents: Sequence["EncounterDocument"], metadata: Optional["EncounterMetaData"]
) -> List["EncounterDocument"]:
    if metadata is None:
        return list(documents or [])
    admit_date = _normalize_service_date(getattr(metadata, "admit_time", None))
    discharge_date = _normalize_service_date(getattr(metadata, "discharge_time", None))
    if admit_date is None and discharge_date is None:
        return list(documents or [])

    def _in_window(document: "EncounterDocument") -> bool:
        timestamp = getattr(document, "service_date", None) or getattr(document, "time_created", None) or getattr(document, "signature_time", None)
        date_value = _normalize_service_date(timestamp)
        if date_value is None:
            return True
        if admit_date and date_value < admit_date:
            return False
        if discharge_date and date_value > discharge_date:
            return False
        return True

    return [document for document in documents or [] if _in_window(document)]


def _filter_excluded_documents(documents: Sequence["EncounterDocument"]) -> List["EncounterDocument"]:
    filtered: List["EncounterDocument"] = []
    for document in documents or []:
        normalized = getattr(document, "document_type_normalized", None)
        if normalized is None:
            normalized = _normalize_document_type(getattr(document, "document_type", None))
        if normalized and normalized in EXCLUDED_DOCUMENT_TYPES:
            continue
        filtered.append(document)
    return filtered


def read_from_json(input_path: str):
    with open(input_path, "r") as fp:
        return json.load(fp)


def write_to_json(data: Any, output_path: str):
    with open(output_path, "w") as fp:
        json.dump(data, fp, indent=4)


def read_from_jsonl(input_path: str):
    with open(input_path, "r") as fp:
        return [json.loads(line) for line in fp]


def load_from_yaml(file_path: str) -> dict[str, dict[str, dict[str, str]]]:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_kwargs(kwargs_str):
    """
    Parses a string representation of a dictionary into a Python dict.

    kwargs_str: A string in the format 'key1=val1,key2=val2,...'.
    output: A dictionary object representing the parsed kwargs.
    """
    try:
        return ast.literal_eval(kwargs_str)
    except (SyntaxError, ValueError) as e:
        raise argparse.ArgumentError(None, f"Invalid kwargs format: {kwargs_str}. Error: {e}")


def is_listed_attribute(schema, label_key):
    """
    Check if the label key in the Input/Output Schema is a list
    """

    label_key_base = label_key.split(".")[0]
    label_key_base_typing = schema.__annotations__[label_key_base]
    second_level_typing = get_args(label_key_base_typing)
    second_level_typing = second_level_typing[0] if (second_level_typing and isinstance(second_level_typing, tuple)) else second_level_typing

    return get_origin(second_level_typing) is list


def flatten_labels(sample) -> list[str]:
    """Return a flat, de-duplicated, alphabetically sorted list of label strings."""
    if sample is None:
        return []

    out = []

    def walk(x):
        if x is None:
            return
        if isinstance(x, (list, tuple, set)):
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            # common case: label dicts like {"code": "E11.9", ...}
            if "code" in x and isinstance(x["code"], (str, int, float)):
                out.append(str(x["code"]).strip())
            else:
                for v in x.values():
                    walk(v)
        elif isinstance(x, (str, int, float)):
            out.append(str(x).strip())
        else:
            # fallback
            out.append(str(x).strip())

    walk(sample)

    # de-dupe (order-insensitive since we’ll sort)
    flat = sorted(set(out), key=lambda s: s.lower())
    return flat


def get_nested_attribute(obj, path):
    attrs = path.split(".")
    for attr in attrs:
        # print(obj, isinstance(obj, list), attr, hasattr(obj, attr))
        if isinstance(obj, list):
            # Apply recursively for each element in the list
            obj = [get_nested_attribute(item, ".".join([attr] + attrs[attrs.index(attr) + 1 :])) for item in obj]
            break
        elif hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            continue
            raise ValueError(f"The output schema doesn't have an attribute {path}!")
    if isinstance(obj, list):
        return flatten_labels(obj)
    return obj


def get_nested_attribute_list(obj, path):
    attrs = path.split(".")
    for attr in attrs:
        if isinstance(obj, list):
            # Apply recursively for each element in the list
            obj = [get_nested_attribute(item, ".".join([attr] + attrs[attrs.index(attr) + 1 :])) for item in obj]
            break
        elif hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            continue
            raise ValueError(f"The output schema doesn't have an attribute {path}!")
    return obj


def get_nested_attribute_with_prefix(obj, path: str, prefix_attr: str = None):
    attrs = path.split(".")
    for i, attr in enumerate(attrs):
        if isinstance(obj, list):
            temp, prefices = [], []
            for item in obj:
                val = None
                if isinstance(item, dict) and attr in item:
                    val = item[attr]
                elif hasattr(item, attr):
                    val = getattr(item, attr)
                else:
                    raise ValueError(f"The output schema doesn't have attribute or key '{attr}' in list element!")
                if isinstance(val, list) and i == len(attrs) - 2:
                    if isinstance(item, dict) and prefix_attr in item:
                        prefix = item[prefix_attr]
                    elif prefix_attr and hasattr(item, prefix_attr):
                        prefix = getattr(item, prefix_attr)
                    else:
                        prefix = None
                    if prefix:
                        val = {prefix: val}
                        prefices.append(prefix)
                if val:
                    temp.append(val)
            # Flatten the list if it contains sub-lists, to ensure full flattening
            flat = []
            for t in temp:
                if isinstance(t, list):
                    flat.extend(t)
                elif isinstance(t, dict) and i == len(attrs) - 2:
                    if all([prefix not in t for prefix in prefices]):
                        flat.append(t)
                        continue
                    prefix = list(t.keys())[0]
                    val, concat_val = t[prefix], prefix
                    last_attr = attrs[-1]
                    for v in val:
                        if isinstance(v, dict) and last_attr in v:
                            concat_val = f"{concat_val}-{v[last_attr]}"
                        elif hasattr(v, last_attr):
                            concat_val = f"{concat_val}-{getattr(v, last_attr)}"
                        else:
                            raise ValueError(f"The output schema doesn't have attribute or key '{last_attr}' in list element!")
                    flat.append({last_attr: concat_val})
                else:
                    flat.append(t)
            obj = flat
        elif isinstance(obj, dict) and attr in obj:
            obj = obj[attr]
        elif hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            raise ValueError(f"The output schema doesn't have an attribute {path}!")
    return obj


def get_values_for_metrics_calculations(y_true, y_pred, label_key):
    """
    Get values for metrics calculations.
    Handles both list and single object.
    For example, label_key=cpt.code:
    - Single: Based on the OutputSchema, the attribute cpt.code is directly accessible.
    - List: Based on the OutputSchema, the attribute cpt is a list and code is a field of the elements in the list.
    """
    try:
        _y_true = get_nested_attribute(y_true, label_key)
        _y_pred = get_nested_attribute(y_pred, label_key)

    except Exception:
        if "." in label_key:
            base_key, sub_key = label_key.split(".")
            _y_true = get_nested_attribute(y_true, base_key)
            _y_pred = get_nested_attribute(y_pred, base_key)

            _y_true = [get_nested_attribute(x, sub_key) for x in _y_true]
            _y_pred = [get_nested_attribute(x, sub_key) for x in _y_pred]
        else:
            raise ValueError(f"Incorrect label_key: {label_key}")

    return _y_true, _y_pred


# Helper: convert list-of-labels → binary indicator matrix
def binarize(y, mapping=None):
    """
    Parameters
    ----------
    y : list[list[hashable]]
        E.g. [['A', 'B'], ['C'], []]
    mapping : dict or None
        {label → column index}. If None it is built from `y`.

    Returns
    -------
    out : (n_samples, n_labels) ndarray of ints {0,1}
    mapping : dict
    """
    if mapping is None:
        labels = sorted({label for sample in y for label in sample})
        mapping = {label: i for i, label in enumerate(labels)}
    out = np.zeros((len(y), len(mapping)), dtype=int)
    for row, sample in enumerate(y):
        for label in sample:
            if label in mapping:  # ignore unknowns
                out[row, mapping[label]] = 1
    return out, mapping


def convert_string_to_type(value):
    if not isinstance(value, str):
        return value

    try:
        # Try to safely evaluate string as literal
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            # Try JSON parsing as fallback
            return json.loads(value)
        except json.JSONDecodeError:
            # Return original value if conversion fails
            return value


def load_evaluation_results(csv_path: str, input_schema: BaseModel, output_schema: BaseModel) -> tuple[BaseModel, BaseModel, BaseModel]:
    """Load evaluation results from CSV and return lookup dict and state keys."""
    df = pd.read_csv(csv_path)

    # Separate prediction columns
    gold_cols = [col for col in df.columns if col.startswith("gold_")]
    output_cols = [col for col in df.columns if col.startswith("output_")]
    input_cols = [col for col in df.columns if col.startswith("input_")]

    inputs = []
    outputs = []
    golds = []
    for _, row in df.iterrows():
        # Create input dictionary using non-prediction columns
        input_dict = {"_".join(col.split("_")[1:]): convert_string_to_type(row[col]) for col in input_cols if pd.notna(row[col])}

        # Create prediction dictionary using prediction columns
        output_dict = {"_".join(col.split("_")[1:]): convert_string_to_type(row[col]) for col in output_cols if pd.notna(row[col])}

        # Create prediction dictionary using prediction columns
        gold_dict = {"_".join(col.split("_")[1:]): convert_string_to_type(row[col]) for col in gold_cols if pd.notna(row[col])}

        # Use string representation of input as key
        inputs.append(input_schema(**input_dict))
        outputs.append(output_schema(**output_dict))
        golds.append(output_schema(**gold_dict))

    return inputs, outputs, golds


def combine_csv_files_into_xlsx(dir_path: str, metadata: dict[str, Any]):
    """
    Combines all CSV files in a directory into a single Excel file with multiple sheets.
    Each CSV becomes a sheet named after the file (without .csv extension).
    Adds a metadata sheet with two columns containing the metadata dictionary items.

    Args:
        dir_path: Directory path containing CSV files
        metadata: Dictionary of metadata to include in a separate sheet
    """

    # Create Excel writer object
    output_path = os.path.join(dir_path, "combined_report.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Process all CSV files in directory
        for file in os.listdir(dir_path):
            if file.endswith(".csv"):
                # Read CSV file
                csv_path = os.path.join(dir_path, file)
                df = pd.read_csv(csv_path)

                # Use filename without extension as sheet name
                sheet_name = Path(file).stem

                # Write dataframe to Excel sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Create metadata sheet
        metadata_df = pd.DataFrame(metadata.items(), columns=["argument", "value"])
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)


def merge_config(args: argparse.Namespace, unknown_args: argparse.Namespace, config: dict) -> argparse.Namespace:
    """
    Merges the parsed command-line arguments (args) with a configuration dictionary (config)
    and includes any values from unknown_args. Command-line arguments take precedence over config values.

    Args:
        args: The Namespace object containing known parsed command-line arguments.
        unknown_args: The list of unknown command-line arguments.
        config: A dictionary containing configuration values read from a YAML file or other source.

    Returns:
        A Namespace object with all merged arguments.
    """
    # Start with the config as the base
    merged_dict = config.copy()

    # Override with known args (including None values)
    merged_dict.update(vars(args))

    # Initialize a parser to parse unknown arguments
    parser = argparse.ArgumentParser()

    # Add unknown arguments dynamically to the parser
    for i in range(0, len(unknown_args), 2):
        if unknown_args[i].startswith("--"):
            if len(unknown_args) > i + 1 and unknown_args[i + 1].lower() in ("true", "false"):
                if unknown_args[i + 1].lower() == "true":
                    parser.add_argument(unknown_args[i], action="store_true")
            else:
                parser.add_argument(unknown_args[i])

    # Parse the unknown arguments
    parsed_unknown_args, _ = parser.parse_known_args(unknown_args)

    # Override with unknown args
    merged_dict.update(vars(parsed_unknown_args))

    # If model_kwargs is a string, evaluate it - to enable passing dicts from command line
    if isinstance(merged_dict.get("model_kwargs"), str):
        merged_dict["model_kwargs"] = eval(merged_dict["model_kwargs"])
        assert isinstance(merged_dict["model_kwargs"], dict), "model_kwargs passed from command line must be a dictionary formatted as a string"
    try:
        if merged_dict.get("filter_kwargs"):
            merged_dict["filter_kwargs"] = eval(merged_dict["filter_kwargs"])
        logger.warning(f"Will use filter '{merged_dict['filter_name']}' with args: {merged_dict['filter_kwargs']}")
    except Exception as e:
        logger.warning(f"Error parsing filter args: {e}")

    # Return the updated Namespace
    return argparse.Namespace(**merged_dict)


@dataclass
class ConcurrentCodingSlice:
    """Representation of a single service-date and specialty view for concurrent coding."""

    service_date: str
    specialty: Optional[str]
    day_input: "InputSchema"
    documents: Sequence["EncounterDocument"]
    orders: Sequence["Order"]


def _normalize_service_date(value: Optional[str]) -> Optional[str]:
    """Coerce various timestamp representations to YYYY-MM-DD."""
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    # Direct YYYY-MM-DD pattern
    if len(value) >= 10:
        candidate = value[:10]
        try:
            return datetime.strptime(candidate, "%Y-%m-%d").date().isoformat()
        except ValueError:
            pass
    sanitized = value.replace("Z", "")
    if "+" in sanitized:
        sanitized = sanitized.split("+", 1)[0]
    if " " in sanitized and "T" not in sanitized:
        sanitized = sanitized.replace(" ", "T", 1)
    try:
        return datetime.fromisoformat(sanitized).date().isoformat()
    except ValueError:
        return None


def _group_by_service_date(items: Iterable[Any], *, extractor, predicate: Optional[Callable[[Any], bool]] = None) -> Dict[str, list]:
    grouped: Dict[str, list] = {}
    for item in items or []:
        if predicate is not None and not predicate(item):
            continue
        date_value = extractor(item)
        if not date_value:
            continue
        grouped.setdefault(date_value, []).append(item)
    return grouped


def _extract_document_specialties(document: "EncounterDocument") -> list[str]:
    participants = getattr(document, "provider_participants", None) or []
    specialties = [participant.get("specialty") for participant in participants if isinstance(participant, dict)]
    return [specialty for specialty in specialties if specialty]


def _group_documents_by_date_and_specialty(documents: Sequence["EncounterDocument"]) -> Dict[Tuple[str, str], list]:
    grouped: Dict[Tuple[str, str], list] = {}
    for document in documents or []:
        date_value = _normalize_service_date(getattr(document, "service_date", None) or getattr(document, "time_created", None))
        if not date_value:
            continue
        specialties = _extract_document_specialties(document)
        if not specialties:
            specialties = ["unknown"]
        for specialty in specialties:
            grouped.setdefault((date_value, specialty), []).append(document)
    return grouped


def _compose_day_text(documents: Sequence["EncounterDocument"], fallback_text: Optional[str], transcript: Optional[str]) -> str:
    bodies = [getattr(doc, "text", "").strip() for doc in documents if getattr(doc, "text", None)]
    candidate = "\n\n".join(body for body in bodies if body)
    if candidate:
        return candidate
    if transcript:
        return transcript.strip()
    return (fallback_text or "").strip()


def split_concurrent_input_by_service_date(payload: Any) -> List[ConcurrentCodingSlice]:
    """Split an inpatient encounter payload into per service-date slices."""
    from cdrc.common.data.schema import EncounterMetaData, InputSchema, format_orders

    if isinstance(payload, dict):
        base_input = InputSchema(**payload)
    elif isinstance(payload, InputSchema):
        base_input = payload
    else:
        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    metadata = base_input.encounter_metadata
    is_inpatient = _is_inpatient_encounter(metadata)

    documents = _filter_excluded_documents(list(base_input.documents or []))
    orders = list(base_input.orders or [])
    if is_inpatient:
        documents = _filter_documents_by_inpatient_window(documents, metadata)
        orders = _filter_orders_by_inpatient_window(orders, metadata)

    document_groups = _group_by_service_date(
        documents,
        extractor=lambda doc: _normalize_service_date(getattr(doc, "service_date", None) or getattr(doc, "time_created", None)),
    )
    order_groups = _group_by_service_date(
        orders,
        extractor=lambda order: _normalize_service_date(getattr(order, "time_created", None) or getattr(order, "time_last_updated", None)),
    )

    if is_inpatient:
        anchor_groups = _group_by_service_date(
            documents,
            extractor=lambda doc: _normalize_service_date(getattr(doc, "service_date", None) or getattr(doc, "time_created", None)),
            predicate=_document_matches_inpatient_anchor,
        )
        service_dates: Set[str] = set(anchor_groups.keys())
        if not service_dates:
            if metadata and metadata.service_dates:
                for raw_date in metadata.service_dates:
                    normalized = _normalize_service_date(raw_date if isinstance(raw_date, str) else str(raw_date))
                    if normalized:
                        service_dates.add(normalized)
        if not service_dates:
            service_dates.update(document_groups.keys())
            service_dates.update(order_groups.keys())
    else:
        service_dates = set()
        if metadata and metadata.service_dates:
            for raw_date in metadata.service_dates:
                normalized = _normalize_service_date(raw_date if isinstance(raw_date, str) else str(raw_date))
                if normalized:
                    service_dates.add(normalized)
        service_dates.update(document_groups.keys())
        service_dates.update(order_groups.keys())

    if not service_dates:
        fallback = _normalize_service_date(base_input.encounter_date) if getattr(base_input, "encounter_date", None) else None
        if not fallback and documents:
            fallback = _normalize_service_date(getattr(documents[0], "service_date", None) or getattr(documents[0], "time_created", None))
        if not fallback:
            raise ValueError("Unable to derive service dates for concurrent coding input")
        service_dates.add(fallback)

    ordered_dates = sorted(service_dates)
    total_days = len(ordered_dates)

    slices: List[ConcurrentCodingSlice] = []
    for service_date in ordered_dates:
        day_docs = document_groups.get(service_date, [])
        day_orders = order_groups.get(service_date, [])

        day_input = base_input.model_copy(deep=True)
        day_input.documents = day_docs
        day_input.orders = day_orders
        day_input.orders_display = format_orders(day_orders)
        day_input.encounter_date = service_date
        if is_inpatient:
            day_text = _concatenate_document_texts(day_docs)
        else:
            day_text = _compose_day_text(day_docs, base_input.text, base_input.transcript)
        day_input.text = day_text
        day_input.transcript = day_text

        if metadata:
            day_input.encounter_metadata = metadata.model_copy(deep=True)
        else:
            day_input.encounter_metadata = EncounterMetaData()
        day_input.encounter_metadata.service_dates = [service_date]
        day_input.encounter_metadata.total_days = total_days

        slices.append(
            ConcurrentCodingSlice(
                service_date=service_date,
                specialty=None,
                day_input=day_input,
                documents=day_docs,
                orders=day_orders,
            )
        )

    return slices


def split_concurrent_input_by_service_date_and_specialty(payload: Any) -> List[ConcurrentCodingSlice]:
    """Split inpatient encounter payload into per-service-date and per-specialty slices."""
    from cdrc.common.data.schema import EncounterMetaData, InputSchema, format_orders

    if isinstance(payload, dict):
        base_input = InputSchema(**payload)
    elif isinstance(payload, InputSchema):
        base_input = payload
    else:
        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    metadata = base_input.encounter_metadata
    is_inpatient = _is_inpatient_encounter(metadata)

    documents = _filter_excluded_documents(list(base_input.documents or []))
    orders = list(base_input.orders or [])
    if is_inpatient:
        documents = _filter_documents_by_inpatient_window(documents, metadata)
        orders = _filter_orders_by_inpatient_window(orders, metadata)

    document_groups = _group_documents_by_date_and_specialty(documents)
    order_groups = _group_by_service_date(
        orders,
        extractor=lambda order: _normalize_service_date(getattr(order, "time_created", None) or getattr(order, "time_last_updated", None)),
    )

    if is_inpatient:
        anchor_dates = {
            date
            for date in (
                _normalize_service_date(getattr(doc, "service_date", None) or getattr(doc, "time_created", None))
                for doc in documents
                if _document_matches_inpatient_anchor(doc)
            )
            if date
        }
        service_dates: Set[str] = set(anchor_dates)
        if not service_dates:
            if metadata and metadata.service_dates:
                for raw_date in metadata.service_dates:
                    normalized = _normalize_service_date(raw_date if isinstance(raw_date, str) else str(raw_date))
                    if normalized:
                        service_dates.add(normalized)
        if not service_dates:
            service_dates.update({date for date, _ in document_groups.keys()})
            service_dates.update(order_groups.keys())
    else:
        service_dates = set()
        if metadata and metadata.service_dates:
            for raw_date in metadata.service_dates:
                normalized = _normalize_service_date(raw_date if isinstance(raw_date, str) else str(raw_date))
                if normalized:
                    service_dates.add(normalized)
        service_dates.update({date for date, _ in document_groups.keys()})
        service_dates.update(order_groups.keys())

    if not service_dates:
        fallback = _normalize_service_date(base_input.encounter_date) if getattr(base_input, "encounter_date", None) else None
        if not fallback and documents:
            fallback = _normalize_service_date(getattr(documents[0], "service_date", None) or getattr(documents[0], "time_created", None))
        if not fallback:
            raise ValueError("Unable to derive service dates for concurrent coding input")
        service_dates.add(fallback)

    ordered_dates = sorted(service_dates)
    total_days = len(ordered_dates)

    slices: List[ConcurrentCodingSlice] = []
    for service_date in ordered_dates:
        day_orders = order_groups.get(service_date, [])
        day_doc_groups = {specialty: docs for (doc_date, specialty), docs in document_groups.items() if doc_date == service_date}
        if not day_doc_groups:
            day_doc_groups = {"unknown": []}

        for specialty, day_docs in day_doc_groups.items():
            day_input = base_input.model_copy(deep=True)
            day_input.documents = day_docs
            day_input.orders = day_orders
            day_input.orders_display = format_orders(day_orders)
            day_input.encounter_date = service_date
            if is_inpatient:
                day_text = _concatenate_document_texts(day_docs)
            else:
                day_text = _compose_day_text(day_docs, base_input.text, base_input.transcript)
            day_input.text = day_text
            day_input.transcript = day_text

            if metadata:
                day_input.encounter_metadata = metadata.model_copy(deep=True)
            else:
                day_input.encounter_metadata = EncounterMetaData()
            day_input.encounter_metadata.service_dates = [service_date]
            day_input.encounter_metadata.total_days = total_days

            slices.append(
                ConcurrentCodingSlice(
                    service_date=service_date,
                    specialty=specialty,
                    day_input=day_input,
                    documents=day_docs,
                    orders=day_orders,
                )
            )

    return slices


def select_latest_concurrent_slice(slices: Sequence[ConcurrentCodingSlice]) -> Optional[ConcurrentCodingSlice]:
    if not slices:
        return None
    return sorted(slices, key=lambda item: item.service_date)[-1]
