import argparse
import ast
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

import pandas as pd
from langchain_core.runnables import RunnableConfig, RunnableLambda
from loguru import logger

from cdrc.common.data.schema import Modifier
from cdrc.common.utils import StringEnum, load_from_yaml

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


class DESIGN(StringEnum):
    LLM_ONLY = "llm_only"
    PARALLEL_MODIFIERS = "parallel_modifiers"
    PARALLEL_MODIFIERS_V2 = "parallel_modifiers_v2"
    PARALLEL_MODIFIERS_V3 = "parallel_modifiers_v3"
    PARALLEL_MODIFIERS_V4 = "parallel_modifiers_v4"
    PARALLEL_MODIFIERS_V5 = "parallel_modifiers_v5"
    PARALLEL_MODIFIERS_V6 = "parallel_modifiers_v6"
    DECISION_RULES = "decision_rules"


def extract_payer_names(payer_name, config: RunnableConfig) -> List[Union[str, dict]]:
    if isinstance(payer_name, list):
        return [x for x in payer_name if x]

    if isinstance(payer_name, str) and payer_name.strip():
        return [payer_name.strip()]

    cfg_value = (config.get("configurable", {}) or {}).get("payer")
    if isinstance(cfg_value, list):
        return [x for x in cfg_value if x]
    if isinstance(cfg_value, str) and cfg_value.strip():
        return [cfg_value.strip()]
    return []


MODIFIERS_IN_SCOPE = {
    "24": "Unrelated evaluation and management service by the same physician or other qualified health care professional during a postoperative period",
    "25": "Significant, separately identifiable evaluation and management service by the same physician or other qualified health care professional on the same day of the procedure or other service",
    "50": "Bilateral procedure",
    "51": "Multiple procedures",
    "52": "Reduced services",
    "57": "Decision for surgery",
    "80": "Assistant surgeon",
    "LT": "Left side (used to identify procedures performed on the left side)",
    "RT": "Right side (used to identify procedures performed on the right side)",
    "GP": "Services delivered under an outpatient physical therapy plan of care",
    "GO": "Services delivered under an outpatient occupational therapy plan of care",
    "GN": "Services delivered under an outpatient speech-language pathology plan of care",
    "GY": "Item or service statutorily excluded or not a Medicare benefit",
    "GZ": "Item or service expected to be denied as not reasonable and necessary (no ABN on file)",
    "GE": "Service performed by a resident without the presence of a teaching physician under the primary care exception",
    "QW": "CLIA-waived test",
    "PO": "Services, procedures and/or surgeries furnished at off-campus provider-based outpatient departments",
    "CQ": "Outpatient physical therapy services furnished in whole or in part by a physical therapist assistant",
    "CO": "Outpatient occupational therapy services furnished in whole or in part by an occupational therapy assistant",
    "PN": "Non-excepted service provided at an off-campus, outpatient, provider-based department of a hospital",
    "UD": "Medicaid level of care (state-specific modifier indicating special dispensing or program requirements)",
    "GC": "Service performed in part by a resident under the direction of a teaching physician",
    "QX": "Qualified non-physician anesthetist service with medical direction by a physician",
    "QZ": "Certified registered nurse anesthetist service without medical direction by a physician",
    "QK": "Medical direction of two, three, or four concurrent anesthesia procedures involving qualified individuals",
    "P1": "A normal healthy patient",
    "P2": "A patient with mild systemic disease",
    "P3": "A patient with severe systemic disease",
    "P4": "A patient with severe systemic disease that is a constant threat to life",
    "P5": "A moribund patient who is not expected to survive without the operation",
    "P6": "A declared brain-dead patient whose organs are being removed for donor purposes",
    "AS": "Physician assistant, nurse practitioner, or clinical nurse specialist services for assistant-at-surgery",
    "AI": "Principal physician of record",
    "KX": "Requirements specified in the medical policy have been met",
    "CG": "Policy criteria applied (service meets medical necessity and documentation requirements)",
    "GT": "Via interactive audio and video telecommunications system (telehealth)",
    "95": "Synchronous telemedicine service rendered via real-time interactive audio and video telecommunications system",
    "93": "Synchronous telemedicine service rendered via telephone or other real-time interactive audio-only telecommunications system",
    "HQ": "Group setting (services delivered in a group session)",
    "PT": "Colorectal cancer screening test; converted to diagnostic test or other procedure",
    "PS": "Postponed surgery (procedure discontinued prior to the administration of anesthesia)",
    "HA": "Child/adolescent program",
    "HN": "Bachelor’s degree level of care",
    "LD": "Left anterior descending coronary artery",
    "AY": "Item or service not related to the condition or diagnosis for which the patient is in the hospital or for which the SNF stay is covered",
    "90": "Reference (outside) laboratory",
}


def get_mod(e) -> str:
    return getattr(e, "code", None) or (e.get("code") if isinstance(e, dict) else None) or str(e)


def get_gold_key(p) -> str:
    return (
        getattr(p, "code", None)
        or getattr(p, "cpt_code", None)
        or (p.get("code") if isinstance(p, dict) else None)
        or (p.get("cpt_code") if isinstance(p, dict) else None)
        or str(p)
    )


def to_modifier(modifier_like: Any) -> Modifier:
    return Modifier(**dump(modifier_like))


def merge_unassigned_by_code(
    unassigned: List[Any],
    justification_cls: Type[Any],
) -> List[Any]:
    if not unassigned:
        return []

    grouped: Dict[str, List[Any]] = defaultdict(list)
    for item in unassigned:
        mod_code = normalize_code(dump(item).get("code") or getattr(item, "code", None))
        if mod_code:
            grouped[mod_code].append(item)

    merged: List[Any] = []
    for mod_code, items in grouped.items():
        rep = items[0]
        seen_evidence: set[str] = set()
        seen_justification: set[str] = set()
        evidence_parts: List[str] = []
        justification_parts: List[str] = []

        for item in items:
            evidence = dump(item).get("evidence") or getattr(item, "evidence", None) or ""
            justification = dump(item).get("justification") or getattr(item, "justification", None) or ""

            if evidence and evidence not in seen_evidence:
                seen_evidence.add(evidence)
                evidence_parts.append(evidence)

            if justification and justification not in seen_justification:
                seen_justification.add(justification)
                justification_parts.append(justification)

        rep_data = dump(rep).copy()
        rep_data["code"] = mod_code
        rep_data["assigned"] = False

        if evidence_parts:
            rep_data["evidence"] = "\n".join(evidence_parts)
        if justification_parts:
            rep_data["justification"] = "\n".join(justification_parts)

        merged.append(justification_cls(**rep_data))

    return merged


def apply_modifiers(items: List[Any], modifiers_by_code: Dict[str, List[Any]], item_type: Any, clear_linked: bool = False):
    output = []
    for item in items or []:
        item_data = dump(item).copy()
        code = item_data.get("code") or item_data.get("cpt_code") or getattr(item, "code", None) or getattr(item, "cpt_code", None)
        normalized_code = normalize_code(code)
        item_data["modifiers"] = modifiers_by_code.get(normalized_code, [])
        if clear_linked and "linked_diagnosis" in item_data:
            item_data["linked_diagnosis"] = []
        output.append(item_type(**item_data))
    return output


def dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        return vars(value)
    return value or {}


def normalize_code(raw: Any) -> str:
    return str(raw or "").strip().upper()


def normalize_modifier_codes(items: Iterable[str]) -> List[str]:
    return [normalize_code(x) for x in items if str(x).strip()]


def get_modifiers_in_scope(raw: Optional[object] = None) -> List[str]:
    """
    Resolve modifiers in scope from CLI/config input.

    Supported formats:
    - None: default to MODIFIERS_IN_SCOPE keys
    - "all": include all known modifiers
    - "25,50,LT": comma-separated list
    - ["25", "50", "LT"]: list/tuple/set (e.g., YAML list)
    - "['25','50']": stringified list
    """
    catalog = set(MODIFIERS_IN_SCOPE.keys())

    if raw is None:
        return list(MODIFIERS_IN_SCOPE.keys())

    if isinstance(raw, str):
        if raw.strip().lower() == "all":
            return list(MODIFIERS_IN_SCOPE.keys())
        try:
            evaluated = ast.literal_eval(raw)
            if isinstance(evaluated, (list, tuple, set)):
                raw = evaluated
        except (SyntaxError, ValueError):
            pass

    if isinstance(raw, (list, tuple, set)):
        selected = normalize_modifier_codes(raw)
    elif isinstance(raw, str):
        selected = normalize_modifier_codes(raw.split(","))
    else:
        logger.warning(f"Unsupported modifiers_in_scope value: {raw!r}. Using defaults.")
        return list(MODIFIERS_IN_SCOPE.keys())

    unknown = [m for m in selected if m not in catalog]
    if unknown:
        logger.warning(f"Ignoring unknown modifiers: {unknown}")

    filtered = [m for m in selected if m in catalog]
    if not filtered:
        logger.warning("No valid modifiers provided; scope will be empty.")
    return filtered


def get_cpt_codes(items):
    cpt_codes = []

    if items is None:
        return cpt_codes

    if items and isinstance(items, list):
        for x in items:
            if isinstance(x, dict):
                code = x["code"]
            else:
                code = getattr(x, "code", None)
            if code is not None:
                cpt_codes.append(code)

    return cpt_codes


def get_cpt_codes_description(items):
    # temporary read
    cpt_desc = pd.read_csv("cdrc/profees_coding/modifiers/data/cpt_codes_from_skg_15-12-25.csv")
    cpt_codes = []

    if items is None:
        return cpt_codes

    if items and isinstance(items, list):
        for x in items:
            if isinstance(x, dict):
                code = x["code"]
                description = x["description"]
            else:
                code = getattr(x, "code", None)
                description = getattr(x, "description", None)

            if code is not None:
                if description == "":
                    lookup = cpt_desc.loc[cpt_desc["code"] == code]
                    if len(lookup) > 0:
                        description = lookup["description"].values[0]

                if description != "":
                    cpt_codes.append(f"{code} ({description})")
                else:
                    cpt_codes.append(f"{code}")

    return cpt_codes


def dedupe_codes(codes: Sequence[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for code in codes or []:
        normalized = normalize_code(code)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def is_em_code(code: str) -> bool:
    try:
        n = int(code)
    except (ValueError, TypeError):
        return False
    # Outpatient/office and other E/M ranges
    em_ranges = [
        (99202, 99499),  # Standard E/M range per CPT
        (92002, 92014),  # General ophthalmological services
    ]
    return any(start <= n <= end for start, end in em_ranges)


def is_anesthesia_cpt(cpt_code: str) -> bool:
    if not cpt_code:
        return False

    code = "".join(filter(str.isdigit, cpt_code))
    if not code:
        return False

    try:
        code_int = int(code)
    except ValueError:
        return False

    return 100 <= code_int <= 1999


def is_procedure_code(code: str) -> bool:
    if code is None:
        return False
    normalized = str(code).strip().upper()
    if not normalized:
        return False
    return not is_em_code(normalized)


def get_guidelines_strings(path: str, prompt_version="v6.1", modifiers_in_scope: Optional[List[str]] = None):
    modifier_guidelines = load_from_yaml(path)
    if prompt_version in modifier_guidelines:
        modifier_guidelines = modifier_guidelines[prompt_version]
    else:
        modifier_guidelines = None
    modifier_guidelines_str = []
    compliance_checks_str = []
    checklist_str = []
    examples_str = []

    scope = modifiers_in_scope or get_modifiers_in_scope()
    for m in scope:
        if modifier_guidelines and m in modifier_guidelines:
            if "guidelines" in modifier_guidelines[m]:
                modifier_guidelines_str.append(modifier_guidelines[m]["guidelines"])
            if "compliance_checks" in modifier_guidelines[m]:
                compliance_checks_str.append(modifier_guidelines[m]["compliance_checks"])
            if "checklist" in modifier_guidelines[m]:
                checklist_str.append(modifier_guidelines[m]["checklist"])
            if "examples" in modifier_guidelines[m]:
                examples_str.append(modifier_guidelines[m]["examples"])
    if len(modifier_guidelines_str) > 0:
        modifier_guidelines_str = "\n-----------------------------\n".join(modifier_guidelines_str)
    else:
        modifier_guidelines_str = ""
    if len(compliance_checks_str) > 0:
        compliance_checks_str = list(dict.fromkeys(compliance_checks_str))
        compliance_checks_str = "\n".join(compliance_checks_str)
    else:
        compliance_checks_str = ""
    if len(checklist_str) > 0:
        checklist_str = list(dict.fromkeys(checklist_str))
        checklist_str = "\n".join(checklist_str)
    else:
        checklist_str = ""
    if len(examples_str) > 0:
        examples_str = list(dict.fromkeys(examples_str))
        examples_str = "\n".join(examples_str)
    else:
        examples_str = ""

    # except Exception:
    #     modifier_guidelines_str = ""
    #     compliance_checks_str = ""
    #     checklist_str = ""
    #     examples_str = ""

    return modifier_guidelines_str, compliance_checks_str, checklist_str, examples_str


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
            # ensures i + 1 is within the valid range before trying to access unknown_args[i + 1]
            if i + 1 < len(unknown_args) and unknown_args[i + 1].lower() in ("true", "false"):
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

    # Return the updated Namespace
    return argparse.Namespace(**merged_dict)


DEFAULT_EHR_LIST = [
    "payer_name",
    "encounter_metadata",
    "health_plan",
    "facility_info",
    "provider_info",
    "lab_info",
    "anaesthesia_staff_info",
    "anaesthesia_concurrent_staff_info",
    "anaesthesia_record_info",
]


def build_ehr_data(state, config: RunnableConfig, ehr_list=None):
    if ehr_list is None:
        ehr_list = DEFAULT_EHR_LIST

    lines: List[str] = []

    if "payer_name" in ehr_list:
        payer_names = state.get("payer_name")
        if payer_names:
            _payer_names = []
            for payer in extract_payer_names(payer_names, config):
                if isinstance(payer, dict):
                    payer_name = str(payer.get("payer_name", "")).strip()
                    is_medicare = payer.get("is_medicare")
                    _payer_names.append(f"{payer_name}, Medicare: {is_medicare}")
                else:
                    payer_name = str(payer).strip()
                    _payer_names.append(f"{payer_name}, Medicare: {payer_name in MEDICARE_PAYER_NAMES}")

            lines.append("- Payer Name:")
            lines.append(",".join(_payer_names))

    for ehr_data in ehr_list:
        if ehr_data == "payer_name":
            continue
        _metadata = state.get(ehr_data)
        if _metadata:
            ehr_data_name = ehr_data.replace("_", " ").capitalize()
            lines.append(f"- {ehr_data_name}:")
            if isinstance(_metadata, list):
                for _m in _metadata:
                    if hasattr(_m, "model_dump_json"):
                        lines.append(_m.model_dump_json(indent=2))
                    elif isinstance(_m, (dict, list)):
                        lines.append(json.dumps(_m, indent=2, default=str))
                    else:
                        lines.append(str(_m))
            else:
                if hasattr(_metadata, "model_dump_json"):
                    lines.append(_metadata.model_dump_json(indent=2))
                elif isinstance(_metadata, (dict, list)):
                    lines.append(json.dumps(_metadata, indent=2, default=str))
                else:
                    lines.append(str(_metadata))

    return "\n".join(lines)


def print_prompt(input):
    print("Printing input of type: ", type(input))
    if hasattr(input, "messages"):
        print(input.messages[0].content)
    else:
        print(input)
    return input


print_runnable = RunnableLambda(print_prompt)

codes_for_95 = [
    "90785",
    "90791",
    "90792",
    "90832",
    "90833",
    "90834",
    "90836",
    "90837",
    "90838",
    "90839",
    "90840",
    "90845",
    "90846",
    "90847",
    "90863",
    "90951",
    "90952",
    "90954",
    "90955",
    "90957",
    "90958",
    "90960",
    "90961",
    "90963",
    "90964",
    "90965",
    "90966",
    "90967",
    "90968",
    "90969",
    "90970",
    "92227",
    "92228",
    "92507",
    "92508",
    "92521",
    "92522",
    "92523",
    "92524",
    "92526",
    "92601",
    "92602",
    "92603",
    "92604",
    "93228",
    "93229",
    "93268",
    "93270",
    "93271",
    "93272",
    "96041",
    "96105",
    "96116",
    "96121",
    "96125",
    "96156",
    "96158",
    "96159",
    "96160",
    "96161",
    "96164",
    "96165",
    "96167",
    "96168",
    "96170",
    "96171",
    "97110",
    "97112",
    "97116",
    "97161",
    "97162",
    "97165",
    "97166",
    "97530",
    "97535",
    "97750",
    "97755",
    "97760",
    "97761",
    "97802",
    "97803",
    "97804",
    "98960",
    "98961",
    "98962",
    "99211",
    "99231",
    "99232",
    "99233",
    "99242",
    "99243",
    "99244",
    "99245",
    "99252",
    "99253",
    "99254",
    "99255",
    "99307",
    "99308",
    "99309",
    "99310",
    "99406",
    "99407",
    "99408",
    "99409",
    "99417",
    "99418",
    "99495",
    "99496",
    "99497",
    "99498",
]
inherently_bilateral_codes = ["27395", "92060", "92065", "92225", "92226", "92230", "92235", "92240", "92250", "92260", "92020"]
addon_codes_mod_50 = [
    "01953",
    "01968",
    "01969",
    "10004",
    "10006",
    "10008",
    "10010",
    "10012",
    "10036",
    "11001",
    "11008",
    "11045",
    "11046",
    "11047",
    "11103",
    "11105",
    "11107",
    "11201",
    "11732",
    "11922",
    "13102",
    "13122",
    "13133",
    "13153",
    "14302",
    "15003",
    "15005",
    "15012",
    "15014",
    "15016",
    "15018",
    "15101",
    "15111",
    "15116",
    "15121",
    "15131",
    "15136",
    "15151",
    "15152",
    "15156",
    "15157",
    "15201",
    "15221",
    "15241",
    "15261",
    "15272",
    "13153",
    "14302",
    "15003",
    "15005",
    "15012",
    "15014",
    "15016",
    "15018",
    "15101",
    "15111",
    "15116",
    "15121",
    "15131",
    "15136",
    "15151",
    "15152",
    "15156",
    "15157",
    "15201",
    "15221",
    "15241",
    "15261",
    "15272",
    "19288",
    "19294",
    "19297",
    "20700",
    "20701",
    "20702",
    "20703",
    "20704",
    "20705",
    "20930",
    "20931",
    "20932",
    "20933",
    "20934",
    "20936",
    "20937",
    "20938",
    "20939",
    "20985",
    "22103",
    "22116",
    "22208",
    "22216",
    "22226",
    "22328",
    "22512",
    "22515",
    "22527",
    "22534",
    "22552",
    "22585",
    "22614",
    "22632",
    "22634",
    "22840",
    "22841",
    "22842",
    "22843",
    "22844",
    "22845",
    "22846",
    "22847",
    "22848",
    "22853",
    "22854",
    "22858",
    "22859",
    "22860",
    "22868",
    "22870",
    "26125",
    "26861",
    "26863",
    "27358",
    "27692",
    "29826",
    "31627",
    "31632",
    "31633",
    "31637",
    "31649",
    "31651",
    "31654",
    "32501",
    "32506",
    "32507",
    "32667",
    "32668",
    "32674",
    "33141",
    "33225",
    "33257",
    "33258",
    "33259",
    "33268",
    "33277",
    "33367",
    "33368",
    "33369",
    "33370",
    "33419",
    "33508",
    "33517",
    "33518",
    "33519",
    "33521",
    "33522",
    "33523",
    "33530",
    "33572",
    "33746",
    "33768",
    "33866",
    "33884",
    "33904",
    "33924",
    "33929",
    "33987",
    "34709",
    "34711",
    "34713",
    "34714",
    "34715",
    "34716",
    "34717",
    "34808",
    "34812",
    "34813",
    "34820",
    "34833",
    "34834",
    "35306",
    "35390",
    "35400",
    "35500",
    "35572",
    "35681",
    "35682",
    "35683",
    "35685",
    "35686",
    "35697",
    "35700",
    "36218",
    "36227",
    "36228",
    "36248",
    "36474",
    "36476",
    "36479",
    "36483",
    "36907",
    "36908",
    "36909",
    "37185",
    "37186",
    "37222",
    "37223",
    "37232",
    "37233",
    "37234",
    "37235",
    "37237",
    "37239",
    "37247",
    "37249",
    "37252",
    "37253",
    "38102",
    "38746",
    "38747",
    "38900",
    "43273",
    "43283",
    "43338",
    "43635",
    "44015",
    "44121",
    "44128",
    "44139",
    "44203",
    "44213",
    "44701",
    "44955",
    "47001",
    "47542",
    "47543",
    "47544",
    "47550",
    "48400",
    "49326",
    "49327",
    "49412",
    "49435",
    "49623",
    "49905",
    "50606",
    "50705",
    "50706",
    "51797",
    "52442",
    "56606",
    "57267",
    "57465",
    "58110",
    "58611",
    "60512",
    "60661",
    "61316",
    "61517",
    "61611",
    "61641",
    "61642",
    "61651",
    "61781",
    "61782",
    "61783",
    "61797",
    "61799",
    "61800",
    "61864",
    "61868",
    "62148",
    "62160",
    "63035",
    "63043",
    "63044",
    "63048",
    "63052",
    "63053",
    "63057",
    "63066",
    "63076",
    "63078",
    "63082",
    "63086",
    "63088",
    "63091",
    "63103",
    "63295",
    "63308",
    "63621",
    "64421",
    "64462",
    "64480",
    "64484",
    "64491",
    "64492",
    "64494",
    "64495",
    "64597",
    "64629",
    "64634",
    "64636",
    "64643",
    "64645",
    "64727",
    "64778",
    "64783",
    "64787",
    "64832",
    "64837",
    "64859",
    "64872",
    "64874",
    "64876",
    "64901",
    "64902",
    "64913",
    "65757",
    "66990",
    "67225",
    "67320",
    "67331",
    "67332",
    "67334",
    "67335",
    "67340",
    "69990",
    "74248",
    "74301",
    "74713",
    "75565",
    "75774",
    "76015",
    "76125",
    "76802",
    "76810",
    "76812",
    "76814",
    "76937",
    "76979",
    "76983",
    "77001",
    "77002",
    "77003",
    "77063",
    "77293",
    "78020",
    "78434",
    "78496",
    "78730",
    "78835",
    "80506",
    "81266",
    "81416",
    "81426",
    "81536",
    "82952",
    "86826",
    "87187",
    "87503",
    "87904",
    "88155",
    "88177",
    "88185",
    "88311",
    "88314",
    "88332",
    "88334",
    "88341",
    "88350",
    "88364",
    "88369",
    "88373",
    "0071U",
    "0072U",
    "0073U",
    "0074U",
    "0075U",
    "0076U",
    "0130U",
    "0131U",
    "0132U",
    "0133U",
    "0134U",
    "0135U",
    "0136U",
    "0137U",
    "0138U",
    "0157U",
    "0158U",
    "0159U",
    "0160U",
    "0161U",
    "0162U",
    "0207U",
    "0427U",
    "90461",
    "90472",
    "90474",
    "90785",
    "90833",
    "90836",
    "90838",
    "90840",
    "90863",
    "90913",
    "91013",
    "92547",
    "92608",
    "92618",
    "92621",
    "92623",
    "92627",
    "92921",
    "92925",
    "92929",
    "92934",
    "92938",
    "92944",
    "92972",
    "92973",
    "92974",
    "92978",
    "92979",
    "92998",
    "93319",
    "93320",
    "93321",
    "93325",
    "93352",
    "93356",
    "93462",
    "93463",
    "93464",
    "93563",
    "93564",
    "93565",
    "93566",
    "93567",
    "93568",
    "93569",
    "93571",
    "93572",
    "93573",
    "93574",
    "93575",
    "93584",
    "93585",
    "93586",
    "93587",
    "93588",
    "93592",
    "93598",
    "93609",
    "93613",
    "93621",
    "93622",
    "93623",
    "93655",
    "93657",
    "93662",
    "93896",
    "93897",
    "93898",
    "94645",
    "94729",
    "94781",
    "95079",
    "95873",
    "95874",
    "95885",
    "95886",
    "95887",
    "95940",
    "95941",
    "95962",
    "95967",
    "95984",
    "96113",
    "96121",
    "96131",
    "96133",
    "96137",
    "96139",
    "96159",
    "96165",
    "96168",
    "96171",
    "96203",
    "96361",
    "96366",
    "96367",
    "96368",
    "96370",
    "96371",
    "96375",
    "96376",
    "96411",
    "96415",
    "96417",
    "96423",
    "96547",
    "96548",
    "96570",
    "96571",
    "96934",
    "96935",
    "96936",
    "97130",
    "97546",
    "97551",
    "97598",
    "97811",
    "97814",
    "98981",
    "99100",
    "99116",
    "99135",
    "99140",
    "99153",
    "99157",
    "99292",
    "99359",
    "99415",
    "99416",
    "99417",
    "99418",
    "99425",
    "99427",
    "99437",
    "99439",
    "99458",
    "99459",
    "99467",
    "99486",
    "99489",
    "99494",
    "99498",
    "99602",
    "99607",
    "0054T",
    "0055T",
    "0076T",
    "0095T",
    "0098T",
    "0164T",
    "0165T",
    "0174T",
    "0214T",
    "0215T",
    "0217T",
    "0218T",
    "0222T",
    "0397T",
    "0437T",
    "0439T",
    "0443T",
    "0450T",
    "0480T",
    "0496T",
    "0513T",
    "0523T",
    "0560T",
    "0562T",
    "0570T",
    "0599T",
    "0628T",
    "0630T",
    "0649T",
    "0663T",
    "0676T",
    "0678T",
    "0690T",
    "0698T",
    "0701T",
    "0709T",
    "0722T",
    "0724T",
    "0735T",
    "0742T",
    "0751T",
    "0752T",
    "0753T",
    "0754T",
    "0755T",
    "0756T",
    "0757T",
    "0758T",
    "0759T",
    "0760T",
    "0761T",
    "0762T",
    "0763T",
    "0764T",
    "0767T",
    "0770T",
    "0772T",
    "0774T",
    "0777T",
    "0791T",
    "0821T",
    "0822T",
    "0827T",
    "0828T",
    "0829T",
    "0830T",
    "0831T",
    "0832T",
    "0833T",
    "0834T",
    "0835T",
    "0836T",
    "0837T",
    "0838T",
    "0839T",
    "0840T",
    "0841T",
    "0842T",
    "0843T",
    "0844T",
    "0845T",
    "0846T",
    "0847T",
    "0848T",
    "0849T",
    "0850T",
    "0851T",
    "0852T",
    "0853T",
    "0854T",
    "0855T",
    "0856T",
    "0857T",
    "0859T",
    "0866T",
    "0882T",
    "0883T",
    "0887T",
    "0896T",
    "0899T",
    "0900T",
    "0907T",
    "0914T",
    "0945T",
]
