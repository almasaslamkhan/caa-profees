import argparse
import random
from typing import Any, Union

import pydantic
from langchain_core.prompts import ChatPromptTemplate

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import LLM, get_llm
from cdrc.common.utils import StringEnum


class DESIGN(StringEnum):
    LLM_ONLY = "llm_only"
    NER_RAG = "ner_rag"
    NER_LLM = "ner_llm"
    NER_LLM_RAG = "ner_llm_rag"


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


def is_imaging_code(code: str) -> bool:
    """
    Return True if the CPT code belongs to any imaging category:
    X-ray, CT, MRI, Ultrasound, Nuclear Medicine, PET.
    """
    try:
        c = int(code)
    except ValueError:
        return False  # skip malformed codes

    # X-ray: 70010–76499
    if 70010 <= c <= 76499:
        return True

    # Ultrasound: 76506–76999
    if 76506 <= c <= 76999:
        return True

    # CT: 70450–70498, 71250–71270, 72125–72133, 73700–73706, 74150–74178
    CT_RANGES = [
        (70450, 70498),
        (71250, 71270),
        (72125, 72133),
        (73700, 73706),
        (74150, 74178),
    ]
    if any(lo <= c <= hi for lo, hi in CT_RANGES):
        return True

    # MRI: many ranges
    MRI_RANGES = [
        (70336, 70336),
        (70540, 70553),
        (72141, 72159),
        (73218, 73223),
        (73718, 73723),
        (75557, 75565),
        (77046, 77049),
    ]
    if any(lo <= c <= hi for lo, hi in MRI_RANGES):
        return True

    # Nuclear medicine: 78012–78999
    if 78012 <= c <= 78999:
        return True

    # PET: 78811–78816
    if 78811 <= c <= 78816:
        return True

    return False


def is_lab_code(code: str) -> bool:
    """
    Return True if the CPT code is a laboratory/pathology code (80000–89999),
    which should NOT be billed on professional-fee claims.
    """
    try:
        c = int(code)
    except ValueError:
        return False  # skip malformed codes

    # All lab & pathology CPT codes fall within 80000–89999
    return 80000 <= c <= 89999


def is_ekg_code(code: str) -> bool:
    ekg_code_list = ["93000", "93005", "93010"]
    return code in ekg_code_list


def is_category_2_code(code: str) -> bool:
    return code.endswith("F")


def is_vaccine_product(code: str) -> bool:
    """
    Return True if the CPT code is a vaccine PRODUCT code.
    (906xx, 907xx, 913xx, COVID A-series codes)
    NOT including admin codes 90460/90461/90471/90472/90480.
    """
    admin_codes = {"90460", "90461", "90471", "90472", "90480"}

    # If it's one of the allowed administration codes → NOT a product
    if code in admin_codes:
        return False

    # Vaccine product codes typically fall into:
    # 1) 906xx, 907xx, 913xx ranges (all numeric)
    try:
        c = int(code)
        # 90600–90699 (flu, Hep, PCV, etc.)
        if 90600 <= c <= 90699:
            return True
        # 90700–90799 (Tdap, MMR, Shingrix, etc.)
        if 90700 <= c <= 90799:
            return True
        # 91300–91399 (COVID vaccine products)
        if 91300 <= c <= 91399:
            return True
    except ValueError:
        pass

    # 2) COVID "A-series" product codes: 0051A–0094A, etc.
    # These are alphanumeric and do NOT parse as int.
    if code.endswith("A") and code[:-1].isdigit():
        # Example: 0051A, 0094A
        return True

    return False
