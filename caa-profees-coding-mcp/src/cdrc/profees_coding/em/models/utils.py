import argparse
import difflib
import math
import random
from typing import Any, Union

import pydantic
from langchain_core.prompts import ChatPromptTemplate

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import LLM, get_llm
from cdrc.common.utils import StringEnum
from cdrc.profees_coding.em.data.schema import AgeCategories, ComplexityLevel, EncounterType, PatientType


class DESIGN(StringEnum):
    CHRG_ASST = "charge_assist"
    LLM_ONLY = "llm_only"
    DEC_TREE = "decision_tree_graph"
    MEDPROMPT = "medprompt"
    MEDPROMPT_DETAILED = "medprompt_detailed"
    MEDPROMPT_DETAILED_MULTI = "medprompt_detailed_multi"
    MEDPROMPT_DETAILED_CRITICS = "medprompt_detailed_critics"
    MEDPROMPT_DETAILED_MDM = "medprompt_detailed_mdm"
    REPREDICTION_MDM_ONLY = "reprediction_mdm_only"
    REPREDICTION_CRITICS_ONLY = "reprediction_critics_only"


def get_chain(
    model_name: str,
    model_kwargs: dict,
    prompt: str,
    output_pydantic_model: pydantic.BaseModel,
    partial_variables: dict[str, Any] = {},
):
    # TODO: add the model config here & investigate why response format is not working
    if model_name in (LLM.GPT_4o_2024_05_13_ORACLE.value, LLM.GPT_4o_2024_05_13_ORACLE_SDS.value, LLM.GPT_4o_mini_2024_07_18_DEV_SDS.value):
        model = get_llm()[model_name](extra_body={"response_format": {"type": "json_object"}}, **model_kwargs)
    else:
        model = get_llm()[model_name](**model_kwargs)

    output_parser = PydanticOutputParserWithRetries(pydantic_object=output_pydantic_model, correction_model=model, num_retries=3)
    prompt = ChatPromptTemplate.from_template(prompt, template_format="jinja2").partial(
        format_instructions=output_parser.get_format_instructions(), **partial_variables
    )
    chain = prompt | model | output_parser

    return chain


def get_mdm_complexity(
    problem_complexity_level: Union[ComplexityLevel, str],
    data_complexity_level: Union[ComplexityLevel, str],
    risk_level: Union[ComplexityLevel, str],
) -> str:
    problem_complexity_level = ComplexityLevel(problem_complexity_level)
    data_complexity_level = ComplexityLevel(data_complexity_level)
    risk_level = ComplexityLevel(risk_level)

    levels = [problem_complexity_level, data_complexity_level, risk_level]

    # sort by order of Complexity fields
    levels.sort(key=lambda x: [elem for elem in ComplexityLevel].index(x))  # Minimal, Low, Moderate, High

    # check for uniqueness
    if len(set(levels)) < 3:
        # return majority
        return max(set(levels), key=lambda x: levels.count(x))
    else:
        return levels[
            1
        ]  # return the middle value in the sorted list because the highest value will be discounted to the middle value so it becomes the majority


def get_age_category(age: int) -> AgeCategories:
    if age < 0:
        return AgeCategories.INVALID
    elif age < 1:
        return AgeCategories.INFANT
    elif age < 5:
        return AgeCategories.EARLY_CHILDHOOD
    elif age < 12:
        return AgeCategories.LATE_CHILDHOOD
    elif age < 18:
        return AgeCategories.ADOLESCENT
    elif age < 40:
        return AgeCategories.YOUNG_ADULT
    elif age < 65:
        return AgeCategories.ADULT
    else:
        return AgeCategories.SENIOR_ADULT


def get_time_based_em_code(
    encounter_type: EncounterType,
    patient_type: PatientType,
    time_spent_minutes: int,
) -> str:
    """Return the time-based E/M code for outpatient encounters, if applicable.

    Uses typical 2021+ outpatient time thresholds:
    - New:        99202 (15–29), 99203 (30–44), 99204 (45–59), 99205 (≥60)
    - Established:99212 (10–19), 99213 (20–29), 99214 (30–39), 99215 (≥40)
    """
    if encounter_type != EncounterType.OUTPATIENT:
        return None

    if time_spent_minutes is None or time_spent_minutes <= 0:
        return None

    # (min_time, max_time, code); max_time = float("inf") means no upper bound
    time_table: PatientType = {
        PatientType.NEW: [
            (15, 29, "99202"),
            (30, 44, "99203"),
            (45, 59, "99204"),
            (60, float("inf"), "99205"),
        ],
        PatientType.ESTABLISHED: [
            (10, 19, "99212"),
            (20, 29, "99213"),
            (30, 39, "99214"),
            (40, float("inf"), "99215"),
        ],
    }

    ranges = time_table.get(patient_type)
    if not ranges:
        return None

    for min_t, max_t, code in ranges:
        if min_t <= time_spent_minutes <= max_t:
            return code

    return None


EM_CODE_DESCRIPTIONS = {
    # Outpatient office/other outpatient (new)
    "99202": "Office o/p new sf 15 min",
    "99203": "Office o/p new low 30 min",
    "99204": "Office o/p new mod 45 min",
    "99205": "Office o/p new hi 60 min",
    # Outpatient office/other outpatient (established)
    "99211": "Off/op est may x req phy/qhp",
    "99212": "Office o/p est sf 10 min",
    "99213": "Office o/p est low 20 min",
    "99214": "Office o/p est mod 30 min",
    "99215": "Office o/p est hi 40 min",
    # Preventive medicine (new)
    "99381": "Init pm e/m new pat infant",
    "99382": "Init pm e/m new pat 1-4 yrs",
    "99383": "Prev visit new age 5-11",
    "99384": "Prev visit new age 12-17",
    "99385": "Prev visit new age 18-39",
    "99386": "Prev visit new age 40-64",
    "99387": "Init pm e/m new pat 65+ yrs",
    # Preventive medicine (established)
    "99391": "Per pm reeval est pat infant",
    "99392": "Prev visit est age 1-4",
    "99393": "Prev visit est age 5-11",
    "99394": "Prev visit est age 12-17",
    "99395": "Prev visit est age 18-39",
    "99396": "Prev visit est age 40-64",
    "99397": "Per pm reeval est pat 65+ yr",
    # Emergency department / other emergency services
    "99281": "Emr dpt vst mayx req phy/qhp",
    "99283": "Emergency dept visit low mdm",
    "99284": "Emergency dept visit mod mdm",
    "99285": "Emergency dept visit hi mdm",
    "99288": "Direct advanced life support",
    # Critical care
    "99291": "Critical care first 30-74 min",
    "99292": "Critical care each addl 30 min",
    # Inpatient / observation
    "99221": "1st hosp ip/obs sf/low 40",
    "99222": "1st hosp ip/obs moderate 55",
    "99223": "1st hosp ip/obs high 75",
    "99231": "Sbsq hosp ip/obs sf/low 25",
    "99232": "Sbsq hosp ip/obs moderate 35",
    "99233": "Sbsq hosp ip/obs high 50",
    "99234": "Hosp ip/obs sm dt sf/low 45",
    "99235": "Hosp ip/obs same date mod 70",
    "99236": "Hosp ip/obs same date hi 85",
    "99238": "Hosp ip/obs dschrg mgmt 30/<",
    "99239": "Hosp ip/obs dschrg mgmt >30",
    # Normal newborn care
    "99460": "Init hosp/birth ctr care normal newborn",
    "99461": "Init care normal newborn other setting",
    "99462": "Subsq hosp care normal newborn",
    "99463": "Init/disch same day normal newborn",
    "INVALID_AGE": "Invalid age for preventive medicine code selection.",
    "NOT_SUPPORTED": "NOT_SUPPORTED",
}


def get_em_code_description(code: str) -> str:
    return EM_CODE_DESCRIPTIONS.get(code, "")


def get_critical_care_codes(time_spent_minutes: int) -> list[str]:
    """Return critical care primary/add-on CPT codes supported by documented time."""
    if time_spent_minutes is None or time_spent_minutes < 30:
        return []

    codes = ["99291"]
    if time_spent_minutes >= 75:
        add_on_units = math.ceil((time_spent_minutes - 74) / 30)
        codes.extend(["99292"] * add_on_units)
    return codes


def get_code(
    time_spent_minutes: int,
    patient_type: PatientType,
    encounter_type: EncounterType,
    problem_complexity_level: ComplexityLevel,
    data_complexity_level: ComplexityLevel,
    risk_level: ComplexityLevel,
    age: int,
    requires_physician_presence: bool = None,
    return_mdm_complexity: bool = False,
    mdm_complexity: Union[ComplexityLevel, None] = None,
) -> str:
    age_cateagory = get_age_category(age)
    code_lookup = {
        EncounterType.OUTPATIENT: {
            PatientType.NEW: {
                ComplexityLevel.MINIMAL: "99202",
                ComplexityLevel.LOW: "99203",
                ComplexityLevel.MODERATE: "99204",
                ComplexityLevel.HIGH: "99205",
            },
            PatientType.ESTABLISHED: {
                ComplexityLevel.MINIMAL: "99212",
                ComplexityLevel.LOW: "99213",
                ComplexityLevel.MODERATE: "99214",
                ComplexityLevel.HIGH: "99215",
            },
        },
        # EncounterType.CONSULTATION: {
        #     ComplexityLevel.MINIMAL: "99242",
        #     ComplexityLevel.LOW: "99243",
        #     ComplexityLevel.MODERATE: "99244",
        #     ComplexityLevel.HIGH: "99245",
        # },
        EncounterType.PREVENTIVE_MEDICINE: {
            PatientType.NEW: {
                AgeCategories.INFANT: "99381",
                AgeCategories.EARLY_CHILDHOOD: "99382",
                AgeCategories.LATE_CHILDHOOD: "99383",
                AgeCategories.ADOLESCENT: "99384",
                AgeCategories.YOUNG_ADULT: "99385",
                AgeCategories.ADULT: "99386",
                AgeCategories.SENIOR_ADULT: "99387",
                AgeCategories.INVALID: "INVALID_AGE",
            },
            PatientType.ESTABLISHED: {
                AgeCategories.INFANT: "99391",
                AgeCategories.EARLY_CHILDHOOD: "99392",
                AgeCategories.LATE_CHILDHOOD: "99393",
                AgeCategories.ADOLESCENT: "99394",
                AgeCategories.YOUNG_ADULT: "99395",
                AgeCategories.ADULT: "99396",
                AgeCategories.SENIOR_ADULT: "99397",
                AgeCategories.INVALID: "INVALID_AGE",
            },
        },
    }
    # 99211 nurse visit: no physician presence (not really time- or MDM-based)
    if patient_type == PatientType.ESTABLISHED:
        if requires_physician_presence is not None and not requires_physician_presence:
            code = "99211"
            source = "MDM"  # or "Other" / "Nurse" if you want a separate category
            description = get_em_code_description(code)
            if return_mdm_complexity:
                return code, description, source, None
            return code, description, source

    if encounter_type == EncounterType.OUTPATIENT:
        # --- MDM-based code ---
        if mdm_complexity is None:
            mdm_complexity = get_mdm_complexity(
                problem_complexity_level,
                data_complexity_level,
                risk_level,
            )
        mdm_code = code_lookup[encounter_type][patient_type][mdm_complexity]

        # --- Time-based code ---
        time_code = get_time_based_em_code(
            encounter_type=encounter_type,
            patient_type=patient_type,
            time_spent_minutes=time_spent_minutes,
        )

        # --- Choose highest possible code (numerically) + source ---
        chosen_code = mdm_code
        source = "MDM"

        if time_code is not None:
            # Compare numerically (e.g., 99215 > 99214)
            if int(time_code) > int(mdm_code):
                chosen_code = time_code
                source = "Time"

        description = get_em_code_description(chosen_code)
        if not return_mdm_complexity:
            return chosen_code, description, source
        else:
            # Still return the MDM complexity (even if time-based code wins)
            return chosen_code, description, source, mdm_complexity.value

    # For non-outpatient encounters, keep existing behavior (no time-based logic)
    # elif encounter_type == EncounterType.CONSULTATION:
    #     if mdm_complexity is None:
    #         mdm_complexity = get_mdm_complexity(
    #             problem_complexity_level,
    #             data_complexity_level,
    #             risk_level,
    #         )
    #     code = code_lookup[encounter_type][mdm_complexity]
    #     source = "MDM"
    #     if return_mdm_complexity:
    #         return code, source, mdm_complexity.value
    #     return code, source

    elif encounter_type == EncounterType.PREVENTIVE_MEDICINE:
        code = code_lookup[encounter_type][patient_type][age_cateagory]
        description = get_em_code_description(code)
        source = "Age"
        # Preventive medicine visits are not MDM-based under this scheme
        if return_mdm_complexity:
            return code, description, source, None
        return code, description, source

    elif encounter_type in {EncounterType.EMERGENCY_DEPARTMENT, EncounterType.EMERGENCY_OTHER_SERVICES}:
        if mdm_complexity is None:
            mdm_complexity = get_mdm_complexity(
                problem_complexity_level,
                data_complexity_level,
                risk_level,
            )
        ed_codes = {
            ComplexityLevel.MINIMAL: "99281",
            ComplexityLevel.LOW: "99283",
            ComplexityLevel.MODERATE: "99284",
            ComplexityLevel.HIGH: "99285",
        }
        code = ed_codes.get(mdm_complexity, "NOT_SUPPORTED")
        if encounter_type == EncounterType.EMERGENCY_OTHER_SERVICES:
            code = "99288"
        description = get_em_code_description(code)
        source = "MDM"
        if return_mdm_complexity:
            return code, description, source, mdm_complexity.value
        return code, description, source

    elif encounter_type == EncounterType.CRITICAL_CARE:
        codes = get_critical_care_codes(time_spent_minutes)
        code = codes[0] if codes else "NOT_SUPPORTED"
        description = get_em_code_description(code)
        source = "Time" if code != "NOT_SUPPORTED" else "NOT_SUPPORTED"
        if return_mdm_complexity:
            return code, description, source, None
        return code, description, source

    elif encounter_type in {
        EncounterType.INPATIENT_INITIAL,
        EncounterType.INPATIENT_SUBSEQUENT,
        EncounterType.INPATIENT_SAME_DAY,
        EncounterType.INPATIENT_DISCHARGE,
    }:
        if mdm_complexity is None:
            mdm_complexity = get_mdm_complexity(
                problem_complexity_level,
                data_complexity_level,
                risk_level,
            )
        inpatient_codes = {
            EncounterType.INPATIENT_INITIAL: {
                ComplexityLevel.MINIMAL: "99221",
                ComplexityLevel.LOW: "99222",
                ComplexityLevel.MODERATE: "99223",
                ComplexityLevel.HIGH: "99223",
            },
            EncounterType.INPATIENT_SUBSEQUENT: {
                ComplexityLevel.MINIMAL: "99231",
                ComplexityLevel.LOW: "99232",
                ComplexityLevel.MODERATE: "99233",
                ComplexityLevel.HIGH: "99233",
            },
            EncounterType.INPATIENT_SAME_DAY: {
                ComplexityLevel.MINIMAL: "99234",
                ComplexityLevel.LOW: "99235",
                ComplexityLevel.MODERATE: "99236",
                ComplexityLevel.HIGH: "99236",
            },
            EncounterType.INPATIENT_DISCHARGE: {
                ComplexityLevel.MINIMAL: "99238",
                ComplexityLevel.LOW: "99238",
                ComplexityLevel.MODERATE: "99239",
                ComplexityLevel.HIGH: "99239",
            },
        }
        code = inpatient_codes.get(encounter_type, {}).get(mdm_complexity, "NOT_SUPPORTED")
        description = get_em_code_description(code)
        source = "MDM"
        if return_mdm_complexity:
            return code, description, source, mdm_complexity.value
        return code, description, source

    elif encounter_type in {
        EncounterType.NEWBORN_INITIAL,
        EncounterType.NEWBORN_SUBSEQUENT,
        EncounterType.NEWBORN_SAME_DAY,
        EncounterType.NEWBORN_OTHER_SETTING,
    }:
        newborn_codes = {
            EncounterType.NEWBORN_INITIAL: "99460",
            EncounterType.NEWBORN_OTHER_SETTING: "99461",
            EncounterType.NEWBORN_SUBSEQUENT: "99462",
            EncounterType.NEWBORN_SAME_DAY: "99463",
        }
        code = newborn_codes[encounter_type] if age_cateagory == AgeCategories.INFANT else "INVALID_AGE"
        description = get_em_code_description(code)
        source = "Age" if code != "INVALID_AGE" else "NOT_SUPPORTED"
        if return_mdm_complexity:
            return code, description, source, None
        return code, description, source

    else:
        return "NOT_SUPPORTED", get_em_code_description("NOT_SUPPORTED"), "NOT_SUPPORTED"


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


def get_random_age_from_code(cpt_code: str) -> int:
    """Returns a random age that would be valid for the given CPT code.

    Args:
        cpt_code: A CPT code string (e.g., '99381', '99214')

    Returns:
        A random integer representing a valid age for the code
    """
    # Age ranges for preventive medicine codes
    age_ranges = {
        # New patient preventive
        "99381": (0, 0),  # infant (< 1)
        "99382": (1, 4),  # early childhood
        "99383": (5, 11),  # late childhood
        "99384": (12, 17),  # adolescent
        "99385": (18, 39),  # young adult
        "99386": (40, 64),  # adult
        "99387": (65, 90),  # senior (65+)
        # Established patient preventive
        "99391": (0, 0),
        "99392": (1, 4),
        "99393": (5, 11),
        "99394": (12, 17),
        "99395": (18, 39),
        "99396": (40, 64),
        "99397": (65, 90),
    }

    if cpt_code in age_ranges:
        min_age, max_age = age_ranges[cpt_code]
        if min_age == 0:  # For infants
            # Generate age in months, then convert to fraction of year
            months = random.randint(0, 11)
            return months / 12
        return random.randint(min_age, max_age)

    # For non-preventive codes (E/M visits), this function should not be used
    raise ValueError(f"Invalid CPT code for age generation: {cpt_code}")


def get_complexity_from_code(cpt_code: str) -> ComplexityLevel:
    """Returns the complexity level associated with a given CPT code.

    Args:
        cpt_code: A CPT code string (e.g., '99214', '99205')

    Returns:
        ComplexityLevel enum value corresponding to the code's complexity

    Raises:
        ValueError: If the CPT code is not a valid E/M code with an associated complexity level
    """
    complexity_mapping = {
        # Outpatient New
        "99202": ComplexityLevel.MINIMAL,
        "99203": ComplexityLevel.LOW,
        "99204": ComplexityLevel.MODERATE,
        "99205": ComplexityLevel.HIGH,
        # Outpatient Established
        "99212": ComplexityLevel.MINIMAL,
        "99213": ComplexityLevel.LOW,
        "99214": ComplexityLevel.MODERATE,
        "99215": ComplexityLevel.HIGH,
        # Consultation
        "99242": ComplexityLevel.MINIMAL,
        "99243": ComplexityLevel.LOW,
        "99244": ComplexityLevel.MODERATE,
        "99245": ComplexityLevel.HIGH,
        # Emergency department
        "99281": ComplexityLevel.MINIMAL,
        "99282": ComplexityLevel.MINIMAL,
        "99283": ComplexityLevel.LOW,
        "99284": ComplexityLevel.MODERATE,
        "99285": ComplexityLevel.HIGH,
        # Inpatient / observation
        "99221": ComplexityLevel.LOW,
        "99222": ComplexityLevel.MODERATE,
        "99223": ComplexityLevel.HIGH,
        "99231": ComplexityLevel.LOW,
        "99232": ComplexityLevel.MODERATE,
        "99233": ComplexityLevel.HIGH,
        "99234": ComplexityLevel.LOW,
        "99235": ComplexityLevel.MODERATE,
        "99236": ComplexityLevel.HIGH,
    }

    if cpt_code not in complexity_mapping:
        return None

    return complexity_mapping[cpt_code].value


def create_markdown_diff(original_text, updated_text):
    """Create a markdown-formatted diff between two texts"""
    # Split text into lines
    original_lines = original_text.splitlines()
    updated_lines = updated_text.splitlines()

    # Generate diff
    differ = difflib.Differ()
    diff = list(differ.compare(original_lines, updated_lines))

    # Convert diff to markdown with colors
    markdown_lines = []
    addition_lines = []
    removal_lines = []
    for line in diff:
        if line.startswith("+ "):
            markdown_lines.append(f"+++ {line[2:]}")
            addition_lines.append(line[2:])
        elif line.startswith("- "):
            markdown_lines.append(f"--- ~~{line[2:]}~~")
            removal_lines.append(line[2:])
        elif line.startswith("  "):
            markdown_lines.append(line[2:])
        # Skip lines starting with '?' which are hints from difflib

    return "\n".join(markdown_lines), addition_lines, removal_lines
