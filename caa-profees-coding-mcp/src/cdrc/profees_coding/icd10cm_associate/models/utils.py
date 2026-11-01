import argparse

from cdrc.common.utils import StringEnum


class DESIGN(StringEnum):
    LLM_ONLY_V1 = "llm_only_v1"
    LLM_ONLY_V2 = "llm_only_v2"
    LLM_ONLY_V3 = "llm_only_v3"
    LLM_ONLY_MERGED_V1 = "llm_only_merged_v1"
    LLM_ONLY_MERGED_V2 = "llm_only_merged_v2"
    LLM_ONLY_MERGED_V3 = "llm_only_merged_v3"
    LLM_ONLY_MERGED_V4 = "llm_only_merged_v4"
    LLM_ONLY_MERGED_V5 = "llm_only_merged_v5"


class CONFIG(StringEnum):
    PROMPTS = "prompts"
    PROMPT_VERSION = "prompt_version"
    PROMPT_KEY = "prompt_key"
    ENHANCER_TYPE = "enhancer_type"
    ENHANCER_PROMPT_VERSION = "enhancer_prompt_version"
    DIAGNOSIS_REGENERATOR = "diagnosis_regenerator"
    REGENERATOR_KEY = "regenerator_key"
    REGENERATOR_VERSION = "regenerator_version"
    R_ENHANCER_PROMPT_VERSION = "r_enhancer_prompt_version"
    MODEL_NAME = "model_name"
    MODEL_KWARGS = "model_kwargs"
    NUM_REPEATS = "num_repeats"
    EXPORT_INTERMEDIATE_PREDICTIONS = "export_intermediate_predictions"


class PROMPTS(StringEnum):
    LLM_ONLY = "llm_only"
    OUTPATIENT = "outpatient"
    ED = "ed"
    INPATIENT = "inpatient"
    ICD_CODING = "icd_coding"
    Z_CODE_ENHANCER = "z_code_enhancer"
    R_CODE_ENHANCER = "r_code_enhancer"


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
