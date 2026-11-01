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
    WORKFLOW = "workflow"
    LLM_MULTI_STAGE = "llm_multistage"
    LLM_SINGLE_STAGE = "llm_singlestage"
    LLM_SINGLE_STAGE_ENHANCED = "llm_singlestage_enhanced"


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
