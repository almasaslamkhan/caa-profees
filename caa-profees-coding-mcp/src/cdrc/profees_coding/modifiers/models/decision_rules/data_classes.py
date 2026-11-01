from cdrc.common.utils import StringEnum


class CONFIG(StringEnum):
    PROMPTS = "prompts"
    PROMPT_VERSION = "prompt_version"
    PROMPT_VERSIONS = "prompt_versions"
    MODEL_NAME = "model_name"
    MODEL_KWARGS = "model_kwargs"
    METADATA_RULES = "metadata_rules"
    TEXT_RULE_FLAGS = "text_rule_flags"


class PROMPTS(StringEnum):
    DECISION_RULES = "decision_rules"
