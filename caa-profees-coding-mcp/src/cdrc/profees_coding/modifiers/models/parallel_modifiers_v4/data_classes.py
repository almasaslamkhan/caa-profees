from cdrc.common.utils import StringEnum


# ------------------------ defining the dictionary keys ---------------------------
class CONFIG(StringEnum):
    PROMPTS = "prompts"
    PROMPT_VERSIONS = "prompt_versions"
    PROMPT_VERSION = "prompt_version"
    PROMPT_VERSION_GY = "prompt_version_gy"
    PROMPT_VERSION_95 = "prompt_version_95"
    PROMPT_VERSION_P1_P6 = "prompt_version_p1_p6"
    PROMPT_VERSION_50_LT_RT = "prompt_version_50_lt_rt"
    MODEL_NAME = "model_name"
    MODEL_KWARGS = "model_kwargs"
    NUM_REPEATS = "num_repeats"


class PROMPTS(StringEnum):
    PARALLEL_MODIFIERS = "parallel_modifiers"
