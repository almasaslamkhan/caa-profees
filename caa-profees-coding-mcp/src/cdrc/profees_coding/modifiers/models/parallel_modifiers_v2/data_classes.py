from cdrc.common.utils import StringEnum


# ------------------------ defining the dictionary keys ---------------------------
class CONFIG(StringEnum):
    PROMPTS = "prompts"
    PROMPT_VERSION = "prompt_version"
    MODEL_NAME = "model_name"
    MODEL_KWARGS = "model_kwargs"
    NUM_REPEATS = "num_repeats"


class PROMPTS(StringEnum):
    PARALLEL_MODIFIERS = "parallel_modifiers"
