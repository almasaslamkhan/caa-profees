from cdrc.common.utils import StringEnum


# ------------------------ defining the dictionary keys ---------------------------
class CONFIG(StringEnum):
    PROMPTS = "prompts"
    PROMPT_VERSION = "prompt_version"
    PROBLEM_EXTRACTION_PROMPT_VERSION = "problem_extraction_prompt_version"
    CODING_PROMPT_VERSION = "coding_prompt_version"
    ENHANCER_PROMPT_VERSION = "z_code_enhancer_prompt_version"
    MODEL_NAME = "model_name"
    MODEL_KWARGS = "model_kwargs"
    EMBEDDING_MODEL = "embedding_model"
    VDB_INDEX_PATH = "vdb_index_path"
    NUM_FEW_SHOT_EXAMPLES = "num_few_shot_examples"
    NUM_REPEATS = "num_repeats"
    OMIT_SOAP_NOTE = "omit_soap_note"
    OMIT_PATIENT_PROBLEMS = "omit_patient_problems"
    ICL_MODE = "icl_mode"
    SEARCH_TOP_K = "search_top_k"


class PROMPTS(StringEnum):
    LLM_MULTI_STAGE = "LLM_MultiStagePrompts"


class METADATA_KEYS(StringEnum):
    SOAP_NOTE = "soap_note"
    CHAIN_OF_THOUGHT_REASONING = "chain_of_thought_reasoning"
    MDM_COMPLEXITY = "mdm_complexity"
