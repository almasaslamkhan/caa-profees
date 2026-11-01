from cdrc.common.utils import StringEnum


# ------------------------ defining the dictionary keys ---------------------------
class CONFIG(StringEnum):
    PROMPTS = "prompts"
    PROMPT_VERSION = "prompt_version"
    MODEL_NAME = "model_name"
    MODEL_KWARGS = "model_kwargs"
    EMBEDDING_MODEL = "embedding_model"
    VDB_INDEX_PATH = "vdb_index_path"
    NUM_FEW_SHOT_EXAMPLES = "num_few_shot_examples"
    NUM_REPEATS = "num_repeats"
    PREDICT_MDM_COMPLEXITY = "predict_mdm_complexity"
    USE_INTERMEDIATE_COMPLEXITIES = "use_intermediate_complexities"
    PROMPT_VERSION_PC = "prompt_version_pc"
    PROMPT_VERSION_DC = "prompt_version_dc"
    PROMPT_VERSION_RC = "prompt_version_rc"
    OMIT_SOAP_NOTE = "omit_soap_note"
    OMIT_PATIENT_PROBLEMS = "omit_patient_problems"
    ICL_MODE = "icl_mode"
    VDB_TEXT = "vdb_text"  # text, signed_note, justifications
    TIME_STATEMENT_PROMPT_VERSION = "time_statement_prompt_version"
    ENCOUNTER_TYPE_PROMPT_VERSION = "encounter_type_prompt_version"


class PROMPTS(StringEnum):
    SYSTEM = "system"
    PROBLEM_COMPLEXITY = "problem_complexity"
    DATA_COMPLEXITY = "data_complexity"
    RISK_OF_COMPLICATIONS = "risk_of_complications"
    TIME_STATEMENT = "time_statement"
    ENCOUNTER = "encounter_type"


class METADATA_KEYS(StringEnum):
    SOAP_NOTE = "soap_note"
    CHAIN_OF_THOUGHT_REASONING = "chain_of_thought_reasoning"
    MDM_COMPLEXITY = "mdm_complexity"
