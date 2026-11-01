from cdrc.common.utils import StringEnum


# ------------------------ defining the dictionary keys ---------------------------
class CONFIG(StringEnum):
    PROMPTS = "prompts"
    PROMPT_VERSION = "prompt_version"
    MODEL_NAME = "model_name"
    MODEL_KWARGS = "model_kwargs"
    EMBEDDING_MODEL = "embedding_model"
    RETRIEVAL = "retrieval"
    QUERY = "query"
    RERANK = "rerank"
    VDB_INDEX_PATH = "vdb_index_path"
    NUM_FEW_SHOT_EXAMPLES = "num_few_shot_examples"
    NUM_REPEATS = "num_repeats"
    OMIT_SOAP_NOTE = "omit_soap_note"
    OMIT_PATIENT_PROBLEMS = "omit_patient_problems"
    ICL_MODE = "icl_mode"


class PROMPTS(StringEnum):
    EXTRACTION = "procedure_extraction"
    VERIFICATION = "verification"
    KNOWLEDGE_ENHANCEMENT = "knowledge_enhancement"
    RETRIEVAL = "retrieval"


class METADATA_KEYS(StringEnum):
    SOAP_NOTE = "soap_note"
    CHAIN_OF_THOUGHT_REASONING = "chain_of_thought_reasoning"
    MDM_COMPLEXITY = "mdm_complexity"
