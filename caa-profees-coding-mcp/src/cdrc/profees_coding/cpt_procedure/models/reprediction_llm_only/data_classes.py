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


class PROMPTS(StringEnum):
    EXTRACTION = "procedure_extraction"
    VERIFICATION = "verification"
    KNOWLEDGE_ENHANCEMENT = "knowledge_enhancement"
    REPREDICTION = "re_prediction"


# --------------------------------------------------------------------------------
