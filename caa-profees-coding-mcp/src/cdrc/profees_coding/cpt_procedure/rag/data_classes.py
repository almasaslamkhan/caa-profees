from cdrc.common.utils import StringEnum


# ------------------------ defining the dictionary keys ---------------------------
class CONFIG(StringEnum):
    EMBEDDING_MODEL = "embedding_model"
    VDB_INDEX_PATH = "vdb_index_path"
    VDB_KWARGS = "vector_store"
    MODEL_NAME = "model_name"
    MODEL_KWARGS = "model_kwargs"
    NUM_FEW_SHOT_EXAMPLES = "num_few_shot_examples"
    QUERY = "query"
    QUERY_SELECT = "select"
    RETRIEVAL = "retrieval"
    RETRIEVAL_FILTER = "filter"
    RERANK = "rerank"
    RERANK_PROMPT_VERSION = "prompt_version"
    SEARCH_KWARGS = "search_kwargs"


class METADATA_KEYS(StringEnum):
    DESCRIPTION = "description"
    CATALOG_TYPE = "catalog_type"
