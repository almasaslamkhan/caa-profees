from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseVectorStoreConfig(BaseModel):
    type: Literal["faiss", "bm25"]
    path: Optional[str] = None
    embedding_model: Optional[str] = None


class BaseRetrieverConfig(BaseModel):
    method: Literal["semantic", "lexical", "hybrid"]
    vector_store: Optional[BaseVectorStoreConfig] = None
    search_kwargs: Optional[Dict[str, Any]] = Field(default=None, validate_default=True)

    @field_validator("search_kwargs", mode="after")
    @classmethod
    def ensure_non_empty(cls, v):
        return v or {}


class SemanticRetrieverConfig(BaseRetrieverConfig):
    method: Literal["semantic"]
    vector_store: BaseVectorStoreConfig
    search_type: Literal["similarity_score_threshold", "similarity", "mmr"]


class LexicalRetrieverConfig(BaseRetrieverConfig):
    method: Literal["lexical"]
    vector_store: BaseVectorStoreConfig


SubRetrieverDiscriminater = Annotated[Union[SemanticRetrieverConfig, LexicalRetrieverConfig], Field(discriminator="method")]


class HybridRetrieverConfig(BaseModel):
    # We force to not let HybridRetrieverConfig accept any search related arguments, e.g. search_kwargs, vector_store, ect.,
    # but rather pass them down to each sub-retriever to avoid unnecessary complexity.
    model_config = ConfigDict(extra="forbid")
    method: Literal["hybrid"]
    weights: List[float]
    sub_retrievers: Dict[str, SubRetrieverDiscriminater]

    @field_validator("weights")
    @classmethod
    def check_weights_sum(cls, v):
        if not abs(sum(v) - 1.0) < 1e-5:
            raise ValueError("Weights in hybrid retriever must sum to 1.0")
        return v
