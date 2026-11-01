from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class BaseRerankerConfig(BaseModel):
    method: Literal["llm_listwise", "llm_listwise_with_reason"]
    top_n: int = Field(gt=0)


class LLMRerankerConfig(BaseRerankerConfig):
    model_name: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = Field(default=None, validate_default=True)
    prompt_path: str
    prompt_version: str

    @field_validator("model_kwargs", mode="after")
    @classmethod
    def ensure_non_empty(cls, v):
        return v or {}
