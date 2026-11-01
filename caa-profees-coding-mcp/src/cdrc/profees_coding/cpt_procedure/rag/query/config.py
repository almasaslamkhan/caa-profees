from typing import Literal, Optional

from pydantic import BaseModel


class BaseStageQuerySelectConfig(BaseModel):
    pass


class BaseQuerySelectConfig(BaseModel):
    """
    Later, we can put some COMMON query part selection config variables here.
    """

    pass


class StageQuerySelectConfig(BaseStageQuerySelectConfig):
    mode: Literal["all", "assessment_and_plan_only"] = "all"


class QuerySelectConfig(BaseQuerySelectConfig):
    retrieval: StageQuerySelectConfig
    rerank: Optional[StageQuerySelectConfig] = None
