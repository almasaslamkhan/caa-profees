from pydantic import BaseModel

from cdrc.common.llm_utils import check_and_reload_env
from cdrc.profees_coding.em.data.schema import ComplexityDetails

check_and_reload_env()


class MDMResult(BaseModel):
    problem_complexity: ComplexityDetails
    data_complexity: ComplexityDetails
    risk_of_complications: ComplexityDetails


def extract_indexable_text(text: str) -> str:
    return text
