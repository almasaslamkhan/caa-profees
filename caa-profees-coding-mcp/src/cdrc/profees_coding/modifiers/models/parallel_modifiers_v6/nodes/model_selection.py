from typing import Any

from cdrc.common.llm_utils import LLM, get_llm

DEFAULT_CORRECTION_MODEL_BY_MODEL = {
    LLM.GPT_54_2026_03_17_mini_ORACLE_DEV_SDS.value: LLM.GPT_54_2026_03_17_nano_ORACLE_DEV_SDS.value,
}


def get_correction_model_name(model_name: str) -> str:
    return DEFAULT_CORRECTION_MODEL_BY_MODEL.get(model_name, model_name)


def get_model_and_correction_model(model_name: str, model_kwargs: dict[str, Any]):
    llms = get_llm()
    model = llms[model_name](**model_kwargs)
    correction_model_name = get_correction_model_name(model_name)
    correction_model = llms[correction_model_name](**model_kwargs) if correction_model_name != model_name else model
    return model, correction_model
