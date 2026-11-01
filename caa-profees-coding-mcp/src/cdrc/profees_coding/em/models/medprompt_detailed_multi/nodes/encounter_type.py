import random
from argparse import Namespace
from typing import Dict, TypedDict

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.em.data.schema import (
    EncounterType,
    EncounterTypeDetails,
    EncounterTypesDetails,
    EncounterTypesDetailsAmbulatory,
    EncounterTypesDetailsED,
    EncounterTypesDetailsInpatient,
    PatientType,
)
from cdrc.profees_coding.em.models.utils import get_chain

from ..data_classes import CONFIG, PROMPTS


def _get_model_kwargs_and_possibly_change_seed(config: Dict) -> Dict:
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))
    if num_repeats > 1:
        model_kwargs["seed"] = random.randint(0, 1000000)
    return model_kwargs


# ------------------------ defining the graph state ---------------------------
class EncounterTypeInputGraphState(TypedDict):
    text: str
    reason_for_visit: str
    patient_type: PatientType
    encounter_metadata: dict


class EncounterTypeOutputGraphState(TypedDict):
    encounter_type: EncounterTypesDetails


class EncounterTypeGraphState(EncounterTypeInputGraphState, EncounterTypeOutputGraphState):
    pass


def _extract_encounter_metadata_type(state: EncounterTypeInputGraphState) -> str:
    metadata = state.get("encounter_metadata")
    if metadata is None:
        return ""
    if isinstance(metadata, dict):
        return str(metadata.get("type", "")).strip().lower()
    return str(getattr(metadata, "type", "") or "").strip().lower()


# ------------------------ defining the graph nodes ---------------------------
@track_tokens("model_usage")
def encounter_type_node(state: EncounterTypeInputGraphState, config: RunnableConfig) -> EncounterTypeOutputGraphState:
    site = str(config["configurable"].get("site", "")).strip().lower()
    if site == "ed":
        metadata_type = _extract_encounter_metadata_type(state)
        encounter_type = EncounterType.EMERGENCY_DEPARTMENT
        if "other emergency" in metadata_type:
            encounter_type = EncounterType.EMERGENCY_OTHER_SERVICES
        return {
            "encounter_type": [
                EncounterTypeDetails(
                    encounter_type=encounter_type,
                    justification="Site is ED; encounter type fixed by workflow.",
                )
            ]
        }

    prompt_version = config["configurable"][CONFIG.ENCOUNTER_TYPE_PROMPT_VERSION]
    prompt_key = config["configurable"].get(CONFIG.ENCOUNTER_PROMPT_KEY, PROMPTS.ENCOUNTER)

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)
    parser_model = EncounterTypesDetails
    if prompt_key == PROMPTS.ENCOUNTER_ED:
        parser_model = EncounterTypesDetailsED
    elif prompt_key == PROMPTS.ENCOUNTER_INPATIENT:
        parser_model = EncounterTypesDetailsInpatient
    elif prompt_key == PROMPTS.ENCOUNTER:
        parser_model = EncounterTypesDetailsAmbulatory

    prompt_bank = config["configurable"][CONFIG.PROMPTS][prompt_key]
    if prompt_version not in prompt_bank:
        prompt_version = "v1" if "v1" in prompt_bank else next(iter(prompt_bank))

    parser = PydanticOutputParserWithRetries(pydantic_object=parser_model, correction_model=model, num_retries=3)
    chain = model | parser
    system_prompt = SystemMessagePromptTemplate.from_template(prompt_bank[prompt_version]["system"], template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(prompt_bank[prompt_version]["user"], template_format="jinja2")
    messages = [
        system_prompt.format(),
        user_prompt.format(soap_note=state["text"], patient_type=state.get("patient_type", ""), format_instructions=parser.get_format_instructions()),
    ]

    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        # o1-preview and mini do not support system messages
        messages[1].content = messages[0].content + "\n\n" + messages[1].content
        messages.pop(0)

    response = chain.invoke(messages)

    return {
        "encounter_type": [EncounterTypeDetails(encounter_type=x.encounter_type, justification=x.justification) for x in response.encounter_types]
    }


# ------------------------ defining the graph builder ---------------------------
def build_encounter_type_subgraph(args: Namespace):
    workflow = StateGraph(EncounterTypeGraphState, input=EncounterTypeInputGraphState, output=EncounterTypeOutputGraphState)
    workflow.add_node("classifier", encounter_type_node)

    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", END)

    return workflow.compile()
