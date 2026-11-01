import random
from argparse import Namespace
from typing import Dict, Literal, TypedDict, Union

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.em.data.schema import PatientType

from ..data_classes import CONFIG, PROMPTS


def _get_model_kwargs_and_possibly_change_seed(config: Dict) -> Dict:
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))
    if num_repeats > 1:
        model_kwargs["seed"] = random.randint(0, 1000000)
    return model_kwargs


class PatientTypePrediction(BaseModel):
    decision: Literal["New", "Established", "Unknown"] = Field(description="The patient type decision from note and appointment type.")
    confidence: Literal["High", "Medium", "Low"] = Field(description="Decision confidence.")
    final_patient_type: Literal["New", "Established", "Unknown"] = Field(description="The final patient type.")
    note_pt: Literal["New", "Established", "Unknown"] = Field(description="Patient type inferred from the note only.")
    evidence: Union[list[str], str] = Field(description="Supporting evidence from the note.")
    evidence_certainity: Literal["Certain", "Strong", "Uncertain"] = Field(description="Evidence certainty level.")
    justification: str = Field(description="Brief source-based justification, such as SOAP note, appointment type, or visit history.")


class PatientTypeInputGraphState(TypedDict):
    text: str
    patient_type: PatientType
    appointment_type: str
    service_date: str


class PatientTypeOutputGraphState(TypedDict):
    patient_type: PatientType
    patient_type_prediction: dict
    patient_type_justification: str


class PatientTypeGraphState(PatientTypeInputGraphState, PatientTypeOutputGraphState):
    pass


def _normalize_patient_type_label(value: object) -> Literal["New", "Established", "Unknown"]:
    raw_value = getattr(value, "value", value)
    normalized = str(raw_value or "").strip().lower()
    if normalized.startswith("new"):
        return "New"
    if normalized.startswith("est"):
        return "Established"
    return "Unknown"


def _normalize_patient_type(decision: str, appointment_type: str) -> PatientType:
    normalized = _normalize_patient_type_label(decision)
    if normalized == "New":
        return PatientType.NEW
    if normalized == "Established":
        return PatientType.ESTABLISHED
    normalized_appointment_type = str(appointment_type or "").strip().lower()
    if "new patient" in normalized_appointment_type or "nw1" in normalized_appointment_type:
        return PatientType.NEW
    return PatientType.ESTABLISHED


def _is_new_appointment_type(appointment_type: str) -> bool:
    normalized_appointment_type = str(appointment_type or "").strip().lower()
    return "new patient" in normalized_appointment_type or "nw1" in normalized_appointment_type


def _apply_final_patient_type_rule(
    prediction: PatientTypePrediction,
    dt_patient_type: str,
) -> PatientTypePrediction:
    provisional_decision = _normalize_patient_type_label(prediction.decision)
    normalized_dt_patient_type = _normalize_patient_type_label(dt_patient_type)
    confidence = str(prediction.confidence or "").strip().lower()
    if (provisional_decision == "Unknown" or confidence == "low") and normalized_dt_patient_type != "Unknown":
        final_patient_type = normalized_dt_patient_type
    else:
        final_patient_type = provisional_decision
    return prediction.model_copy(update={"decision": provisional_decision, "final_patient_type": final_patient_type})


def _friendly_patient_type_justification(
    prediction: PatientTypePrediction,
    appointment_type: str,
    dt_patient_type: str,
) -> str:
    final_patient_type = _normalize_patient_type_label(prediction.final_patient_type)
    note_patient_type = _normalize_patient_type_label(prediction.note_pt)
    decision = _normalize_patient_type_label(prediction.decision)
    dt_patient_type = _normalize_patient_type_label(dt_patient_type)
    confidence = str(prediction.confidence or "").strip().lower()

    if final_patient_type == "Unknown":
        return "Patient type could not be determined."

    article = "a" if final_patient_type == "New" else "an"
    if note_patient_type == final_patient_type:
        return f"SOAP note supports {article} {final_patient_type.lower()} patient visit."

    if final_patient_type == "New" and _is_new_appointment_type(appointment_type):
        return "Appointment type indicates a new patient visit."

    if (decision == "Unknown" or confidence == "low" or decision != final_patient_type) and dt_patient_type == final_patient_type:
        return f"Visit history supports {article} {final_patient_type.lower()} patient visit."

    if final_patient_type == "Established":
        return "Appointment type supports an established patient visit."

    if decision == final_patient_type:
        return "Appointment type indicates a new patient visit."

    return "Visit history supports a new patient visit."


def _prediction_to_dict(
    prediction: PatientTypePrediction,
    appointment_type: str = "",
    dt_patient_type: str = "",
) -> dict:
    payload = prediction.model_dump()
    if isinstance(payload.get("evidence"), list):
        payload["evidence"] = [str(item).strip() for item in payload["evidence"] if str(item).strip()]
    else:
        payload["evidence"] = str(payload.get("evidence") or "").strip()
    payload["justification"] = _friendly_patient_type_justification(prediction, appointment_type, dt_patient_type)
    return payload


def _prediction_justification(
    prediction: PatientTypePrediction,
    appointment_type: str = "",
    dt_patient_type: str = "",
) -> str:
    return _friendly_patient_type_justification(prediction, appointment_type, dt_patient_type)


def _predict_without_note(appointment_type: str, dt_patient_type: str) -> PatientTypePrediction:
    if _is_new_appointment_type(appointment_type):
        prediction = PatientTypePrediction(
            decision="New",
            confidence="Medium",
            final_patient_type="New",
            note_pt="Unknown",
            evidence="No note text available.",
            evidence_certainity="Uncertain",
            justification="Appointment type indicates a new patient visit.",
        )
        return _apply_final_patient_type_rule(prediction, dt_patient_type)
    prediction = PatientTypePrediction(
        decision="Established",
        confidence="Medium",
        final_patient_type="Established",
        note_pt="Unknown",
        evidence="No note text available.",
        evidence_certainity="Uncertain",
        justification="Appointment type supports an established patient visit.",
    )
    return _apply_final_patient_type_rule(prediction, dt_patient_type)


@track_tokens("model_usage")
def patient_type_node(state: PatientTypeInputGraphState, config: RunnableConfig) -> PatientTypeOutputGraphState:
    note_text = str(state.get("text", "") or "").strip()
    dt_patient_type = _normalize_patient_type_label(state.get("patient_type", ""))
    appointment_type = str(state.get("appointment_type", "") or "").strip()
    service_date = str(state.get("service_date", "") or "").strip()

    if not note_text:
        prediction = _predict_without_note(appointment_type, dt_patient_type)
        return {
            "patient_type": _normalize_patient_type(prediction.final_patient_type, appointment_type),
            "patient_type_prediction": _prediction_to_dict(prediction, appointment_type, dt_patient_type),
            "patient_type_justification": _prediction_justification(prediction, appointment_type, dt_patient_type),
        }

    prompt_version = config["configurable"][CONFIG.PATIENT_TYPE_PROMPT_VERSION]
    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)
    parser = PydanticOutputParserWithRetries(pydantic_object=PatientTypePrediction, correction_model=model, num_retries=3)
    chain = model | parser
    system_prompt = SystemMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.PATIENT_TYPE][prompt_version]["system"], template_format="jinja2"
    )
    user_prompt = HumanMessagePromptTemplate.from_template(
        config["configurable"][CONFIG.PROMPTS][PROMPTS.PATIENT_TYPE][prompt_version]["user"], template_format="jinja2"
    )
    messages = [
        system_prompt.format(),
        user_prompt.format(
            note=note_text,
            appt_type=appointment_type,
            service_date=service_date,
            dt_patient_type=dt_patient_type,
            format_instructions=parser.get_format_instructions(),
        ),
    ]

    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        messages[1].content = messages[0].content + "\n\n" + messages[1].content
        messages.pop(0)

    prediction = _apply_final_patient_type_rule(chain.invoke(messages), dt_patient_type)
    return {
        "patient_type": _normalize_patient_type(prediction.final_patient_type, appointment_type),
        "patient_type_prediction": _prediction_to_dict(prediction, appointment_type, dt_patient_type),
        "patient_type_justification": _prediction_justification(prediction, appointment_type, dt_patient_type),
    }


def build_patient_type_subgraph(args: Namespace):
    workflow = StateGraph(PatientTypeGraphState, input=PatientTypeInputGraphState, output=PatientTypeOutputGraphState)
    workflow.add_node("classifier", patient_type_node)

    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", END)

    return workflow.compile()
