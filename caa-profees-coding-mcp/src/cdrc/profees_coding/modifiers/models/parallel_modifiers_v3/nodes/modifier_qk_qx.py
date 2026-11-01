import json
import os
from argparse import Namespace
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, TypedDict, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from cdrc.common.data.schema import AnaesthesiaConcurrentStaffInfo, AnaesthesiaRecordInfo, AnaesthesiaStaffInfo, ProviderInfo
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.profees_coding.modifiers.data.schema import EMItem, Modifier, Order, Problem, Procedure
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v3.data_classes import CONFIG
from cdrc.profees_coding.modifiers.models.utils import get_cpt_codes

QK_QX_MODIFIERS: tuple[str, ...] = ("QK", "QX")


class ModifierJustification(BaseModel):
    code: str = Field(..., description="Modifier code (QK or QX).")
    evidence: Optional[str] = Field(None, description="Direct quote or structured field reference supporting the decision.")
    justification: Optional[str] = Field(None, description="Brief reasoning explaining why the modifier is or is not appropriate.")
    assigned: bool = Field(..., description="True if the modifier should be applied to this CPT/HCPCS code.")


class CodeAndModifier(BaseModel):
    cpt_code: str = Field(..., description="CPT or HCPCS code evaluated for modifiers QK/QX.")
    modifiers: List[ModifierJustification] = Field(default_factory=list, description="List of modifier decisions for this CPT code.")


class ModifierOutput(BaseModel):
    code_and_modifiers: List[CodeAndModifier] = Field(default_factory=list, description="Assigned modifiers by CPT/HCPCS code.")


class ModifierGraphInput(TypedDict):
    example_id: Optional[Union[int, str]]
    payer_name: Optional[List[str]]
    text: str
    patient_sex: Optional[str]
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    provider_info: Optional[List[ProviderInfo]]
    anaesthesia_staff_info: Optional[List[AnaesthesiaStaffInfo]]
    anaesthesia_concurrent_staff_info: Optional[List[AnaesthesiaConcurrentStaffInfo]]
    anaesthesia_record_info: Optional[List[AnaesthesiaRecordInfo]]
    all_patient_problems: Optional[List[Problem]]
    procedure: List[Procedure]
    em: List[EMItem]


class ModifierGraphOutput(TypedDict):
    procedure: List[Procedure]
    em: List[EMItem]
    unassigned_modifiers: List[ModifierJustification]


class ModifierGraphState(ModifierGraphInput, ModifierGraphOutput):
    output_qk_qx: Optional[ModifierOutput]


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    return config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})


def _dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        return vars(value)
    return value or {}


def _normalize_code(raw: Any) -> str:
    return str(raw or "").strip().upper()


def _upsert_modifier(mods: List[Any], mj: ModifierJustification) -> List[Any]:
    chosen_code = _normalize_code(getattr(mj, "code", None) or _dump(mj).get("code"))
    if chosen_code not in QK_QX_MODIFIERS:
        return list(mods or [])

    justification = (getattr(mj, "justification", None) or _dump(mj).get("justification") or "").strip() or None
    updated = list(mods or [])

    for idx, mod in enumerate(updated):
        mod_data = _dump(mod)
        mod_code = _normalize_code(mod_data.get("code") or getattr(mod, "code", None))
        if mod_code != chosen_code:
            continue
        existing_justification = (mod_data.get("justification") or "").strip()
        mod_data["code"] = chosen_code
        mod_data["justification"] = existing_justification or justification
        updated[idx] = Modifier(**mod_data)
        return updated

    updated.append(Modifier(code=chosen_code, justification=justification))
    return updated


@track_tokens("model_usage")
def modifier_qk_qx_decision_node(state: ModifierGraphState, config: RunnableConfig) -> Dict[str, ModifierOutput]:
    cpt_codes = [_normalize_code(code) for code in (get_cpt_codes(state.get("em")) + get_cpt_codes(state.get("procedure"))) if code]
    eligible_cpt_codes: List[str] = []
    seen: Set[str] = set()
    for code in cpt_codes:
        if code in seen:
            continue
        seen.add(code)
        eligible_cpt_codes.append(code)

    if not eligible_cpt_codes:
        # print("No eligible codes for modifiers QK/QX")
        return {"output_qk_qx": ModifierOutput()}

    # prompt_version = (config.get("configurable", {}) or {}).get("prompt_version_qk_qx", "v1")
    prompt_version = json.loads(config["configurable"][CONFIG.PROMPT_VERSIONS])["QK-QX"]
    prompt_template = config["configurable"][CONFIG.PROMPTS]["modifier_qk_qx"][prompt_version]

    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)

    parser = PydanticOutputParserWithRetries(
        pydantic_object=ModifierOutput,
        correction_model=model,
        num_retries=2,
    )
    prompt = ChatPromptTemplate.from_template(prompt_template, template_format="jinja2").partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser

    chain_inputs = {
        "soap_note": state.get("text"),
        "orders": state.get("orders_display"),
        "cpt_code": ",".join(eligible_cpt_codes),
        "provider_info": state.get("provider_info"),
        "anaesthesia_staff_info": state.get("anaesthesia_staff_info"),
        "anaesthesia_concurrent_staff_info": state.get("anaesthesia_concurrent_staff_info"),
        "anaesthesia_record_info": state.get("anaesthesia_record_info"),
    }

    response = chain.invoke(chain_inputs)
    return {"output_qk_qx": response}


def validate_and_apply_qk_qx(
    state: ModifierGraphState,
    config: RunnableConfig,
    *,
    allowed_modifiers: Set[str],
) -> ModifierGraphOutput:
    del config

    allowed = {str(x).strip().upper() for x in (allowed_modifiers or set())}
    modifier_raw = state.get("output_qk_qx")
    modifier_output: ModifierOutput = modifier_raw if isinstance(modifier_raw, ModifierOutput) else ModifierOutput(**(modifier_raw or {}))

    decisions_by_cpt: Dict[str, List[ModifierJustification]] = defaultdict(list)
    unassigned: List[ModifierJustification] = []

    for cam in modifier_output.code_and_modifiers or []:
        cpt = _normalize_code(cam.cpt_code)
        for mj in cam.modifiers or []:
            mod_code = _normalize_code(mj.code)
            if mod_code not in allowed:
                continue
            if mj.assigned:
                decisions_by_cpt[cpt].append(mj)
                continue
            mj_data = _dump(mj).copy()
            justification = (mj_data.get("justification") or "").strip()
            mj_data["justification"] = f"{cpt}: {justification}" if justification else f"{cpt}:"
            unassigned.append(ModifierJustification(**mj_data))

    new_em: List[EMItem] = []
    for em in state.get("em") or []:
        d = (_dump(em) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        for mj in decisions_by_cpt.get(cpt, []):
            mods = _upsert_modifier(mods, mj)
        d["modifiers"] = mods
        new_em.append(EMItem(**d))

    new_proc: List[Procedure] = []
    for p in state.get("procedure") or []:
        d = (_dump(p) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        for mj in decisions_by_cpt.get(cpt, []):
            mods = _upsert_modifier(mods, mj)
        d["modifiers"] = mods
        if "linked_diagnosis" in d:
            d["linked_diagnosis"] = []
        new_proc.append(Procedure(**d))

    return {
        "em": new_em,
        "procedure": new_proc,
        "unassigned_modifiers": _merge_unassigned_by_code((state.get("unassigned_modifiers") or []) + unassigned),
    }


def _merge_unassigned_by_code(unassigned: List[ModifierJustification]) -> List[ModifierJustification]:
    if not unassigned:
        return []
    grouped: Dict[str, List[ModifierJustification]] = defaultdict(list)
    for item in unassigned:
        mod_code = _normalize_code(_dump(item).get("code") or getattr(item, "code", None))
        if mod_code:
            grouped[mod_code].append(item)

    merged: List[ModifierJustification] = []
    for mod_code, items in grouped.items():
        rep = items[0]
        seen_evidence: set[str] = set()
        seen_justification: set[str] = set()
        evidence_parts: List[str] = []
        justification_parts: List[str] = []
        for item in items:
            evidence = _dump(item).get("evidence") or getattr(item, "evidence", None) or ""
            justification = _dump(item).get("justification") or getattr(item, "justification", None) or ""
            if evidence and evidence not in seen_evidence:
                seen_evidence.add(evidence)
                evidence_parts.append(evidence)
            if justification and justification not in seen_justification:
                seen_justification.add(justification)
                justification_parts.append(justification)

        rep_data = _dump(rep).copy()
        rep_data["code"] = mod_code
        rep_data["assigned"] = False
        if evidence_parts:
            rep_data["evidence"] = "\n".join(evidence_parts)
        if justification_parts:
            rep_data["justification"] = "\n".join(justification_parts)
        merged.append(ModifierJustification(**rep_data))

    return merged


def build_modifier_qk_qx_subgraph(args: Namespace, modifier_names: List[str]):
    requested = {_normalize_code(name) for name in (modifier_names or [])}
    allowed_modifiers = requested.intersection(QK_QX_MODIFIERS)
    if not allowed_modifiers:
        raise ValueError(f"modifier_qk_qx subgraph only supports QK/QX, got: {modifier_names}")

    workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    workflow.add_node("modifier_qk_qx_decision", modifier_qk_qx_decision_node)
    workflow.add_node(
        "modifier_qk_qx_validate",
        lambda state, config: validate_and_apply_qk_qx(state, config, allowed_modifiers=allowed_modifiers),
    )

    workflow.add_edge(START, "modifier_qk_qx_decision")
    workflow.add_edge("modifier_qk_qx_decision", "modifier_qk_qx_validate")
    workflow.add_edge("modifier_qk_qx_validate", END)

    return workflow.compile()
