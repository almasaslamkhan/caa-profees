from argparse import Namespace
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, TypedDict, Union

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from cdrc.profees_coding.modifiers.data.schema import EMItem, Modifier, Order, Problem, Procedure

CLIA_WAIVED_CERT_TYPES = {
    "WAIVED",
    "CLIA WAIVED",
    "CERTIFICATE OF WAIVER",
    "COW",
}

CLIA_NON_WAIVED_CERT_TYPES = {
    "PPM",
    "PROVIDER PERFORMED MICROSCOPY",
    "MODERATE",
    "HIGH",
    "MODERATE/HIGH",
    "COMPLEXITY",
}

MEDICARE_PAYER_NAMES = {
    "MEDICARE",
    "MEDICARE FFS",
    "CMS",
    "MEDICARE PART B",
    "MEDICARE PART A",
    "BCBS",
    "UHC",
    "UMR",
    "HEALTHPARTNERS",
    "WPS",
    "MVA",
    "AETNA",
}

NOTE_KEYWORDS = (
    "clia waived",
    "clia-waived",
    "point of care",
    "point-of-care",
    "poc test",
    "poct",
    "rapid test",
)


class AssignedModifier(BaseModel):
    code: str = Field("QW", description="Assigned modifier code. Always 'QW' for this task.")
    justification: Optional[str] = Field(None, description="Brief reasoning explaining why QW is appropriate.")


class CodeAndModifier(BaseModel):
    cpt_code: str = Field(..., description="CPT or HCPCS code evaluated for Modifier QW.")
    modifiers: List[AssignedModifier] = Field(default_factory=list, description="List of modifier decisions for this CPT code (QW only).")


class ModifierOutput(BaseModel):
    code_and_modifiers: List[CodeAndModifier] = Field(default_factory=list, description="Modifier QW decisions grouped by CPT code.")


class ModifierGraphInput(TypedDict):
    example_id: Optional[Union[int, str]]
    payer_name: Optional[List[str]]
    text: str
    patient_sex: Optional[str]
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    all_patient_problems: Optional[List[Problem]]
    procedure: List[Procedure]
    em: List[EMItem]


class ModifierGraphOutput(TypedDict):
    procedure: List[Procedure]
    em: List[EMItem]


class ModifierGraphState(ModifierGraphInput, ModifierGraphOutput):
    output_qw: Optional[ModifierOutput]


def _dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        return vars(value)
    return value or {}


def _normalize_code(raw: Any) -> str:
    return str(raw or "").strip().upper()


def _normalize_text(raw: Any) -> str:
    return str(raw or "").strip().upper()


def _extract_payer_names(state: ModifierGraphState, config: RunnableConfig) -> List[str]:
    state_value = state.get("payer_name")
    if isinstance(state_value, list):
        names = [str(x).strip() for x in state_value if str(x).strip()]
        if names:
            return names
    if isinstance(state_value, str):
        raw = state_value.strip()
        if raw:
            return [raw]

    cfg_value = (config.get("configurable", {}) or {}).get("payer")
    if isinstance(cfg_value, list):
        return [str(x).strip() for x in cfg_value if str(x).strip()]
    if isinstance(cfg_value, str) and cfg_value.strip():
        return [cfg_value.strip()]
    return []


def _payer_follows_cms_rules(state: ModifierGraphState, config: RunnableConfig) -> bool:
    configurable = config.get("configurable", {}) or {}
    if configurable.get("qw_apply_for_all_payers", False):
        return True
    if configurable.get("qw_apply_for_non_medicare", False):
        return True
    return any(_normalize_code(payer) in MEDICARE_PAYER_NAMES for payer in _extract_payer_names(state, config))


def _extract_clia_value(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (list, tuple, set)):
        for item in raw:
            val = _extract_clia_value(item)
            if val:
                return val
        return None
    if isinstance(raw, dict):
        for key in (
            "clia_certificate_type",
            "facility_clia_certificate_type",
            "performing_lab_clia_type",
            "lab_clia_certificate_type",
            "clia_type",
            "certificate_type",
        ):
            val = raw.get(key)
            if val:
                return _extract_clia_value(val)
        return None
    return str(raw).strip() or None


def _extract_clia_certificate_type(state: ModifierGraphState, config: RunnableConfig, cpt_code: str) -> Optional[str]:
    for item in state.get("procedure") or []:
        d = _dump(item) or {}
        code = _normalize_code(d.get("code") or d.get("cpt_code"))
        if code != _normalize_code(cpt_code):
            continue
        val = _extract_clia_value(d)
        if val:
            return val

    for key in (
        "clia_certificate_type",
        "facility_clia_certificate_type",
        "performing_lab_clia_type",
        "lab_clia_certificate_type",
        "clia_type",
    ):
        val = state.get(key)
        if val:
            return _extract_clia_value(val)

    cfg = config.get("configurable", {}) or {}
    for key in (
        "clia_certificate_type",
        "facility_clia_certificate_type",
        "performing_lab_clia_type",
        "lab_clia_certificate_type",
        "clia_type",
    ):
        val = cfg.get(key)
        if val:
            return _extract_clia_value(val)

    return None


def _is_clia_waived(cert_type: Optional[str]) -> bool:
    if not cert_type:
        return False
    normalized = _normalize_text(cert_type)
    if normalized in CLIA_WAIVED_CERT_TYPES:
        return True
    if normalized in CLIA_NON_WAIVED_CERT_TYPES:
        return False
    return "WAIV" in normalized


def _note_mentions_qw(text: str) -> Optional[str]:
    if not text:
        return None
    lower = text.lower()
    for kw in NOTE_KEYWORDS:
        if kw in lower:
            return f"Note mentions '{kw}'."
    return None


def _upsert_qw_modifier(mods: List[Any], mj: AssignedModifier) -> List[Any]:
    justification = (getattr(mj, "justification", None) or _dump(mj).get("justification") or "").strip() or None
    updated = list(mods or [])

    for idx, mod in enumerate(updated):
        mod_data = _dump(mod)
        if _normalize_code(mod_data.get("code") or getattr(mod, "code", None)) != "QW":
            continue
        existing_justification = (mod_data.get("justification") or "").strip()
        if existing_justification:
            return updated
        mod_data["code"] = "QW"
        mod_data["justification"] = justification
        updated[idx] = Modifier(**mod_data)
        return updated

    updated.append(Modifier(code="QW", justification=justification))
    return updated


def _iter_all_cpt_codes(state: ModifierGraphState) -> List[str]:
    codes: List[str] = []
    for item in (state.get("em") or []) + (state.get("procedure") or []):
        d = _dump(item) or {}
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        if cpt:
            codes.append(cpt)
    return codes


def _build_decision(
    cpt: str,
    cert_type: Optional[str],
    payer_follows_cms: bool,
    note_hint: Optional[str],
) -> Optional[AssignedModifier]:
    if not cert_type:
        return None

    if not _is_clia_waived(cert_type):
        return None

    if not payer_follows_cms:
        return None

    return AssignedModifier(
        code="QW",
        justification="Lab certificate is waived and payer follows CMS lab rules.",
    )


def modifier_qw_decision_node(state: ModifierGraphState, config: RunnableConfig) -> Dict[str, ModifierOutput]:
    note_hint = _note_mentions_qw(state.get("text") or "")
    payer_follows_cms = _payer_follows_cms_rules(state, config)

    seen: Set[str] = set()
    code_and_modifiers: List[CodeAndModifier] = []
    for cpt in _iter_all_cpt_codes(state):
        if cpt in seen:
            continue
        seen.add(cpt)
        cert_type = _extract_clia_certificate_type(state, config, cpt)
        decision = _build_decision(cpt, cert_type, payer_follows_cms, note_hint)
        # print(cpt, cert_type, decision)
        code_and_modifiers.append(CodeAndModifier(cpt_code=cpt, modifiers=[decision]))

    return {"output_qw": ModifierOutput(code_and_modifiers=code_and_modifiers)}


def validate_and_apply_qw(state: ModifierGraphState, config: RunnableConfig) -> ModifierGraphOutput:
    del config

    modifier_raw = state.get("output_qw")
    modifier_output: ModifierOutput = modifier_raw if isinstance(modifier_raw, ModifierOutput) else ModifierOutput(**(modifier_raw or {}))

    decision_by_cpt: Dict[str, AssignedModifier] = {}

    for cam in modifier_output.code_and_modifiers or []:
        cpt = _normalize_code(cam.cpt_code)
        for mj in cam.modifiers or []:
            if _normalize_code(mj.code) != "QW":
                continue
            decision_by_cpt[cpt] = mj
            break

    new_em: List[EMItem] = []
    for em in state.get("em") or []:
        d = (_dump(em) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        mj = decision_by_cpt.get(cpt)
        if mj:
            mods = _upsert_qw_modifier(mods, mj)
        d["modifiers"] = mods
        new_em.append(EMItem(**d))

    new_proc: List[Procedure] = []
    for p in state.get("procedure") or []:
        d = (_dump(p) or {}).copy()
        cpt = _normalize_code(d.get("code") or d.get("cpt_code"))
        mods = list(d.get("modifiers") or [])
        mj = decision_by_cpt.get(cpt)
        if mj:
            mods = _upsert_qw_modifier(mods, mj)
        d["modifiers"] = mods
        if "linked_diagnosis" in d:
            d["linked_diagnosis"] = []
        new_proc.append(Procedure(**d))

    return {
        "em": new_em,
        "procedure": new_proc,
    }


def build_modifier_qw_subgraph(args: Namespace, modifier_names: List[str]):
    normalized = {_normalize_code(name) for name in (modifier_names or [])}
    if "QW" not in normalized:
        raise ValueError(f"modifier_qw subgraph only supports QW, got: {modifier_names}")

    workflow = StateGraph(ModifierGraphState, input=ModifierGraphInput, output=ModifierGraphOutput)
    workflow.add_node("modifier_qw_decision", modifier_qw_decision_node)
    workflow.add_node("modifier_qw_validate", validate_and_apply_qw)

    workflow.add_edge(START, "modifier_qw_decision")
    workflow.add_edge("modifier_qw_decision", "modifier_qw_validate")
    workflow.add_edge("modifier_qw_validate", END)

    try:
        num_repeats = int(getattr(args, "num_repeats", 1))
    except Exception:
        num_repeats = 1

    if num_repeats <= 1:
        return workflow.compile()

    # QW is rules-based and deterministic; repeat runs are equivalent, so return a single compiled graph.
    return workflow.compile()
