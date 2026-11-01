# cdrc/profees_coding/icd10cm/models/workflow/workflow_icd10cm.py
from __future__ import annotations

import json
import random
from argparse import Namespace
from typing import Any, Dict, Final, List, Optional, TypedDict, Union

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

# Tool shims
from cdrc.common.icd10_cm_tools.agent_tools import get_full_data as tool_full
from cdrc.common.icd10_cm_tools.agent_tools import is_billable as tool_is_billable
from cdrc.common.icd10_cm_tools.agent_tools import is_valid_item as tool_is_valid
from cdrc.common.icd10_cm_tools.agent_tools import search as tool_search
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.icd10cm.data.schema import ICD10CM, Order, Problem
from cdrc.profees_coding.icd10cm.models.workflow.data_classes import CONFIG, PROMPTS

# from SPARQLWrapper import JSON, SPARQLWrapper


SKOS_PREF_LABEL: Final[str] = "http://www.w3.org/2004/02/skos/core#prefLabel"
MODEL = "default"
VERSION = "stable"
SKG_SPARQL_ENDPOINT = (
    f"https://ehrc-5108139e91a14c328263fbd97d0ff2e4-dadciad.data.cda.us-ashburn-1.oci.oc-test.com/skg/api/sparql/1.1/graphs/versions/{VERSION}"
)

# ------------------------ Pydantic IO models ------------------------


class ExtractedContext(BaseModel):
    problem: str
    evidence: str


class CombinedExtractQueriesOut(BaseModel):
    contexts: List[ExtractedContext] = Field(default_factory=list)
    queries_by_problem: Dict[str, List[str]] = Field(default_factory=dict)


class DirectPredictionItem(BaseModel):
    rationale: Optional[str] = ""
    description: Optional[str] = ""
    code: str


class DirectPredictionOut(BaseModel):
    predictions: List[DirectPredictionItem] = Field(default_factory=list)


class CandidateForChoice(BaseModel):
    code: str
    description: str
    rationale: Optional[str] = ""
    source: str
    score: float = 0


class PrimaryPick(BaseModel):
    rationale: Optional[str] = ""
    description: Optional[str] = ""
    code: str


class SecondaryPick(BaseModel):
    rationale: Optional[str] = ""
    description: Optional[str] = ""
    code: str


class ChooseCodesOut(BaseModel):
    primary: Optional[PrimaryPick] = None
    secondary: List[SecondaryPick] = Field(default_factory=list)


class RetrievalTrace(BaseModel):
    # 1) Extraction & query planning
    contexts: List[ExtractedContext] = Field(default_factory=list)
    queries_by_problem: Dict[str, List[str]] = Field(default_factory=dict)
    # 2) Retrieval (per-problem code hits). Use CandidateForChoice for homogeneity.
    search_hits: Dict[str, List[CandidateForChoice]] = Field(default_factory=dict)  # problem -> union candidates from multiple queries
    per_query_hits: Dict[str, Dict[str, List[CandidateForChoice]]] = Field(default_factory=dict)  # problem -> query -> candidates
    # 3) Validation/filtering and final selection (also CandidateForChoice)
    filtered_candidates: List[CandidateForChoice] = Field(default_factory=list)
    final_selection: List[CandidateForChoice] = Field(default_factory=list)
    # 4) filtered candidates per problem & query, keeping full CandidateForChoice
    per_query_filtered: Dict[str, Dict[str, List[CandidateForChoice]]] = Field(default_factory=dict)  # problem -> query -> [CandidateForChoice]


# ------------------------ State ------------------------


class InputState(TypedDict):
    example_id: Optional[Union[int, str]]
    text: str
    patient_sex: Optional[str]
    patient_age: Optional[int]
    orders: Optional[List[Order]]
    orders_display: Optional[str]
    all_patient_problems: Optional[List[Problem]]


class OutputState(TypedDict):
    diagnosis: List[ICD10CM]
    clarifications: List[str]
    retrieval_trace: Optional[RetrievalTrace] = None
    # (optional but helpful) keep any future extras instead of dropping silently
    model_config = ConfigDict(extra="allow")


class WorkingState(TypedDict):
    extracted_contexts: List[dict]
    search_queries: Dict[str, List[str]]  # problem -> [queries]

    search_hits: Dict[str, List[CandidateForChoice]]  # query -> candidates
    per_query_hits: Dict[str, Dict[str, List[CandidateForChoice]]]  # problem -> query -> candidates
    direct_candidates: List[CandidateForChoice]
    merged_candidates: List[CandidateForChoice]
    filtered_candidates: List[CandidateForChoice]
    final_selection: List[CandidateForChoice]

    per_query_filtered: Dict[str, Dict[str, List[CandidateForChoice]]]  # problem -> query -> [candidates]


class ICD10CMCodeGraphStates(InputState, WorkingState, OutputState):
    pass


# ------------------------ LLM utils ------------------------


def _get_model_kwargs_and_possibly_change_seed(config: dict) -> dict:
    model_kwargs = _get_model_kwargs_and_possibly_change_seed(config)
    num_repeats = int(config["configurable"].get(CONFIG.NUM_REPEATS, 1))
    if num_repeats > 1:
        model_kwargs["seed"] = random.randint(0, 1000000)
    return model_kwargs


def _get_llm_and_parser(config: dict, pydantic_model: Any):
    model_kwargs = config["configurable"].get(CONFIG.MODEL_KWARGS, {"seed": 42})
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**model_kwargs)
    parser = PydanticOutputParserWithRetries(pydantic_object=pydantic_model, correction_model=model, num_retries=3)
    return model, parser


def _format_messages(system_tmpl: str, user_tmpl: str, parser, **vars_):
    system_prompt = SystemMessagePromptTemplate.from_template(system_tmpl, template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(user_tmpl, template_format="jinja2")
    return [system_prompt.format(), user_prompt.format(format_instructions=parser.get_format_instructions(), **vars_)]


def _maybe_collapse_system_for_o1(messages: List, config: dict):
    if "o1" in config["configurable"][CONFIG.MODEL_NAME]:
        messages[1].content = messages[0].content + "\n\n" + messages[1].content
        messages.pop(0)
    return messages


# ------------------------ Tool helpers ------------------------


# def _tool_valid(code: str) -> bool:
#     """
#     Check if an ICD-10-CM code exists in SKG by verifying it has a skos:prefLabel.

#     Returns:
#         True  - if at least one prefLabel triple exists for the code.
#         False - if no result is found or an error occurs.
#     """
#     query = f"""
#         SELECT ?s ?p ?o
#         WHERE {{
#             BIND(<http://skg.health.oraclecloud.com/icd10cm/{code}> AS ?s) .
#             ?s <{SKOS_PREF_LABEL}> ?o .
#             BIND(<{SKOS_PREF_LABEL}> AS ?p)
#         }}
#         LIMIT 1
#     """

#     sparql = SPARQLWrapper(SKG_SPARQL_ENDPOINT)
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#     sparql.addCustomHttpHeader("Accept", "application/sparql-results+json")

#     try:
#         results = sparql.query().convert()
#         bindings = results.get("results", {}).get("bindings", [])

#         # Optional: debug output
#         # for result in bindings:
#         #     logger.debug(f"SPARQL result for code='{code}': s={result.get('s', {}).get('value')}, \
#         # p={result.get('p', {}).get('value')}, o={result.get('o', {}).get('value')}")
#         #     logger.debug("----------")
#         return bool(bindings)
#     except Exception as e:
#         logger.debug(f"SPARQL query failed for code='{code}': {e}")
#         return False


# ------------------------ Nodes ------------------------


@track_tokens("model_usage")
def node_extract_and_queries(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """Combined extraction of contexts + per-problem search queries."""
    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.WORKFLOW]
    model, parser = _get_llm_and_parser(config, CombinedExtractQueriesOut)
    chain = model | parser

    system_tmpl = prompts["extract_and_queries"][config["configurable"][CONFIG.EXTRACT_AND_QUERIES_PROMPT_VERSION]]["system"]
    user_tmpl = prompts["extract_and_queries"][config["configurable"][CONFIG.EXTRACT_AND_QUERIES_PROMPT_VERSION]]["user"]
    logger.debug(f"[node_extract_and_queries] system prompt: {system_tmpl[:500]}")
    messages = _format_messages(
        system_tmpl,
        user_tmpl,
        parser,
        note=state["text"],
        patient_age=state.get("patient_age", ""),
        patient_sex=state.get("patient_sex", ""),
    )
    messages = _maybe_collapse_system_for_o1(messages, config)

    out: CombinedExtractQueriesOut = chain.invoke(messages)

    # Persist contexts
    extracted = [c.model_dump() for c in out.contexts]

    # Ensure each problem includes the raw problem as a query
    queries_by_problem: Dict[str, List[str]] = {}
    for ctx in extracted:
        p = ctx["problem"]
        qs = list(dict.fromkeys([p, *out.queries_by_problem.get(p, [])]))
        queries_by_problem[p] = qs
    logger.debug(f"Extracted contexts NO.: {len(extracted)}; total search queries NO.: {sum(len(qs) for qs in queries_by_problem.values())}")
    return {
        "extracted_contexts": extracted,
        "search_queries": queries_by_problem,
    }


def node_search_codes(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """
    Search ICD-10-CM codes per problem (and its queries), ensuring:
      - each problem's search space includes the problem text itself
      - queries are unique and stripped
      - keep per-query ranked results in order (per_query_hits)
      - deduplicate across queries for a summary list per problem (search_hits)
    """
    hits: Dict[str, List[CandidateForChoice]] = {}
    per_query_hits: Dict[str, Dict[str, List[CandidateForChoice]]] = {}

    problems_to_queries: Dict[str, List[str]] = state.get("search_queries", {}) or {}
    for problem, queries in problems_to_queries.items():
        # Build a unique, non-empty query set that includes the problem itself
        qset: List[str] = []
        seen_q: set[str] = set()
        for q in [problem, *(queries or [])]:
            q = (q or "").strip()
            if not q or q in seen_q:
                continue
            seen_q.add(q)
            qset.append(q)

        # Per-query ranked lists (preserve order as returned by the tool)
        per_query_hits[problem] = {}

        # Deduplicate by ICD-10 code across all queries; keep best tool score for summary
        by_code: Dict[str, dict] = {}

        for q in qset:
            try:
                top_k = config["configurable"].get(CONFIG.SEARCH_TOP_K, 30)
                results = tool_search(query=q, top_k=top_k)
            except Exception as e:
                logger.warning(f"tool_search failed for query='{q}': {e}")
                per_query_hits[problem][q] = []
                continue

            # tool_search may return a JSON string or a list[dict]
            if isinstance(results, str):
                try:
                    results = json.loads(results)
                except Exception as e:
                    logger.warning(f"Failed to json.loads tool_search result for query='{q}': {e}")
                    results = []

            # Normalize to typed candidates, in the order returned
            ranked_cands: List[CandidateForChoice] = []
            for r in results or []:
                code = (r.get("code") or "").strip()
                if not code:
                    continue
                desc = (r.get("description") or "").strip()
                score = float(r.get("score", 0.0) or 0.0)

                ranked_cands.append(
                    CandidateForChoice(
                        code=code,
                        description=desc,
                        rationale="",  # none from search branch
                        source="search",
                        score=score,
                    )
                )

                # maintain best-by-code for summary (search_hits)
                keep = by_code.get(code)
                if keep is None or score > keep.get("score", float("-inf")):
                    by_code[code] = {
                        "code": code,
                        "description": desc,
                        "score": score,
                        "source": "search",
                    }

            per_query_hits[problem][q] = ranked_cands

        # Convert to typed summary list (deduped across queries)
        cand_list: List[CandidateForChoice] = [
            CandidateForChoice(
                code=entry["code"],
                description=entry.get("description", ""),
                rationale="",
                source="search",
                score=float(entry.get("score", 0.0) or 0.0),
            )
            for entry in by_code.values()
        ]
        hits[problem] = cand_list

        logger.debug(
            f"Search hits for problem='{problem}': {len(qset)} queries → "
            f"{sum(len(v) for v in per_query_hits[problem].values())} total raw results, "
            f"{len(cand_list)} unique codes (summary)"
        )

    logger.debug(f"Search hits completed for {len(hits)} problems.")
    return {"search_hits": hits, "per_query_hits": per_query_hits}


@track_tokens("model_usage")
def node_direct_predict(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """Direct ICD-10-CM proposals from the SOAP (parallel branch)."""
    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.WORKFLOW]
    model, parser = _get_llm_and_parser(config, DirectPredictionOut)
    chain = model | parser

    system_tmpl = prompts["direct"][config["configurable"][CONFIG.LLM_ONLY_PROMPT_VERSION]]["system"]
    user_tmpl = prompts["direct"][config["configurable"][CONFIG.LLM_ONLY_PROMPT_VERSION]]["user"]

    messages = _format_messages(
        system_tmpl,
        user_tmpl,
        parser,
        note=state["text"],
        patient_age=state.get("patient_age", ""),
        patient_sex=state.get("patient_sex", ""),
    )
    messages = _maybe_collapse_system_for_o1(messages, config)

    out: DirectPredictionOut = chain.invoke(messages)

    direct: List[CandidateForChoice] = []
    for p in out.predictions:
        code = (p.code or "").strip()
        if code:
            direct.append(
                CandidateForChoice(
                    code=code,
                    description=(p.description or "").strip(),
                    rationale=(p.rationale or "").strip(),
                    source="direct",
                )
            )
    logger.debug(f"Direct predictions NO. of codes: {len(direct)}")
    return {"direct_candidates": direct}


def node_merge_candidates(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """Merge search-based and direct candidates; run only after BOTH are present."""
    # ---- barrier: wait until both upstream results exist ----
    search_hits = state.get("search_hits")
    direct_candidates = state.get("direct_candidates")

    if not search_hits or direct_candidates is None:
        # Returning {} makes this node a no-op; it will be triggered again
        # when the other branch writes its part of the state.
        logger.debug("merge_candidates waiting: search_hits or direct_candidates not ready.")
        return {}

    # ---- actual merge once both are ready ----
    all_candidates: List[CandidateForChoice] = []
    for _, cands in (search_hits or {}).items():
        # cands is List[CandidateForChoice]
        all_candidates.extend(cands or [])
    all_candidates.extend(direct_candidates or [])

    def _score(c: CandidateForChoice) -> float:
        # If you later add tool_score to CandidateForChoice, this will pick it up.
        # Otherwise it returns 0.0 (keeps first seen).
        try:
            s = getattr(c, "tool_score", None)
            return float(s) if s is not None else 0.0
        except Exception:
            return 0.0

    by_code: Dict[str, CandidateForChoice] = {}
    best_score: Dict[str, float] = {}

    for c in all_candidates:
        code = (c.code or "").strip()
        if not code:
            continue
        s = _score(c)
        keep = by_code.get(code)
        if keep is None or s > best_score.get(code, float("-inf")):
            by_code[code] = c
            best_score[code] = s

    merged: List[CandidateForChoice] = list(by_code.values())
    logger.debug(f"Merged candidates NO. of codes: {len(merged)}")
    return {"merged_candidates": merged}


def node_validate_and_filter(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """Validate existence and billability, preserving typed CandidateForChoice,
    and record filtered candidates per problem/query.
    """
    filtered: List[CandidateForChoice] = []
    merged: List[CandidateForChoice] = state.get("merged_candidates", []) or []
    per_query_hits: Dict[str, Dict[str, List[CandidateForChoice]]] = state.get("per_query_hits", {}) or {}

    def _is_code_kept(raw_code: str) -> bool:
        code = (raw_code or "").strip()
        if not code:
            return False

        is_valid = tool_is_valid(code=code)
        # is_valid_skg = _tool_valid(code=code)
        is_valid_skg = True
        is_billable = tool_is_billable(code=code)

        keep = bool(is_valid and is_billable and is_valid_skg)
        if not keep:
            logger.debug(f"Code {code} valid status: {is_valid}, " f"billable status: {is_billable}, " f"skg status: {is_valid_skg}")
        return keep

    # ---- flat filtered list (used downstream for reranker/choose) ----
    for c in merged:
        code = (c.code or "").strip()
        if not code:
            continue
        if not _is_code_kept(code):
            continue
        filtered.append(c)

    # ---- per-problem, per-query filtered candidates ----
    per_query_filtered: Dict[str, Dict[str, List[CandidateForChoice]]] = {}

    for problem, qdict in per_query_hits.items():
        per_query_filtered[problem] = {}
        for query, cand_list in (qdict or {}).items():
            kept: List[CandidateForChoice] = []
            seen: set[str] = set()
            for cand in cand_list or []:
                code = (cand.code or "").strip()
                if not code:
                    continue
                if not _is_code_kept(code):
                    continue
                key = code.upper()
                if key in seen:
                    continue
                seen.add(key)
                kept.append(cand)
            per_query_filtered[problem][query] = kept

    logger.debug(f"Validated & filtered candidates NO. of codes: {len(filtered)}")
    return {
        "filtered_candidates": filtered,
        "per_query_filtered": per_query_filtered,
    }


@track_tokens("model_usage")
def node_choose_codes(state: ICD10CMCodeGraphStates, config: RunnableConfig) -> dict:
    """LLM selection step with guideline awareness (choose + clarifications + enforced primary)."""
    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.WORKFLOW]
    model, parser = _get_llm_and_parser(config, ChooseCodesOut)
    chain = model | parser

    # Minimize payload to LLM: filtered_candidates is already List[CandidateForChoice]
    cand_min: List[CandidateForChoice] = list(state.get("filtered_candidates", []) or [])

    system_tmpl = prompts["choose"][config["configurable"][CONFIG.CODE_RERANKER_PROMPT_VERSION]]["system"]
    user_tmpl = prompts["choose"][config["configurable"][CONFIG.CODE_RERANKER_PROMPT_VERSION]]["user"]
    logger.debug(f"[node_choose_codes] system prompt: {system_tmpl[:500]}")
    # Prefer per_query_hits structure for v8 choose prompt; keep backward compatibility otherwise.
    choose_ver = str(config["configurable"][CONFIG.CODE_RERANKER_PROMPT_VERSION])
    # Interpret versions like "v8", "v9", etc.; route pqc_json path only for versions > v8
    try:
        _ver_num = int(str(choose_ver).lower().lstrip("v"))
    except Exception:
        _ver_num = 0
    if _ver_num >= 8:
        # Build problems → queries → candidates view directly from per_query_hits
        pqc_from_pqh: List[dict] = []
        pqh: Dict[str, Dict[str, List[CandidateForChoice]]] = state.get("per_query_hits", {}) or {}
        # Build evidence spans per problem from extracted_contexts
        evidence_by_problem: Dict[str, List[str]] = {}
        for ctx in state.get("extracted_contexts", []) or []:
            try:
                p = (ctx.get("problem") if isinstance(ctx, dict) else getattr(ctx, "problem", "")) or ""
                ev = (ctx.get("evidence") if isinstance(ctx, dict) else getattr(ctx, "evidence", "")) or ""
            except Exception:
                p, ev = "", ""
            if p:
                lst = evidence_by_problem.setdefault(p, [])
                if ev and ev not in lst:
                    lst.append(ev)
        for problem, qdict in pqh.items():
            q_entries: List[dict] = []
            for query, clist in (qdict or {}).items():
                q_entries.append(
                    {
                        "query": query,
                        "candidates": [
                            {
                                "code": c.code,
                                "description": c.description,
                                # "source": getattr(c, "source", ""),
                                # "score": float(getattr(c, "score", 0.0) or 0.0),
                            }
                            for c in (clist or [])
                        ],
                    }
                )
            pqc_from_pqh.append({"problem": problem, "evidence_of_problem": evidence_by_problem.get(problem, []), "queries": q_entries})

        # Include direct-branch candidates under a synthetic "direct" query if present
        direct_only = [c for c in (state.get("direct_candidates", []) or []) if getattr(c, "source", "") == "direct"]
        if direct_only:
            pqc_from_pqh.append(
                {
                    "problem": "",
                    "queries": [
                        {
                            "query": "direct",
                            "candidates": [
                                {
                                    "code": c.code,
                                    "description": c.description,
                                    # "source": getattr(c, "source", ""),
                                    # "score": float(getattr(c, "score", 0.0) or 0.0),
                                }
                                for c in direct_only
                            ],
                        }
                    ],
                }
            )

        prompt_vars = dict(
            note=state["text"],
            patient_age=state.get("patient_age", ""),
            patient_sex=state.get("patient_sex", ""),
            pqc_json=json.dumps(pqc_from_pqh),
        )
    else:
        # Older prompts expect contexts_json and candidates_json; keep them for compatibility
        prompt_vars = dict(
            note=state["text"],
            patient_age=state.get("patient_age", ""),
            patient_sex=state.get("patient_sex", ""),
            contexts_json=json.dumps(state.get("extracted_contexts", [])),
            candidates_json=json.dumps([c.model_dump() for c in cand_min]),
        )

    messages = _format_messages(system_tmpl, user_tmpl, parser, **prompt_vars)
    messages = _maybe_collapse_system_for_o1(messages, config)

    out: ChooseCodesOut = chain.invoke(messages)

    # Index filtered candidates for enrichment
    filtered: List[CandidateForChoice] = state.get("filtered_candidates", []) or []
    by_code = {(c.code or "").strip(): c for c in filtered}

    chosen_internal: List[CandidateForChoice] = []  # internal with rationale & full_data
    primary_entry: Optional[CandidateForChoice] = None

    def _lookup_candidate(code: str, role: str) -> Optional[CandidateForChoice]:
        code_norm = (code or "").strip()
        if not code_norm:
            return None
        base = by_code.get(code_norm)
        if base is None:
            logger.warning(f"[node_choose_codes] LLM selected {role} code '{code_norm}' " "which is not in filtered_candidates; skipping.")
            return None
        return base

    # --- Primary from LLM ---
    if out.primary:
        cand = _lookup_candidate(out.primary.code, "primary")
        if cand is not None:
            primary_entry = cand.model_copy(update={"rationale": (out.primary.rationale or "")})
            chosen_internal.append(primary_entry)  # keep first

    # --- Secondary from LLM ---
    for sec in out.secondary:
        cand = _lookup_candidate(sec.code, "secondary")
        if cand is None:
            continue
        chosen_internal.append(cand.model_copy(update={"rationale": (sec.rationale or "")}))

    # --- Enforce a primary if none selected ---
    def _score(c: CandidateForChoice) -> float:
        try:
            v = getattr(c, "score", None)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    def _billable(c: CandidateForChoice) -> bool:
        v = getattr(c, "billable", None)
        return bool(v) if v is not None else False

    if not chosen_internal:
        # pick best billable by score; else best valid by tool_score
        filt: List[CandidateForChoice] = state.get("filtered_candidates", []) or []
        best_billable: List[CandidateForChoice] = [c for c in filt if _billable(c)]
        if best_billable:
            fallback = max(best_billable, key=_score)
        elif filt:
            fallback = max(filt, key=_score)
        else:
            # No valid/billable candidates at all: we cannot assign any diagnosis.
            retrieval_trace = RetrievalTrace(
                contexts=state.get("extracted_contexts", []) or [],
                queries_by_problem=state.get("search_queries", {}) or {},
                per_query_hits=state.get("per_query_hits", {}) or {},
                search_hits=state.get("search_hits", {}) or {},
                filtered_candidates=state.get("filtered_candidates", []) or [],
                final_selection=[],
                per_query_filtered=state.get("per_query_filtered", {}) or {},
            )
            # As a last resort, use merged (validated not guaranteed) – but we avoid this to keep validity.
            return {
                "final_selection": [],
                "clarifications": ["No valid/billable candidates found; cannot assign a primary diagnosis."],
                "diagnosis": [],
                "retrieval_trace": retrieval_trace,
            }

        primary_entry = fallback.model_copy(
            update={
                "rationale": (
                    "Selected as fallback primary: highest-scoring billable "
                    "(or otherwise highest-scoring) candidate when LLM returned no selection."
                )
            }
        )
        chosen_internal.append(primary_entry)

        # If we have codes but primary_entry is still None (shouldn't happen), promote first
    if primary_entry is None and chosen_internal:
        primary_entry = chosen_internal[0]

    # ---------- Apply combo-code rules (post-processing) ----------
    primary_code: Optional[str] = primary_entry.code if primary_entry else None
    combo_rules = config["configurable"].get("combo_rules") or {}
    if combo_rules:
        chosen_internal, primary_code = _apply_combo_rules_to_candidates(
            chosen_internal,
            primary_code,
            combo_rules,
        )

    # ---------- Map to your ICD10CM schema ----------
    diagnosis: List[ICD10CM] = []
    for c in chosen_internal:
        is_primary = bool(primary_code) and c.code == primary_code
        diagnosis.append(
            ICD10CM(
                code=c.code,
                is_primary=is_primary,
                system="ICD-10-CM",
                description=(c.description or ""),
                justification=(c.rationale or ""),
            )
        )

    # --- RetrievalTrace ---
    retrieval_trace = RetrievalTrace(
        contexts=state.get("extracted_contexts", []) or [],
        queries_by_problem=state.get("search_queries", {}) or {},
        per_query_hits=state.get("per_query_hits", {}) or {},
        search_hits=state.get("search_hits", {}) or {},
        filtered_candidates=state.get("filtered_candidates", []) or [],
        final_selection=chosen_internal or [],
        per_query_filtered=state.get("per_query_filtered", {}) or {},
    )

    # Keep primary first for readability
    diagnosis.sort(key=lambda d: (not d.is_primary, d.code))
    logger.debug(f"diagnosis output: {diagnosis}")
    return {"diagnosis": diagnosis, "retrieval_trace": retrieval_trace}


# ---------------------- Helper Functions -----------------------
def _apply_combo_rules_to_candidates(
    candidates: List[CandidateForChoice],
    primary_code: Optional[str],
    combo_rules: Dict[str, str],
) -> tuple[List[CandidateForChoice], Optional[str]]:
    """
    Apply combo-code rules to a list of chosen candidates.

    combo_rules: mapping like
        "E09.9,N18.33" -> "E09.22"
    Keys are comma-separated ICD-10-CM codes, value is the combo code.

    Algorithm:
      - For each rule, if all component codes are present in candidates,
        remove those components and insert (or reuse) the combo code.
      - If any component was primary, promote the combo code to primary.
      - Applies repeatedly until no further rule matches.
    """
    if not combo_rules:
        return candidates, primary_code

    # Normalize YAML rules to a mapping from frozenset(component_codes) -> combo_code
    normalized_rules: Dict[frozenset[str], str] = {}
    for key, val in combo_rules.items():
        if not key:
            continue
        parts = [p.strip().upper() for p in str(key).split(",") if p.strip()]
        if not parts:
            continue
        normalized_rules[frozenset(parts)] = str(val).strip().upper()

    # Work on a copy to avoid mutating caller's list in-place
    cands = list(candidates)
    # Iterate until no rule applies (handles overlapping/chain rules reasonably)
    changed = True
    while changed:
        changed = False
        codes_present = {c.code.upper() for c in cands}

        for code_set, combo_code in normalized_rules.items():
            # Do we have all components?
            if not code_set.issubset(codes_present):
                continue

            # We will apply this rule
            changed = True
            component_codes = {c.upper() for c in code_set}

            # Collect component candidates and others
            component_cands = [c for c in cands if c.code.upper() in component_codes]
            remaining_cands = [c for c in cands if c.code.upper() not in component_codes]

            # If combo code already present, reuse it; else create a new candidate
            existing_combo = None
            for c in remaining_cands:
                if c.code.upper() == combo_code:
                    existing_combo = c
                    break

            # Build a new/updated combo candidate
            if existing_combo is not None:
                # Enrich rationale to record combo application
                extra_rationale = f"Combo code applied for components: {', '.join(sorted(component_codes))}."
                new_rationale = existing_combo.rationale or ""
                if extra_rationale not in new_rationale:
                    new_rationale = (new_rationale + " " if new_rationale else "") + extra_rationale
                combo_cand = existing_combo.model_copy(update={"rationale": new_rationale})
            else:
                # Merge descriptions / rationales from components if you like
                comp_descr = ", ".join(sorted({c.description for c in component_cands if c.description}))
                comp_rats = " ".join([c.rationale for c in component_cands if c.rationale])
                rationale = f"Combo code applied for components: {', '.join(sorted(component_codes))}."
                if comp_rats:
                    rationale += " Component rationales: " + comp_rats
                combo_cand = CandidateForChoice(
                    code=combo_code,
                    description=comp_descr,
                    rationale=rationale,
                    source="combo_rule",
                    score=0.0,
                )

            # Update primary code if any component was primary
            if primary_code is not None and primary_code.upper() in component_codes:
                primary_code = combo_code

            # Insert/reinsert combo candidate
            # (Avoid duplicates if already present and we just updated it.)
            new_cands: List[CandidateForChoice] = []
            seen_combo = False
            for c in remaining_cands:
                if c.code.upper() == combo_code:
                    if not seen_combo:
                        new_cands.append(combo_cand)
                        seen_combo = True
                    # skip duplicates
                else:
                    new_cands.append(c)
            if not seen_combo:
                new_cands.append(combo_cand)

            cands = new_cands
            # break to recompute codes_present and re-scan rules
            break

    return cands, primary_code


# ------------------------ Graph builder ------------------------


def build_workflow_graph(args: Namespace):
    """
    Returns (CompiledStateGraph, model_config).
    Expects your prompts dict to include the three blocks exactly as provided:
      WORKFLOW.extract_and_queries.v1, WORKFLOW.direct.v1, WORKFLOW.choose.v1
    """
    prompts = getattr(args, "prompts", None) or load_from_yaml(args.prompt_path)
    config = vars(args)

    # load combo code rules
    combo_rules = None
    combo_rules_path = getattr(args, "combo_rules_path", None)
    if combo_rules_path:
        combo_rules = load_from_yaml(combo_rules_path)
        logger.info(
            f"Loaded combo code rules from: {combo_rules_path} " f"({len(combo_rules) if isinstance(combo_rules, dict) else 'unknown'} rules)"
        )

    model_config = {"configurable": {"prompts": prompts, **config}}
    if combo_rules is not None:
        # store rules dict into config so nodes can use it
        model_config["configurable"]["combo_rules"] = combo_rules

    g = StateGraph(
        ICD10CMCodeGraphStates,
        input=InputState,
        output=OutputState,
    )

    g.add_node("extract_and_queries", node_extract_and_queries)  # combined
    g.add_node("search_codes", node_search_codes)
    g.add_node("direct_predict", node_direct_predict)
    g.add_node("merge_candidates", node_merge_candidates)
    g.add_node("validate_filter", node_validate_and_filter)
    g.add_node("choose_codes", node_choose_codes)

    # Parallel fan-out from START
    g.add_edge(START, "extract_and_queries")
    g.add_edge(START, "direct_predict")

    # Retrieval path
    g.add_edge("extract_and_queries", "search_codes")

    # Join
    g.add_edge("search_codes", "merge_candidates")
    g.add_edge("direct_predict", "merge_candidates")

    # Filter → choose → END
    g.add_edge("merge_candidates", "validate_filter")
    g.add_edge("validate_filter", "choose_codes")
    g.add_edge("choose_codes", END)

    return g.compile(), model_config
