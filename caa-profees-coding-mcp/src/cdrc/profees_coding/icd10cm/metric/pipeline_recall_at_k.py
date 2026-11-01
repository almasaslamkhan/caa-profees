# cdrc/common/metric/pipeline_recall_at_k.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from pydantic import BaseModel

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.utils import get_nested_attribute_list

# ----------------------------- Helpers -----------------------------


def _to_plain(obj: Any) -> Any:
    """Recursively convert Pydantic models to plain dicts/lists; pass through dicts/lists."""
    if isinstance(obj, BaseModel):
        return _to_plain(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def _get_retrieval_trace(output: Any) -> Dict[str, Any]:
    """
    Prefer additional_output['retrieval_trace']; fallback to output.retrieval_trace.
    Returns a plain (non-Pydantic) dict.
    """
    addl = getattr(output, "additional_output", None) or {}
    rt = addl.get("retrieval_trace")
    if rt is None:
        rt = getattr(output, "retrieval_trace", None)
    rt = _to_plain(rt)
    return rt or {}


def _score_of(item: Any) -> float:
    """Extract per-item score (dict or typed), missing/None -> -inf."""
    try:
        if isinstance(item, dict):
            s = item.get("score", None)
        else:
            s = getattr(item, "score", None)
        return float(s) if s is not None else float("-inf")
    except Exception:
        return float("-inf")


def _union_per_query_topk(rt: Dict[str, Any], k: int) -> Dict[str, Dict[str, Any]]:
    """
    For each problem, take top-k results *per query* (ranking only within the query),
    then form the union across queries for that problem.

    Expected rt structure (plain dicts/lists):
      {
        "per_query_hits": {
          "<problem>": {
            "<query_1>": [ {code, description, score?, ...}, ... ],
            "<query_2>": [ ... ],
            ...
          },
          ...
        }
      }

    Returns:
      {
        "<problem>": {
          "per_query_topk": { "<query>": [candidate_dict, ...], ... },
          "union_codes": ["CODE1", "CODE2", ...]  # deduped union across queries
        },
        ...
      }
    """
    out: Dict[str, Dict[str, Any]] = {}
    per_query_hits = rt.get("per_query_hits", {}) or {}

    for problem, query_map in per_query_hits.items():
        per_q_topk: Dict[str, List[Dict[str, Any]]] = {}
        union_codes: set[str] = set()

        for query, results in (query_map or {}).items():
            # Normalize each query's results to plain dicts
            norm: List[Dict[str, Any]] = []
            for r in results or []:
                r_plain = _to_plain(r)
                if isinstance(r_plain, dict):
                    norm.append(r_plain)

            # Sort within the query if any scores present; otherwise keep tool order
            if any(("score" in x and x["score"] is not None) for x in norm):
                norm = sorted(norm, key=_score_of, reverse=True)

            # Take top-k in query-local rank order
            topk = norm[:k] if k > 0 else norm
            per_q_topk[query] = topk

            # Accumulate union of codes
            for e in topk:
                code = (e.get("code") or "").strip()
                if code:
                    union_codes.add(code)

        out[problem] = {
            "per_query_topk": per_q_topk,
            "union_codes": sorted(union_codes),  # sorted for determinism
        }

    return out


def _fallback_union_from_search_hits(rt: Dict[str, Any], k: int) -> set[str]:
    """
    Fallback when per_query_hits is unavailable:
      - Flatten search_hits per problem
      - Sort candidates by score desc (if present), otherwise keep order
      - Take top-k per problem
      - Union across problems
    """
    union: set[str] = set()
    search_hits = rt.get("search_hits", {}) or {}
    for _problem, cand_list in search_hits.items():
        # normalize
        norm = []
        for c in cand_list or []:
            c_plain = _to_plain(c)
            if isinstance(c_plain, dict):
                norm.append(c_plain)
        if any(("score" in x and x["score"] is not None) for x in norm):
            norm = sorted(norm, key=_score_of, reverse=True)
        topk = norm[:k] if k > 0 else norm
        for e in topk:
            code = (e.get("code") or "").strip()
            if code:
                union.add(code)
    return union


# ----------------------------- Metric -----------------------------


class PipelineRecallAtK(BaseMetric):
    """
    Whole-pipeline Recall@K for diagnosis extraction + code search.

    Definition
    ----------
    For each sample i:
      • Let G_i be the set of gold codes (after normalization).
      • For each problem extracted by the model, each query returns a ranked list
        of candidate codes. We keep the top-K results per query (as returned by
        the tool for that query), then take the UNION across all queries and all
        problems to form R_i^(K), the retrieved set for sample i.
      • Per-sample recall@K is:
            recall@K(i) = | G_i ∩ R_i^(K) | / | G_i |    if |G_i| > 0
                           None (excluded from macro)     if |G_i| = 0

    Aggregates
    ----------
      • micro: sum of per-code hits over sum of gold codes across samples.
        Samples with more gold codes have greater weight.
            micro_recall@K = (Σ_i |G_i ∩ R_i^(K)|) / (Σ_i |G_i|)

      • macro: average of per-sample recall over samples with support > 0.
        Equal weight for all samples.
            macro_recall@K = (1 / |I|) Σ_{i∈I} recall@K(i),
        where I = { i : |G_i| > 0 }.

    Notes & Edge Cases
    ------------------
      • Duplicates are removed by set semantics when forming G_i and R_i^(K).
      • If a query returns fewer than K results, all are used; their order is
        taken as given by the tool for that query (no cross-query re-ranking).
      • If |G_i| = 0, the instance contributes nothing to micro (0/0 is avoided)
        and is excluded from macro averaging.
      • This metric isolates the *recall capacity* of the retrieval stage,
        assuming a later re-ranker chooses from R_i^(K).

    Requires
    --------
    output.additional_output["retrieval_trace"]["per_query_hits"] (preferred),
    or (fallback) output["retrieval_trace"]["search_hits"] with scores/order.
    """

    def __init__(self, label_key: str, ks: Sequence[int] = (1, 5, 10)):
        """
        Args:
            label_key: nested path to gold codes list (e.g., 'diagnosis.code')
            ks: K values to report (e.g., (1, 5, 10))
        """
        super().__init__(supervised=True)
        self.label_key = label_key
        self.ks = sorted({int(k) for k in ks})

    def calculate(self, outputs: List[Any], gold: List[Any]) -> MetricResult:
        instance_metrics: List[Dict[str, Any]] = []
        micro_hits = {k: 0 for k in self.ks}
        micro_gold = 0
        macro_sums = {k: 0.0 for k in self.ks}
        macro_count = 0

        for i, out in enumerate(outputs):
            # ---- Gold labels ----
            gold_codes: List[str] = get_nested_attribute_list(gold[i], self.label_key) or []
            gold_set = {(c or "").strip() for c in gold_codes if c}
            support = len(gold_set)

            rt = _get_retrieval_trace(out)
            has_per_query = bool((rt or {}).get("per_query_hits"))

            row: Dict[str, Any] = {"support": support}
            if support == 0:
                for k in self.ks:
                    row[f"pipeline_recall@{k}"] = None
                instance_metrics.append(row)
                continue

            micro_gold += support
            for k in self.ks:
                if has_per_query:
                    # compute union at this specific K
                    per_problem_k = _union_per_query_topk(rt, k=k)
                    union_all_k: set[str] = set()
                    for _p, payload in per_problem_k.items():
                        for code in payload.get("union_codes", []):
                            if code:
                                union_all_k.add(code)
                else:
                    # fallback from search_hits
                    union_all_k = _fallback_union_from_search_hits(rt, k=k)

                hits_k = len(gold_set & union_all_k)
                r_at_k = hits_k / support
                row[f"pipeline_recall@{k}"] = r_at_k
                micro_hits[k] += hits_k
                macro_sums[k] += r_at_k

            macro_count += 1
            instance_metrics.append(row)

        aggregated = {
            "micro": {f"pipeline_recall@{k}": (micro_hits[k] / micro_gold) if micro_gold > 0 else None for k in self.ks},
            "macro": {f"pipeline_recall@{k}": (macro_sums[k] / macro_count) if macro_count > 0 else None for k in self.ks},
            "support_total": micro_gold,
        }
        return {"instance_metrics": instance_metrics, "aggregated_metric": aggregated}
