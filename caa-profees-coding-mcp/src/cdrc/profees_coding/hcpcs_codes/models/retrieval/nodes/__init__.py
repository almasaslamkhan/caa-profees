"""Deprecated compatibility shim for the standalone HCPCS retrieval nodes."""

from functools import lru_cache

from cdrc.profees_coding.hcpcs_codes.rag.service import AMBULATORY_ROUTE, FACILITY_ROUTE, route_encounter
from cdrc.profees_coding.hcpcs_codes.rag.vector_store import RetrievalHit
from cdrc.profees_coding.hcpcs_codes.rag.vector_store import embed_text as embed_summary
from cdrc.profees_coding.hcpcs_codes.rag.vector_store import load_vector_store as load_vdb
from cdrc.profees_coding.hcpcs_codes.rag.vector_store import search_vector_store as search_vdb


@lru_cache(maxsize=8)
def _load_cached_vdb(vdb_dir: str):
    return load_vdb(vdb_dir)


__all__ = [
    "AMBULATORY_ROUTE",
    "FACILITY_ROUTE",
    "RetrievalHit",
    "_load_cached_vdb",
    "embed_summary",
    "load_vdb",
    "route_encounter",
    "search_vdb",
]
