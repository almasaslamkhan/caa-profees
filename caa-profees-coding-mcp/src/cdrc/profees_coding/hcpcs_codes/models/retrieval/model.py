"""Deprecated compatibility shim for the standalone HCPCS RAG package."""

from cdrc.profees_coding.hcpcs_codes.rag.model import build_rag_graph, build_retrieval_graph

__all__ = ["build_rag_graph", "build_retrieval_graph"]
