from __future__ import annotations

import json
from importlib.resources import files
from typing import TYPE_CHECKING, List, TypedDict

# from agents import function_tool
from cdrc.common.icd10_cm_tools.src.tools.data_config import DEFAULT_ICD10_CM_DATA_PATHS
from cdrc.common.icd10_cm_tools.src.tools.validator import cm_code_validator as validator

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from cdrc.common.icd10_cm_tools.src.tools.searcher.bm25_search import BM25sICD10Search


class SearchResultDict(TypedDict):
    """Serializable search result for ICD-10-CM queries."""

    code: str
    score: float
    description: str | None
    short_description: str | None


def _load_searcher() -> "BM25sICD10Search":
    """Create a BM25 searcher from the packaged prebuilt index.

    Returns:
        BM25sICD10Search: Ready-to-use searcher instance backed by a prebuilt index.
    """

    index_path = files("cdrc.common.icd10_cm_tools.data.source").joinpath(DEFAULT_ICD10_CM_DATA_PATHS.search_index_file)
    from cdrc.common.icd10_cm_tools.src.tools.searcher.bm25_search import BM25sICD10Search

    return BM25sICD10Search(index_path=index_path)


# Initialize a module-level searcher lazily on first use to keep import time fast.
_SEARCHER: "BM25sICD10Search" | None = None


def _get_searcher() -> "BM25sICD10Search":
    global _SEARCHER
    if _SEARCHER is None:
        _SEARCHER = _load_searcher()
    return _SEARCHER


def search(query: str, top_k: int = 10) -> List[SearchResultDict]:
    """Search ICD-10-CM codes by natural language query.

    Args:
        query: Free-text query such as "heart failure".
        top_k: Maximum number of results to return.

    Returns:
        A list of dictionaries containing `code`, `description`, `short_description`, and `score`.
    """

    results = _get_searcher().search(query=query, top_k=top_k)
    serialized: List[SearchResultDict] = []
    for r in results:
        # r.code may be a CMCode model or a string, normalize here
        code_value: str
        description: str | None = None
        short_description: str | None = None

        code_obj = getattr(r, "code", None)
        if hasattr(code_obj, "code"):
            code_value = getattr(code_obj, "code")
            description = getattr(code_obj, "description", None)
            short_description = getattr(code_obj, "short_description", None)
        else:
            code_value = str(code_obj)
            try:
                description = validator.get_description(code_value)
            except Exception:
                description = None

        serialized.append(
            {
                "code": code_value,
                "score": float(r.score),
                "description": description,
                "short_description": short_description,
            }
        )
    return serialized


def is_valid_item(code: str) -> bool:
    """Check whether the provided ICD-10-CM code exists.

    The function is tolerant to the absence of the decimal point (e.g., "I10" or "I10.0").

    Args:
        code: ICD-10-CM code to validate.

    Returns:
        True if the code exists in ICD-10-CM; False otherwise.
    """

    return bool(validator.is_valid_item(code))


def is_billable(code: str) -> bool:
    """Determine whether the ICD-10-CM code is billable.

    Args:
        code: ICD-10-CM code.

    Returns:
        True if the code is billable; False otherwise.
    """

    return bool(validator.is_billable(code))


def get_description(code: str, prioritize_blocks: bool = False) -> str | None:
    """Get the official description for an ICD-10-CM code.
    Args:
        code: ICD-10-CM code.
    Returns:
        The description string if the code exists; otherwise None.
    """

    return validator.get_description(code, prioritize_blocks=prioritize_blocks)


def get_full_data(code: str) -> str | None:
    """Get a formatted string with full information for the code.

    Includes description, parent, includes/excludes, inclusion terms, seven-character
    notes/definitions, additional coding guidance, and children list if present.

    Args:
        code: ICD-10-CM code.

    Returns:
        A human-readable string with the code's details, or None if the code is invalid.
    """

    return validator.get_full_data(code)


class SEARCHER:
    """Convenience wrapper with a `search` method for direct Python use.

    Example:
        results = SEARCHER.search("heart failure")
    """

    @staticmethod
    def search(query: str, top_k: int = 10) -> List[SearchResultDict]:
        return search(query=query, top_k=top_k)


class VALIDATOR:
    """Convenience wrapper mirroring validator operations for direct Python use.

    Example:
        VALIDATOR.is_valid_item("I10")
    """

    @staticmethod
    def is_valid_item(code: str) -> bool:
        return is_valid_item(code)

    @staticmethod
    def is_billable(code: str) -> bool:
        return is_billable(code)

    @staticmethod
    def get_full_data(code: str) -> str | None:
        return get_full_data(code)


# @function_tool
# def search_icd10cm(query: str, top_k: int = 10) -> str:
#     """Search ICD-10-CM codes using a semantic BM25 index across code titles, short descriptions, and inclusion terms.

#     Args:
#         query: Free-text query such as "heart failure" (e.g., symptoms, diagnoses, or terms).
#         top_k: Maximum number of results to return (default 10). Must be a positive integer.

#     Returns:
#         JSON string containing a list of objects with `code`, `description`, `short_description`, and `score`.
#         The list is sorted by relevance score descending and de-duplicated at the code level.
#     """

#     return json.dumps(search(query=query, top_k=top_k))


# @function_tool
# def is_valid_icd10cm_code(code: str) -> str:
#     """Validate whether an ICD-10-CM code exists in the ontology (with or without a decimal point).

#     Args:
#         code: ICD-10-CM code to validate (e.g., "I10", "I10.0"). Case-insensitive; punctuation other than the decimal point is unsupported.

#     Returns:
#         JSON string like {"is_valid": true}.
#     """

#     return json.dumps({"is_valid": is_valid_item(code)})


# @function_tool
# def is_billable_icd10cm_code(code: str) -> str:
#     """Check whether an ICD-10-CM code is billable based on the official billable indicator.

#     Args:
#         code: ICD-10-CM code (with or without decimal point). Must be a valid code.

#     Returns:
#         JSON string like {"is_billable": true}.
#     """

#     return json.dumps({"is_billable": is_billable(code)})


# @function_tool
# def get_icd10cm_full_data(code: str) -> str:
#     """Get the full, human-readable data for an ICD-10-CM code including relationships and coding guidance.

#     The output contains description, parent, includes/excludes, inclusion terms, seven-character note/definition,
#     additional coding guidance, and any child codes.

#     Args:
#         code: ICD-10-CM code (with or without decimal point).

#     Returns:
#         If the code is valid: a single formatted string aggregating all relevant fields.
#         If invalid: JSON string {"error": "Invalid ICD-10-CM code"}.
#     """

#     data = get_full_data(code)
#     if data is None:
#         return json.dumps({"error": "Invalid ICD-10-CM code"})
#     return data


# @function_tool
# def get_icd10cm_description(code: str, prioritize_blocks: bool = False) -> str:
#     """Get the official description for an ICD-10-CM code.

#     Includes optional block prioritization (returns the parent block description when applicable).

#     Args:
#         code: ICD-10-CM code (with or without decimal point).
#         prioritize_blocks: If True and the code is a block-level code, return the parent block's description.

#     Returns:
#         The description string if the code exists; otherwise a JSON error string.
#     """

#     try:
#         return validator.get_description(code, prioritize_blocks=prioritize_blocks)
#     except Exception as exc:
#         return json.dumps({"error": str(exc)})

__all__ = [
    # Direct-use helpers
    "SEARCHER",
    "VALIDATOR",
    "search",
    "is_valid_item",
    "is_billable",
    "get_description",
    "get_full_data",
    # Agent tools
    # "search_icd10cm",
    # "is_valid_icd10cm_code",
    # "is_billable_icd10cm_code",
    # "get_icd10cm_full_data",
    # "get_icd10cm_description",
]
