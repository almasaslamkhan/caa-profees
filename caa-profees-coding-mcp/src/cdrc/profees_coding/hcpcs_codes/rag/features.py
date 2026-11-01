"""Backward-compatible shim for retrieval feature profiles."""

from .profiles import G2211_PROFILE
from .profiles import RetrievalProfile as RetrievalFeatureProfile
from .profiles import resolve_retrieval_profile


def resolve_feature_profile(*, feature_name=None, candidate_codes=None):
    return resolve_retrieval_profile(feature_name=feature_name, candidate_codes=candidate_codes)


__all__ = [
    "G2211_PROFILE",
    "RetrievalFeatureProfile",
    "resolve_feature_profile",
]
