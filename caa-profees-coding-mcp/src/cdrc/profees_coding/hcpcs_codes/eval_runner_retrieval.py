"""Deprecated entrypoint shim for the standalone HCPCS RAG runner."""

from __future__ import annotations

import warnings

from cdrc.profees_coding.hcpcs_codes.rag import eval_runner as rag_eval_runner

get_parser = rag_eval_runner.get_parser
load_from_yaml = rag_eval_runner.load_from_yaml
merge_config = rag_eval_runner.merge_config
set_debug = rag_eval_runner.set_debug
_apply_runtime_defaults = rag_eval_runner._apply_runtime_defaults


def main():
    warnings.warn(
        "cdrc.profees_coding.hcpcs_codes.eval_runner_retrieval is deprecated; " "use cdrc.profees_coding.hcpcs_codes.rag.eval_runner instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rag_eval_runner.main()


if __name__ == "__main__":
    main()
