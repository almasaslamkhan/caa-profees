import argparse
import pathlib
import typing

import langgraph.graph

import cdrc.profees_coding.icd10cm_associate.analysis.llm_as_judge.export_results_node as export_results_node
import cdrc.profees_coding.icd10cm_associate.analysis.llm_as_judge.guideline_context_node as guideline_context_node
import cdrc.profees_coding.icd10cm_associate.analysis.llm_as_judge.llm_as_judge_node as llm_as_judge_node
import cdrc.profees_coding.icd10cm_associate.analysis.llm_as_judge.llm_suggestion_node as llm_suggestion_node
import cdrc.profees_coding.icd10cm_associate.analysis.llm_as_judge.load_sample_report_node as load_sample_report_node
import cdrc.profees_coding.icd10cm_associate.analysis.llm_as_judge.prompt_generator_node as prompt_generator_node

# uv run python cdrc/profees_coding/icd10cm_associate/analysis/llm_as_judge/main.py

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent.parent.parent.parent.parent

# Default input and output files.
DEFAULT_ANALYSIS_DATA = [
    # ANALYSIS_DIR / "sample_report_hudson.csv",
    # ANALYSIS_DIR / "sample_report_stjohns.csv",
    # ANALYSIS_DIR / "sample_report_uom.csv",
    # ANALYSIS_DIR / "sample_report_torrance.csv",
    ANALYSIS_DIR / "sample_report_billings_silver.csv",
    ANALYSIS_DIR / "sample_report_billings_gold.csv",
]
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "sample_report_ed_llm_judge.csv"
DEFAULT_GENERATED_PROMPT_FILE = PROJECT_ROOT / "generated_ed_prompt.yaml"

# Suggestion generation scope. Use "all", "fp", or "fn".
SUGGESTION_ERROR_TYPES = "fp"

# Default model settings.
DEFAULT_MODEL_NAME = "caa-gpt-5.5-2026-04-23-dev-sds"
DEFAULT_MODEL_KWARGS = {"reasoning_effort": "low", "temperature": 1.0}

# Prompt files and YAML key paths.
PROMPT_FILE = ANALYSIS_DIR / "prompt.yaml"
JUDGE_PROMPT_KEY_PATH = ("llm-as-judge", "v1")
SUGGESTION_PROMPT_KEY_PATH = ("llm-suggestion", "v1")
PROMPT_GENERATOR_KEY_PATH = ("prompt-generator", "v1")
ORIGINAL_PROMPT_FILE = ANALYSIS_DIR.parent.parent / "models" / "llm_only_merged_v5" / "config" / "prompts.yaml"
ORIGINAL_PROMPT_KEY_PATH = ("ed", "v3")
GENERATED_PROMPT_KEY_PATH = ("ed", "v5")


def main():

    # Parse CLI settings for the input report, output report, and LLM configuration.
    args = _parse_args()

    # Build and invoke the LangGraph workflow.
    graph = _build_llm_as_judge_graph()

    # Seed the graph state with only user-configurable inputs; nodes add everything else.
    graph.invoke(
        {
            "analysis_data": args.analysis_data,
            "output_file": args.output_file,
            "generated_prompt_file": args.generated_prompt_file,
            "suggestion_error_types": SUGGESTION_ERROR_TYPES,
            "model_name": args.model_name,
            "model_kwargs": args.model_kwargs,
            "default_model_kwargs": DEFAULT_MODEL_KWARGS,
            "prompt_file": PROMPT_FILE,
            "judge_prompt_key_path": JUDGE_PROMPT_KEY_PATH,
            "suggestion_prompt_key_path": SUGGESTION_PROMPT_KEY_PATH,
            "prompt_generator_key_path": PROMPT_GENERATOR_KEY_PATH,
            "original_prompt_file": ORIGINAL_PROMPT_FILE,
            "original_prompt_key_path": ORIGINAL_PROMPT_KEY_PATH,
            "generated_prompt_key_path": GENERATED_PROMPT_KEY_PATH,
        }
    )


def _parse_args():

    # Keep the command line surface small so this can be run as a simple analysis script.
    parser = argparse.ArgumentParser(description="Use an LLM as judge for ICD-10 association errors in one or more sample reports.")
    parser.add_argument(
        "--analysis-data", type=pathlib.Path, nargs="+", default=DEFAULT_ANALYSIS_DATA, help="Path(s) to experiment sample report CSV file(s)."
    )
    parser.add_argument("--output-file", type=pathlib.Path, default=DEFAULT_OUTPUT_FILE, help="Path to write the judged CSV.")
    parser.add_argument(
        "--generated-prompt-file", type=pathlib.Path, default=DEFAULT_GENERATED_PROMPT_FILE, help="Path to write the generated prompt YAML."
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Model name registered in get_llm().")
    parser.add_argument("--model-kwargs", type=str, default=None, help="Optional JSON object for model kwargs.")
    return parser.parse_args()


def _build_llm_as_judge_graph():

    # The graph state is a mutable dictionary that grows as each workflow phase completes.
    workflow = langgraph.graph.StateGraph(JudgeGraphState, input_schema=JudgeInputState)

    # Keep graph nodes coarse-grained so the script reads as the analysis workflow.
    workflow.add_node("load_sample_report_node", load_sample_report_node.load_sample_report_node)
    workflow.add_node("guideline_context_node", guideline_context_node.guideline_context_node)
    workflow.add_node("llm_as_judge_node", llm_as_judge_node.llm_as_judge_node)
    workflow.add_node("llm_suggestion_node", llm_suggestion_node.llm_suggestion_node)
    workflow.add_node("export_results_node", export_results_node.export_results_node)
    workflow.add_node("prompt_generator_node", prompt_generator_node.prompt_generator_node)

    # Full workflow: extract errors, add guideline context, judge errors, suggest prompt fixes, export CSV, generate prompt.
    workflow.add_edge(langgraph.graph.START, "load_sample_report_node")
    workflow.add_edge("load_sample_report_node", "guideline_context_node")
    workflow.add_edge("guideline_context_node", "llm_as_judge_node")
    workflow.add_edge("llm_as_judge_node", "llm_suggestion_node")
    workflow.add_edge("llm_suggestion_node", "export_results_node")
    workflow.add_edge("export_results_node", "prompt_generator_node")
    workflow.add_edge("prompt_generator_node", langgraph.graph.END)

    return workflow.compile()


class JudgeInputState(typing.TypedDict):

    # User-configurable inputs provided to the graph at invocation time.
    analysis_data: typing.List[pathlib.Path]
    output_file: pathlib.Path
    generated_prompt_file: pathlib.Path
    suggestion_error_types: str
    model_name: str
    model_kwargs: typing.Optional[str]
    default_model_kwargs: typing.Dict[str, typing.Any]
    prompt_file: pathlib.Path
    judge_prompt_key_path: typing.Tuple[str, ...]
    suggestion_prompt_key_path: typing.Tuple[str, ...]
    prompt_generator_key_path: typing.Tuple[str, ...]
    original_prompt_file: pathlib.Path
    original_prompt_key_path: typing.Tuple[str, ...]
    generated_prompt_key_path: typing.Tuple[str, ...]


class JudgeGraphState(JudgeInputState, total=False):

    # Later graph nodes add these fields as each workflow stage completes.
    encounters: typing.List[typing.Any]
    icd10_descriptions: typing.Dict[str, str]
    analysis_results: typing.List[typing.Dict[str, typing.Any]]
    suggestion_results: typing.List[typing.Dict[str, typing.Any]]
    generated_prompt: typing.Dict[str, str]


if __name__ == "__main__":
    main()
