import argparse
import atexit
import getpass
import json
import os
import time

from langchain_core.globals import set_debug
from loguru import logger

from cdrc.common.data import JsonDataReader
from cdrc.common.evaluator import BaseEvaluator
from cdrc.common.metric import AccuracyMultiLabel, FScoreMultilabel, Latency
from cdrc.common.report_generation import CSVReportGenerator
from cdrc.common.utils import combine_csv_files_into_xlsx, load_from_yaml, merge_config
from cdrc.profees_coding.em.metric import ComplexityLevelMetrics
from cdrc.profees_coding.icd10cm.data import get_default_reader
from cdrc.profees_coding.icd10cm.data.filters import AVAILABLE_FILTERS
from cdrc.profees_coding.icd10cm.data.schema import InputSchema, OutputSchema
from cdrc.profees_coding.icd10cm.metric.pipeline_recall_at_k import PipelineRecallAtK
from cdrc.profees_coding.icd10cm.metric.primary_dx_metrics import PrimaryDXMetrics
from cdrc.profees_coding.icd10cm.models.llm_multistage.model import (
    build_llm_multistage_graph,
    build_llm_singlestage_enhanced_graph,
    build_llm_singlestage_graph,
)
from cdrc.profees_coding.icd10cm.models.utils import DESIGN
from cdrc.profees_coding.icd10cm.models.workflow.model import build_workflow_graph


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="path to runner config yaml file. either this or the rest of the arguments need to be specified.",
    )
    # Prompt
    parser.add_argument(
        "--extract_and_queries_prompt_version",
        type=str,
        required=False,
        help="Version of the prompt for diagnosis extraction and query generation.",
    )
    parser.add_argument(
        "--llm_only_prompt_version",
        type=str,
        required=False,
        help="Version of the prompt for llm only component.",
    )
    parser.add_argument(
        "--code_reranker_prompt_version",
        type=str,
        required=False,
        help="Version of the prompt for code reranking.",
    )
    parser.add_argument(
        "--problem_extraction_prompt_version",
        type=str,
        required=False,
        help="Version of the prompt for problem extraction.",
    )
    parser.add_argument(
        "--coding_prompt_version",
        type=str,
        required=False,
        help="Version of the prompt for coding.",
    )
    parser.add_argument(
        "--z_code_enhancer_prompt_version",
        type=str,
        required=False,
        help="Version of the prompt for z code enhancer.",
    )
    parser.add_argument(
        "--search_top_k",
        type=int,
        required=False,
        default=10,
        help="The number of candidate codes to keep for each query (there can be multiple queries per extracted problem/diagnosis)",
    )

    # Input
    parser.add_argument("--data_path", type=str, required=True, help="path to the dataset file")
    # Data filtering options
    parser.add_argument(
        "--filter_name",
        type=str,
        default=None,
        choices=list(AVAILABLE_FILTERS.keys()),
        help=f"Name of filter to apply. Available filters: {list(AVAILABLE_FILTERS.keys())}",
    )
    parser.add_argument("--filter_kwargs", type=str, default="{}", help="JSON string of arguments to pass to the filter function")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")

    # Model
    parser.add_argument("--design", type=DESIGN, required=True, help="cpt e/m model design")
    parser.add_argument("--debug", action="store_true", help="debug mode?")
    parser.add_argument(
        "--site",
        type=str,
        default="ambulatory",
        choices=["ambulatory", "inpatient", "ed", "auto"],
        help="Encounter site used to select the default reader.",
    )

    # Loggers
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"],
        default="INFO",
        help="Set logging verbosity. Options: CRITICAL, ERROR, WARNING, INFO, DEBUG, TRACE.",
    )

    # Output
    parser.add_argument("--run_name", type=str, default="medcoding-evaluation-test", help="Run name")
    parser.add_argument("--output_dir", type=str, default="./results", help="Path to output dir")
    parser.add_argument("--contributor", type=str, default=getpass.getuser(), help="contributor name")

    parser.add_argument(
        "--previous_run",
        type=str,
        default=None,
        help="running evaluation on a previous run? remember to set this to the sample report csv file path.",
    )
    parser.add_argument("--n_instances", type=int, default=-1, help="Number of instances to run model on. -1 means run all instances.")
    return parser


def _infer_site_from_data(data_path: str) -> str:
    def normalize(value: str) -> str:
        return value.strip().lower()

    try:
        if data_path.lower().endswith(".jsonl"):
            with open(data_path, "r") as f:
                line = f.readline().strip()
                if not line:
                    return "ambulatory"
                payload = json.loads(line)
        else:
            with open(data_path, "r") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                if not payload:
                    return "ambulatory"
                payload = payload[0]
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return "ambulatory"

    input_payload = payload.get("input", payload)
    metadata = input_payload.get("encounter_metadata") or {}
    encounter_type = normalize(str(metadata.get("type", "")))

    if "inpatient" in encounter_type:
        return "inpatient"
    if "emergency" in encounter_type or encounter_type in {"ed", "er"}:
        return "ed"
    if "ambulatory" in encounter_type or "outpatient" in encounter_type:
        return "ambulatory"
    return "ambulatory"


def main():
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()

    config = load_from_yaml(args.config_path)
    args = merge_config(args, unknown_args, config)
    args.site = _infer_site_from_data(args.data_path)

    os.makedirs(args.output_dir, exist_ok=True)
    set_debug(args.debug)
    # ------------------ Logging setup ------------------
    # Remove default handlers to avoid duplicate logs
    logger.remove()
    level = args.logging_level.upper()
    # Add console sink with nice formatting
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    logger.info(f"Logger initialized with level: {level}")
    # ----------------------------------------------------

    os.makedirs(args.output_dir, exist_ok=True)
    set_debug(args.debug)
    logger.debug(f"Arguments: {args}")
    # if a previous run is specified, no new generation would be performed!
    if args.previous_run:
        icd10cm_model = None
        icd10cm_config = {}
    else:
        # compile the graph
        if args.design == DESIGN.WORKFLOW:
            icd10cm_model, icd10cm_config = build_workflow_graph(args)
        elif args.design == DESIGN.LLM_MULTI_STAGE:
            icd10cm_model, icd10cm_config = build_llm_multistage_graph(args)
        elif args.design == DESIGN.LLM_SINGLE_STAGE:
            icd10cm_model, icd10cm_config = build_llm_singlestage_graph(args)
        elif args.design == DESIGN.LLM_SINGLE_STAGE_ENHANCED:
            icd10cm_model, icd10cm_config = build_llm_singlestage_enhanced_graph(args)
        else:
            raise ValueError(f"Unsupported design: {args.design}")

    if getattr(args, "site", "ambulatory") in {"inpatient"}:
        data_reader = get_default_reader()
    else:
        data_reader = JsonDataReader(input_schema=InputSchema, output_schema=OutputSchema)

    metrics = {
        "icd10cm_accuracy": AccuracyMultiLabel(label_key="diagnosis.code"),
        "primary_dx_accuracy": PrimaryDXMetrics(label_key="diagnosis"),
        "f1_score": FScoreMultilabel(label_key="diagnosis.code"),
        "latency": Latency(),
        "pipeline_recall": PipelineRecallAtK(label_key="diagnosis.code", ks=(1, 5, 10, 30, 50, 70, 100)),
    }

    csv_report_gen = CSVReportGenerator(save_path=os.path.join(args.output_dir, args.run_name))

    # set the evaluator and run evaluation
    evaluator = BaseEvaluator(
        data_reader=data_reader,
        model=icd10cm_model,
        model_config=icd10cm_config,
        metrics=metrics,
        report_generator=csv_report_gen,
        batch_size=args.batch_size,
        previous_run_path=args.previous_run,
        cache_dir=os.path.join(args.output_dir, args.run_name),
    )

    metrics_result, sample_report, summary_reports = evaluator.run(
        data_path=args.data_path, filter_function=AVAILABLE_FILTERS.get(args.filter_name), **args.filter_kwargs
    )

    combine_csv_files_into_xlsx(dir_path=os.path.join(args.output_dir, args.run_name), metadata=vars(args))


if __name__ == "__main__":
    main()
