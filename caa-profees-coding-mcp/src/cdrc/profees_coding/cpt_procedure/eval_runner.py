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
from cdrc.profees_coding.cpt_procedure.data import get_default_reader
from cdrc.profees_coding.cpt_procedure.data.filters import AVAILABLE_FILTERS
from cdrc.profees_coding.cpt_procedure.data.schema import InputSchema, OutputSchema
from cdrc.profees_coding.cpt_procedure.models.llm_only.model import build_llm_only_graph
from cdrc.profees_coding.cpt_procedure.models.ner_rag.model import build_ner_llm_graph, build_ner_llm_rag_graph, build_ner_rag_graph
from cdrc.profees_coding.cpt_procedure.models.reprediction_llm_only.model import build_reprediction_llm_only_graph
from cdrc.profees_coding.cpt_procedure.models.utils import DESIGN


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="path to runner config yaml file. either this or the rest of the arguments need to be specified.",
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

    # Metrics
    parser.add_argument(
        "--weighted_by_provider",
        type=str,
        choices=["hudson", "uni_missouri", "st_johns"],
        default=None,
        help="Add weighted metrics with prior label distribution from a specific provider.",
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

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/mnt/home/zezhai/projects/clinical-rev-cycle/checkpoints/checkpoint.sqlite",
    )

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
    # if a previous run is specified, no new generation would be performed!
    if args.previous_run:
        procedure_model = None
        procedure_config = {}
    elif args.design == DESIGN.LLM_ONLY:
        # compile the graph
        procedure_model, procedure_config = build_llm_only_graph(args)
    elif args.design == DESIGN.NER_RAG:
        # compile the graph
        procedure_model, procedure_config = build_ner_rag_graph(args)
    elif args.design == DESIGN.NER_LLM:
        # compile the graph
        procedure_model, procedure_config = build_ner_llm_graph(args)
    elif args.design == DESIGN.NER_LLM_RAG:
        # compile the graph
        procedure_model, procedure_config = build_ner_llm_rag_graph(args)
    elif args.design == DESIGN.REPREDICTION_LLM_ONLY:
        # compile the graph
        procedure_model, procedure_config = build_reprediction_llm_only_graph(args)

    if getattr(args, "site", "ambulatory") in {"inpatient"}:
        data_reader = get_default_reader()
    else:
        data_reader = JsonDataReader(input_schema=InputSchema, output_schema=OutputSchema)

    metrics = {
        "accuracy": AccuracyMultiLabel(label_key="procedure.code"),
        "f1_score": FScoreMultilabel(label_key="procedure.code"),
        # "f1_weighted_all": FScoreMultilabel(label_key="procedure.code", weight_by="all"),
        "latency": Latency(),
    }

    # if args.weighted_by_provider is not None:
    #     provider = args.weighted_by_provider
    #     metrics.update(
    #         {
    #             f"f1_score_weighted_by_{provider}": FScoreMultilabel(label_key="procedure.code", weight_by=provider),
    #         }
    #     )

    # Read gold data with filters if specified
    _, gold_data = data_reader.read_data(args.data_path, filter_function=AVAILABLE_FILTERS.get(args.filter_name), **args.filter_kwargs)

    csv_report_gen = CSVReportGenerator(save_path=os.path.join(args.output_dir, args.run_name))

    # set the evaluator and run evaluation
    evaluator = BaseEvaluator(
        data_reader=data_reader,
        model=procedure_model,
        model_config=procedure_config,
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

    # if args.design == DESIGN.CHRG_ASST:
    #     server_instance.stop()


if __name__ == "__main__":
    main()
