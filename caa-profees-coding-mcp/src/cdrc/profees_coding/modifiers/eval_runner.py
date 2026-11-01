import argparse
import atexit
import getpass
import json
import os
import time
from importlib import util
from pathlib import Path

from langchain_core.globals import set_debug
from loguru import logger

from cdrc.common.data.reader import JsonDataReader
from cdrc.common.evaluator import BaseEvaluator
from cdrc.common.metric import Latency
from cdrc.common.metric.accuracy import AccuracyPairedCode
from cdrc.common.report_generation import CSVReportGenerator
from cdrc.common.utils import combine_csv_files_into_xlsx, load_from_yaml, merge_config
from cdrc.profees_coding.modifiers.data import get_default_reader
from cdrc.profees_coding.modifiers.data.filters import AVAILABLE_FILTERS
from cdrc.profees_coding.modifiers.data.schema import InputSchema, OutputSchema
from cdrc.profees_coding.modifiers.metric.modifier_metrics import CombinedModifierMicroMetric, ModifierMetric
from cdrc.profees_coding.modifiers.models.decision_rules.model import build_decision_rules_graph
from cdrc.profees_coding.modifiers.models.llm_only.model import build_llm_only_graph
from cdrc.profees_coding.modifiers.models.parallel_modifiers.model import build_parallel_modifiers_graph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v2.model import build_parallel_modifiers_graph as build_parallel_modifiers_v2_graph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v3.model import build_parallel_modifiers_graph as build_parallel_modifiers_v3_graph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v4.model import build_parallel_modifiers_graph as build_parallel_modifiers_v4_graph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v5.model import build_parallel_modifiers_graph as build_parallel_modifiers_v5_graph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.model import build_parallel_modifiers_graph as build_parallel_modifiers_v6_graph
from cdrc.profees_coding.modifiers.models.utils import DESIGN


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
    parser.add_argument("--design", type=DESIGN, required=True, help="modifier model design")
    parser.add_argument("--debug", action="store_true", help="debug mode?")
    parser.add_argument(
        "--modifiers_in_scope",
        type=str,
        default=None,
        help="Comma-separated list of modifier codes to include, or 'all' for all known modifiers.",
    )
    parser.add_argument(
        "--ehr_data_in_scope",
        type=str,
        default=argparse.SUPPRESS,
        help=(
            "Comma-separated list of structured EHR fields to include for parallel_modifiers_v6. "
            "Use 'default'/'all' for the v6 default, or 'none' to suppress structured EHR context."
            "Choices include: patient_sex, patient_age, payer_name, encounter_metadata, health_plan, facility_info, provider_info, lab_info, anaesthesia_staff_info"
        ),
    )
    parser.add_argument(
        "--modifier_relevant_ehr_scope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For parallel_modifiers_v6, narrow structured EHR context per modifier using the ablation-based mapping. "
            "Use --no-modifier_relevant_ehr_scope to restore the previous mixed behavior where generic nodes use ehr_data_in_scope "
            "and specialized PO/PN and QK/QX nodes pass only their original targeted fields."
        ),
    )
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
        modifier_model = None
        modifier_config = {}
    else:
        # compile the graph
        if args.design == DESIGN.LLM_ONLY:
            modifier_model, modifier_config = build_llm_only_graph(args)
        elif args.design == DESIGN.PARALLEL_MODIFIERS:
            modifier_model, modifier_config = build_parallel_modifiers_graph(args)
        elif args.design == DESIGN.PARALLEL_MODIFIERS_V2:
            modifier_model, modifier_config = build_parallel_modifiers_v2_graph(args)
        elif args.design == DESIGN.PARALLEL_MODIFIERS_V3:
            modifier_model, modifier_config = build_parallel_modifiers_v3_graph(args)
        elif args.design == DESIGN.PARALLEL_MODIFIERS_V4:
            modifier_model, modifier_config = build_parallel_modifiers_v4_graph(args)
        elif args.design == DESIGN.PARALLEL_MODIFIERS_V5:
            modifier_model, modifier_config = build_parallel_modifiers_v5_graph(args)
        elif args.design == DESIGN.PARALLEL_MODIFIERS_V6:
            modifier_model, modifier_config = build_parallel_modifiers_v6_graph(args)
        elif args.design == DESIGN.DECISION_RULES:
            modifier_model, modifier_config = build_decision_rules_graph(args)
        else:
            raise NotImplementedError(f"the design {args.design} is not implemented yet!")
    if getattr(args, "site", "ambulatory") in {"inpatient"}:
        data_reader = get_default_reader()
    else:
        data_reader = JsonDataReader(input_schema=InputSchema, output_schema=OutputSchema)

    metrics = {
        "em_accuracy": AccuracyPairedCode(label_key=["em.modifiers.code"], prefix="code"),
        "procedure_accuracy": AccuracyPairedCode(label_key=["procedure.modifiers.code"], prefix="code"),
        "em_modifier_metrics": ModifierMetric(label_key="em.modifiers.code", scope=getattr(args, "modifiers_in_scope", None)),
        "procedure_modifier_metrics": ModifierMetric(label_key="procedure.modifiers.code", scope=getattr(args, "modifiers_in_scope", None)),
        "combined_modifier_micro_metrics": CombinedModifierMicroMetric(
            em_label_key="em.modifiers.code", procedure_label_key="procedure.modifiers.code", scope=getattr(args, "modifiers_in_scope", None)
        ),
        "latency": Latency(),
    }

    csv_report_gen = CSVReportGenerator(save_path=os.path.join(args.output_dir, args.run_name))

    # set the evaluator and run evaluation
    evaluator = BaseEvaluator(
        data_reader=data_reader,
        model=modifier_model,
        model_config=modifier_config,
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
