import argparse
import atexit
import getpass
import json
import os
import time
from pathlib import Path

from langchain_core.globals import set_debug
from loguru import logger

from cdrc.common.data import JsonDataReader
from cdrc.common.evaluator import BaseEvaluator
from cdrc.common.metric import AccuracyMultiLabel, FScoreMultilabel, Latency
from cdrc.common.report_generation import CSVReportGenerator
from cdrc.common.utils import combine_csv_files_into_xlsx, load_from_yaml, merge_config
from cdrc.profees_coding.em.data import get_default_reader
from cdrc.profees_coding.em.data.filters import AVAILABLE_FILTERS
from cdrc.profees_coding.em.data.schema import InputSchema, OutputSchema
from cdrc.profees_coding.em.metric import ComplexityLevelMetrics
from cdrc.profees_coding.em.models.medprompt_detailed_critics.model import (
    build_medprompt_cpt_em_graph as build_medprompt_detailed_critics_cpt_em_graph,
)
from cdrc.profees_coding.em.models.medprompt_detailed_mdm.model import build_medprompt_cpt_em_graph as build_medprompt_detailed_mdm_cpt_em_graph
from cdrc.profees_coding.em.models.medprompt_detailed_multi.model import build_medprompt_cpt_em_graph as build_medprompt_detailed_multi_cpt_em_graph
from cdrc.profees_coding.em.models.reprediction_critics_only.model import build_reprediction_critics_only_graph
from cdrc.profees_coding.em.models.reprediction_mdm_only.model import build_reprediction_mdm_only_graph
from cdrc.profees_coding.em.models.utils import DESIGN


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
        choices=["auto", "ambulatory", "ed", "inpatient"],
        default="auto",
        help="Select prompt/data behavior for ambulatory, ED, or inpatient workflows.",
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
        # default="/home/chantan/clinical-rev-cycle/cdrc/profees_coding/em/models/medprompt_detailed_multi/checkpoints/checkpoint.sqlite",
    )

    return parser


def _infer_site_from_data(data_path: str) -> str:
    def normalize(value: str) -> str:
        return value.strip().lower()

    if data_path.casefold().endswith(".jsonl"):
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
    if args.site in {"ed", "inpatient"}:
        args.encounter_prompt_key = "encounter_type_ed" if args.site == "ed" else "encounter_type_inpatient"
        args.time_statement_prompt_key = "time_statement_ed" if args.site == "ed" else "time_statement_inpatient"
        args.mdm_prompt_key = "medprompt_inference_ed" if args.site == "ed" else "medprompt_inference_inpatient"
        args.rci_multi_critic_prompt_key = "rci_multi_critic_ed" if args.site == "ed" else "rci_multi_critic_inpatient"

    if args.checkpoint_path:
        Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    os.makedirs(args.output_dir, exist_ok=True)
    set_debug(args.debug)
    print("args:", args)
    reprediction = False
    if args.design == DESIGN.REPREDICTION_MDM_ONLY or args.design == DESIGN.REPREDICTION_CRITICS_ONLY:
        reprediction = True

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
        cpt_model = None
        cpt_config = {}
    else:
        # compile the graph
        if args.design == DESIGN.MEDPROMPT_DETAILED_MULTI:
            cpt_model, cpt_config = build_medprompt_detailed_multi_cpt_em_graph(args)
        elif args.design == DESIGN.MEDPROMPT_DETAILED_CRITICS:
            cpt_model, cpt_config = build_medprompt_detailed_critics_cpt_em_graph(args)
        elif args.design == DESIGN.MEDPROMPT_DETAILED_MDM:
            cpt_model, cpt_config = build_medprompt_detailed_mdm_cpt_em_graph(args)
        elif args.design == DESIGN.REPREDICTION_MDM_ONLY:
            cpt_model, cpt_config = build_reprediction_mdm_only_graph(args)
        elif args.design == DESIGN.REPREDICTION_CRITICS_ONLY:
            cpt_model, cpt_config = build_reprediction_critics_only_graph(args)
        else:
            raise NotImplementedError(f"the design {args.design} is not implemented yet!")

    if args.site in {"inpatient"}:
        data_reader = get_default_reader()
    else:
        data_reader = JsonDataReader(input_schema=InputSchema, output_schema=OutputSchema)

    metrics = {
        "cpt_accuracy": AccuracyMultiLabel(label_key="em.code"),
        "encounter_type_accuracy": AccuracyMultiLabel(label_key="em.encounter_type.encounter_type"),
        "f1_score": FScoreMultilabel(label_key="em.code"),
        # "f1_weighted_all": FScoreMultilabel(label_key="em.code", weight_by="all"),
        "latency": Latency(),
        "mdm_complexity_metrics": ComplexityLevelMetrics(label_key="em.mdm_complexity.level"),
    }

    # if args.weighted_by_provider is not None:
    #     provider = args.weighted_by_provider
    #     metrics.update(
    #         {
    #             f"f1_score_weighted_by_{provider}": FScoreMultilabel(label_key="em.code", weight_by=provider),
    #         }
    #     )

    # Read gold data with filters if specified
    _, gold_data = data_reader.read_data(args.data_path, filter_function=AVAILABLE_FILTERS.get(args.filter_name), **args.filter_kwargs)
    intermediate = ["problem_complexity", "data_complexity", "risk_of_complications"]
    gold_has_intermediate = all([all([getattr(gold_instance, elem, None) is not None for elem in intermediate]) for gold_instance in gold_data])
    gold_has_intermediate = True

    reprediction = False
    if args.design == DESIGN.REPREDICTION_MDM_ONLY or args.design == DESIGN.REPREDICTION_CRITICS_ONLY:
        reprediction = True

    add_intermediate_metrics = args.design == DESIGN.MEDPROMPT_DETAILED_MULTI
    if getattr(DESIGN, "MEDPROMPT_DETAILED_MDM", None) is not None:
        add_intermediate_metrics = add_intermediate_metrics or args.design == DESIGN.MEDPROMPT_DETAILED_MDM

    if add_intermediate_metrics and gold_has_intermediate:
        metrics.update(
            {
                "mdm_complexity_metrics": ComplexityLevelMetrics(label_key="em.mdm_complexity.level"),
                "problem_complexity_metrics": ComplexityLevelMetrics(label_key="em.problem_complexity.level"),
                "data_complexity_metrics": ComplexityLevelMetrics(label_key="em.data_complexity.level"),
                "risk_of_complications_metrics": ComplexityLevelMetrics(label_key="em.risk_of_complications.level"),
            }
        )
    elif getattr(DESIGN, "MEDPROMPT", None) is not None and args.design == DESIGN.MEDPROMPT:
        metrics.update({"mdm_complexity_metrics": ComplexityLevelMetrics(label_key="em.mdm_complexity.level")})

    csv_report_gen = CSVReportGenerator(save_path=os.path.join(args.output_dir, args.run_name))

    # set the evaluator and run evaluation
    evaluator = BaseEvaluator(
        data_reader=data_reader,
        model=cpt_model,
        model_config=cpt_config,
        metrics=metrics,
        report_generator=csv_report_gen,
        batch_size=args.batch_size,
        previous_run_path=args.previous_run,
        cache_dir=os.path.join(args.output_dir, args.run_name),
        reprediction=reprediction,
    )

    metrics_result, sample_report, summary_reports = evaluator.run(
        data_path=args.data_path, filter_function=AVAILABLE_FILTERS.get(args.filter_name), **args.filter_kwargs
    )

    combine_csv_files_into_xlsx(dir_path=os.path.join(args.output_dir, args.run_name), metadata=vars(args))


if __name__ == "__main__":
    main()
