import argparse
import getpass
import json
import os

from langchain_core.globals import set_debug
from loguru import logger

from cdrc.common.data import JsonDataReader
from cdrc.common.evaluator import BaseEvaluator
from cdrc.common.report_generation import CSVReportGenerator
from cdrc.common.utils import combine_csv_files_into_xlsx, load_from_yaml, merge_config
from cdrc.profees_coding.hcpcs_codes.data import get_default_reader
from cdrc.profees_coding.hcpcs_codes.data.filters import AVAILABLE_FILTERS
from cdrc.profees_coding.hcpcs_codes.data.schema import InputSchema, OutputSchema
from cdrc.profees_coding.hcpcs_codes.evaluation.metric import PrecRecF1Metric
from cdrc.profees_coding.hcpcs_codes.models.baseline_llm.model import build_baseline_llm_graph
from cdrc.profees_coding.hcpcs_codes.models.helix.model import build_helix_graph
from cdrc.profees_coding.hcpcs_codes.models.utils import MODEL_VARIANT


def _parse_optional_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, received '{value}'.")


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
        default=argparse.SUPPRESS,
        choices=list(AVAILABLE_FILTERS.keys()),
        help=f"Name of filter to apply. Available filters: {list(AVAILABLE_FILTERS.keys())}",
    )
    parser.add_argument(
        "--filter_kwargs",
        type=str,
        default=argparse.SUPPRESS,
        help="JSON string of arguments to pass to the filter function",
    )
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS, help="batch size")

    # Model
    parser.add_argument("--debug", action="store_true", help="debug mode?")
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=[MODEL_VARIANT.BASELINE_LLM.value, MODEL_VARIANT.HELIX.value],
        default=argparse.SUPPRESS,
        help="HCPCS model variant to run.",
    )
    parser.add_argument(
        "--retrieval_enabled",
        nargs="?",
        const=True,
        type=_parse_optional_bool,
        default=argparse.SUPPRESS,
        help="Enable Helix retrieval augmentation. Accepts optional bool value.",
    )
    parser.add_argument(
        "--awv_precheck_mode",
        type=str,
        choices=["deterministic", "llm"],
        default=argparse.SUPPRESS,
        help="AWV precheck mode for Helix.",
    )
    parser.add_argument("--g2211_prompt_version", type=str, default=None, help="Override the G2211 branch prompt version.")
    parser.add_argument("--awv_prompt_version", type=str, default=None, help="Override the AWV classifier branch prompt version.")
    parser.add_argument(
        "--awv_eligibility_prompt_version",
        type=str,
        default=argparse.SUPPRESS,
        help="Override the AWV eligibility precheck prompt version.",
    )
    parser.add_argument(
        "--other_prompt_version",
        type=str,
        default=argparse.SUPPRESS,
        help="Override the facility-context other-HCPCS branch prompt version.",
    )
    parser.add_argument(
        "--hcpcs_in_scope",
        type=str,
        default=argparse.SUPPRESS,
        help="Comma-separated HCPCS branch families to run: all, g2211, awv, or other_hcpcs.",
    )
    parser.add_argument(
        "--input_ablation_mode",
        type=str,
        choices=["none", "soap_note_only"],
        default=argparse.SUPPRESS,
        help="Optional model-visible input ablation mode.",
    )

    # Loggers
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"],
        default=argparse.SUPPRESS,
        help="Set logging verbosity. Options: CRITICAL, ERROR, WARNING, INFO, DEBUG, TRACE.",
    )

    # Output
    parser.add_argument("--run_name", type=str, default=argparse.SUPPRESS, help="Run name")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS, help="Path to output dir")
    parser.add_argument("--contributor", type=str, default=argparse.SUPPRESS, help="contributor name")

    parser.add_argument(
        "--previous_run",
        type=str,
        default=argparse.SUPPRESS,
        help="running evaluation on a previous run? remember to set this to the sample report csv file path.",
    )

    parser.add_argument(
        "--include_justification",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Include debug justifications for HCPCS codes (e.g., for SME review).",
    )

    return parser


def _apply_runtime_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "filter_name": None,
        "filter_kwargs": {},
        "batch_size": 1,
        "debug": False,
        "model_variant": MODEL_VARIANT.HELIX.value,
        "retrieval_enabled": False,
        "awv_precheck_mode": "deterministic",
        "hcpcs_in_scope": "all",
        "input_ablation_mode": "none",
        "logging_level": "INFO",
        "run_name": "g2211-evaluation-test",
        "output_dir": "./results",
        "contributor": getpass.getuser(),
        "previous_run": None,
        "include_justification": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def _build_model_graph(args: argparse.Namespace):
    model_variant = str(getattr(args, "model_variant", MODEL_VARIANT.HELIX.value) or MODEL_VARIANT.HELIX.value)
    if model_variant == MODEL_VARIANT.BASELINE_LLM.value:
        return build_baseline_llm_graph(args)
    if model_variant == MODEL_VARIANT.HELIX.value:
        return build_helix_graph(args)
    raise ValueError(f"Unsupported HCPCS model_variant '{model_variant}'.")


def _load_first_payload(data_path: str) -> dict | None:
    try:
        if data_path.lower().endswith(".jsonl"):
            with open(data_path, "r") as f:
                line = f.readline().strip()
                if not line:
                    return None
                payload = json.loads(line)
        else:
            with open(data_path, "r") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                if not payload:
                    return None
                payload = payload[0]
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None

    return payload if isinstance(payload, dict) else None


def _infer_site_from_data(data_path: str) -> str:
    def normalize(value: str) -> str:
        return value.strip().lower()

    payload = _load_first_payload(data_path)
    if payload is None:
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


def _infer_gold_container_shape(data_path: str) -> str:
    payload = _load_first_payload(data_path)
    if payload is None:
        return "dict"

    gold_payload = payload.get("gold")
    return "list" if isinstance(gold_payload, list) else "dict"


def _select_data_reader(data_path: str, site: str):
    gold_shape = _infer_gold_container_shape(data_path)
    if gold_shape == "list":
        logger.info("Using ConcurrentHCPCSReader for dataset '{}' (site='{}', gold_shape='list').", data_path, site)
        return get_default_reader()

    if site == "inpatient":
        logger.warning(
            "Inferred inpatient site for dataset '{}' but gold payload is single-record; falling back to JsonDataReader.",
            data_path,
        )
    else:
        logger.info("Using JsonDataReader for dataset '{}' (site='{}', gold_shape='dict').", data_path, site)
    return JsonDataReader(input_schema=InputSchema, output_schema=OutputSchema)


def main():
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()

    config = load_from_yaml(args.config_path)
    args = merge_config(args, unknown_args, config)
    args = _apply_runtime_defaults(args)
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
        hcpcs_model = None
        hcpcs_config = {}
    else:
        hcpcs_model, hcpcs_config = _build_model_graph(args)
    data_reader = _select_data_reader(args.data_path, getattr(args, "site", "ambulatory"))

    metrics = {
        "prec_rec_f1": PrecRecF1Metric(),
    }

    # Read gold data with filters if specified
    _, gold_data = data_reader.read_data(args.data_path, filter_function=AVAILABLE_FILTERS.get(args.filter_name), **args.filter_kwargs)

    csv_report_gen = CSVReportGenerator(save_path=os.path.join(args.output_dir, args.run_name))

    # set the evaluator and run evaluation
    evaluator = BaseEvaluator(
        data_reader=data_reader,
        model=hcpcs_model,
        model_config=hcpcs_config,
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
