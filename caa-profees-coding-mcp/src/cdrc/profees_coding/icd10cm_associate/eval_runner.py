import argparse
import getpass
import json
import os

from langchain_core.globals import set_debug
from loguru import logger

from cdrc.common.data import JsonDataReader
from cdrc.common.evaluator import BaseEvaluator
from cdrc.common.metric import AccuracyCodeLevelMultiLabel, FScoreCodeLevelMultilabel, FScoreMultilabel, Latency
from cdrc.common.utils import combine_csv_files_into_xlsx, load_from_yaml, merge_config
from cdrc.profees_coding.icd10cm_associate.data import get_default_reader
from cdrc.profees_coding.icd10cm_associate.data.filters import AVAILABLE_FILTERS
from cdrc.profees_coding.icd10cm_associate.data.schema import InputSchema, OutputSchema
from cdrc.profees_coding.icd10cm_associate.metric.primary_dx_metrics import PrimaryDXMetrics
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v1.model import build_llm_only_merged_v1_graph
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v2.model import build_llm_only_merged_v2_graph
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v3.model import build_llm_only_merged_v3_graph
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v4.model import build_llm_only_merged_v4_graph
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v5.model import build_llm_only_merged_v5_graph
from cdrc.profees_coding.icd10cm_associate.models.llm_only_v1.model import build_llm_only_v1_graph
from cdrc.profees_coding.icd10cm_associate.models.llm_only_v2.model import build_llm_only_v2_graph
from cdrc.profees_coding.icd10cm_associate.models.llm_only_v3.model import build_llm_only_v3_graph
from cdrc.profees_coding.icd10cm_associate.models.utils import DESIGN, PROMPTS
from cdrc.profees_coding.icd10cm_associate.report_generation import ICD10CMAssociateCSVReportGenerator as CSVReportGenerator


def get_parser():

    # Initialize argument parser.
    parser = argparse.ArgumentParser()

    # Configuration section.
    parser.add_argument("--config_path", type=str, required=True, help="Path to the runner configuration YAML file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input dataset file.")

    # Data filtering options.
    parser.add_argument(
        "--filter_name", type=str, default=None, choices=list(AVAILABLE_FILTERS.keys()), help=f"Filter to apply: {list(AVAILABLE_FILTERS.keys())}."
    )
    parser.add_argument("--filter_kwargs", type=str, default="{}", help="JSON string of filter function arguments, e.g. '{\"min_age\":20}'.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples to process per batch.")

    # Model hyper-parameters.
    parser.add_argument("--design", type=DESIGN, required=True, help="Specify the model design for CPT E/M evaluation.")
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="outpatient",
        choices=[PROMPTS.OUTPATIENT, PROMPTS.ED, PROMPTS.INPATIENT],
        help="Prompt bank key to use in prompt YAML.",
    )
    parser.add_argument(
        "--regenerator_key",
        type=str,
        default="diagnosis_regenerator",
        help="Prompt bank key to use for the diagnosis regenerator node.",
    )
    parser.add_argument(
        "--regenerator_version",
        type=str,
        default="v1",
        help="Prompt version to use under the diagnosis regenerator prompt bank key.",
    )
    parser.add_argument(
        "--enhancer_type",
        type=str,
        default="none",
        choices=["none", "z", "r"],
        help="Optional target enhancer to run after the main prediction stage.",
    )
    parser.add_argument(
        "--enhancer_prompt_version",
        type=str,
        default="v1",
        help="Prompt version to use for the selected enhancer prompt bank.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for additional logging.")
    parser.add_argument(
        "--site",
        type=str,
        default="ambulatory",
        choices=["ambulatory", "inpatient", "ed", "auto"],
        help="Encounter site used to select the default reader.",
    )

    # Model settings and flags.
    parser.add_argument(
        "--cpt_procedure_filtering",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Flag for whether CPT procedures should be filtered using National Coverage Determinations (NCD) rules.",
    )
    parser.add_argument(
        "--hcpcs_procedure_filtering",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Flag for whether CPT procedures should be filtered using National Coverage Determinations (NCD) rules.",
    )
    parser.add_argument(
        "--populate_code_descriptions",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Flag for whether code descriptions should be automatically populated.",
    )
    parser.add_argument(
        "--validate_codes",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Flag for whether linked ICD-10-CM diagnosis codes should be validated and filtered by billability.",
    )
    parser.add_argument(
        "--diagnosis_regenerator",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Flag for whether diagnosis regenerator stage should run.",
    )
    parser.add_argument(
        "--export_intermediate_predictions",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help=(
            "Flag for whether intermediate association snapshots should be persisted in output payload. "
            "When enabled this adds per-stage intermediate output columns to sample_report.csv."
        ),
    )
    # Data files used by the association coding model.
    parser.add_argument(
        "--cpt_icd_mapping_path",
        type=str,
        default="./cdrc/profees_coding/icd10cm_associate/data/cpt_icd10_eligibility.json",
        help="Path to json file containing CPT to ICD LCD/NCD rules.",
    )
    parser.add_argument(
        "--hcpcs_icd_mapping_path",
        type=str,
        default="./cdrc/profees_coding/icd10cm_associate/data/hcpcs_icd10_eligibility.json",
        help="Path to json file containing HCPCS to ICD LCD/NCD rules.",
    )
    parser.add_argument(
        "--cpt_icd_noncovered_mapping_path",
        type=str,
        default="./cdrc/profees_coding/icd10cm_associate/data/2026/cpt_icd10_noncovered_2026.json",
        help="Path to json file containing CPT to explicitly noncovered ICD rules.",
    )
    parser.add_argument(
        "--hcpcs_icd_noncovered_mapping_path",
        type=str,
        default="./cdrc/profees_coding/icd10cm_associate/data/2026/hcpcs_icd10_noncovered_2026.json",
        help="Path to json file containing HCPCS to explicitly noncovered ICD rules.",
    )
    parser.add_argument(
        "--cpt_description_path",
        type=str,
        default="./cdrc/profees_coding/icd10cm_associate/data/cpt_descriptions.json",
        help="Path to json file containing CPT code descriptions.",
    )
    parser.add_argument(
        "--hcpcs_description_path",
        type=str,
        default="./cdrc/profees_coding/icd10cm_associate/data/hcpcs_descriptions.json",
        help="Path to json file containing CPT code descriptions.",
    )
    parser.add_argument(
        "--icd10_descriptions_path",
        type=str,
        default="./cdrc/profees_coding/icd10cm_associate/data/icd10_descriptions.json",
        help="Path to json file containing CPT code descriptions.",
    )
    parser.add_argument(
        "--icd10_validity_path",
        type=str,
        default="./cdrc/profees_coding/icd10cm_associate/data/2026/icd10_validity_2026.json",
        help="Path to json file containing ICD-10-CM validity indicators.",
    )
    parser.add_argument(
        "--icd10_billable_path",
        type=str,
        default="./cdrc/profees_coding/icd10cm_associate/data/2026/icd10_billable_2026.json",
        help="Path to json file containing ICD-10-CM billable indicators.",
    )

    # Output configurations.
    parser.add_argument("--run_name", type=str, default="medcoding-evaluation-test", help="Name for this run instance.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save output files.")
    parser.add_argument("--contributor", type=str, default=getpass.getuser(), help="Name of the person running the evaluation.")
    parser.add_argument("--previous_run", type=str, default=None, help="Path to previous run's sample report CSV file if resuming.")

    # Logging level.
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"],
        default="INFO",
        help="Set logging verbosity. Options: CRITICAL, ERROR, WARNING, INFO, DEBUG, TRACE.",
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

    # Parse arguments from command line and config files.
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()

    # Load and merge YAML config with CLI args and any additional unknown args
    config = load_from_yaml(args.config_path)
    args = merge_config(args, unknown_args, config)
    args.site = _infer_site_from_data(args.data_path)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Enable debugging if requested
    set_debug(args.debug)

    # Remove default handlers to avoid duplicate logs
    logger.remove()
    level = args.logging_level.upper()

    # Add formatted logger.
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    # Printing the log level, and experiment args.
    logger.info(f"Logger initialized with level: {level}")
    print("args:", args)

    # if a previous run is specified, no new generation is needed.
    if args.previous_run:
        association_model = None
        association_config = {}
    else:
        # Association Coding only models.
        if args.design == DESIGN.LLM_ONLY_V1:
            association_model, association_config = build_llm_only_v1_graph(args)
        elif args.design == DESIGN.LLM_ONLY_V2:
            association_model, association_config = build_llm_only_v2_graph(args)
        elif args.design == DESIGN.LLM_ONLY_V3:
            association_model, association_config = build_llm_only_v3_graph(args)

        # ICD-10 + Association Coding (joint) models.
        elif args.design == DESIGN.LLM_ONLY_MERGED_V1:
            association_model, association_config = build_llm_only_merged_v1_graph(args)
        elif args.design == DESIGN.LLM_ONLY_MERGED_V2:
            association_model, association_config = build_llm_only_merged_v2_graph(args)
        elif args.design == DESIGN.LLM_ONLY_MERGED_V3:
            association_model, association_config = build_llm_only_merged_v3_graph(args)
        elif args.design == DESIGN.LLM_ONLY_MERGED_V4:
            association_model, association_config = build_llm_only_merged_v4_graph(args)
        elif args.design == DESIGN.LLM_ONLY_MERGED_V5:
            association_model, association_config = build_llm_only_merged_v5_graph(args)

        else:  # If no valid model design is given throw an error.
            raise ValueError("Not a valid model design", args.design)

    # Initialize data reader with expected input/output schema.
    if getattr(args, "site", "ambulatory") in {"inpatient"}:
        data_reader = get_default_reader()
    else:
        data_reader = JsonDataReader(input_schema=InputSchema, output_schema=OutputSchema)

    # Define all metrics to compute for the evaluation
    metrics = {
        "icd10_f1_score": FScoreMultilabel(label_key="diagnosis.code"),
        # Evaluating the combined accuracy of E/M codes + procedures linked diagnoses.
        "combined_association_accuracy": AccuracyCodeLevelMultiLabel(label_key=["em.linked_diagnosis.code", "procedure.linked_diagnosis.code"]),
        "combined_primary_association_accuracy": PrimaryDXMetrics(label_key=["em.linked_diagnosis", "procedure.linked_diagnosis"]),
        "combined_association_f1_score": FScoreCodeLevelMultilabel(label_key=["em.linked_diagnosis.code", "procedure.linked_diagnosis.code"]),
        # Evaluating the accuracy of the CPT-E/M linked diagnoses.
        "em_association_accuracy": AccuracyCodeLevelMultiLabel(label_key="em.linked_diagnosis.code"),
        "em_primary_association_accuracy": PrimaryDXMetrics(label_key="em.linked_diagnosis"),
        "em_association_f1_score": FScoreCodeLevelMultilabel(label_key="em.linked_diagnosis.code"),
        # Evaluating the accuracy of the HCPCS procedure linked diagnoses.
        "procedure_association_accuracy": AccuracyCodeLevelMultiLabel(label_key="procedure.linked_diagnosis.code"),
        "procedure_primary_association_accuracy": PrimaryDXMetrics(label_key="procedure.linked_diagnosis"),
        "procedure_association_f1_score": FScoreCodeLevelMultilabel(label_key="procedure.linked_diagnosis.code"),
        "latency": Latency(),
    }

    # Read gold data with filters if specified
    _, gold_data = data_reader.read_data(args.data_path, filter_function=AVAILABLE_FILTERS.get(args.filter_name), **args.filter_kwargs)

    # Initialize CSV report generator for this run
    csv_report_gen = CSVReportGenerator(save_path=os.path.join(args.output_dir, args.run_name))

    # Run model evaluation.
    evaluator = BaseEvaluator(
        data_reader=data_reader,
        model=association_model,
        model_config=association_config,
        metrics=metrics,
        report_generator=csv_report_gen,
        batch_size=args.batch_size,
        previous_run_path=args.previous_run,
        cache_dir=os.path.join(args.output_dir, args.run_name),
    )

    # Run the evaluation: generates metrics, per-sample report, summary
    metrics_result, sample_report, summary_reports = evaluator.run(
        data_path=args.data_path, filter_function=AVAILABLE_FILTERS.get(args.filter_name), **args.filter_kwargs
    )

    # Combine any CSV reports into a XLSX summary file for easier sharing/consumption.
    combine_csv_files_into_xlsx(dir_path=os.path.join(args.output_dir, args.run_name), metadata=vars(args))


if __name__ == "__main__":
    main()
