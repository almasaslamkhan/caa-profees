# Release: 26.06
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, Literal, Optional

from loguru import logger
from pyhocon import ConfigTree
from pyhocon.exceptions import ConfigMissingException

# from aoa_agent.profees_coding_agent.agent_obs.metrics import METRICS_REPORTING_NAMESPACE, counter, gauge, timeit
from cdrc.common.llm_utils import TokenStats, TokenTracker
from cdrc.config.utils import settings  # ConfigTree
from cdrc.profees_coding.cpt_procedure.models.ner_rag.model import build_ner_llm_graph as build_cpt_procedure_graph
from cdrc.profees_coding.deploy.api.utils import (
    CareSetting,
    CDA_note,
    DeliveryVehicle,
    Diagnosis,
    EMItem,
    InputData,
    MetricNames,
    ModelData,
    Procedure,
    ProfeesResponse,
    build_hcpcs_input,
    build_icd_associate_input,
    build_modifiers_input,
    calculate_patient_age,
    convert_orders_to_display,
    derive_care_setting,
    exclude_depression_screening_when_awv_present,
    exclude_g2211_when_base_em_missing,
    exclude_preventive_em_when_awv_present,
    flatten_multiline,
    normalize_nested_models,
    predict_patient_type,
    sanitize_for_prod,
)
from cdrc.profees_coding.em.models.medprompt_detailed_multi.model import build_medprompt_cpt_em_graph as build_em_graph
from cdrc.profees_coding.hcpcs_codes.models.helix.model import build_helix_graph as build_hcpcs_graph
from cdrc.profees_coding.icd10cm_associate.models.llm_only_merged_v5.model import build_llm_only_merged_v5_graph as build_icd10cm_associate_graph
from cdrc.profees_coding.modifiers.models.parallel_modifiers_v6.model import build_parallel_modifiers_graph as build_modifiers_graph

Mode = Literal["SDS", "LOCAL", "DEV", "PROD"]


async def timed(stage: str, name: str, coro: Awaitable[Any], *, encounter_id: str, request_id: str):
    start = time.perf_counter()
    logger.info("Encounter ID: {} Request ID: {} {} {} START", encounter_id, request_id, stage, name)
    try:
        return await coro
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Encounter ID: {} Request ID: {} {} {} END in {:.3f}s", encounter_id, request_id, stage, name, elapsed)


async def limited_timed(stage: str, name: str, coro: Awaitable[Any], sem: asyncio.Semaphore, *, encounter_id: str, request_id: str):
    async with sem:
        return await timed(stage, name, coro, encounter_id=encounter_id, request_id=request_id)


async def safe_await(coro: Awaitable[Any]) -> Any:
    """Return exception as a value (for 'continue-on-error' stages)."""
    try:
        return await coro
    except Exception as e:
        return e


class ProfeesModel:
    MODES = ("SDS", "LOCAL", "DEV", "PROD")

    def __init__(self, delivery_vehicle: DeliveryVehicle, mode: Optional[Mode] = None, log_level: str = "info"):
        self.delivery_vehicle = delivery_vehicle

        # Dispatch registry (by delivery vehicle)
        self._registry: Dict[DeliveryVehicle, Callable[[InputData, Optional[str]], Awaitable[tuple[ProfeesResponse, TokenStats]]]] = {
            DeliveryVehicle.SCRATCHPAD: self._predict_scratchpad,
            DeliveryVehicle.CAAD: self._predict_caad,
        }

        # Resolve mode + logging
        self.mode = (mode or os.getenv("MODE_PROFEES") or ("SDS" if os.path.exists("/sds") else "LOCAL")).upper()
        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {self.mode}")
        os.environ["MODE_PROFEES"] = self.mode

        logger.remove()
        logger.add(
            sink=sys.stderr,
            level=log_level.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )

        self.settings = settings
        self.log_dir: str = self.settings.get_string("logging.log_dir")

        logger.info(
            "Initialized ProfeesModel | log_dir={}",
            self.mode,
            self.log_dir,
        )

        logger.info(
            "Initialized ProfeesModel | vehicle={} | mode={}",
            delivery_vehicle.value,
            self.mode,
        )

        # Graphs/configs
        self.em_graph = None
        self.em_model_config = None
        self.ed_em_graph = None
        self.ed_em_model_config = None
        self.inpatient_em_graph = None
        self.inpatient_em_model_config = None
        self.em_recode_graph = None
        self.em_recode_model_config = None
        self.modifiers_graph = None
        self.modifiers_model_config = None
        self.cpt_procedure_graph = None
        self.cpt_procedure_model_config = None
        self.icd10cm_associate_graph = None
        self.icd10cm_associate_model_config = None

        self._initialize_graphs()

    # ---------------- Graph Initialization (HOCON only) ---------------- #
    def _initialize_graphs(self) -> None:
        """Initialize graphs using HOCON settings only."""

        # ---------- EM config ----------
        try:
            em_cfg_tree = self.settings.get("em_model_config")
        except ConfigMissingException:
            raise ValueError("Missing 'em_model_config' section in settings configuration file")

        if not isinstance(em_cfg_tree, ConfigTree):
            raise ValueError("'em_model_config' must be an object (HOCON block)")

        em_cfg_dict = em_cfg_tree.as_plain_ordered_dict()
        logger.info(
            "Loaded em_model_config: design={}, model={}, embedding={}, prompt_version={}",
            em_cfg_dict.get("design"),
            em_cfg_dict.get("model_name"),
            em_cfg_dict.get("embedding_model"),
            em_cfg_dict.get("prompt_version"),
        )

        self.em_graph, self.em_model_config = self._build_graph_from_settings(
            em_cfg_dict,
            builder=build_em_graph,
        )

        logger.info("E/M graph ready.")

        # ---------- Modifiers config ----------
        mod_cfg_tree = self.settings.get("modifiers_model_config")
        mod_cfg_dict = mod_cfg_tree.as_plain_ordered_dict()
        logger.info(
            "Loaded modifiers_model_config: design={}, model={}, embedding={}, prompt_version={}",
            mod_cfg_dict.get("design"),
            mod_cfg_dict.get("model_name"),
            mod_cfg_dict.get("embedding_model"),
            mod_cfg_dict.get("prompt_version"),
        )

        self.modifiers_graph, self.modifiers_model_config = self._build_graph_from_settings(
            mod_cfg_dict,
            builder=build_modifiers_graph,
        )
        logger.info("Modifiers graph ready.")

        # ---------- CPT procedure config ----------
        cpt_cfg_tree = self.settings.get("cpt_procedure_model_config")
        cpt_cfg_dict = cpt_cfg_tree.as_plain_ordered_dict()
        logger.info(
            "Loaded cpt_procedure_model_config: design={}, model={}, embedding={}, prompt_version={}",
            cpt_cfg_dict.get("design"),
            cpt_cfg_dict.get("model_name"),
            cpt_cfg_dict.get("embedding_model"),
            cpt_cfg_dict.get("prompt_version"),
        )

        self.cpt_procedure_graph, self.cpt_procedure_model_config = self._build_graph_from_settings(
            cpt_cfg_dict,
            builder=build_cpt_procedure_graph,
        )
        logger.info("CPT procedure graph ready.")

        # ---------- HCPCS config ----------
        hcpcs_cfg_tree = self.settings.get("hcpcs_model_config")
        hcpcs_cfg_dict = hcpcs_cfg_tree.as_plain_ordered_dict()
        logger.info(
            "Loaded hcpcs_model_config: design={}, model={}, embedding={}, prompt_version={}",
            hcpcs_cfg_dict.get("design"),
            hcpcs_cfg_dict.get("model_name"),
            hcpcs_cfg_dict.get("embedding_model"),
            hcpcs_cfg_dict.get("prompt_version"),
        )
        self.hcpcs_graph, self.hcpcs_model_config = self._build_graph_from_settings(
            hcpcs_cfg_dict,
            builder=build_hcpcs_graph,
        )
        logger.info("HCPCS graph ready.")

        # ---------- ICD10CM associate config ----------
        icd_assoc_cfg_tree = self.settings.get("icd10cm_associate_model_config")
        icd_assoc_cfg_dict = icd_assoc_cfg_tree.as_plain_ordered_dict()
        logger.info(
            "Loaded icd10cm_associate_model_config: design={}, model={}, embedding={}, prompt_version={}",
            icd_assoc_cfg_dict.get("design"),
            icd_assoc_cfg_dict.get("model_name"),
            icd_assoc_cfg_dict.get("embedding_model"),
            icd_assoc_cfg_dict.get("prompt_version"),
        )

        self.icd10cm_associate_graph, self.icd10cm_associate_model_config = self._build_graph_from_settings(
            icd_assoc_cfg_dict,
            builder=build_icd10cm_associate_graph,
        )
        logger.info("ICD10CM associate graph ready.")

    def _build_graph_from_settings(self, cfg_dict: dict, *, builder):
        """
        Convert HOCON dict -> SimpleNamespace, apply runtime overrides, probe embedding, then build graph.
        """
        ns = SimpleNamespace(**cfg_dict)

        # Ensure key fields exist
        if not hasattr(ns, "output_dir"):
            ns.output_dir = "./results"
        if not hasattr(ns, "batch_size"):
            ns.batch_size = 1
        if not hasattr(ns, "log_level"):
            ns.log_level = "info"

        os.makedirs(ns.output_dir, exist_ok=True)

        # Probe embedding/LLM model availability without blocking graph build.
        if self.mode in self.MODES:
            embedding_model_name = getattr(ns, "embedding_model", None)
            if not embedding_model_name:
                msg = "Missing required 'embedding_model' in model config."
                logger.warning(msg)
            else:
                try:
                    from cdrc.common.embeddings_utils import get_embedding_models

                    model = get_embedding_models()[embedding_model_name]()
                    _ = model.embed_query("hi")
                    logger.info(f"✅ Embedding model '{embedding_model_name}' is valid.")
                except Exception as e:
                    msg = f"Embedding model '{embedding_model_name}' failed startup validation: {e}"
                    logger.warning(f"⚠️ {msg}")
                    if hasattr(ns, "num_few_shot_examples"):
                        previous_num_few_shot_examples = ns.num_few_shot_examples
                        ns.num_few_shot_examples = 0
                        logger.warning(
                            "Set num_few_shot_examples from {} to 0 because embedding model '{}' failed startup validation.",
                            previous_num_few_shot_examples,
                            embedding_model_name,
                        )

            llm_model_name = getattr(ns, "model_name", None)
            if not llm_model_name:
                msg = "Missing required 'model_name' in model config."
                logger.warning(msg)
            else:
                fallback_model_names = self._llm_startup_fallbacks(llm_model_name)
                ns.model_name = self._resolve_available_llm_model(llm_model_name, fallback_model_names)

        graph, model_config = builder(ns)
        return graph, model_config

    def _llm_startup_fallbacks(self, llm_model_name: str) -> list[str]:
        """Return mode-compatible fallback models, ordered from preferred to last resort."""
        if self.mode == "SDS" or llm_model_name.endswith("-sds"):
            fallbacks = [
                "caa-gpt-5.2-2025-12-11-dev-sds",
                "caa-gpt-4.1-2025-04-14-dev-sds",
            ]
        elif self.mode == "PROD" or llm_model_name.endswith("-prod"):
            fallbacks = [
                "caa-gpt-5.2-2025-12-11-prod",
                "caa-gpt-4.1-2025-04-14-prod",
            ]
        elif llm_model_name.endswith("-aifactory"):
            fallbacks = [
                "caa-gpt-5.2-2025-12-11-dev-aifactory",
                "caa-gpt-4.1-2025-04-14-dev-aifactory",
            ]
        else:
            fallbacks = [
                "caa-gpt-5.2-2025-12-11-dev",
                "caa-gpt-4.1-2025-04-14-dev",
            ]

        return [model_name for model_name in fallbacks if model_name != llm_model_name]

    def _resolve_available_llm_model(self, llm_model_name: str, fallback_model_names: list[str]) -> str:
        from cdrc.common.llm_utils import get_llm

        llms = get_llm()
        candidate_model_names = [llm_model_name, *fallback_model_names]
        for candidate_model_name in candidate_model_names:
            try:
                model = llms[candidate_model_name]()
                _ = model.invoke("hi")
                if candidate_model_name == llm_model_name:
                    logger.info(f"✅ LLM model '{candidate_model_name}' is valid.")
                else:
                    logger.warning(
                        "Using fallback LLM model '{}' because '{}' failed startup validation.",
                        candidate_model_name,
                        llm_model_name,
                    )
                return candidate_model_name
            except Exception as e:
                msg = f"LLM model '{candidate_model_name}' failed startup validation: {e}"
                logger.warning(f"⚠️ {msg}")

        logger.warning(
            "No startup-validated LLM model is available. Continuing with configured model '{}'.",
            llm_model_name,
        )
        return llm_model_name

    # ---------------- Prediction Entry ---------------- #

    async def predict(self, data: InputData, request_id: Optional[str] = None) -> tuple[ProfeesResponse, TokenStats]:
        handler = self._registry.get(self.delivery_vehicle)
        if not handler:
            raise ValueError(f"Unsupported delivery vehicle: {self.delivery_vehicle}")
        return await handler(data, request_id)

    # ---------------- Vehicle-specific Predictors ---------------- #

    async def _predict_scratchpad(self, data: InputData, request_id: Optional[str] = None) -> tuple[ProfeesResponse, TokenStats]:
        logger.debug("Running Scratchpad prediction pipeline.")
        return await self._bpredict__scratchpad(data, request_id=request_id)

    async def _predict_caad(self, data: InputData, request_id: Optional[str] = None) -> tuple[ProfeesResponse, TokenStats]:
        logger.error("Running CAAD prediction is not supported.")
        raise NotImplementedError("CAAD delivery vehicle is not supported in app_2603 ProfeesModel.")

    # ---------------- Core Async Prediction ---------------- #
    async def _bpredict__scratchpad(self, data: InputData, request_id: Optional[str] = None) -> tuple[ProfeesResponse, TokenStats]:
        """
        Stage 1: EM + HCPCS + CPT procedure in parallel.
        Stage 2: Modifiers + ICD10CM associate (using stage 1 outputs).
        Then normalize and return ProfeesResponse.
        """
        if not self.em_graph:
            raise RuntimeError("Graphs not initialized.")

        generated_request_id = request_id is None
        request_id = request_id or f"req_{uuid.uuid4()}_pid_{os.getpid()}"
        cleanup_local_tracker = generated_request_id

        def _to_dict_list(items):
            out = []
            for x in items or []:
                if isinstance(x, dict):
                    out.append(x)
                elif hasattr(x, "model_dump"):
                    out.append(x.model_dump())
                else:
                    try:
                        out.append(dict(x))
                    except Exception:
                        raise TypeError(f"Unsupported item type in list: {type(x)!r}")
            return out

        # with timeit(MetricNames.API_REQUEST_TIME, {"namespace": METRICS_REPORTING_NAMESPACE}):
        try:
            # counter(MetricNames.API_REQUEST_RECEIVED, 1, {"namespace": METRICS_REPORTING_NAMESPACE})

            # ---------- preprocess ----------
            patient_age = calculate_patient_age(data.patient_birthdate, data.current_encounter_period.start)
            patient_type = predict_patient_type(patient_type=data.patient_type)

            selected_orders = data.orders
            if data.soap_text:
                cda_note = data.soap_text
            else:
                cda_note = CDA_note(data.ambient_note)
            orders_display = convert_orders_to_display(selected_orders)
            appointment_type = str(data.appointment_type).strip() if data.appointment_type else None

            # ---------- select model configs by care setting ----------
            em_graph = self.em_graph
            em_model_config = self.em_model_config
            modifiers_graph = self.modifiers_graph
            modifiers_model_config = self.modifiers_model_config

            # ---------- base model_data used by multiple graphs ----------
            base_model_data = ModelData(
                text=cda_note,
                patient_type=patient_type,
                patient_age=patient_age,
                orders_display=orders_display,
                medications=[],
                reason_for_visit="",
                appointment_type=appointment_type,
                service_date=data.current_encounter_period.start,
                soap_text_previous=data.soap_text_previous,
                orders_display_previous=convert_orders_to_display(data.orders_previous) if data.orders_previous else "",
                response_previous=data.response_previous,
            )
            hcpcs_input = build_hcpcs_input(data, cda_note, patient_age, orders_display)

            # ---------- Stage 1 configs ----------
            em_cfg = dict(em_model_config or {})
            em_cfg.setdefault("metadata", {})["token_tracker_id"] = request_id

            cpt_cfg = dict(self.cpt_procedure_model_config or {})
            cpt_cfg.setdefault("metadata", {})["token_tracker_id"] = request_id

            hcpcs_cfg = dict(self.hcpcs_model_config or {})
            hcpcs_cfg.setdefault("metadata", {})["token_tracker_id"] = request_id

            # ---------- Stage 1: EM + ICD10CM + CPT + HCPCS concurrently ----------
            em_coro = em_graph.ainvoke(base_model_data, em_cfg, debug=False)
            cpt_coro = self.cpt_procedure_graph.ainvoke(base_model_data, cpt_cfg, debug=False)
            hcpcs_coro = self.hcpcs_graph.ainvoke(hcpcs_input, hcpcs_cfg, debug=False)

            async with asyncio.TaskGroup() as tg:
                t_em = tg.create_task(em_coro)
                t_cpt = tg.create_task(cpt_coro)
                t_hcpcs = tg.create_task(hcpcs_coro)

            raw_em, raw_cpt, raw_hcpcs = t_em.result(), t_cpt.result(), t_hcpcs.result()

            # ---------- Normalize Stage 1 outputs ----------
            em_data = normalize_nested_models(dict(raw_em), ["mdm_results", "encounter_type"])
            cpt_data = normalize_nested_models(dict(raw_cpt), ["mdm_results", "encounter_type"])
            hcpcs_data = normalize_nested_models(dict(raw_hcpcs), ["mdm_results", "encounter_type"])

            em_dicts = _to_dict_list(em_data.get("em"))
            diagnosis_dicts = []
            procedure_dicts = _to_dict_list(cpt_data.get("procedure") or []) + _to_dict_list(hcpcs_data.get("procedure") or [])

            em_dicts = exclude_preventive_em_when_awv_present(
                em_dicts,
                procedure_dicts,
                request_id=request_id,
            )
            procedure_dicts = exclude_depression_screening_when_awv_present(
                procedure_dicts,
                request_id=request_id,
            )
            procedure_dicts = exclude_g2211_when_base_em_missing(
                em_dicts,
                procedure_dicts,
                request_id=request_id,
            )
            # ---------- Stage 2: Modifiers + ICD10 associate concurrently (continue-on-error) ----------

            stage2_input = build_icd_associate_input(data, cda_note, patient_age, orders_display, em_dicts, procedure_dicts)
            modifiers_input = build_modifiers_input(data, stage2_input, diagnosis_dicts)

            async with asyncio.TaskGroup() as tg:
                mod_cfg = dict(modifiers_model_config or {})
                mod_cfg.setdefault("metadata", {})["token_tracker_id"] = request_id
                mod_coro = modifiers_graph.ainvoke(modifiers_input, mod_cfg, debug=False)
                t_modifiers = tg.create_task(mod_coro)

                icd_assoc_cfg = dict(self.icd10cm_associate_model_config or {})
                icd_assoc_cfg.setdefault("metadata", {})["token_tracker_id"] = request_id
                icd_assoc_coro = self.icd10cm_associate_graph.ainvoke(stage2_input, icd_assoc_cfg, debug=False)
                t_icd_assoc = tg.create_task(icd_assoc_coro)

            raw_modifiers, raw_icd_assoc = t_modifiers.result(), t_icd_assoc.result()

            # # ---------- Apply Stage 2 results (preserve your semantics) ----------
            if isinstance(raw_modifiers, Exception):
                logger.error("{} Modifiers graph error (continuing without modifiers): {}", request_id, raw_modifiers, exc_info=True)
            else:
                modifiers_data = normalize_nested_models(dict(raw_modifiers), ["mdm_results", "encounter_type"])
                em_dicts = _to_dict_list(modifiers_data.get("em") or em_dicts)
                procedure_dicts = _to_dict_list(modifiers_data.get("procedure") or procedure_dicts)
                procedure_dicts = exclude_depression_screening_when_awv_present(
                    procedure_dicts,
                    request_id=request_id,
                )

            if isinstance(raw_icd_assoc, Exception):
                logger.error("{} ICD10CM associate graph error (continuing without associates): {}", request_id, raw_icd_assoc, exc_info=True)
            else:
                icd_assoc_data = normalize_nested_models(dict(raw_icd_assoc), ["mdm_results", "encounter_type"])
                assoc_em = _to_dict_list(icd_assoc_data.get("em") or [])
                assoc_procedure = _to_dict_list(icd_assoc_data.get("procedure") or [])
                assoc_diagnosis = _to_dict_list(icd_assoc_data.get("diagnosis") or [])

                if assoc_em and len(assoc_em) == len(em_dicts):
                    for base_item, assoc_item in zip(em_dicts, assoc_em):
                        if "linked_diagnosis" in assoc_item:
                            base_item["linked_diagnosis"] = assoc_item["linked_diagnosis"]

                if assoc_procedure and len(assoc_procedure) == len(procedure_dicts):
                    for base_item, assoc_item in zip(procedure_dicts, assoc_procedure):
                        if "linked_diagnosis" in assoc_item:
                            base_item["linked_diagnosis"] = assoc_item["linked_diagnosis"]
                diagnosis_dicts = assoc_diagnosis

            # ---------- Build typed response ----------
            response = ProfeesResponse(
                em=em_dicts,
                procedure=procedure_dicts,
                diagnosis=diagnosis_dicts,
            )
            # ---------- metrics ----------
            stats = TokenTracker.get_stats(request_id)
            # self._emit_token_stats(stats)
            # counter(MetricNames.API_REQUEST_SUCCESS, 1, {"namespace": METRICS_REPORTING_NAMESPACE})
            return response, stats

        except Exception as e:
            logger.error("{} Prediction error: {}", request_id, e, exc_info=True)
            if isinstance(e, ExceptionGroup):
                for i, sub_exc in enumerate(e.exceptions, start=1):
                    logger.error("{} Prediction sub-exception [{}]: {}", request_id, i, repr(sub_exc))
            # counter(MetricNames.API_REQUEST_SERVER_ERROR, 1, {"namespace": METRICS_REPORTING_NAMESPACE})
            raise
        finally:
            # If predict() is used outside api.py (no shared request_id lifecycle),
            # ensure this request's token tracker does not leak.
            if cleanup_local_tracker:
                TokenTracker.remove_instance(request_id)

    # ---------------- Metrics ---------------- #

    # def _emit_token_stats(self, stats: TokenStats) -> None:
    #     counter(MetricNames.LLM_TOTAL_TOKENS, stats.total_tokens, {"namespace": METRICS_REPORTING_NAMESPACE})
    #     counter(MetricNames.LLM_SUCCESSFUL_REQUESTS, stats.successful_requests, {"namespace": METRICS_REPORTING_NAMESPACE})
    #     counter(MetricNames.LLM_TOTAL_COST, stats.total_cost, {"namespace": METRICS_REPORTING_NAMESPACE})
    #     counter(MetricNames.LLM_PROMPT_TOKENS, stats.prompt_tokens, {"namespace": METRICS_REPORTING_NAMESPACE})
    #     counter(MetricNames.LLM_COMPLETION_TOKENS, stats.completion_tokens, {"namespace": METRICS_REPORTING_NAMESPACE})
    #     counter(MetricNames.LLM_PROMPT_TOKENS_CACHED, stats.prompt_tokens_cached, {"namespace": METRICS_REPORTING_NAMESPACE})
    #     counter(MetricNames.LLM_REASONING_TOKENS, stats.reasoning_tokens, {"namespace": METRICS_REPORTING_NAMESPACE})
    #     counter(MetricNames.LLM_REQUEST_TIME, sum(stats.requests_time_in_seconds or []), {"namespace": METRICS_REPORTING_NAMESPACE})
    #     counter(MetricNames.LLM_RATE_LIMIT_DELAYS, sum(stats.rate_limit_delays or []), {"namespace": METRICS_REPORTING_NAMESPACE})
