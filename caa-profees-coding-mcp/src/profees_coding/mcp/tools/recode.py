from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from adkruntime.agent_environment import AgentEnvironment
from adkruntime.mcp.decorators import extract_messaging_context
from adkruntime.messaging import MessagingContext
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastmcp.server import Context as FastMCPContext
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from oci_signing.service_accounts.service_accounts import ServiceAccountRequestDetails
from semantic_object_model.core.core_pb2 import Resource

from profees_coding.client.coding_service_call_factory import CodingClient
from profees_coding.client.constants import MR_SERVICE_ACCOUNT
from profees_coding.client.model_mapper_to_service import CodingModelMapper
from profees_coding.client.scratchpad_store import EM_CANDIDATES_KEY, ScratchpadStore
from profees_coding.common.debug_capture import capture_payload

AGENT_ENVIRONMENT: AgentEnvironment = cast(AgentEnvironment, None)
_CACHED_CODING_MODEL: Any | None = None

logger = logging.getLogger(__name__)


def _use_sample_model_response() -> bool:
    return os.getenv("PROFEES_RECODE_USE_SAMPLE_RESPONSE", "false").lower() in {"1", "true", "yes"}


def _log_recode_phase(phase: str, **values: Any) -> None:
    extra = ""
    if values:
        extra = " " + " ".join(f"{key}={value}" for key, value in values.items())
    message = f"[profees-mcp][recode_phase] {phase}{extra}"
    logger.info(message)


_AMBULATORY_CODING_NOTIFICATION_PLUGIN_NAME = "ambulatorycodingnotification"
_AMBULATORY_RECODE_COMPLETION_RENDERING = {
    "plugin_name": _AMBULATORY_CODING_NOTIFICATION_PLUGIN_NAME,
    "agent_name": "ambulatory_coding_notification",
    "agent_group_name": "ambulatory.coding",
    "render_as": "card",
    "plugin_title": "Generated codes",
    "plugin_description": "Ambulatory coding suggestions.",
}
_AMBULATORY_RECODE_TOOL_META = {
    "UIPluginRendering": _AMBULATORY_RECODE_COMPLETION_RENDERING,
}


def build_recode_completion_payload(
    *,
    result: dict,
    request_id: str | None,
    patient_id: str | None,
    encounter_id: str | None,
    document_ref_id: str | None,
    elapsed_ms: int,
) -> dict:
    result_payload = result.get("result") if isinstance(result, dict) else None
    if not isinstance(result_payload, dict):
        result_payload = {}

    payload_success = bool(result_payload.get("valid"))
    completion_payload = {
        "renderedType": "notification",
        "success": payload_success,
        "action": "recode",
        "status": "completed" if payload_success else "failed",
        "requestId": request_id or "",
        "patientId": patient_id or "",
        "encounterId": encounter_id or "",
        "document_ref_id": document_ref_id or "",
        "docrefId": document_ref_id or "",
        "documentRefId": document_ref_id or "",
        "toolStatus": result_payload.get("status"),
        "valid": bool(result_payload.get("valid")),
        "message": result_payload.get("message") or "ProFees recode completed.",
        "title": "Generated codes",
        "description": "Ambulatory coding new suggestions",
        "patientContext": {
            "contextType": "patient-chart",
            "patientId": patient_id or "",
            "encounterId": encounter_id or "",
        },
        "actions": {"openEnabled": True, "openImmediately": False},
        "elapsedMs": elapsed_ms,
        "elapsedSeconds": round(elapsed_ms / 1000, 3),
        "result": result_payload,
    }
    launch_parameters = {
        **completion_payload,
        "data": dict(completion_payload),
    }

    return {
        **completion_payload,
        "pluginName": _AMBULATORY_CODING_NOTIFICATION_PLUGIN_NAME,
        "renderMode": "card",
        "launchParameters": launch_parameters,
        "_meta": _AMBULATORY_RECODE_TOOL_META,
    }


DOCUMENT_REFERENCE_QUERY_PAGE_SIZE = 10_000
LOINC_CODE_ROS = "LOINC!10187-3"
LOINC_CODE_PE = "LOINC!29545-1"
LOINC_CODE_ANP = "LOINC!51847-2"

PROFEES_SERVICE_ACCOUNT = ServiceAccountRequestDetails(
    name=MR_SERVICE_ACCOUNT,
    serviceResourceId="defaultId",
    serviceResourceName="Medical coding agent",
    scopes="internal",
)


class ProFeesCPTCodingService:
    """MCP-local ProFees CPT coding service scaffold.

    This mirrors the agent service boundary without importing the ProFees agent
    package. Runtime steps will be filled in incrementally as the MCP flow grows.
    """

    def __init__(
        self,
        *,
        si_helper: Any,
        authenticated_environment: Any,
        coding_model: Any | None = None,
        scratchpad_store_factory: Any | None = None,
    ) -> None:
        self.si_helper = si_helper
        self.authenticated_environment = authenticated_environment
        self.coding_model = coding_model
        self.scratchpad_store_factory = scratchpad_store_factory
        self.scratchpad_store = None

    def _get_scratchpad_store(self):
        if self.scratchpad_store is None:
            if self.scratchpad_store_factory is None:
                raise RuntimeError("Agent-style scratchpad access has not been configured")
            self.scratchpad_store = self.scratchpad_store_factory()
        return self.scratchpad_store

    def readiness(self) -> dict:
        return {
            "service_created": True,
            "has_si_helper": self.si_helper is not None,
            "has_authenticated_environment": self.authenticated_environment is not None,
            "has_coding_model": self.coding_model is not None,
        }

    async def read_doc_content(self, document_ref_id: str) -> dict[str, str]:
        document_references = await self.si_helper.execute_query(
            resource_type=Resource.ResourceType.DOCUMENT_REFERENCE,
            query={"match": {"id": document_ref_id}},
            page_size=1,
        )
        document_reference = document_references[0] if document_references else None
        doc_content_id = f"{document_ref_id}-dc"

        document_contents = await self.si_helper.execute_query(
            resource_type=Resource.ResourceType.DOCUMENT_CONTENT,
            query={"match": {"id": doc_content_id}},
            page_size=1,
        )

        if not document_contents:
            return await self._read_doc_content_from_lso(document_ref_id, document_reference, None)

        embedded_data = _nested_value(
            document_contents[0],
            ("details", "embeddedDetails", "data"),
            ("details", "embedded_details", "data"),
            ("details", "embeddedDetails", "value"),
            ("details", "embedded_details", "value"),
        )
        if embedded_data is None:
            return await self._read_doc_content_from_lso(document_ref_id, document_reference, document_contents[0])

        try:
            if isinstance(embedded_data, str):
                decoded_content = base64.b64decode(embedded_data).decode("utf-8")
            else:
                decoded_content = bytes(embedded_data).decode("utf-8")
        except Exception:
            logger.exception("ProFees recode failed to decode embedded DocumentContent. document_ref_id=%s", document_ref_id)
            return {}

        return {document_ref_id: decoded_content}

    async def _read_doc_content_from_lso(
        self,
        document_ref_id: str,
        document_reference: Any,
        document_content: Any,
    ) -> dict[str, str]:
        document_content_lso_id = _document_content_lso_id(document_content)
        document_reference_lso_id = _document_reference_lso_id(document_reference)
        lso_candidates = [
            ("document_reference", document_reference_lso_id),
            ("document_content", document_content_lso_id),
        ]
        seen_lso_ids: set[str] = set()

        for lso_source, lso_id in lso_candidates:
            if not lso_id or lso_id in seen_lso_ids:
                continue
            seen_lso_ids.add(lso_id)

            try:
                content = await self._get_large_object_content(lso_id)
            except Exception:
                logger.exception(
                    "ProFees recode failed to read LSO through AOA SI client large object content API. document_ref_id=%s lso_id=%s lso_source=%s",
                    document_ref_id,
                    lso_id,
                    lso_source,
                )
                continue
            if content is None:
                continue

            try:
                if isinstance(content, str):
                    try:
                        decoded_content = base64.b64decode(content).decode("utf-8")
                    except Exception:
                        decoded_content = content
                else:
                    decoded_content = bytes(content).decode("utf-8")
            except Exception:
                logger.exception("ProFees recode failed to decode LSO content. document_ref_id=%s lso_id=%s", document_ref_id, lso_id)
                continue

            return {document_ref_id: decoded_content}

        return {}

    async def _get_large_object_content(self, lso_id: str) -> bytes | str | None:
        si_client = getattr(self.si_helper, "_si_client", None)
        tenant_id = getattr(self.si_helper, "_si_tenant_id", None)
        x_user_auth_context = getattr(self.si_helper, "_x_user_auth_context", None)
        if si_client is None or not tenant_id:
            return None

        def read_content() -> bytes | str | None:
            response = si_client.get_large_object_content(
                lso_id,
                tenant_id,
                x_user_auth_context=x_user_auth_context,
            )
            data = getattr(response, "data", None)
            if data is None:
                return None
            raw = getattr(data, "raw", None)
            if raw is not None and hasattr(raw, "read"):
                return raw.read()
            if hasattr(data, "read"):
                return data.read()
            return data

        return await asyncio.to_thread(read_content)

    async def check_filters(
        self,
        *,
        document_ref_id: str,
        patient_id: str,
        encounter_id: str,
    ) -> dict:
        doc_contents = await self.read_doc_content(document_ref_id)
        doc_content = doc_contents.get(document_ref_id)
        if doc_content is None:
            return {
                "filter_matched": False,
                "content_matched": False,
                "doc_content_available": False,
                "doc_content_length": 0,
                "reason": "DocumentContent was not found or could not be decoded for the selected DocumentReference.",
            }

        loinc_sections = _check_document_content_for_loinc_codes(doc_content)
        content_matched = loinc_sections["is_ros_present"] or loinc_sections["is_pe_present"] or loinc_sections["is_anp_present"]
        return {
            "filter_matched": content_matched,
            "content_matched": content_matched,
            "doc_content_available": True,
            "doc_content_length": len(doc_content),
            "loinc_sections": loinc_sections,
            "reason": (
                "Document content contains at least one required ProFees coding section."
                if content_matched
                else "Document content did not contain a required ProFees coding section."
            ),
        }

    async def build_coding_request_context(
        self,
        *,
        patient_id: str,
        encounter_id: str,
        semantic_patient_id: str | None,
        semantic_encounter_id: str | None,
    ) -> dict:
        patient_lookup_id = semantic_patient_id or patient_id
        encounter_lookup_id = semantic_encounter_id or encounter_id
        patient_info = await self._get_patient_info(
            patient_id=patient_lookup_id,
            source_patient_id=patient_id,
            encounter_id=encounter_lookup_id,
        )
        encounter_period = await self._get_encounter_period_in_str(
            encounter_id=encounter_lookup_id,
            source_encounter_id=encounter_id,
        )
        medication_orders = await self._read_medication_orders_by_encounter_id(encounter_lookup_id)
        non_medication_orders = await self._read_non_medication_orders_by_encounter_id(encounter_lookup_id)
        previous_doc_content = await self._get_previous_doc_content_from_scratchpad(encounter_lookup_id)
        order_history = medication_orders + non_medication_orders

        enriched_context = {
            "patient_info": patient_info,
            "encounter_period": encounter_period,
            "orders": order_history,
            "soap_text_previous": previous_doc_content,
            "semantic_patient_id": semantic_patient_id,
            "semantic_encounter_id": semantic_encounter_id,
        }
        return enriched_context

    async def _get_patient_info(
        self,
        *,
        patient_id: str,
        source_patient_id: str,
        encounter_id: str,
    ) -> dict:
        patient = await self._first_resource(
            Resource.ResourceType.PATIENT,
            {"match": {"id": patient_id}},
        )
        if patient is None and source_patient_id != patient_id:
            patient = await self._first_resource(
                Resource.ResourceType.PATIENT,
                {"match": {"subject.idAtSource.PERSON.id.longValue": source_patient_id}},
            )
        if patient is None and source_patient_id != patient_id:
            patient = await self._first_resource(
                Resource.ResourceType.PATIENT,
                {"match": {"meta.idAtSource.PERSON.id.longValue": source_patient_id}},
            )

        if patient is None:
            return {
                "type": "Unknown",
                "birthdate": "Unknown",
                "gender": "Unknown",
            }

        patient_type = await self._predict_patient_type(patient_id=patient_id, encounter_id=encounter_id)
        return {
            "type": patient_type,
            "birthdate": _iso_from_semantic_date(
                _nested_value(
                    patient,
                    ("birthDateTime", "birthDate"),
                    ("birth_date_time", "birth_date"),
                    ("birthDate",),
                    ("birth_date",),
                )
            ),
            "gender": _nested_value(patient, ("gender", "text"), ("gender", "code"), ("gender",)) or "Unknown",
        }

    async def _predict_patient_type(self, *, patient_id: str, encounter_id: str) -> str:
        try:
            three_years_ago = int((datetime.now(timezone.utc) - timedelta(days=3 * 365)).timestamp())
            historical_encounters = await self.si_helper.execute_query(
                resource_type=Resource.ResourceType.ENCOUNTER,
                query={
                    "bool": {
                        "must": [
                            {"match": {"meta.patient.value": patient_id}},
                            {"range": {"actualPeriod.start.seconds": {"gte": three_years_ago}}},
                        ],
                        "must_not": [{"match": {"id": encounter_id}}],
                    }
                },
                page_size=1,
            )
            return "Established" if historical_encounters else "New"
        except Exception:
            logger.exception(
                "ProFees recode failed to infer patient type; defaulting to Established. patient_id=%s encounter_id=%s",
                patient_id,
                encounter_id,
            )
            return "Established"

    async def _get_encounter_period_in_str(
        self,
        *,
        encounter_id: str,
        source_encounter_id: str,
    ) -> dict:
        encounter = await self._first_resource(
            Resource.ResourceType.ENCOUNTER,
            {"match": {"id": encounter_id}},
        )
        if encounter is None and source_encounter_id != encounter_id:
            encounter = await self._first_resource(
                Resource.ResourceType.ENCOUNTER,
                {"match": {"meta.idAtSource.ENCOUNTER.id.longValue": source_encounter_id}},
            )
        if encounter is None and source_encounter_id != encounter_id:
            encounter = await self._first_resource(
                Resource.ResourceType.ENCOUNTER,
                {"match": {"context.encounter.idAtSource.ENCOUNTER.id.longValue": source_encounter_id}},
            )

        if encounter is None:
            return {"start": "Unknown", "end": "Unknown"}

        start_time = _nested_value(
            encounter,
            ("actualPeriod", "start"),
            ("actual_period", "start"),
            ("resource", "actualPeriod", "start"),
            ("semantic_object", "actualPeriod", "start"),
            ("semantic_object_", "actual_period", "start"),
        )
        end_time = _nested_value(
            encounter,
            ("actualPeriod", "end"),
            ("actual_period", "end"),
            ("resource", "actualPeriod", "end"),
            ("semantic_object", "actualPeriod", "end"),
            ("semantic_object_", "actual_period", "end"),
        )
        period = {
            "start": _iso_from_semantic_date(start_time),
            "end": _iso_from_semantic_date(end_time),
        }
        return period

    async def _read_medication_orders_by_encounter_id(self, encounter_id: str) -> list[dict]:
        try:
            medication_requests = await self.si_helper.execute_query(
                resource_type=Resource.ResourceType.MEDICATION_REQUEST,
                query={"match": {"meta.encounter.value": encounter_id}},
                page_size=250,
            )
        except Exception:
            logger.exception("ProFees recode failed to fetch medication orders. encounter_id=%s", encounter_id)
            return []

        orders = [
            {
                "order_type": "Medication Order",
                "catalog_type": _nested_value(data, ("millenniumMeta", "catalogTypeCd"), ("millennium_meta", "catalog_type_cd")) or "",
                "display_line": _nested_value(data, ("renderedDosageInstruction",), ("rendered_dosage_instruction",)) or "",
                "mnemonic": _nested_value(data, ("name",)) or "",
                "status": _nested_value(data, ("status", "text"), ("status", "code"), ("status",)) or "",
            }
            for data in medication_requests or []
        ]
        return orders

    async def _read_non_medication_orders_by_encounter_id(self, encounter_id: str) -> list[dict]:
        try:
            service_requests = await self.si_helper.execute_query(
                resource_type=Resource.ResourceType.SERVICE_REQUEST,
                query={"match": {"meta.encounter.value": encounter_id}},
                page_size=250,
            )
        except Exception:
            logger.exception("ProFees recode failed to fetch non-medication orders. encounter_id=%s", encounter_id)
            return []

        orders = [
            {
                "order_type": "Non Medication Order",
                "catalog_type": _nested_value(data, ("millenniumMeta", "catalogTypeCd"), ("millennium_meta", "catalog_type_cd")) or "",
                "display_line": _nested_value(data, ("renderedInstruction",), ("rendered_instruction",)) or "",
                "mnemonic": _nested_value(data, ("name",)) or "",
                "status": _nested_value(data, ("status", "text"), ("status", "code"), ("status",)) or "",
            }
            for data in service_requests or []
        ]
        return orders

    async def _get_previous_doc_content_from_scratchpad(self, encounter_id: str) -> str | None:
        try:
            scratchpad = await self._get_scratchpad_store().read_previous(encounter_id)
        except Exception:
            logger.exception("ProFees recode failed to read previous scratchpad through agent-style SI authentication")
            return None
        if scratchpad is None:
            return None
        return _extract_scratchpad_document_content(scratchpad)

    async def _first_resource(self, resource_type: Any, query: dict[str, Any]) -> Any | None:
        results = await self.si_helper.execute_query(
            resource_type=resource_type,
            query=query,
            page_size=1,
        )
        return results[0] if results else None

    async def run_model_prediction_for_prototype(
        self,
        *,
        document_ref_id: str,
        patient_id: str,
        encounter_id: str,
        coding_request_context: dict,
    ) -> dict:
        if self.coding_model is None:
            return {
                "called": False,
                "success": False,
                "error": "coding_model is unavailable",
            }

        doc_contents = await self.read_doc_content(document_ref_id)
        doc_content = doc_contents.get(document_ref_id)
        if not doc_content:
            return {
                "called": False,
                "success": False,
                "error": "document content is unavailable",
            }

        try:
            model_response, additional_argument, sds_record = await self.process_encounter(
                patient_id=patient_id,
                encounter_id=encounter_id,
                document_ref_id=document_ref_id,
                doc_content=doc_content,
                coding_request_context=coding_request_context,
            )
        except Exception as exc:
            logger.exception(
                "ProFees recode process_encounter failed. patient_id=%s encounter_id=%s document_ref_id=%s",
                patient_id,
                encounter_id,
                document_ref_id,
            )
            return {
                "called": True,
                "success": False,
                "error": str(exc),
            }

        input_summary = sds_record.get("input") if isinstance(sds_record, dict) else None
        response_dict = _model_dump(model_response)
        output_summary = _summarize_model_output(response_dict)
        return {
            "called": True,
            "success": True,
            "input_summary": _summarize_model_input(input_summary) if input_summary is not None else None,
            "output_summary": output_summary,
            "model_response": response_dict,
            "additional_argument": additional_argument,
            "sds_record": sds_record,
        }

    async def process_encounter(
        self,
        patient_id: str,
        encounter_id: str,
        document_ref_id: str,
        doc_content: str,
        coding_request_context: dict | None = None,
    ) -> tuple[dict, dict, dict]:
        from cdrc.profees_coding.deploy.api.utils import InputData, InputPatientType

        if coding_request_context is None:
            coding_request_context = await self.build_coding_request_context(
                patient_id=patient_id,
                encounter_id=encounter_id,
                semantic_patient_id=None,
                semantic_encounter_id=None,
            )

        patient_info = coding_request_context["patient_info"]
        encounter_period = coding_request_context["encounter_period"]
        add_argument = {"encounter_period": encounter_period}
        model_data = InputData(
            encounter_id=encounter_id,
            soap_text=doc_content,
            patient_type=_input_patient_type(patient_info.get("type"), InputPatientType),
            patient_birthdate=patient_info.get("birthdate") or "Unknown",
            patient_sex=patient_info.get("gender") or "Unknown",
            current_encounter_period=encounter_period,
            problem_list=[],
            orders=coding_request_context["orders"],
            soap_text_previous=coding_request_context.get("soap_text_previous"),
        )

        repo_root = Path(__file__).resolve().parents[3]
        previous_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            if _use_sample_model_response():
                fixture_path = Path(__file__).resolve().parents[2] / "common" / "fixtures" / "profees_semantic_search_sample.json"
                raw_response = json.loads(fixture_path.read_text(encoding="utf-8"))
                logger.warning("[profees-mcp] Using SYNTHETIC test response; model predict is disabled")
            else:
                raw_response = self.coding_model.predict(model_data)
                raw_response = await raw_response if inspect.isawaitable(raw_response) else raw_response
        finally:
            os.chdir(previous_cwd)

        response = raw_response[0] if isinstance(raw_response, tuple) else raw_response
        response_dict = _model_dump(response)
        if not isinstance(response_dict, dict):
            raise TypeError(f"Unsupported response type from coding_model.predict: {type(raw_response)!r}")

        capture_payload("model-response", response_dict)

        sds_record = {
            "input": _model_dump(model_data),
            "output": response_dict,
        }
        return response_dict, add_argument, sds_record

    def save_billing_code(
        self,
        *,
        agent_env: AgentEnvironment,
        context: MessagingContext,
        model_response: dict,
        encounter_id: str,
        patient_id: str,
        document_ref_id: str,
        encounter_period: dict,
    ) -> dict:
        try:
            mapper = CodingModelMapper()
            client = CodingClient(agent_env=agent_env, context=context)
            request_body = mapper.map_billing_code_summary(
                model_response,
                encounter_id,
                patient_id,
                document_ref_id,
                encounter_period,
            )
            request_summary = _summarize_billing_code_request(request_body)
            response = client.save_billing_code(request_body)
            response_summary = _summarize_save_response(response)
            return {
                "called": True,
                "success": bool(response_summary.get("ok")),
                "request_summary": request_summary,
                "response_summary": response_summary,
            }
        except Exception as exc:
            logger.exception(
                "ProFees recode failed to save billing code. patient_id=%s encounter_id=%s document_ref_id=%s",
                patient_id,
                encounter_id,
                document_ref_id,
            )
            return {
                "called": True,
                "success": False,
                "error": str(exc),
            }

    async def save_generated_draft_scratchpad(
        self,
        *,
        agent_env,
        context,
        model_response,
        patient_id,
        encounter_id,
        document_ref_id,
        doc_content=None,
    ) -> dict:
        try:
            groups = model_response.get(EM_CANDIDATES_KEY)
            if groups is not None:
                groups = CodingModelMapper().map_em_order_catalog_groups(groups)
            if not doc_content:
                doc_content = (await self.read_doc_content(document_ref_id)).get(document_ref_id)
            if self.scratchpad_store_factory is None:
                self.scratchpad_store_factory = lambda: ScratchpadStore.from_instance_environment(agent_env.get_instance_environment(context))
            store = self._get_scratchpad_store()
            scratchpad = await store.save(
                patient_id=patient_id,
                encounter_id=encounter_id,
                document_ref_id=document_ref_id,
                doc_content=doc_content,
                groups=groups,
            )
            return {"called": True, "success": True, "scratchpad_id": scratchpad.id, "em_group_count": len(groups or [])}
        except Exception:
            logger.exception("ProFees recode failed to persist generated draft scratchpad")
            return {"called": True, "success": False, "error": "Generated draft scratchpad persistence failed"}


def _service_account_name() -> str | None:
    if isinstance(PROFEES_SERVICE_ACCOUNT, dict):
        return PROFEES_SERVICE_ACCOUNT.get("name")
    return getattr(PROFEES_SERVICE_ACCOUNT, "name", None)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _validate_recode_inputs(
    *,
    patient_id: str | None = None,
    encounter_id: str | None = None,
    document_ref_id: str | None = None,
) -> dict:
    clean_patient_id = _clean(patient_id)
    clean_encounter_id = _clean(encounter_id)
    clean_document_ref_id = _clean(document_ref_id)

    missing_fields = []
    if not clean_patient_id:
        missing_fields.append("patient_id")
    if not clean_encounter_id:
        missing_fields.append("encounter_id")

    return {
        "valid": not missing_fields,
        "missing_fields": missing_fields,
        "inputs": {
            "patient_id": clean_patient_id,
            "encounter_id": clean_encounter_id,
            "document_ref_id": clean_document_ref_id,
        },
    }


def _resolve_runtime_context(context: MessagingContext | None) -> dict:
    if context is None:
        return {
            "available": False,
            "error": "AOA MessagingContext was not injected into the ProFees recode tool.",
        }
    if AGENT_ENVIRONMENT is None:
        return {
            "available": False,
            "error": "AGENT_ENVIRONMENT was not injected into the ProFees recode toolkit.",
        }

    try:
        authenticated_environment = AGENT_ENVIRONMENT.get_authenticated_environment(context, PROFEES_SERVICE_ACCOUNT)
        instance_environment = AGENT_ENVIRONMENT.get_instance_environment(context)
        si_helper = authenticated_environment.get_si_helper()
    except Exception as exc:
        logger.exception("ProFees recode failed to resolve AOA authenticated environment and SI helper.")
        return {
            "available": False,
            "error": "Unable to resolve AOA authenticated environment for ProFees recode.",
            "details": str(exc),
        }

    return {
        "available": True,
        "has_authenticated_environment": authenticated_environment is not None,
        "has_si_helper": si_helper is not None,
        "semantic_index_endpoint": instance_environment.get_semantic_index_endpoint(),
        "si_tenant_id": getattr(instance_environment, "si_tenant_id", None),
        "si_instance_id": getattr(instance_environment, "si_instance_id", None),
        "cda_instance_id": getattr(instance_environment, "instance_id", None),
    }


def _nested_value(value: Any, *paths: tuple[str | int, ...]) -> Any:
    for path in paths:
        current = value
        for key in path:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list) and isinstance(key, int):
                current = current[key] if 0 <= key < len(current) else None
            elif isinstance(key, int):
                current = None
            else:
                current = getattr(current, key, None)
            if current is None:
                break
        if current is not None:
            return current
    return None


def _document_reference_id(document_reference: Any) -> str | None:
    value = _nested_value(
        document_reference,
        ("id",),
        ("resource", "id"),
        ("semantic_object", "id"),
        ("semantic_object_", "id"),
        ("row", "id"),
    )
    return str(value) if value else None


def _document_reference_last_updated_seconds(document_reference: Any) -> int:
    value = _nested_value(
        document_reference,
        ("meta", "lastUpdated", "seconds"),
        ("meta", "last_updated", "seconds"),
        ("resource", "meta", "lastUpdated", "seconds"),
        ("semantic_object", "meta", "lastUpdated", "seconds"),
        ("semantic_object_", "meta", "last_updated", "seconds"),
        ("row", "meta", "lastUpdated", "seconds"),
    )
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _document_reference_lso_id(document_reference: Any) -> str | None:
    sections = _nested_value(
        document_reference,
        ("common", "section"),
        ("resource", "common", "section"),
        ("semantic_object", "common", "section"),
        ("semantic_object_", "common", "section"),
    )
    if isinstance(sections, list) and sections:
        first_section = sections[0]
        value = _nested_value(
            first_section,
            ("contentLso", "value"),
            ("content_lso", "value"),
            ("contentLSO", "value"),
        )
        return str(value) if value else None
    return None


def _document_reference_patient_id(document_reference: Any) -> str | None:
    value = _nested_value(
        document_reference,
        ("subject", "value"),
        ("resource", "subject", "value"),
        ("semantic_object", "subject", "value"),
        ("semantic_object_", "subject", "value"),
    )
    return str(value) if value else None


def _document_reference_encounter_id(document_reference: Any) -> str | None:
    value = _nested_value(
        document_reference,
        ("meta", "encounter", "value"),
        ("resource", "meta", "encounter", "value"),
        ("semantic_object", "meta", "encounter", "value"),
        ("semantic_object_", "meta", "encounter", "value"),
    )
    if value:
        return str(value)

    encounters = _nested_value(
        document_reference,
        ("context", "encounter"),
        ("resource", "context", "encounter"),
        ("semantic_object", "context", "encounter"),
        ("semantic_object_", "context", "encounter"),
    )
    if isinstance(encounters, list) and encounters:
        value = _nested_value(encounters[0], ("value",))
        return str(value) if value else None
    return None


def _document_content_lso_id(document_content: Any) -> str | None:
    value = _nested_value(
        document_content,
        ("details", "attachmentDetails", "documentContent", "value"),
        ("details", "attachment_details", "document_content", "value"),
        ("details", "attachmentDetails", "document_content", "value"),
        ("details", "attachment_details", "documentContent", "value"),
    )
    return str(value) if value else None


def _source_id_values(value: Any, source_type: str, source_name: str | None = None) -> list[str]:
    found: list[str] = []

    def visit(current: Any) -> None:
        if isinstance(current, dict):
            id_at_source = current.get("idAtSource")
            if isinstance(id_at_source, dict):
                source = id_at_source.get(source_type)
                if isinstance(source, dict):
                    ids = source.get("id")
                    if isinstance(ids, list):
                        for item in ids:
                            if not isinstance(item, dict):
                                continue
                            if source_name is not None and item.get("name") != source_name:
                                continue
                            long_value = item.get("longValue")
                            if long_value is not None:
                                found.append(str(long_value))
            for child in current.values():
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    return found


def _is_loinc_section_present(html_content: str, loinc_code: str) -> bool:
    """Return whether a LOINC section contains non-empty ``ddfreetext`` content."""
    try:
        from lxml import html as lxml_html

        soup = BeautifulSoup(html_content, "html.parser")
        xml_output_str = soup.prettify()
        xml_output_str = re.sub(r"<\?xml[^>]*\?>", "", xml_output_str)
        xml_document = lxml_html.fromstring(xml_output_str)
        matched_elements = xml_document.xpath(f"//*[local-name()='div' and @*[local-name()='dd:sectioncode']='{loinc_code}']")
        for element in matched_elements:
            free_text = element.xpath(".//*[contains(concat(' ', normalize-space(@class), ' '), " "' ddfreetext ')]//text()")
            if any(text.strip() for text in free_text):
                return True
        return False
    except Exception:
        logger.exception("ProFees recode failed to parse LOINC section with lxml. loinc_code=%s", loinc_code)
    return False


def _check_document_content_for_loinc_codes(html_content: str) -> dict[str, bool]:
    return {
        "is_ros_present": _is_loinc_section_present(html_content, LOINC_CODE_ROS),
        "is_pe_present": _is_loinc_section_present(html_content, LOINC_CODE_PE),
        "is_anp_present": _is_loinc_section_present(html_content, LOINC_CODE_ANP),
    }


def _iso_from_semantic_date(value: Any) -> str:
    seconds = _nested_value(value, ("seconds",))
    try:
        seconds_int = int(seconds)
    except (TypeError, ValueError):
        return "Unknown"
    if seconds_int <= 0:
        return "Unknown"
    return datetime.fromtimestamp(seconds_int, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _extract_scratchpad_document_content(scratchpad: Any) -> str | None:
    if isinstance(scratchpad, Message):
        scratchpad = MessageToDict(scratchpad)
    value = _nested_value(
        scratchpad,
        ("content", "document", "payload", "value", "content"),
        ("content", "document", "payload", "content"),
        ("content", "document", "content"),
    )
    return str(value) if value else None


def _input_patient_type(value: Any, input_patient_type: Any) -> Any:
    normalized = str(value or "Unknown").strip().lower()
    if normalized == "new":
        return input_patient_type.NEW
    if normalized == "established":
        return input_patient_type.ESTABLISHED
    return input_patient_type.UNKNOWN


def _summarize_document_reference(document_reference: Any) -> dict:
    return {
        "document_ref_id": _document_reference_id(document_reference),
        "last_updated_seconds": _document_reference_last_updated_seconds(document_reference),
        "semantic_patient_id": _document_reference_patient_id(document_reference),
        "semantic_encounter_id": _document_reference_encounter_id(document_reference),
        "source_encounter_ids": _source_id_values(document_reference, "ENCOUNTER", "ENCNTR_ID"),
        "source_patient_ids": _source_id_values(document_reference, "PERSON", "PERSON_ID"),
        "encounter_value": _nested_value(
            document_reference,
            ("meta", "encounter", "value"),
            ("resource", "meta", "encounter", "value"),
            ("semantic_object", "meta", "encounter", "value"),
            ("semantic_object_", "meta", "encounter", "value"),
        ),
        "encounter_reference": _nested_value(
            document_reference,
            ("meta", "encounter", "reference"),
            ("resource", "meta", "encounter", "reference"),
            ("semantic_object", "meta", "encounter", "reference"),
            ("semantic_object_", "meta", "encounter", "reference"),
        ),
        "type_code": _nested_value(
            document_reference,
            ("common", "type", "code"),
            ("resource", "common", "type", "code"),
            ("semantic_object", "common", "type", "code"),
            ("semantic_object_", "common", "type", "code"),
        ),
        "raw_type": type(document_reference).__name__,
        "raw_preview": str(document_reference)[:1000],
    }


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return value


def _split_model_prediction_response(response: Any) -> tuple[Any, Any | None]:
    if isinstance(response, tuple):
        if len(response) == 2:
            return response[0], response[1]
        if len(response) > 0:
            return response[0], response[1:]
    return response, None


def _summarize_model_input(model_data: Any) -> dict:
    data = _model_dump(model_data)
    soap_text = data.get("soap_text") if isinstance(data, dict) else getattr(model_data, "soap_text", None)
    orders = data.get("orders") if isinstance(data, dict) else getattr(model_data, "orders", None)
    problem_list = data.get("problem_list") if isinstance(data, dict) else getattr(model_data, "problem_list", None)
    return {
        "encounter_id": data.get("encounter_id") if isinstance(data, dict) else getattr(model_data, "encounter_id", None),
        "patient_type": data.get("patient_type") if isinstance(data, dict) else getattr(model_data, "patient_type", None),
        "patient_birthdate": data.get("patient_birthdate") if isinstance(data, dict) else getattr(model_data, "patient_birthdate", None),
        "patient_sex": data.get("patient_sex") if isinstance(data, dict) else getattr(model_data, "patient_sex", None),
        "current_encounter_period": (
            data.get("current_encounter_period") if isinstance(data, dict) else _model_dump(getattr(model_data, "current_encounter_period", None))
        ),
        "orders_count": len(orders or []),
        "problem_list_count": len(problem_list or []),
        "has_soap_text_previous": (
            bool(data.get("soap_text_previous")) if isinstance(data, dict) else bool(getattr(model_data, "soap_text_previous", None))
        ),
        "soap_text_length": len(soap_text or ""),
        "soap_text_preview": (soap_text or "")[:500],
    }


def _summarize_model_output(response: Any) -> dict:
    response_dict = _model_dump(response)
    if not isinstance(response_dict, dict):
        return {
            "raw_type": type(response).__name__,
            "raw_preview": str(response)[:1000],
        }
    return {
        "procedure_count": len(response_dict.get("procedure") or []),
        "em_count": len(response_dict.get("em") or []),
        "diagnosis_count": len(response_dict.get("diagnosis") or []),
        "procedure_codes": [item.get("code") for item in response_dict.get("procedure") or [] if isinstance(item, dict) and item.get("code")],
        "em_codes": [item.get("code") for item in response_dict.get("em") or [] if isinstance(item, dict) and item.get("code")],
        "diagnosis_codes": [item.get("code") for item in response_dict.get("diagnosis") or [] if isinstance(item, dict) and item.get("code")],
    }


def _summarize_billing_code_request(request_body: Any) -> dict:
    data = _model_dump(request_body)
    if not isinstance(data, dict):
        return {
            "raw_type": type(request_body).__name__,
            "raw_preview": str(request_body)[:1000],
        }
    procedure = data.get("procedure") or []
    em = data.get("em") or []
    diagnosis = data.get("diagnosis") or []
    return {
        "encounter_id": data.get("encounterId") or data.get("encounter_id"),
        "patient_id": data.get("patientId") or data.get("patient_id"),
        "owner_id": data.get("ownerId") or data.get("owner_id"),
        "is_proposed": data.get("isProposed") if "isProposed" in data else data.get("is_proposed"),
        "procedure_count": len(procedure),
        "em_count": len(em),
        "diagnosis_count": len(diagnosis),
        "procedure_codes": [item.get("code") for item in procedure if isinstance(item, dict)][:10],
        "em_codes": [item.get("code") for item in em if isinstance(item, dict)][:10],
        "diagnosis_codes": [item.get("code") for item in diagnosis if isinstance(item, dict)][:10],
    }


def _summarize_save_response(response: Any) -> dict:
    if response is None:
        return {
            "raw_type": None,
            "ok": False,
        }
    data = _model_dump(response)
    if isinstance(data, dict):
        is_error_response = type(response).__name__ == "ErrorResponse" or "code" in data and "message" in data
        return {
            "raw_type": type(response).__name__,
            "ok": not is_error_response,
            "keys": sorted(data.keys())[:20],
            "id": data.get("id"),
            "status": data.get("status"),
            "code": data.get("code"),
            "message": data.get("message"),
        }
    return {
        "raw_type": type(response).__name__,
        "ok": True,
        "raw_preview": str(response)[:1000],
    }


def _summarize_coding_request_context(context: dict) -> dict:
    return {
        "semantic_patient_id": context.get("semantic_patient_id"),
        "semantic_encounter_id": context.get("semantic_encounter_id"),
        "patient_info": context.get("patient_info"),
        "encounter_period": context.get("encounter_period"),
        "orders_count": len(context.get("orders") or []),
        "has_soap_text_previous": bool(context.get("soap_text_previous")),
        "soap_text_previous_length": len(context.get("soap_text_previous") or ""),
    }


def _create_cpt_coding_service(authenticated_environment: Any) -> ProFeesCPTCodingService:
    coding_model = _get_or_create_coding_model()
    service = ProFeesCPTCodingService(
        si_helper=authenticated_environment.get_si_helper(),
        authenticated_environment=authenticated_environment,
        coding_model=coding_model,
    )
    return service


def _get_or_create_coding_model() -> Any | None:
    global _CACHED_CODING_MODEL

    if _use_sample_model_response():
        # Only a readiness sentinel; fixture mode never calls its predict method.
        return object()

    coding_model = getattr(AGENT_ENVIRONMENT, "coding_model", None)
    if coding_model is not None:
        _log_recode_phase("coding_model_cache_hit", cache="agent_environment", has_agent_environment=AGENT_ENVIRONMENT is not None)
        return coding_model

    if _CACHED_CODING_MODEL is not None:
        _log_recode_phase("coding_model_cache_hit", cache="module", has_agent_environment=AGENT_ENVIRONMENT is not None)
        if AGENT_ENVIRONMENT is not None:
            setattr(AGENT_ENVIRONMENT, "coding_model", _CACHED_CODING_MODEL)
        return _CACHED_CODING_MODEL

    repo_root = Path(__file__).resolve().parents[3]
    previous_cwd = os.getcwd()
    try:
        _load_profees_model_environment(repo_root)
        os.chdir(repo_root)
        incoming_model_mode = os.getenv("MODE_PROFEES")
        model_mode = os.getenv("MODE_PROFEES", "LOCAL").upper()
        os.environ["MODE_PROFEES"] = model_mode
        logger.info("[profees-mcp][recode_profees_coding] MODE_PROFEES incoming=%s effective=%s", incoming_model_mode, model_mode)
        _log_recode_phase(
            "coding_model_init_start",
            mode=model_mode,
            repo_root=repo_root,
            has_agent_environment=AGENT_ENVIRONMENT is not None,
        )
        from cdrc.profees_coding.deploy.api.model.app_gen2 import ProfeesModel
        from cdrc.profees_coding.deploy.api.utils import DeliveryVehicle

        coding_model = ProfeesModel(delivery_vehicle=DeliveryVehicle.CAAD, mode=model_mode)
        logger.info("[profees-mcp][recode_profees_coding] ProfeesModel initialized with delivery_vehicle=CAAD mode=%s", model_mode)
        _log_recode_phase("coding_model_init_done", mode=model_mode, coding_model_created=coding_model is not None)
        _CACHED_CODING_MODEL = coding_model
        if AGENT_ENVIRONMENT is not None:
            setattr(AGENT_ENVIRONMENT, "coding_model", coding_model)
        return coding_model
    except Exception:
        logger.exception("ProFees recode failed to import or initialize ProfeesModel for lazy MCP initialization.")
        return None
    finally:
        os.chdir(previous_cwd)


def warmup_coding_model() -> bool:
    """Initialize and cache the ProFees model before the first recode call."""
    _log_recode_phase("coding_model_warmup_start", has_agent_environment=AGENT_ENVIRONMENT is not None)
    coding_model = _get_or_create_coding_model()
    _log_recode_phase("coding_model_warmup_done", coding_model_created=coding_model is not None)
    return coding_model is not None


def _load_profees_model_environment(repo_root: Path) -> None:
    """Load PPCA env files before ProFees model init changes cwd.

    The copied ProFees model stack reads Path(".env") from the current working
    directory. In the PPCA/MCP runtime that cwd is not guaranteed to be the PPCA
    repo root, so load the expected env files explicitly and keep existing
    process variables authoritative.
    """
    master_root = repo_root.parent
    candidate_env_paths = [
        Path.cwd() / ".env",
        master_root / "inpatient-coding-proactive-pca" / ".env",
        master_root / "inpatient-coding-proactive-pca" / "personas" / "coder" / ".env",
        repo_root / ".env",
    ]
    loaded_paths: list[str] = []
    for env_path in candidate_env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            loaded_paths.append(str(env_path))


async def _execute_document_reference_query(si_helper: Any, query: dict[str, Any], *, page_from: int = 0) -> list[Any]:
    return await si_helper.execute_query(
        resource_type=Resource.ResourceType.DOCUMENT_REFERENCE,
        query=query,
        page_size=DOCUMENT_REFERENCE_QUERY_PAGE_SIZE,
        page_from=page_from,
    )


async def _find_document_reference(
    *,
    si_helper: Any,
    cpt_coding_service: ProFeesCPTCodingService,
    patient_id: str,
    encounter_id: str,
    document_ref_id: str | None = None,
) -> dict:
    queries = _document_reference_queries_for_encounter(encounter_id)
    query: dict[str, Any] = queries[0]
    candidate_summaries: list[dict] = []
    filter_summaries: list[dict] = []
    raw_candidate_count = 0
    strict_candidate_count = 0
    selected_document = None
    selected_filter_result = None

    async def evaluate_documents(candidate_documents: list[Any]) -> None:
        nonlocal selected_document, selected_filter_result
        for document in candidate_documents:
            candidate_document_ref_id = _document_reference_id(document)
            if candidate_document_ref_id is None:
                continue
            filter_result = await cpt_coding_service.check_filters(
                document_ref_id=candidate_document_ref_id,
                patient_id=patient_id,
                encounter_id=encounter_id,
            )
            filter_summary = {
                "document_ref_id": candidate_document_ref_id,
                "last_updated_seconds": _document_reference_last_updated_seconds(document),
                "filter_matched": filter_result.get("filter_matched"),
                "content_matched": filter_result.get("content_matched"),
                "doc_content_available": filter_result.get("doc_content_available"),
                "doc_content_length": filter_result.get("doc_content_length"),
                "loinc_sections": filter_result.get("loinc_sections"),
                "reason": filter_result.get("reason"),
            }
            _log_recode_phase(
                "document_filter_result",
                document_ref_id=filter_summary["document_ref_id"],
                filter_matched=filter_summary["filter_matched"],
                content_matched=filter_summary["content_matched"],
                doc_content_available=filter_summary["doc_content_available"],
                doc_content_length=filter_summary["doc_content_length"],
                loinc_sections=filter_summary["loinc_sections"],
                reason=filter_summary["reason"],
            )
            filter_summaries.append(filter_summary)
            if not filter_result.get("filter_matched"):
                continue

            selected_document = document
            selected_filter_result = filter_result
            break

    if document_ref_id:
        direct_query = {"match": {"id": document_ref_id}}
        direct_results = await _execute_document_reference_query(si_helper, direct_query)
        raw_candidate_count += len(direct_results or [])
        direct_documents = sorted(
            [
                document
                for document in list(direct_results or [])
                if _document_reference_id(document) == document_ref_id
                and _document_reference_matches_recode_request(
                    document,
                    patient_id=patient_id,
                    encounter_id=encounter_id,
                )
            ],
            key=_document_reference_last_updated_seconds,
            reverse=True,
        )
        if direct_documents:
            query = direct_query
            strict_candidate_count += len(direct_documents)
            candidate_summaries = [_summarize_document_reference(document) for document in direct_documents[:5]]
            await evaluate_documents(direct_documents)

    if selected_document is None:
        for candidate_query in queries:
            candidate_results = await _execute_document_reference_query(si_helper, candidate_query)
            raw_candidate_count += len(candidate_results or [])

            documents = sorted(
                [
                    document
                    for document in list(candidate_results or [])
                    if (document_ref_id is None or _document_reference_id(document) == document_ref_id)
                    and _document_reference_matches_recode_request(
                        document,
                        patient_id=patient_id,
                        encounter_id=encounter_id,
                    )
                ],
                key=_document_reference_last_updated_seconds,
                reverse=True,
            )
            if not documents:
                continue

            query = candidate_query
            strict_candidate_count += len(documents)
            if not candidate_summaries:
                candidate_summaries = [_summarize_document_reference(document) for document in documents[:5]]

            await evaluate_documents(documents)
            if selected_document is not None:
                break

    selected_document_ref_id = _document_reference_id(selected_document) if selected_document is not None else None

    return {
        "found": selected_document_ref_id is not None,
        "query": query,
        "candidate_count": raw_candidate_count,
        "strict_candidate_count": strict_candidate_count,
        "filter_checked_count": len(filter_summaries),
        "selected_document_ref_id": selected_document_ref_id,
        "selected_patient_id": _document_reference_patient_id(selected_document) if selected_document is not None else None,
        "selected_encounter_id": _document_reference_encounter_id(selected_document) if selected_document is not None else None,
        "selected_last_updated_seconds": (_document_reference_last_updated_seconds(selected_document) if selected_document is not None else None),
        "selected_filter_result": selected_filter_result,
        "candidates": candidate_summaries,
        "filter_results": filter_summaries[:5],
    }


def _document_reference_queries_for_encounter(encounter_id: str) -> list[dict[str, Any]]:
    if encounter_id.startswith("en-"):
        return [
            {"match": {"meta.encounter.value": encounter_id}},
            {"match": {"context.encounter.value": encounter_id}},
            {"match": {"context.encounter.reference": f"Encounter/{encounter_id}"}},
            {"match": {"meta.encounter.reference": f"Encounter/{encounter_id}"}},
        ]
    return [
        {"match": {"context.encounter.idAtSource.ENCOUNTER.id.longValue": encounter_id}},
        {"match": {"meta.idAtSource.ENCOUNTER.id.longValue": encounter_id}},
    ]


def _document_reference_matches_recode_request(
    document_reference: Any,
    *,
    patient_id: str,
    encounter_id: str,
) -> bool:
    encounter_matches = _document_reference_matches_encounter(document_reference, encounter_id)
    patient_matches = _document_reference_matches_patient(document_reference, patient_id)
    if not encounter_matches or not patient_matches:
        return False
    return True


def _document_reference_matches_encounter(document_reference: Any, encounter_id: str) -> bool:
    values = {
        _document_reference_encounter_id(document_reference),
        _nested_value(document_reference, ("meta", "encounter", "reference")),
        _nested_value(document_reference, ("context", "encounter", 0, "reference")),
        _nested_value(document_reference, ("resource", "meta", "encounter", "reference")),
        _nested_value(document_reference, ("resource", "context", "encounter", 0, "reference")),
        _nested_value(document_reference, ("semantic_object", "meta", "encounter", "reference")),
        _nested_value(document_reference, ("semantic_object", "context", "encounter", 0, "reference")),
        _nested_value(document_reference, ("semantic_object_", "meta", "encounter", "reference")),
        _nested_value(document_reference, ("semantic_object_", "context", "encounter", 0, "reference")),
    }
    normalized_values = {str(value) for value in values if value}
    if encounter_id in normalized_values or f"Encounter/{encounter_id}" in normalized_values:
        return True
    return encounter_id in _source_id_values(document_reference, "ENCOUNTER", "ENCNTR_ID")


def _document_reference_matches_patient(document_reference: Any, patient_id: str) -> bool:
    values = {
        _document_reference_patient_id(document_reference),
        _nested_value(document_reference, ("subject", "reference")),
        _nested_value(document_reference, ("meta", "patient", "value")),
        _nested_value(document_reference, ("meta", "patient", "reference")),
        _nested_value(document_reference, ("resource", "subject", "reference")),
        _nested_value(document_reference, ("resource", "meta", "patient", "value")),
        _nested_value(document_reference, ("resource", "meta", "patient", "reference")),
        _nested_value(document_reference, ("semantic_object", "subject", "reference")),
        _nested_value(document_reference, ("semantic_object", "meta", "patient", "value")),
        _nested_value(document_reference, ("semantic_object", "meta", "patient", "reference")),
        _nested_value(document_reference, ("semantic_object_", "subject", "reference")),
        _nested_value(document_reference, ("semantic_object_", "meta", "patient", "value")),
        _nested_value(document_reference, ("semantic_object_", "meta", "patient", "reference")),
    }
    normalized_values = {str(value) for value in values if value}
    if patient_id in normalized_values or f"Patient/{patient_id}" in normalized_values:
        return True
    return patient_id in _source_id_values(document_reference, "PERSON", "PERSON_ID")


async def recode_profees_coding(
    patient_id: str | None = None,
    encounter_id: str | None = None,
    document_ref_id: str | None = None,
    save_billing_codes: bool = True,
    context: FastMCPContext = None,
) -> dict:
    """Validate a ProFees recode request and resolve the target DocumentReference.

    :param patient_id: Required patient identifier from PPCA/UI context.
    :param encounter_id: Required encounter identifier from PPCA/UI context.
    :param document_ref_id: Optional document reference ID for selected-document recode.
    :param save_billing_codes: When true, persist predicted codes through the Coding Service after model prediction.
    :param context: Injected AOA MessagingContext. This is not a user-facing tool input.
    :return: Validation, runtime-context, and DocumentReference lookup status.
    """
    logger.info(
        "[profees-mcp][recode_profees_coding] start patient_id=%s encounter_id=%s document_ref_id=%s save_billing_codes=%s",
        patient_id,
        encounter_id,
        document_ref_id,
        save_billing_codes,
    )
    validation = _validate_recode_inputs(
        patient_id=patient_id,
        encounter_id=encounter_id,
        document_ref_id=document_ref_id,
    )
    if not validation["valid"]:
        return {
            "result": {
                "status": "invalid",
                "valid": False,
                "message": "patient_id and encounter_id are required for ProFees recode.",
                **validation,
            }
        }

    messaging_context = extract_messaging_context(context)
    runtime_context = _resolve_runtime_context(messaging_context)
    if not runtime_context["available"]:
        return {
            "result": {
                "status": "runtime_context_unavailable",
                "message": "ProFees recode input validation passed, but AOA runtime context is unavailable.",
                **validation,
                "valid": False,
                "runtime_context": runtime_context,
                "next_steps": [
                    "query Semantic Index for candidate DocumentReference records",
                    "find document reference to recode",
                    "run ProFees CodingEventFilter checks",
                    "call CPTCodingService.process_encounter",
                    "save billing code through Coding Service",
                ],
            }
        }

    authenticated_environment = AGENT_ENVIRONMENT.get_authenticated_environment(messaging_context, PROFEES_SERVICE_ACCOUNT)
    _log_recode_phase("coding_service_create_start")
    cpt_coding_service = _create_cpt_coding_service(authenticated_environment)
    cpt_coding_service.scratchpad_store_factory = lambda: ScratchpadStore.from_instance_environment(
        AGENT_ENVIRONMENT.get_instance_environment(messaging_context)
    )
    cpt_coding_service_readiness = cpt_coding_service.readiness()
    _log_recode_phase("coding_service_create_done", readiness=cpt_coding_service_readiness)
    _log_recode_phase(
        "document_lookup_start",
        patient_id=validation["inputs"]["patient_id"],
        encounter_id=validation["inputs"]["encounter_id"],
        document_ref_id=validation["inputs"]["document_ref_id"],
    )
    document_reference_lookup = await _find_document_reference(
        si_helper=authenticated_environment.get_si_helper(),
        cpt_coding_service=cpt_coding_service,
        patient_id=validation["inputs"]["patient_id"],
        encounter_id=validation["inputs"]["encounter_id"],
        document_ref_id=validation["inputs"]["document_ref_id"],
    )
    _log_recode_phase(
        "document_lookup_done",
        found=document_reference_lookup["found"],
        candidate_count=document_reference_lookup.get("candidate_count"),
        strict_candidate_count=document_reference_lookup.get("strict_candidate_count"),
        selected_document_ref_id=document_reference_lookup.get("selected_document_ref_id"),
    )
    if not document_reference_lookup["found"]:
        return {
            "result": {
                "status": "document_reference_not_found",
                "message": "No DocumentReference was found for the ProFees recode request.",
                **validation,
                "valid": False,
                "runtime_context": runtime_context,
                "document_reference_lookup": document_reference_lookup,
                "cpt_coding_service": cpt_coding_service_readiness,
                "next_steps": [
                    "confirm the encounter_id/document_ref_id values from PPCA context",
                    "confirm that at least one candidate DocumentReference matches patient/encounter and passes ProFees CodingEventFilter checks",
                    "call CPTCodingService.process_encounter",
                    "save billing code through Coding Service",
                ],
            }
        }

    filter_result = document_reference_lookup["selected_filter_result"]
    _log_recode_phase("coding_context_start")
    coding_request_context = await cpt_coding_service.build_coding_request_context(
        patient_id=validation["inputs"]["patient_id"],
        encounter_id=validation["inputs"]["encounter_id"],
        semantic_patient_id=document_reference_lookup["selected_patient_id"],
        semantic_encounter_id=document_reference_lookup["selected_encounter_id"],
    )
    _log_recode_phase("coding_context_done")
    _log_recode_phase("model_prediction_start")
    model_prediction = await cpt_coding_service.run_model_prediction_for_prototype(
        document_ref_id=document_reference_lookup["selected_document_ref_id"],
        patient_id=validation["inputs"]["patient_id"],
        encounter_id=validation["inputs"]["encounter_id"],
        coding_request_context=coding_request_context,
    )
    _log_recode_phase(
        "model_prediction_done",
        called=model_prediction.get("called"),
        success=model_prediction.get("success"),
        error=model_prediction.get("error"),
        predicted_codes=_summarize_model_output(model_prediction.get("model_response")),
    )
    save_result = {
        "called": False,
        "success": False,
        "reason": "save_billing_codes is false",
    }
    scratchpad_result = {"called": False, "success": False, "reason": "persistence was not requested or prediction failed"}
    fixture_mode = _use_sample_model_response()
    fixture_save_allowed = os.getenv("PROFEES_FIXTURE_ALLOW_PERSISTENCE", "false").lower() in {"1", "true", "yes"}
    fixture_preview = fixture_mode and model_prediction["success"] and not fixture_save_allowed
    if fixture_preview or (save_billing_codes and model_prediction["success"]):
        preview_request = CodingModelMapper().map_billing_code_summary(
            model_prediction["model_response"],
            validation["inputs"]["encounter_id"],
            validation["inputs"]["patient_id"],
            document_reference_lookup["selected_document_ref_id"],
            (model_prediction.get("additional_argument") or {}).get("encounter_period", coding_request_context["encounter_period"]),
        )
        preview_path = capture_payload("savebillingcodes-payload", preview_request.to_dict())
        save_result = {
            "called": False,
            "success": False,
            "reason": "Synthetic fixture preview: billing and scratchpad writes are blocked" if fixture_preview else "Billing API call temporarily disabled; scratchpad-only persistence",
            "payload_capture_path": str(preview_path) if preview_path else None,
        }
    if save_billing_codes and model_prediction["success"] and not fixture_preview:
        if fixture_mode:
            logger.warning("[profees-mcp] SYNTHETIC fixture persistence explicitly enabled; only scratchpad save will be attempted")
        # Temporarily skip cpt_coding_service.save_billing_code; keep its mapped payload preview above.
        _log_recode_phase("billing_save_skipped", reason=save_result["reason"])
        _log_recode_phase("scratchpad_save_start")
        scratchpad_result = await cpt_coding_service.save_generated_draft_scratchpad(
            agent_env=AGENT_ENVIRONMENT,
            context=messaging_context,
            model_response=model_prediction["model_response"],
            patient_id=document_reference_lookup["selected_patient_id"] or validation["inputs"]["patient_id"],
            encounter_id=document_reference_lookup["selected_encounter_id"] or validation["inputs"]["encounter_id"],
            document_ref_id=document_reference_lookup["selected_document_ref_id"],
            doc_content=((model_prediction.get("sds_record") or {}).get("input") or {}).get("soap_text"),
        )
        _log_recode_phase("scratchpad_save_done", success=scratchpad_result["success"])
    elif save_billing_codes and not model_prediction["success"]:
        save_result = {
            "called": False,
            "success": False,
            "reason": "model prediction did not succeed",
        }

    _log_recode_phase(
        "save_done",
        called=save_result.get("called"),
        success=save_result.get("success"),
        reason=save_result.get("reason"),
    )

    result = {
        "result": {
            "status": (
                "scratchpad_saved"
                if scratchpad_result["success"]
                else (
                    "scratchpad_save_failed"
                    if scratchpad_result["called"]
                    else (
                        "billing_code_save_failed"
                        if save_result["called"]
                        else (
                            "model_prediction_completed"
                            if model_prediction["success"]
                            else "model_prediction_failed" if model_prediction["called"] else "model_prediction_skipped"
                        )
                    )
                )
            ),
            **validation,
            "valid": model_prediction["success"] and (not save_billing_codes or scratchpad_result["success"]),
            "message": (
                (
                    "ProFees recode saved the generated draft scratchpad. Billing API call was intentionally skipped."
                    if scratchpad_result["success"]
                    else "Scratchpad persistence failed. Billing API call was intentionally skipped."
                )
                if scratchpad_result["called"]
                else (
                    "ProFees recode called ProfeesModel.predict, but saving billing codes failed."
                    if save_result["called"]
                    else (
                        "ProFees recode called ProfeesModel.predict and received a model response."
                        if model_prediction["success"]
                        else (
                            "ProFees recode reached the model prediction step, but the ProFees model call did not complete successfully."
                            if model_prediction["called"]
                            else "ProFees recode resolved DocumentReference and created the MCP-local CPT coding service, but skipped model prediction."
                        )
                    )
                )
            ),
            "save_billing_codes": save_billing_codes,
            "runtime_context": runtime_context,
            "document_reference_lookup": document_reference_lookup,
            "cpt_coding_service": cpt_coding_service_readiness,
            "filter_result": filter_result,
            "filter_gate_bypassed": False,
            "coding_request_context": _summarize_coding_request_context(coding_request_context),
            "model_prediction": {key: value for key, value in model_prediction.items() if key != "model_response"},
            "save_billing_code": save_result,
            "save_scratchpad": scratchpad_result,
            "next_steps": [
                "inspect the scratchpad save outcome; no billing API call was made",
            ],
        }
    }
    if fixture_preview:
        result["result"].update(
            status="synthetic_fixture_preview",
            valid=True,
            synthetic=True,
            message="Synthetic test response mapped to a billing payload preview. No model prediction or billing/scratchpad writes were performed.",
            next_steps=["inspect the local model-response and savebillingcodes-payload JSON captures"],
        )
    elif fixture_mode and model_prediction["success"]:
        result["result"]["synthetic"] = True
        result["result"]["message"] = (
            "Used a synthetic test fixture, not live model prediction. "
            f"Billing save success={save_result['success']}; scratchpad save success={scratchpad_result['success']}."
        )
    return result
