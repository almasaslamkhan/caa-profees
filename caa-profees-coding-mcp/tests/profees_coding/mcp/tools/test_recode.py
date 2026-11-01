from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from profees_coding.mcp import aoa_toolkit
from profees_coding.mcp.tools import recode


def _run(coro):
    return asyncio.run(coro)


def test_build_recode_completion_payload_handles_success_and_invalid_results() -> None:
    success = recode.build_recode_completion_payload(
        result={"result": {"valid": True, "status": "saved", "message": "Saved"}},
        request_id="request-1",
        patient_id="patient-1",
        encounter_id="encounter-1",
        document_ref_id="document-1",
        elapsed_ms=1250,
    )
    failed = recode.build_recode_completion_payload(
        result={},
        request_id=None,
        patient_id=None,
        encounter_id=None,
        document_ref_id=None,
        elapsed_ms=0,
    )

    assert success["success"] is True
    assert success["status"] == "completed"
    assert success["launchParameters"]["data"]["elapsedSeconds"] == 1.25
    assert success["pluginName"] == "ambulatorycodingnotification"
    assert success["renderedType"] == "notification"
    assert success["launchParameters"]["patientContext"]["encounterId"] == "encounter-1"
    assert success["_meta"]["UIPluginRendering"]["plugin_name"] == "ambulatorycodingnotification"
    assert failed["success"] is False
    assert failed["status"] == "failed"
    assert failed["message"] == "ProFees recode completed."


def test_ehrm_recode_wrapper_returns_recode_complete_semantic_action_properties(monkeypatch) -> None:
    completion_payload = {
        "action": "recode",
        "requestId": "request-1",
        "success": True,
        "toolStatus": "billing_code_saved",
        "valid": True,
    }
    tool_call = AsyncMock(return_value=completion_payload)
    monkeypatch.setattr(aoa_toolkit, "recode_profees_coding", tool_call)

    result = _run(
        aoa_toolkit.recode_profees_coding_ehrm(
            patient_id="patient-1",
            encounter_id="encounter-1",
            document_ref_id="document-1",
            request_id="request-1",
        )
    )

    tool_call.assert_awaited_once_with(
        patient_id="patient-1",
        encounter_id="encounter-1",
        document_ref_id="document-1",
        request_id="request-1",
        save_billing_codes=True,
        mcp_context=None,
    )
    assert result == {
        "patientId": "patient-1",
        "isError": "false",
        "semanticAction": "true",
    }


def test_ehrm_recode_wrapper_returns_semantic_action_error_payload(monkeypatch) -> None:
    tool_call = AsyncMock(
        return_value={
            "success": False,
            "toolStatus": "document_reference_not_found",
            "message": "No matching DocumentReference was found.",
        }
    )
    monkeypatch.setattr(aoa_toolkit, "recode_profees_coding", tool_call)

    result = _run(
        aoa_toolkit.recode_profees_coding_ehrm(
            patient_id="patient-1",
            encounter_id="encounter-1",
            document_ref_id="document-1",
        )
    )

    assert result == {
        "patientId": "patient-1",
        "isError": "true",
        "message": "No matching DocumentReference was found.",
        "semanticAction": "true",
    }


def test_validation_and_document_reference_matching_support_nested_source_ids() -> None:
    document = {
        "id": "document-1",
        "subject": {"reference": "Patient/patient-1"},
        "context": {"encounter": [{"reference": "Encounter/encounter-1"}]},
        "meta": {"lastUpdated": {"seconds": "42"}},
        "idAtSource": {
            "ENCOUNTER": {"id": [{"name": "ENCNTR_ID", "longValue": "source-encounter"}]},
            "PERSON": {"id": [{"name": "PERSON_ID", "longValue": "source-patient"}]},
        },
    }

    validation = recode._validate_recode_inputs(patient_id=" patient-1 ", encounter_id=" ", document_ref_id=" doc-1 ")

    assert validation["valid"] is False
    assert validation["missing_fields"] == ["encounter_id"]
    assert validation["inputs"]["patient_id"] == "patient-1"
    assert recode._document_reference_last_updated_seconds(document) == 42
    assert recode._document_reference_matches_recode_request(document, patient_id="patient-1", encounter_id="encounter-1")
    assert recode._document_reference_matches_recode_request(document, patient_id="source-patient", encounter_id="source-encounter")
    assert not recode._document_reference_matches_recode_request(document, patient_id="other", encounter_id="encounter-1")


def test_document_helpers_and_summaries_handle_dicts_and_models() -> None:
    model = SimpleNamespace(to_dict=lambda: {"procedure": [{"code": "99213"}], "em": [], "diagnosis": [{"code": "A01"}]})
    request = {"encounterId": "encounter-1", "patientId": "patient-1", "procedure": [{"code": "123"}], "em": [], "diagnosis": []}

    assert recode._nested_value({"items": [{"value": "found"}]}, ("items", 0, "value")) == "found"
    assert recode._nested_value({"items": []}, ("items", 0, "value")) is None
    assert recode._iso_from_semantic_date({"seconds": 1}) == "1970-01-01T00:00:01Z"
    assert recode._iso_from_semantic_date({"seconds": 0}) == "Unknown"
    assert recode._extract_scratchpad_document_content({"content": {"document": {"content": "note"}}}) == "note"
    assert recode._summarize_model_output(model)["procedure_codes"] == ["99213"]
    assert recode._summarize_billing_code_request(request)["procedure_codes"] == ["123"]
    assert recode._summarize_save_response(None) == {"raw_type": None, "ok": False}


def test_read_doc_content_decodes_embedded_content_and_uses_lso_fallback() -> None:
    encoded = base64.b64encode(b"clinical note").decode("ascii")
    helper = SimpleNamespace(
        execute_query=AsyncMock(
            side_effect=[
                [{"id": "document-1"}],
                [{"details": {"embeddedDetails": {"data": encoded}}}],
            ]
        )
    )
    service = recode.ProFeesCPTCodingService(si_helper=helper, authenticated_environment=object())

    assert _run(service.read_doc_content("document-1")) == {"document-1": "clinical note"}

    fallback_service = recode.ProFeesCPTCodingService(
        si_helper=SimpleNamespace(execute_query=AsyncMock(side_effect=[[{"common": {"section": [{"contentLso": {"value": "lso-1"}}]}}], []])),
        authenticated_environment=object(),
    )
    fallback_service._get_large_object_content = AsyncMock(return_value=b"fallback note")

    assert _run(fallback_service.read_doc_content("document-2")) == {"document-2": "fallback note"}


def test_service_filters_context_and_prediction_failure_paths(monkeypatch) -> None:
    service = recode.ProFeesCPTCodingService(si_helper=SimpleNamespace(), authenticated_environment=object())
    service.read_doc_content = AsyncMock(return_value={})
    missing_content = _run(service.check_filters(document_ref_id="document-1", patient_id="patient-1", encounter_id="encounter-1"))

    service.read_doc_content = AsyncMock(return_value={"document-1": "note"})
    monkeypatch.setattr(
        recode, "_check_document_content_for_loinc_codes", lambda _: {"is_ros_present": True, "is_pe_present": False, "is_anp_present": False}
    )
    matching_content = _run(service.check_filters(document_ref_id="document-1", patient_id="patient-1", encounter_id="encounter-1"))

    service._get_patient_info = AsyncMock(return_value={"type": "New"})
    service._get_encounter_period_in_str = AsyncMock(return_value={"start": "start", "end": "end"})
    service._read_medication_orders_by_encounter_id = AsyncMock(return_value=[{"order_type": "Medication Order"}])
    service._read_non_medication_orders_by_encounter_id = AsyncMock(return_value=[{"order_type": "Non Medication Order"}])
    service._get_previous_doc_content_from_scratchpad = AsyncMock(return_value="previous note")
    context = _run(
        service.build_coding_request_context(
            patient_id="source-patient", encounter_id="source-encounter", semantic_patient_id="patient-1", semantic_encounter_id="encounter-1"
        )
    )

    assert missing_content["doc_content_available"] is False
    assert matching_content["filter_matched"] is True
    assert len(context["orders"]) == 2
    assert context["soap_text_previous"] == "previous note"
    assert _run(
        service.run_model_prediction_for_prototype(
            document_ref_id="document-1", patient_id="patient-1", encounter_id="encounter-1", coding_request_context={}
        )
    ) == {
        "called": False,
        "success": False,
        "error": "coding_model is unavailable",
    }


def test_recode_tool_reports_invalid_input_and_unavailable_runtime(monkeypatch) -> None:
    invalid = _run(recode.recode_profees_coding(patient_id=" ", encounter_id=None))
    monkeypatch.setattr(recode, "extract_messaging_context", lambda _: object())
    monkeypatch.setattr(recode, "AGENT_ENVIRONMENT", None)
    unavailable = _run(recode.recode_profees_coding(patient_id="patient-1", encounter_id="encounter-1"))

    assert invalid["result"]["status"] == "invalid"
    assert invalid["result"]["missing_fields"] == ["patient_id", "encounter_id"]
    assert unavailable["result"]["status"] == "runtime_context_unavailable"
    assert unavailable["result"]["runtime_context"]["available"] is False


def test_service_reads_patient_encounter_orders_and_scratchpad() -> None:
    patient = {
        "birthDateTime": {"birthDate": {"seconds": 1}},
        "gender": {"text": "female"},
    }
    encounter = {
        "actualPeriod": {
            "start": {"seconds": 1},
            "end": {"seconds": 2},
        }
    }
    helper = SimpleNamespace(
        execute_query=AsyncMock(
            side_effect=[
                [patient],
                [{"id": "historical-encounter"}],
                [encounter],
                [
                    {
                        "millenniumMeta": {"catalogTypeCd": "medication"},
                        "renderedDosageInstruction": "once daily",
                        "name": "Medication A",
                        "status": {"text": "active"},
                    }
                ],
                [
                    {
                        "millennium_meta": {"catalog_type_cd": "procedure"},
                        "rendered_instruction": "perform test",
                        "name": "Procedure B",
                        "status": {"code": "draft"},
                    }
                ],
            ]
        )
    )
    service = recode.ProFeesCPTCodingService(si_helper=helper, authenticated_environment=object())
    store = SimpleNamespace(read_previous=AsyncMock(return_value={"content": {"document": {"payload": {"content": "previous note"}}}}))
    service.scratchpad_store_factory = lambda: store

    patient_info = _run(
        service._get_patient_info(
            patient_id="patient-1",
            source_patient_id="source-patient-1",
            encounter_id="encounter-1",
        )
    )
    period = _run(service._get_encounter_period_in_str(encounter_id="encounter-1", source_encounter_id="source-encounter-1"))
    medication_orders = _run(service._read_medication_orders_by_encounter_id("encounter-1"))
    non_medication_orders = _run(service._read_non_medication_orders_by_encounter_id("encounter-1"))
    previous_note = _run(service._get_previous_doc_content_from_scratchpad("encounter-1"))

    assert patient_info == {"type": "Established", "birthdate": "1970-01-01T00:00:01Z", "gender": "female"}
    assert period == {"start": "1970-01-01T00:00:01Z", "end": "1970-01-01T00:00:02Z"}
    assert medication_orders[0]["mnemonic"] == "Medication A"
    assert non_medication_orders[0]["display_line"] == "perform test"
    assert previous_note == "previous note"
    store.read_previous.assert_awaited_once_with("encounter-1")


def test_model_prediction_and_billing_code_save_use_service_boundaries(monkeypatch) -> None:
    service = recode.ProFeesCPTCodingService(si_helper=SimpleNamespace(), authenticated_environment=object(), coding_model=object())
    service.read_doc_content = AsyncMock(return_value={"document-1": "note"})
    service.process_encounter = AsyncMock(
        return_value=(
            {"procedure": [{"code": "99213"}], "em": [], "diagnosis": []},
            {"encounter_period": {}},
            {"input": {"soap_text": "note", "orders": [], "problem_list": []}},
        )
    )

    prediction = _run(
        service.run_model_prediction_for_prototype(
            document_ref_id="document-1",
            patient_id="patient-1",
            encounter_id="encounter-1",
            coding_request_context={},
        )
    )

    class Mapper:
        def map_billing_code_summary(self, *_):
            return {"encounterId": "encounter-1", "patientId": "patient-1", "procedure": [{"code": "99213"}]}

    class Client:
        def __init__(self, **_):
            pass

        def save_billing_code(self, _):
            return {"id": "billing-code-1", "status": "saved"}

    monkeypatch.setattr(recode, "CodingModelMapper", Mapper)
    monkeypatch.setattr(recode, "CodingClient", Client)
    saved = service.save_billing_code(
        agent_env=object(),
        context=object(),
        model_response=prediction["model_response"],
        encounter_id="encounter-1",
        patient_id="patient-1",
        document_ref_id="document-1",
        encounter_period={},
    )

    assert prediction["success"] is True
    assert prediction["output_summary"]["procedure_codes"] == ["99213"]
    assert saved["success"] is True
    assert saved["response_summary"]["id"] == "billing-code-1"


def test_find_document_reference_selects_latest_matching_document() -> None:
    older_document = {
        "id": "document-older",
        "subject": {"value": "patient-1"},
        "meta": {"encounter": {"value": "en-1"}, "lastUpdated": {"seconds": 1}},
    }
    latest_document = {
        "id": "document-latest",
        "subject": {"value": "patient-1"},
        "meta": {"encounter": {"value": "en-1"}, "lastUpdated": {"seconds": 2}},
    }
    helper = SimpleNamespace(execute_query=AsyncMock(return_value=[older_document, latest_document]))
    coding_service = SimpleNamespace(
        check_filters=AsyncMock(
            return_value={
                "filter_matched": True,
                "content_matched": True,
                "doc_content_available": True,
                "doc_content_length": 10,
                "loinc_sections": {"is_ros_present": True},
                "reason": "matched",
            }
        )
    )

    result = _run(
        recode._find_document_reference(
            si_helper=helper,
            cpt_coding_service=coding_service,
            patient_id="patient-1",
            encounter_id="en-1",
        )
    )

    assert result["found"] is True
    assert result["selected_document_ref_id"] == "document-latest"
    assert result["filter_checked_count"] == 1
    assert result["selected_filter_result"]["filter_matched"] is True


def test_recode_tool_completes_prediction_and_saves_only_scratchpad(monkeypatch) -> None:
    messaging_context = object()
    authenticated_environment = SimpleNamespace(get_si_helper=lambda: object())
    agent_environment = SimpleNamespace(get_authenticated_environment=lambda *_: authenticated_environment)

    class CodingService:
        def readiness(self):
            return {"service_created": True}

        async def build_coding_request_context(self, **_):
            return {
                "patient_info": {"type": "Established"},
                "encounter_period": {"start": "start", "end": "end"},
                "orders": [],
                "soap_text_previous": None,
            }

        async def run_model_prediction_for_prototype(self, **_):
            return {
                "called": True,
                "success": True,
                "model_response": {"procedure": [{"code": "99213"}], "em": [], "diagnosis": []},
                "additional_argument": {"encounter_period": {"start": "start", "end": "end"}},
            }

        def save_billing_code(self, **_):
            raise AssertionError("Billing API must not be called in scratchpad-only mode")

        async def save_generated_draft_scratchpad(self, **_):
            return {"called": True, "success": True}

    document_lookup = {
        "found": True,
        "candidate_count": 1,
        "strict_candidate_count": 1,
        "selected_document_ref_id": "document-1",
        "selected_patient_id": "semantic-patient-1",
        "selected_encounter_id": "semantic-encounter-1",
        "selected_filter_result": {"filter_matched": True},
    }
    monkeypatch.setattr(recode, "AGENT_ENVIRONMENT", agent_environment)
    monkeypatch.setattr(recode, "extract_messaging_context", lambda _: messaging_context)
    monkeypatch.setattr(recode, "_resolve_runtime_context", lambda _: {"available": True})
    monkeypatch.setattr(recode, "_create_cpt_coding_service", lambda _: CodingService())
    monkeypatch.setattr(recode, "_find_document_reference", AsyncMock(return_value=document_lookup))

    result = _run(
        recode.recode_profees_coding(
            patient_id="patient-1",
            encounter_id="encounter-1",
            document_ref_id="document-1",
            context=object(),
        )
    )

    assert result["result"]["status"] == "scratchpad_saved"
    assert result["result"]["valid"] is True
    assert result["result"]["save_billing_code"]["called"] is False
    assert result["result"]["save_scratchpad"]["success"] is True


def test_large_object_content_reads_raw_response_and_handles_missing_client() -> None:
    response = SimpleNamespace(data=SimpleNamespace(raw=SimpleNamespace(read=lambda: b"large object content")))
    client = SimpleNamespace(get_large_object_content=lambda *_args, **_kwargs: response)
    service = recode.ProFeesCPTCodingService(
        si_helper=SimpleNamespace(_si_client=client, _si_tenant_id="tenant-1", _x_user_auth_context="auth"),
        authenticated_environment=object(),
    )
    unavailable_service = recode.ProFeesCPTCodingService(si_helper=SimpleNamespace(), authenticated_environment=object())

    assert _run(service._get_large_object_content("lso-1")) == b"large object content"
    assert _run(unavailable_service._get_large_object_content("lso-1")) is None


def test_process_encounter_builds_model_input_and_returns_prediction(monkeypatch) -> None:
    capture = Mock()
    monkeypatch.setattr(recode, "capture_payload", capture)
    class Model:
        def __init__(self):
            self.model_data = None

        def predict(self, model_data):
            self.model_data = model_data
            return ({"procedure": [{"code": "99213"}], "em": [], "diagnosis": []}, {"ignored": True})

    model = Model()
    service = recode.ProFeesCPTCodingService(si_helper=SimpleNamespace(), authenticated_environment=object(), coding_model=model)
    context = {
        "patient_info": {"type": "New", "birthdate": "1970-01-01", "gender": "female"},
        "encounter_period": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z"},
        "orders": [
            {
                "order_type": "Medication Order",
                "catalog_type": "pharmacy",
                "display_line": "Medication A",
                "mnemonic": "Medication A",
                "status": "active",
            }
        ],
        "soap_text_previous": "previous note",
    }

    response, additional_argument, sds_record = _run(
        service.process_encounter(
            patient_id="patient-1",
            encounter_id="encounter-1",
            document_ref_id="document-1",
            doc_content="current note",
            coding_request_context=context,
        )
    )

    assert response["procedure"][0]["code"] == "99213"
    assert additional_argument == {"encounter_period": context["encounter_period"]}
    assert sds_record["input"]["soap_text"] == "current note"
    assert model.model_data.encounter_id == "encounter-1"
    capture.assert_called_once_with("model-response", response)


def test_loinc_and_model_summary_helpers_handle_expected_content() -> None:
    clinical_html = '<div dd:sectioncode="LOINC!10187-3"><span class="ddfreetext">normal finding</span></div>'
    empty_section_html = '<div dd:sectioncode="LOINC!10187-3"><span class="ddfreetext">  </span></div>'
    non_free_text_html = '<div dd:sectioncode="LOINC!10187-3"><span>normal finding</span></div>'

    assert recode._is_loinc_section_present(clinical_html, recode.LOINC_CODE_ROS) is True
    assert recode._is_loinc_section_present(empty_section_html, recode.LOINC_CODE_ROS) is False
    assert recode._is_loinc_section_present(non_free_text_html, recode.LOINC_CODE_ROS) is False
    assert recode._check_document_content_for_loinc_codes(clinical_html) == {
        "is_ros_present": True,
        "is_pe_present": False,
        "is_anp_present": False,
    }
    assert recode._split_model_prediction_response(("response", "extra")) == ("response", "extra")
    assert recode._split_model_prediction_response(("response", "first", "second")) == ("response", ("first", "second"))
    assert recode._summarize_model_input({"encounter_id": "encounter-1", "soap_text": "note", "orders": [{}]})["orders_count"] == 1
