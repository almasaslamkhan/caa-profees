import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from semantic_object_model.core.op_type_pb2 import OpType
from semantic_object_model.workflow.scratchpad_pb2 import Scratchpad

from profees_coding.client import coding_service_call_factory as factory
from profees_coding.client.model_mapper_to_service import CodingModelMapper
from profees_coding.client.scratchpad_store import EM_CANDIDATES_KEY, SCRATCHPAD_CATEGORY, ScratchpadStore
from profees_coding.mcp.tools import recode


@pytest.fixture
def sample():
    return json.loads((Path(__file__).parent / "fixtures/profees_semantic_search_sample.json").read_text())


def map_sample(sample):
    return CodingModelMapper().map_billing_code_summary(sample, "en-sample", "pa-sample", "df-sample", {"start": "2026-10-07T09:00:00Z"})


def test_sample_mapping_preserves_candidates_and_model_response(sample):
    sample["procedure"][0]["order_catalog_candidates"][0]["orderableId"] = "orderable-1"
    before = deepcopy(sample)
    result = map_sample(sample).to_dict()
    assert sample == before
    for procedure, original in zip(result["procedure"], sample["procedure"]):
        assert procedure["orderCatalogCandidates"] == original["order_catalog_candidates"]
        assert "order_catalog_candidates" not in procedure
        assert procedure["quantity"] == original["quantity"]
        assert procedure["quantity_evidence"] == original["quantity_evidence"]
        assert procedure["sources"] == [{"id": "df-sample", "type": "DOCUMENT_REFERENCE"}]
    assert result["procedure"][2]["modifiers"] == []
    assert result["procedure"][2]["linked_diagnosis"] == []
    assert result["em"][0]["orderCatalogCandidates"] == sample[EM_CANDIDATES_KEY][1]["order_catalog_candidates"]
    assert [item["code"] for item in result["em"]] == ["99213"]
    assert EM_CANDIDATES_KEY not in result


def test_legacy_missing_and_empty_candidate_groups(sample):
    sample.pop(EM_CANDIDATES_KEY)
    for procedure in sample["procedure"]:
        procedure.pop("order_catalog_candidates")
        procedure["quantity"] = None
        procedure["quantity_evidence"] = None
    result = map_sample(sample).to_dict()
    assert "orderCatalogCandidates" not in result["em"][0]
    assert all("orderCatalogCandidates" not in p and "quantity" not in p for p in result["procedure"])
    sample[EM_CANDIDATES_KEY] = []
    assert "orderCatalogCandidates" not in map_sample(sample).to_dict()["em"][0]
    sample[EM_CANDIDATES_KEY] = [{"system": "CPT", "code": "99213", "order_catalog_candidates": []}]
    assert map_sample(sample).to_dict()["em"][0]["orderCatalogCandidates"] == []


def test_em_matching_uses_system_and_code_without_hardcoded_ranges(sample):
    sample["em"][0]["code"] = "G0001"
    sample["em"][0]["system"] = "HCPCS"
    sample[EM_CANDIDATES_KEY][0].update(code="G0001", system="HCPCS")
    sample[EM_CANDIDATES_KEY][1].update(code="G0001", system="CPT")
    assert map_sample(sample).to_dict()["em"][0]["orderCatalogCandidates"] == sample[EM_CANDIDATES_KEY][0]["order_catalog_candidates"]


@pytest.mark.parametrize("bad", [None, {}, [None], [{"CATALOG_CD": "0"}], [{"CATALOG_CD": "abc"}], [{"CATALOG_CD": 123}]])
def test_invalid_candidates_fail_before_mapping(sample, bad):
    sample[EM_CANDIDATES_KEY][0]["order_catalog_candidates"] = bad
    with pytest.raises(ValueError):
        map_sample(sample)


def test_duplicate_groups_and_catalogs_rejected(sample):
    sample[EM_CANDIDATES_KEY].append(deepcopy(sample[EM_CANDIDATES_KEY][0]))
    with pytest.raises(ValueError, match="Duplicate E/M"):
        map_sample(sample)
    sample[EM_CANDIDATES_KEY].pop()
    candidates = sample["procedure"][0]["order_catalog_candidates"]
    candidates.append(deepcopy(candidates[0]))
    with pytest.raises(ValueError, match="Duplicate catalog"):
        map_sample(sample)


def make_store(existing=None, http_status=200, record_success=True):
    captured = []

    def write(scratchpad, operation):
        captured.append((scratchpad, operation))
        if http_status != 200 or not record_success:
            raise RuntimeError("SI write failed")

    access = SimpleNamespace(read=Mock(return_value=[existing] if existing else []), write=Mock(side_effect=write))
    store = ScratchpadStore(access=access, tenant_id="tenant", si_instance_id="si-instance")
    return store, captured


def save_scratchpad(store, sample):
    return asyncio.run(
        store.save(
            patient_id="pa-sample",
            encounter_id="en-sample",
            document_ref_id="df-sample",
            doc_content="Current synthetic note",
            groups=sample.get(EM_CANDIDATES_KEY),
        )
    )


def test_scratchpad_contains_all_groups_and_readable_document(sample):
    store, captured = make_store()
    saved = save_scratchpad(store, sample)
    wire, operation = captured[0]
    collection = Struct()
    assert wire.content[EM_CANDIDATES_KEY].payload.Unpack(collection)
    assert MessageToDict(collection)[EM_CANDIDATES_KEY] == sample[EM_CANDIDATES_KEY]
    assert MessageToDict(collection)["document_ref_id"] == "df-sample"
    assert recode._extract_scratchpad_document_content(saved) == "Current synthetic note"
    assert recode._extract_scratchpad_document_content(MessageToDict(saved)) == "Current synthetic note"
    assert operation == OpType.INSERT
    assert wire.category.code == SCRATCHPAD_CATEGORY
    assert wire.meta.patient.value == "pa-sample"
    assert wire.meta.encounter.value == "en-sample"
    assert wire.time_to_live_seconds == 864000
    store.access.read.assert_called_once_with(patient_id="pa-sample", encounter_id="en-sample")


@pytest.mark.parametrize("groups", [[], None])
def test_recode_updates_existing_record_and_clears_stale_alternatives(sample, groups):
    store, _ = make_store()
    old = save_scratchpad(store, sample)
    unrelated = Struct()
    unrelated.update({"preserve": True})
    old.content["other"].payload.Pack(unrelated)
    old_json = MessageToDict(old)
    sample[EM_CANDIDATES_KEY] = groups
    store, captured = make_store(old_json)
    updated = save_scratchpad(store, sample)
    assert updated.id == old.id
    assert captured[0][1] == OpType.UPDATE
    assert updated.content["other"] == old.content["other"]
    assert old_json == MessageToDict(old)
    if groups is None:
        assert EM_CANDIDATES_KEY not in updated.content
    else:
        collection = Struct()
        updated.content[EM_CANDIDATES_KEY].payload.Unpack(collection)
        assert MessageToDict(collection)[EM_CANDIDATES_KEY] == []


@pytest.mark.parametrize("http_status,success", [(500, True), (200, False)])
def test_scratchpad_http_and_record_failures_are_not_success(sample, http_status, success):
    store, _ = make_store(http_status=http_status, record_success=success)
    with pytest.raises(RuntimeError):
        save_scratchpad(store, sample)


def test_sample_model_response_save_prints_both_wire_payloads(sample, monkeypatch):
    captured = []

    def handle(request):
        assert request.method == "POST" and request.url.path == "/billingCodes"
        captured.append(json.loads(request.content))
        print("\nAgent request to POST /billingCodes:")
        print(json.dumps(captured[0], indent=2))
        return httpx.Response(200, text="saved")

    with httpx.Client(transport=httpx.MockTransport(handle), base_url="https://coding.test") as http_client:
        monkeypatch.setattr(factory, "get_service_url", lambda *_: "https://coding.test")
        monkeypatch.setattr(factory, "get_httpx_client", lambda *_args, **_kwargs: http_client)
        assert factory.CodingClient(object(), object()).save_billing_code(map_sample(sample)) == b"saved"
    assert captured[0]["em"][0]["orderCatalogCandidates"] == sample[EM_CANDIDATES_KEY][1]["order_catalog_candidates"]
    store, _ = make_store()
    saved = save_scratchpad(store, sample)
    print("\nScratchpad SO submitted to SI:")
    print(json.dumps(MessageToDict(saved), indent=2))


@pytest.mark.parametrize("status", [400, 500])
def test_billing_http_failure_propagates(sample, monkeypatch, status):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json={"code": "error", "message": "failed"})), base_url="https://coding.test"
    ) as http_client:
        monkeypatch.setattr(factory, "get_service_url", lambda *_: "https://coding.test")
        monkeypatch.setattr(factory, "get_httpx_client", lambda *_args, **_kwargs: http_client)
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            factory.CodingClient(object(), object()).save_billing_code(map_sample(sample))


@pytest.mark.parametrize(
    "save_enabled,prediction_success,billing_status,scratchpad_status,expected_status,valid",
    [
        (True, True, 200, 200, "scratchpad_saved", True),
        (True, True, 500, 200, "scratchpad_saved", True),
        (True, True, 200, 500, "scratchpad_save_failed", False),
        (False, True, 200, 200, "model_prediction_completed", True),
        (True, False, 200, 200, "model_prediction_failed", False),
        (True, True, 200, 200, "synthetic_fixture_preview", True),
    ],
)
@pytest.mark.parametrize("fixture_persistence", [False, True])
def test_recode_coordinates_both_real_client_boundaries(
    sample, monkeypatch, save_enabled, prediction_success, billing_status, scratchpad_status, expected_status, valid, fixture_persistence
):
    fixture_mode = expected_status == "synthetic_fixture_preview" or fixture_persistence
    fixture_preview = expected_status == "synthetic_fixture_preview" and not fixture_persistence
    if expected_status == "synthetic_fixture_preview" and fixture_persistence:
        expected_status = "scratchpad_saved"
    monkeypatch.setenv("PROFEES_RECODE_USE_SAMPLE_RESPONSE", str(fixture_mode).lower())
    monkeypatch.setenv("PROFEES_FIXTURE_ALLOW_PERSISTENCE", str(fixture_persistence).lower())
    store, si_writes = make_store(http_status=scratchpad_status)
    helper = SimpleNamespace(execute_query=AsyncMock())
    environment = SimpleNamespace(get_si_helper=lambda: helper)
    agent = SimpleNamespace(
        get_authenticated_environment=lambda *_: environment,
        get_instance_environment=lambda _: SimpleNamespace(si_tenant_id="tenant", si_instance_id="si-instance"),
    )
    service = recode.ProFeesCPTCodingService(si_helper=helper, authenticated_environment=environment, coding_model=object())
    monkeypatch.setattr(ScratchpadStore, "from_instance_environment", lambda _: store)
    service.build_coding_request_context = AsyncMock(return_value={"encounter_period": {"start": "2026-10-07T09:00:00Z"}})
    service.run_model_prediction_for_prototype = AsyncMock(
        return_value={
            "called": True,
            "success": prediction_success,
            "model_response": sample,
            "sds_record": {"input": {"soap_text": "Current synthetic note"}, "output": sample},
        }
    )
    lookup = {
        "found": True,
        "selected_document_ref_id": "df-sample",
        "selected_patient_id": "pa-semantic",
        "selected_encounter_id": "en-semantic",
        "selected_filter_result": {"filter_matched": True},
    }
    monkeypatch.setattr(recode, "AGENT_ENVIRONMENT", agent)
    monkeypatch.setattr(recode, "extract_messaging_context", lambda _: object())
    monkeypatch.setattr(recode, "_resolve_runtime_context", lambda _: {"available": True})
    monkeypatch.setattr(recode, "_create_cpt_coding_service", lambda _: service)
    monkeypatch.setattr(recode, "_find_document_reference", AsyncMock(return_value=lookup))
    billing_writes = []

    def handle(request):
        billing_writes.append(json.loads(request.content))
        return (
            httpx.Response(billing_status, text="saved")
            if billing_status == 200
            else httpx.Response(billing_status, json={"code": "error", "message": "failed"})
        )

    with httpx.Client(transport=httpx.MockTransport(handle), base_url="https://coding.test") as http_client:
        monkeypatch.setattr(factory, "get_service_url", lambda *_: "https://coding.test")
        monkeypatch.setattr(factory, "get_httpx_client", lambda *_args, **_kwargs: http_client)
        result = asyncio.run(
            recode.recode_profees_coding(patient_id="source-patient", encounter_id="source-encounter", save_billing_codes=save_enabled)
        )["result"]
    assert result["status"] == expected_status
    assert result["valid"] is valid
    if fixture_mode and prediction_success:
        assert result["synthetic"] is True
    assert len(billing_writes) == 0
    assert result["save_billing_code"]["called"] is False
    assert result["save_billing_code"]["success"] is False
    assert len(si_writes) == int(save_enabled and prediction_success and not fixture_preview)
    if fixture_preview:
        assert result["synthetic"] is True
        assert result["save_billing_code"]["called"] is False
        assert result["save_scratchpad"]["called"] is False
    if si_writes:
        assert si_writes[0][0].meta.patient.value == "pa-semantic"
        assert si_writes[0][0].meta.encounter.value == "en-semantic"
        collection = Struct()
        si_writes[0][0].content[EM_CANDIDATES_KEY].payload.Unpack(collection)
        assert MessageToDict(collection)[EM_CANDIDATES_KEY] == sample[EM_CANDIDATES_KEY]
        assert sample == service.run_model_prediction_for_prototype.return_value["sds_record"]["output"]
    if expected_status == "scratchpad_save_failed":
        assert result["save_billing_code"]["success"] is False
        assert result["save_scratchpad"]["success"] is False


def test_scratchpad_query_failure_does_not_attempt_a_write(sample):
    store, captured = make_store()
    store.access.read.side_effect = RuntimeError("SI unavailable")
    with pytest.raises(RuntimeError, match="SI unavailable"):
        save_scratchpad(store, sample)
    assert captured == []
