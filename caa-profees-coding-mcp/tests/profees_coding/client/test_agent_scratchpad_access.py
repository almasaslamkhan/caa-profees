import asyncio
import sys
import subprocess
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from google.protobuf.struct_pb2 import Struct
from semantic_object_model.core.op_type_pb2 import OpType
from semantic_object_model.workflow.scratchpad_pb2 import Scratchpad

from profees_coding.client.scratchpad_store import AgentScratchpadAccess, SCRATCHPAD_CATEGORY, ScratchpadStore
from profees_coding.mcp.tools import recode


def test_installed_agent_dependencies_import_in_fresh_process():
    # Mocked routing tests cannot catch incompatibilities inside vendor packages.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cda_semantic_index.connection.index import SemanticIndexDetails; "
            "from cda_semantic_index.semantic_index import SemanticIndex; "
            "from cda_utils.jwt_utils import generate_real_auth_token; "
            "assert callable(generate_real_auth_token)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def agent_dependencies(monkeypatch):
    client = Mock(signer=object())
    client_factory = Mock(return_value=client)
    token_factory = Mock(return_value="synthetic-service-context")
    details_factory = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    modules = {
        "cda_semantic_index": {},
        "cda_semantic_index.connection": {},
        "cda_semantic_index.connection.index": {"SemanticIndexDetails": details_factory},
        "cda_semantic_index.semantic_index": {"SemanticIndex": client_factory},
        "cda_utils": {},
        "cda_utils.jwt_utils": {"generate_real_auth_token": token_factory},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return client, client_factory, token_factory, details_factory


def record(patient, updated):
    value = Scratchpad(id=f"sq-{updated}")
    value.meta.patient.value = patient
    value.meta.last_updated.seconds = updated
    return value


def test_reads_with_agent_token_category_and_latest_patient_record(agent_dependencies):
    client, factory, tokens, _ = agent_dependencies
    newest = record("pa-target", 3)
    client.read.return_value = [record("pa-other", 4), record("pa-target", 1), newest]
    details = object()
    access = AgentScratchpadAccess(details)
    assert access.read(encounter_id="en-'quoted", patient_id="pa-target") == [newest]
    factory.assert_called_once_with(details)
    tokens.assert_called_once_with("clinical-rev-cycle", semantic_index_details=details)
    kwargs = client.read.call_args.kwargs
    assert "SCRATCHPAD_ENCOUNTER_ID='en-''quoted'" in kwargs["sql_query"]
    assert f"SCRATCHPAD_CATEGORY='{SCRATCHPAD_CATEGORY}'" in kwargs["sql_query"]
    assert kwargs["semantic_object_type"] is Scratchpad
    assert kwargs["auth_token"] == "synthetic-service-context"


def test_library_wrappers_are_unpacked_before_payload_handling(agent_dependencies):
    client, _, _, _ = agent_dependencies
    scratchpad = record("pa-target", 1)
    client.read.return_value = [SimpleNamespace(semantic_object_=scratchpad)]
    assert AgentScratchpadAccess(object()).read(encounter_id="en-target", patient_id="pa-target") == [scratchpad]


@pytest.mark.parametrize("operation,method", [(OpType.INSERT, "create"), (OpType.UPDATE, "update")])
def test_writes_use_agent_context_not_adk_or_delete(agent_dependencies, operation, method):
    client, _, tokens, _ = agent_dependencies
    scratchpad = record("pa-target", 1)
    details = object()
    AgentScratchpadAccess(details).write(scratchpad, operation)
    kwargs = getattr(client, method).call_args.kwargs
    assert kwargs["semantic_object"] is scratchpad
    assert kwargs["auth_token"] == "synthetic-service-context"
    assert kwargs["opc_request_id"]
    tokens.assert_called_once_with("clinical-rev-cycle", semantic_index_details=details)
    client.delete.assert_not_called()


def test_no_unsigned_request_or_token_when_signer_unavailable(agent_dependencies):
    client, _, tokens, _ = agent_dependencies
    client.signer = None
    with pytest.raises(RuntimeError, match="configured OCI request signer"):
        AgentScratchpadAccess(object()).read(encounter_id="en-target")
    client.read.assert_not_called()
    tokens.assert_not_called()


def test_real_token_failure_does_not_fall_back_or_send_si_request(agent_dependencies):
    client, _, tokens, _ = agent_dependencies
    tokens.side_effect = RuntimeError("Service-account exchange denied")
    with pytest.raises(RuntimeError, match="Service-account exchange denied"):
        AgentScratchpadAccess(object()).read(encounter_id="en-target")
    client.read.assert_not_called()
    client.create.assert_not_called()
    client.update.assert_not_called()


def test_instance_configuration_used_for_scratchpad_only(agent_dependencies):
    client, _, _, details_factory = agent_dependencies
    instance = SimpleNamespace(si_tenant_id="tenant", si_instance_id="si-instance", get_semantic_index_endpoint=lambda: "https://si.example.test")
    store = ScratchpadStore.from_instance_environment(instance)
    details_factory.assert_called_once_with(
        index_name="profees_mcp_scratchpad", endpoint="https://si.example.test", tenant="tenant", instance_id="si-instance", config=None
    )
    client.read.return_value = []
    assert asyncio.run(store.read_previous("en-target")) is None


def test_previous_document_read_uses_agent_store(agent_dependencies):
    client, _, _, _ = agent_dependencies
    scratchpad = record("pa-target", 1)
    document = Struct()
    document.update({"content": "Synthetic previous note"})
    scratchpad.content["document"].payload.Pack(document)
    client.read.return_value = [scratchpad]
    store = ScratchpadStore(access=AgentScratchpadAccess(object()), tenant_id="tenant", si_instance_id="si")
    helper = Mock()
    service = recode.ProFeesCPTCodingService(si_helper=helper, authenticated_environment=object(), scratchpad_store_factory=lambda: store)
    assert asyncio.run(service._get_previous_doc_content_from_scratchpad("en-target")) == "Synthetic previous note"
    helper.execute_query.assert_not_called()


def test_denied_read_is_logged_and_never_triggers_write(agent_dependencies, caplog):
    client, _, _, _ = agent_dependencies
    client.read.side_effect = RuntimeError("SI-101: Not authorized to perform read")
    store = ScratchpadStore(access=AgentScratchpadAccess(object()), tenant_id="tenant", si_instance_id="si")
    service = recode.ProFeesCPTCodingService(si_helper=Mock(), authenticated_environment=object(), scratchpad_store_factory=lambda: store)
    assert asyncio.run(service._get_previous_doc_content_from_scratchpad("en-target")) is None
    assert "SI-101" in caplog.text
    with pytest.raises(RuntimeError, match="SI-101"):
        asyncio.run(
            store.save(patient_id="pa-target", encounter_id="en-target", document_ref_id="df-target", doc_content="Synthetic note", groups=[])
        )
    client.create.assert_not_called()
    client.update.assert_not_called()
