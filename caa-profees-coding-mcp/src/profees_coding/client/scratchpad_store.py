"""Persist the latest generated draft and its complete E/M catalog collection."""

import asyncio
import logging
from time import time_ns
from uuid import NAMESPACE_URL, uuid4, uuid5

from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct
from semantic_object_model.core.core_pb2 import Resource
from semantic_object_model.core.op_type_pb2 import OpType
from semantic_object_model.workflow.scratchpad_pb2 import Scratchpad

SCRATCHPAD_CATEGORY = "GENERATED_DRAFT_BILLING_CODE"
EM_CANDIDATES_KEY = "ambulatory_em_order_catalog_candidates"
SCRATCHPAD_TTL_SECONDS = 864000
SCRATCHPAD_SERVICE_ACCOUNT = "clinical-rev-cycle"
logger = logging.getLogger(__name__)


class AgentScratchpadAccess:
    """Use the agent's SI client and service context, never the ADK context."""

    def __init__(self, si_details):
        self.si_details = si_details

    def _client_and_token(self):
        from cda_semantic_index.semantic_index import SemanticIndex
        from cda_utils.jwt_utils import generate_real_auth_token

        client = SemanticIndex(self.si_details)
        # The service-context JWT is not a substitute for authenticated transport.
        if client.signer is None:
            raise RuntimeError("Agent-style scratchpad access requires a configured OCI request signer; no unsigned SI request was sent")
        return client, generate_real_auth_token(SCRATCHPAD_SERVICE_ACCOUNT, semantic_index_details=self.si_details)

    def read(self, *, encounter_id, patient_id=None):
        # Identifiers enter SQL literals; quote them rather than interpolating raw input.
        encounter = encounter_id.replace("'", "''")
        query = (
            "SELECT SCRATCHPAD_VALUE FROM SCRATCHPAD "
            f"WHERE SCRATCHPAD_ENCOUNTER_ID='{encounter}' "
            f"AND SCRATCHPAD_CATEGORY='{SCRATCHPAD_CATEGORY}'"
        )
        client, token = self._client_and_token()
        records = [
            getattr(record, "semantic_object_", record) for record in client.read(sql_query=query, semantic_object_type=Scratchpad, auth_token=token)
        ]
        if patient_id:
            records = [record for record in records if record.meta.patient.value == patient_id]
        records.sort(key=lambda record: (record.meta.last_updated.seconds, record.meta.last_updated.nanos), reverse=True)
        logger.info("[profees-mcp] Agent-style scratchpad read succeeded; matching_records=%d", len(records))
        return records[:1]

    def write(self, scratchpad, operation):
        client, token = self._client_and_token()
        method = client.update if operation == OpType.UPDATE else client.create
        method(semantic_object=scratchpad, auth_token=token, opc_request_id=str(uuid4()))


class ScratchpadStore:
    def __init__(self, *, access, tenant_id, si_instance_id):
        self.access = access
        self.tenant_id = tenant_id
        self.si_instance_id = si_instance_id

    @classmethod
    def from_instance_environment(cls, instance):
        from cda_semantic_index.connection.index import SemanticIndexDetails

        details = SemanticIndexDetails(
            index_name="profees_mcp_scratchpad",
            endpoint=instance.get_semantic_index_endpoint(),
            tenant=instance.si_tenant_id,
            instance_id=instance.si_instance_id,
            config=None,
        )
        return cls(access=AgentScratchpadAccess(details), tenant_id=instance.si_tenant_id, si_instance_id=instance.si_instance_id)

    async def read_previous(self, encounter_id):
        records = await asyncio.to_thread(self.access.read, encounter_id=encounter_id)
        return records[0] if records else None

    async def save(self, *, patient_id, encounter_id, document_ref_id, doc_content, groups):
        if not all((self.tenant_id, self.si_instance_id, patient_id, encounter_id, document_ref_id)):
            raise ValueError("Scratchpad persistence requires SI, patient, encounter and document identifiers")
        if not isinstance(doc_content, str) or not doc_content:
            raise ValueError("Scratchpad persistence requires current document content")
        existing = await asyncio.to_thread(self.access.read, patient_id=patient_id, encounter_id=encounter_id)
        scratchpad = Scratchpad()
        if existing:
            if isinstance(existing[0], Scratchpad):
                scratchpad.CopyFrom(existing[0])
            else:
                ParseDict(existing[0], scratchpad)
            if not scratchpad.id:
                raise ValueError("Existing scratchpad is missing its ID")
        else:
            scratchpad.id = f"sq-{uuid5(NAMESPACE_URL, f'profees-generated-draft:{patient_id}:{encounter_id}')}"
        scratchpad.meta.patient.type = Resource.ResourceType.PATIENT
        scratchpad.meta.patient.value = patient_id
        scratchpad.meta.encounter.type = Resource.ResourceType.ENCOUNTER
        scratchpad.meta.encounter.value = encounter_id
        seconds, nanos = divmod(time_ns(), 1_000_000_000)
        scratchpad.meta.last_updated.seconds = seconds
        scratchpad.meta.last_updated.nanos = nanos
        scratchpad.category.code = SCRATCHPAD_CATEGORY
        scratchpad.category.text = "GeneratedDraftBillingCode"
        scratchpad.category.coding_key = "GENERATED_DRAFT_BILLING_CODE_KEY"
        scratchpad.time_to_live_seconds = SCRATCHPAD_TTL_SECONDS

        document = Struct()
        ParseDict({"content": doc_content, "document_ref_id": document_ref_id}, document)
        scratchpad.content["document"].payload.Pack(document)
        # Replace the entire collection, including an explicit empty result. Never retain stale alternatives.
        if groups is None:
            if EM_CANDIDATES_KEY in scratchpad.content:
                del scratchpad.content[EM_CANDIDATES_KEY]
        else:
            collection = Struct()
            ParseDict({"schema_version": 1, "document_ref_id": document_ref_id, EM_CANDIDATES_KEY: groups}, collection)
            scratchpad.content[EM_CANDIDATES_KEY].payload.Pack(collection)

        await asyncio.to_thread(self._write, scratchpad, OpType.UPDATE if existing else OpType.INSERT)
        return scratchpad

    def _write(self, scratchpad, operation):
        self.access.write(scratchpad, operation)
