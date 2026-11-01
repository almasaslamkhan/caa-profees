# Recode Order Catalog Persistence

The model input and `src/cdrc/` are unchanged. The MCP mapper accepts the semantic-search response and prepares procedure candidates and matching predicted E/M candidates as `orderCatalogCandidates` for the existing `POST /billingCodes` endpoint. Billing API persistence is currently temporarily disabled; its payload is still mapped and captured. E/M matching uses `(system, code)`, not a hard-coded code list. Candidate identifiers and additional fields are preserved, without fabricating orderable IDs or rankings.

The bundled legacy `ProfeesResponse` schema in `src/cdrc/profees_coding/deploy/api/utils.py` does not declare the new catalog fields. The model team must supply the updated response/schema so these fields reach the MCP boundary; fields already discarded inside the model cannot be recovered by the mapper. The tests use the supplied synthetic response rather than claiming the bundled model already generates catalog candidates.

## Scratchpad Contract (Pending Consumer Confirmation)

Category: `GENERATED_DRAFT_BILLING_CODE`. Scope: the latest generated draft per patient/encounter, matching the existing previous-note reader. TTL: 864000 seconds (10 days), matching the agent's existing policy. Confirm these choices with Coding Service before rollout.

The Scratchpad SO has patient/encounter references in `meta`. Its `content` map contains protobuf `Any` payloads packing `google.protobuf.Struct`:

- `document`: `{ "content": "current note", "document_ref_id": "df-..." }`. The existing previous-note reader remains compatible.
- `ambulatory_em_order_catalog_candidates`: `{ "schema_version": 1, "document_ref_id": "df-...", "ambulatory_em_order_catalog_candidates": [ ...all returned code groups... ] }`.

Each group preserves the model's `system`, `code`, `order_catalog_candidates`, and additional fields. The complete collection includes the predicted E/M group as well as alternatives. A consumer should verify the document reference and read the group matching `(system, code)`. This content key and shape are an implementation proposal, not a confirmed Coding Service contract.

The latest existing record is updated in place, preserving unrelated content keys. No existing record is deleted before writing. If none exists, a deterministic patient/encounter-scoped ID is inserted. An explicitly empty collection replaces old alternatives with an empty collection. A missing/null collection removes the old candidates entry while saving the current note, so alternatives from a prior generation are not reused.

## Save Outcomes

The billing request is mapped/validated first, including every returned E/M candidate group. Only scratchpad persistence is currently attempted when `save_billing_codes=True`; `False` skips persistence. Synthetic fixtures remain preview-only unless `PROFEES_FIXTURE_ALLOW_PERSISTENCE=true` explicitly enables persistence.

A failed scratchpad read or write during persistence returns `scratchpad_save_failed` and `valid: false`. There is no distributed transaction, rollback, or automated retry of the whole recode.

Concurrent whole-recode calls are not serialized. Query/update is not compare-and-swap; last-write behavior, index visibility, multiple-document encounters, TTL expiry and compatibility with the agent's delete/recreate writer require end-to-end consumer validation. No arbitrary encounter-wide deletion is performed by MCP.

## Scratchpad Authentication

For the current getOrCreate diagnostic comparison, ADK operations request `system:serviceaccount:clinical-rev-cycle:default` instead of `medical-coding-agent`. Resource ID `defaultId`, resource name `Medical coding agent`, and scope `internal` are unchanged. This is a diagnostic account switch, not confirmation that authentication succeeds. Both the previous-note scratchpad reader and persistence use a separate SI library path: `cda_semantic_index.SemanticIndex` plus `cda_utils.jwt_utils.generate_real_auth_token("clinical-rev-cycle", semantic_index_details=...)`. There is no fallback to the ADK authorization context or locally constructed test tokens.

Compatibility note: the coworker's agent uses `cda-utils==2.24.2` with an older OCI SDK, and locally constructs service-context tokens. That helper imports `oci._vendor.jwt`, which the newer MCP SDK lacks. Downgrading OCI conflicts with ADK's cryptography requirements. The MCP therefore uses `cda-utils==3.1.0`, whose real token helper performs an authenticated service-account exchange. This fixes package compatibility but is not the exact legacy authentication behavior: account resolution can still fail. The library needs approved token-service routing and OCI signing configuration; live authorization is not established by import tests.

The SI endpoint, tenant and instance are taken from the injected instance environment, not hard-coded. Reads use the agent's encounter/category SQL query; persistence filters by patient and selects the latest record. Updates preserve existing content instead of the agent's delete/recreate pattern. The existing MCP payload schema is unchanged.

The service-context token is not an authentication credential by itself. This path requires the library to obtain a real OCI request signer; if its signer is `None`, MCP rejects the operation before sending an unsigned request. An approved OCI runtime or approved local metadata/signing configuration is required. Merely choosing a service-account name does not grant permission. Configure this through your environment owner, not by copying tokens into source code.

Read failures are logged with exception details and remain optional for prediction. Successful empty reads are logged as `matching_records=0`; persistence read failures stop the write. Tokens are never logged. This implementation has mock-based tests, not evidence of live SI authorization success.

## Local Verification

Run from the MCP repository with its virtual environment activated:

```bash
pytest tests/profees_coding/client/test_order_catalog_mapping.py -q --no-cov
pytest tests/profees_coding/client/test_agent_scratchpad_access.py -q --no-cov
pytest tests/profees_coding/client/test_order_catalog_mapping.py::test_sample_model_response_save_prints_both_wire_payloads -s -q --no-cov
```

The second command prints the serialized billing request and Scratchpad SO using the agent's synthetic sample fixture. HTTP and SI are mocked; no live model or SI data is accessed.
