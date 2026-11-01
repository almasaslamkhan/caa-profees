# MCP Tool Definition

Complete this file only if the submission includes an MCP tool.

## Tool source code
`cdrc/profees_coding/mcp/tools/coding.py`

Supporting files:

- `cdrc/profees_coding/mcp/server.py`
- `cdrc/profees_coding/mcp/service.py`
- `cdrc/profees_coding/mcp/prompts/coding.py`
- `profees_coding_skills/src/profees_ambulatory_coding_skills/skills/ambulatory-coding/schemas/ambulatory_coding_canvas.json`

## Tool name
`render_coding_summary`

Persona-facing hot tool name in the physician persona: `profees_coding_render_coding_summary`.

## Tool purpose
`render_coding_summary` launches the ambulatory ProFees generated-codes canvas. When called with no arguments, it returns a plugin launch payload and A2UI metadata so the `ambulatorycoding` canvas plugin can fetch billing code data from the active A2UI patient and encounter context. When called with an encounter note, payload JSON, or payload file, it invokes `ProfeesModel.predict`, maps the returned coding summary into an ambulatory generated-codes payload, and returns A2UI metadata that launches the `ambulatorycoding` canvas plugin.

The tool is read-only from a billing-system perspective. It produces generated coding recommendations for review and rendering. The **Submit codes** button is a canvas action handled by the UI/plugin workflow, not by this MCP render tool.

## Tool exposure

| Field | Value |
|---|---|
| Exposure | Skill-scoped public MCP; also exposed as one bounded hot tool in the physician persona |
| Owning skill or allowed skills | `ambulatory-coding` |
| MCP server name and endpoint, if MCP-backed | Server name: `profees_coding`; deployed endpoint: `http://caa-profees-coding-mcp.caa-profees-coding-mcp.svc.cluster.local:80/mcp`; local override can use `MCP_PROFEES_CODING_URL=http://127.0.0.1:9000/mcp` |
| Hot tool config, if hot | `profees_coding_render_coding_summary` |

Use the narrowest correct exposure. In PPCA physician, only the render/open
canvas tool is exposed as hot; the full ProFees MCP server is not enabled through
`auto_hot_tool_servers`.

## Tool contract

### Inputs

| Parameter | Type | Required | Source | LLM-settable | Server-injected | Description |
|---|---|---|---|---|---|---|
| patient_id | string | No | UI / System | Yes | No | Optional patient identifier to pass to the plugin launch payload. |
| encounter_id | string | No | UI / System | Yes | No | Optional encounter identifier to pass to the plugin launch payload. |

### Outputs

| Field | Type | Description | Notes |
|---|---|---|---|
| pluginName | string | Name of the CAAD plugin to launch. | Successful result uses `ambulatorycoding`. |
| renderMode | string | Rendering mode for the plugin. | Successful result uses `canvas`. |
| launchParameters | object | Patient, encounter, and billing response data included for plugin compatibility. | The A2UI schema passes `parameters.path: "/launchParameters"` to the CAAD plugin. |
| result | object | UI-ready generated-codes payload passed to A2UI/CAAD when the tool generates a coding summary. | Includes patient, encounter, billing response, office visit, preventive care, linked diagnosis IDs, diagnosis slots, and options. Omitted for the no-argument plugin-launch path. |
| _meta | object | FastMCP/A2UI rendering metadata. | Includes `UIPluginRendering` with plugin title, catalog ID, schema path, and root component ID. |
| success | boolean | Error status for recoverable failures. | Present as `false` on failure responses; omitted from successful plugin payloads. |
| error | string | Recoverable error message. | Returned when repo root, payload loading, model prediction, or subprocess fallback fails. |
| timeout_seconds | number | Resolved timeout used for prediction. | Included in prediction failure responses. |

Successful `result` payload fields include:

- `encounterId`
- `patientId`
- `officeVisit.selectedCode`
- `officeVisit.modifier`
- `officeVisit.lengthMinutes`
- `officeVisit.detailText`
- `preventiveCare.selectedCode`
- `preventiveCare.modifier`
- `linkedDiagnoses`
- `procedures`

## Idempotency
The tool is functionally idempotent for the same payload, note, model configuration, and prompt/model version. Repeated calls may differ if the underlying model, prompts, environment, or sampled model behavior changes. Duplicate invocation is not blocked because the tool is read-only and does not submit or persist codes.

## Preconditions

- The MCP server can locate a repository root containing `cdrc/`, either from `CDRC_REPO_ROOT` or by walking parent directories from the tool source file.
- A valid ProFees `InputData` payload is available when `payload_json`, `payload_path`, or `encounter_note` is used. For the no-argument plugin-launch path, patient and encounter context can be supplied by A2UI and the plugin fetches billing code data.
- The ProFees model dependencies are importable in-process or a valid CDRC Python interpreter is available through `CDRC_PYTHON` or `.venv/bin/python`.
- Required AI Factory/model credentials and model runtime environment variables are configured for the selected runtime mode.
- The caller is authorized to call the MCP server.

## Postconditions
After successful completion, the caller receives an A2UI-compatible plugin launch payload for the generated-codes canvas. The payload contains generated coding recommendations and option lists for review. No source system, chart, claim, or billing record is modified.

## Failure modes

| Failure mode | Cause | User-visible effect | Retryable | Orchestrator action |
|---|---|---|---|---|
| Repository root not found | `CDRC_REPO_ROOT` is unset/invalid and parent search cannot find `cdrc/`. | Returns `success: false` with guidance to set `CDRC_REPO_ROOT`. | Yes | Ask for corrected runtime configuration or run from the clinical-rev-cycle checkout. |
| Payload load failure | `payload_json` is malformed, `payload_path` does not exist, or payload is not valid JSON. | Returns `success: false` with payload loading error. | Yes | Ask caller for valid payload JSON/path or encounter note. |
| Invalid model input | Payload does not satisfy ProFees `InputData` validation. | Prediction fails and tool returns `success: false` after fallback also fails. | Yes | Ask caller to correct missing or invalid payload fields. |
| In-process model failure | Import error, dependency mismatch, model initialization error, credential failure, or prediction exception. | Tool attempts subprocess fallback. If fallback succeeds, user sees normal canvas payload. | Yes | Allow fallback; if both fail, surface structured error. |
| Subprocess fallback failure | CDRC Python interpreter missing, subprocess returns non-zero, dependencies unavailable, or no JSON object is emitted. | Returns `success: false` with in-process and subprocess error details. | Yes | Ask for environment/dependency fix; avoid blind retries. |
| Timeout | Prediction exceeds `timeout_seconds` or `PROFEES_TOOL_TIMEOUT_SEC`. | Returns `success: false` with timeout metadata if both execution paths fail. | Yes | Retry only with a valid reason, smaller payload, or adjusted timeout. |
| Unauthorized MCP call | Missing or invalid `x-api-key` when API-key enforcement is enabled. | HTTP 401 or 503 from middleware before tool execution. | Yes | Reauthenticate or configure `MCP_API_KEY` / `MCP_REQUIRE_API_KEY`. |

## Extra security model
The MCP server can require an API key through `MCP_REQUIRE_API_KEY=true` and `MCP_API_KEY`. Health, readiness, and liveness endpoints bypass this check; MCP tool calls do not. The tool itself should receive only encounter-scoped payloads or notes and should not be used to fetch unbounded records.

Secrets, credentials, MAX session IDs, MRNs, FINs, and environment-specific identifiers must not be committed to documentation or test fixtures.

## Will this tool be called by external apps such as ChatGPT?
No. This tool is intended for skill-scoped MCP use inside the ProFees/PCA coding runtime, not as a globally exposed external-app tool. External callers would need additional authorization, PHI handling controls, and product review before direct access is safe.

## Rate limits
There are no tool-specific rate limits defined in `render_coding_summary`. Effective limits come from the MCP deployment, worker count, model provider/AI Factory quotas, and any upstream runtime controls. Calls with note or payload input can invoke a model prediction, so orchestrators should avoid blind retries and duplicate recode calls.
