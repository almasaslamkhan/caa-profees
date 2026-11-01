import logging
import os
import time
from typing import Any, cast

from adkruntime.agent_environment import AgentEnvironment
from adkruntime.mcp.decorators import tool
from aoacore.types import ToolkitMetadata
from fastmcp import Context
from fastmcp.tools import ToolResult

from profees_coding.mcp.tools import coding, recode

try:
    from proactive_pca_extensions.mcp import UI_PLUGIN_SEMANTIC_ACTION_METADATA_KEY
except ImportError:  # pragma: no cover - supplied by the PCA runtime
    UI_PLUGIN_SEMANTIC_ACTION_METADATA_KEY = "UIPluginSemanticAction"

logger = logging.getLogger(__name__)

PCA_POLICY_META = "com.oracle.pca/policy"

RECODE_TOOL_POLICY_META = {
    PCA_POLICY_META: {
        "timeoutSeconds": 180,
    },
}

EHRM_RECODE_COMPLETE_TOOL_META = {
    **RECODE_TOOL_POLICY_META,
    UI_PLUGIN_SEMANTIC_ACTION_METADATA_KEY: {
        "context": "ehrc",
        "custom_action": "documentation.codingAgentRecodeCompleteEHRMEvent",
    },
}

TOOLKIT_METADATA = ToolkitMetadata(
    name="profees_coding.mcp.ProfeesCodingToolkit",
    version="0.1.0",
    description="Tools for ProFees coding.",
)

AGENT_ENVIRONMENT: AgentEnvironment = cast(AgentEnvironment, None)


def _warmup_recode_model_on_toolkit_load() -> None:
    enabled = os.getenv("PROFEES_WARM_MODEL_ON_TOOLKIT_LOAD", "true").lower() not in {"0", "false", "no"}
    if not enabled:
        _log_tool_event("recode_profees_coding", "warmup_skipped", reason="disabled_by_env")
        return

    started_at = time.monotonic()
    _log_tool_event("recode_profees_coding", "warmup_start")
    try:
        warmed = recode.warmup_coding_model()
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        _log_tool_event("recode_profees_coding", "warmup_done", success=warmed, elapsed_ms=elapsed_ms)
    except Exception:
        logger.exception("ProFees recode model warmup failed during toolkit load; first tool call will lazy initialize.")


def _log_tool_event(tool_name: str, event: str, **values: Any) -> None:
    extra = ""
    if values:
        extra = " " + " ".join("%s=%s" % (key, value) for key, value in values.items())
    logger.info("[profees-mcp][%s] %s%s", tool_name, event, extra)


_warmup_recode_model_on_toolkit_load()


@tool(meta={"UIPluginRendering": coding._AMBULATORY_CODING_A2UI_RENDERING})
async def render_coding_summary(
    patient_id: str | None = None,
    encounter_id: str | None = None,
    description: str | None = None,
    notification_title: str | None = None,
    action_context: dict | None = None,
    mcp_context: Context | None = None,
) -> ToolResult:
    """Return ambulatory coding plugin render metadata.
    :param patient_id: Optional patient identifier for plugin launch.
    :param encounter_id: Optional encounter identifier for plugin launch.
    :param description: Optional generated-code description for notification rendering.
    :param notification_title: Optional notification title override.
    :param action_context: Optional UI event context containing launch properties.
    :param mcp_context: Runtime MCP context, unused by this tool.
    :return: Plugin render metadata and launch parameters.
    """
    _log_tool_event(
        "render_coding_summary",
        "start",
        patient_id=patient_id,
        encounter_id=encounter_id,
        has_description=bool(description),
        has_action_context=bool(action_context),
    )
    result = await coding.render_coding_summary(
        patient_id=patient_id,
        encounter_id=encounter_id,
        description=description,
        notification_title=notification_title,
        action_context=action_context,
    )
    _log_tool_event(
        "render_coding_summary",
        "final_result",
        plugin_name=result.structured_content.get("pluginName"),
        render_mode="canvas",
        has_launch_parameters=bool(result.structured_content.get("launchParameters")),
    )
    return result


@tool(meta={"UIPluginRendering": coding._AMBULATORY_CODING_NOTIFICATION_RENDERING})
async def render_ambulatory_coding_notification(
    patient_id: str | None = None,
    encounter_id: str | None = None,
    description: str | None = None,
    notification_title: str | None = None,
    action_context: dict | None = None,
    mcp_context: Context | None = None,
):
    """Render the native Generated Codes notification card from a managed UI event."""
    _ = mcp_context
    _log_tool_event(
        "render_ambulatory_coding_notification",
        "start",
        patient_id=patient_id,
        encounter_id=encounter_id,
        has_description=bool(description),
    )
    result = await coding.render_ambulatory_coding_notification(
        patient_id=patient_id,
        encounter_id=encounter_id,
        description=description,
        notification_title=notification_title,
        action_context=action_context,
    )
    _log_tool_event(
        "render_ambulatory_coding_notification",
        "final_result",
        rendered_type=result.structured_content.get("renderedType"),
        has_patient_context=bool(result.structured_content.get("patientContext")),
    )
    return result


@tool(meta=RECODE_TOOL_POLICY_META)
async def recode_profees_coding(
    patient_id: str | None = None,
    encounter_id: str | None = None,
    document_ref_id: str | None = None,
    request_id: str | None = None,
    save_billing_codes: bool = True,
    mcp_context: Context | None = None,
) -> dict:
    """Run ProFees recode for a patient/encounter and optional selected document.

    :param patient_id: Required patient identifier from PPCA/UI context.
    :param encounter_id: Required encounter identifier from PPCA/UI context.
    :param document_ref_id: Optional document reference ID for selected-document recode.
    :param request_id: Optional UI request correlation ID to echo in the completion payload.
    :param save_billing_codes: When true, persist predicted codes through the HIM Coding Service.
    :param mcp_context: Injected AOA/FastMCP context.
    :return: Validation, DocumentReference lookup, model prediction, and optional save status.
    """
    _log_tool_event(
        "recode_profees_coding",
        "start",
        patient_id=patient_id,
        encounter_id=encounter_id,
        document_ref_id=document_ref_id,
        save_billing_codes=save_billing_codes,
    )
    recode.AGENT_ENVIRONMENT = AGENT_ENVIRONMENT
    started_at = time.monotonic()
    result = await recode.recode_profees_coding(
        patient_id=patient_id,
        encounter_id=encounter_id,
        document_ref_id=document_ref_id,
        save_billing_codes=save_billing_codes,
        context=mcp_context,
    )
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    completion_payload = recode.build_recode_completion_payload(
        result=result,
        request_id=request_id,
        patient_id=patient_id,
        encounter_id=encounter_id,
        document_ref_id=document_ref_id,
        elapsed_ms=elapsed_ms,
    )
    _log_tool_event(
        "recode_profees_coding",
        "final_result",
        status=completion_payload.get("toolStatus"),
        success=completion_payload.get("success"),
        valid=completion_payload.get("valid"),
        elapsed_ms=completion_payload.get("elapsedMs"),
    )
    return completion_payload


@tool(meta=EHRM_RECODE_COMPLETE_TOOL_META)
async def recode_profees_coding_ehrm(
    patient_id: str | None = None,
    encounter_id: str | None = None,
    document_ref_id: str | None = None,
    request_id: str | None = None,
    save_billing_codes: bool = True,
    mcp_context: Context | None = None,
) -> dict[str, str]:
    """Agent completed the Coding now UI needs to display the new data.

    This has the same inputs and Recode behavior as ``recode_profees_coding``.
    On completion, it triggers the EHRM coding completion semantic action.
    """
    _log_tool_event(
        "recode_profees_coding_ehrm",
        "start",
        patient_id=patient_id,
        encounter_id=encounter_id,
        document_ref_id=document_ref_id,
        save_billing_codes=save_billing_codes,
    )
    try:
        completion_payload = await recode_profees_coding(
            patient_id=patient_id,
            encounter_id=encounter_id,
            document_ref_id=document_ref_id,
            request_id=request_id,
            save_billing_codes=save_billing_codes,
            mcp_context=mcp_context,
        )
    except Exception:
        logger.exception("[profees-mcp][recode_profees_coding_ehrm] recode failed before completion payload")
        return {
            "patientId": patient_id or "",
            "isError": "true",
            "message": "error while generating codes",
            "semanticAction": "true",
        }

    if not completion_payload.get("success"):
        error_message = completion_payload.get("message") or "ProFees recode did not complete successfully."
        _log_tool_event(
            "recode_profees_coding_ehrm",
            "error_result",
            status=completion_payload.get("toolStatus"),
            message=error_message,
        )
        return {
            "patientId": patient_id or "",
            "isError": "true",
            "message": str(error_message),
            "semanticAction": "true",
        }

    # PCA builds executeApplicationActionCommand from this annotated tool's
    # successful result.  The result must contain semantic-action properties,
    # not a command envelope.
    # The tool metadata identifies the EHRM completion action, while this flag
    # marks the returned properties as a semantic-action result.  Without it,
    # the PCA runtime treats a successful recode as ordinary tool data and does
    # not build or publish the executeApplicationActionCommand for the UI.
    ehrm_payload = {
        "patientId": patient_id or "",
        "isError": "false",
        "semanticAction": "true",
    }
    _log_tool_event(
        "recode_profees_coding_ehrm",
        "final_result",
        status=completion_payload.get("toolStatus"),
        success=completion_payload.get("success"),
        valid=completion_payload.get("valid"),
        completion_action="codingAgentRecodeCompleteEHRMEvent",
    )
    return ehrm_payload
