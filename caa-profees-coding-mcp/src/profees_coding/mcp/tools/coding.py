from __future__ import annotations

import json
import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from mcp.types import TextContent

logger = logging.getLogger(__name__)

_AMBULATORY_CODING_PLUGIN_NAME = "ambulatorycoding"
_AMBULATORY_CODING_NOTIFICATION_PLUGIN_NAME = "ambulatorycodingnotification"
_AMBULATORY_CODING_A2UI_PLUGIN_NAME = "a2ui"
_AMBULATORY_CODING_UI_SCHEMA_PATH = "ambulatory-coding/schemas/ambulatory_coding_canvas.json"
_AMBULATORY_CODING_A2UI_ROOT_COMPONENT_ID = "ambulatoryCodingPluginRoot"
_AMBULATORY_CODING_A2UI_RENDERING = {
    "plugin_name": _AMBULATORY_CODING_A2UI_PLUGIN_NAME,
    "agent_name": "ambulatory_coding_canvas",
    "agent_group_name": "ambulatory.coding",
    "render_as": "canvas",
    "plugin_title": "Generated codes",
    "plugin_description": "Ambulatory coding suggestions",
    "catalog_id": "https://oracle.com/specification/v0_8_1_13/oh_catalog.json",
    "ui_schema_path": _AMBULATORY_CODING_UI_SCHEMA_PATH,
    "a2ui_root_component_id": _AMBULATORY_CODING_A2UI_ROOT_COMPONENT_ID,
    "ui_schema": {
        "components": [
            {
                "id": _AMBULATORY_CODING_A2UI_ROOT_COMPONENT_ID,
                "component": {
                    "CaafPluginComponent": {
                        "pluginName": {"path": "/pluginName"},
                        "parameters": {"path": "/launchParameters"},
                    }
                },
            }
        ]
    },
}
_AMBULATORY_CODING_NOTIFICATION_RENDERING = {
    "plugin_name": _AMBULATORY_CODING_NOTIFICATION_PLUGIN_NAME,
    "agent_name": "ambulatory_coding_notification",
    "agent_group_name": "ambulatory.coding",
    "render_as": "card",
    "plugin_title": "Generated codes",
    "plugin_description": "Ambulatory coding suggestions.",
    "plugin_icon": "oj-ux-ico-financial-list",
    "properties_schema": {},
}
_AMBULATORY_CODING_TOOL_META = {
    "UIPluginRendering": _AMBULATORY_CODING_A2UI_RENDERING,
}
_AMBULATORY_CODING_NOTIFICATION_TITLE = "Generated codes"
_NOTIFICATION_LOG_PREFIX = "[profees-mcp][show_ambulatory_coding]"
_NOTIFICATION_MCP_LOG_PREFIX = "[AmbulatoryCodingNotification_MCP]"


def _attach_fastmcp_meta(func: Callable[..., Any], *, meta: dict[str, Any]) -> Callable[..., Any]:
    func.__fastmcp__ = SimpleNamespace(meta=meta)
    return func


def _add_notification_description(target: dict[str, Any], notification_description: str) -> None:
    if notification_description:
        target["description"] = notification_description
        target["notificationDescription"] = notification_description


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    return next((mapping[key] for key in keys if mapping.get(key)), None)


def _first_present_in_event_context(mapping: dict[str, Any], *keys: str) -> Any:
    """Read a managed-event value whether it is direct or nested by the host."""
    value = _first_present(mapping, *keys)
    if value:
        return value

    for container_key in ("properties", "auxiliaryProps", "data", "result"):
        nested = mapping.get(container_key)
        if isinstance(nested, dict):
            value = _first_present_in_event_context(nested, *keys)
            if value:
                return value
    return None


def _build_ambulatory_coding_plugin_response(
    *,
    patient_id: str | None = None,
    encounter_id: str | None = None,
    notification_title: str | None = None,
    notification_description: str | None = None,
) -> dict[str, Any]:
    notification_title = str(notification_title or _AMBULATORY_CODING_NOTIFICATION_TITLE)
    notification_description = str(notification_description or "")
    launch_data = {
        "renderedType": "canvas",
        "patientId": str(patient_id or ""),
        "encounterId": str(encounter_id or ""),
        "notificationTitle": notification_title,
    }
    _add_notification_description(launch_data, notification_description)
    launch_parameters = {
        **launch_data,
        "data": dict(launch_data),
    }
    return {
        "pluginName": _AMBULATORY_CODING_PLUGIN_NAME,
        "launchParameters": launch_parameters,
    }


def _build_ambulatory_coding_notification_payload(
    *,
    patient_id: str | None = None,
    encounter_id: str | None = None,
    description: str | None = None,
    notification_title: str | None = None,
    open_immediately: bool = True,
) -> dict[str, Any]:
    """Build the native conversational notification contract.

    This deliberately does not contain a ``pluginName`` launch instruction.
    The Agent Panel selects the published notification card from the tool's
    static UIPluginRendering metadata.
    """
    title = str(notification_title or _AMBULATORY_CODING_NOTIFICATION_TITLE)
    return {
        "renderedType": "notification",
        "contractVersion": "1.0",
        "title": title,
        "description": str(description or ""),
        "patientContext": {
            "contextType": "patient-chart",
            "patientId": str(patient_id or ""),
            "encounterId": str(encounter_id or ""),
        },
        "actions": {"openEnabled": True, "openImmediately": open_immediately},
    }


async def render_coding_summary(
    patient_id: str | None = None,
    encounter_id: str | None = None,
    description: str | None = None,
    notification_title: str | None = None,
    action_context: dict[str, Any] | None = None,
) -> ToolResult:
    """Return the A2UI ambulatory coding canvas metadata."""
    action_context = action_context or {}
    logger.info(
        "%s open_requested supplied_patient_id=%s supplied_encounter_id=%s " "supplied_description=%s action_context_keys=%s",
        _NOTIFICATION_LOG_PREFIX,
        bool(patient_id),
        bool(encounter_id),
        bool(description),
        sorted(action_context.keys()),
    )

    patient_id = patient_id or _first_present(action_context, "patient_id", "patientId")
    encounter_id = encounter_id or _first_present(
        action_context,
        "encounter_id",
        "encounterId",
        "semantic_object_id",
        "semanticObjectId",
        "id",
    )

    notification_title = notification_title or _first_present(action_context, "notificationTitle", "title") or _AMBULATORY_CODING_NOTIFICATION_TITLE
    notification_title = str(notification_title)
    # A notification is an explicit UI request.  Do not use descriptive
    # metadata retained in action_context: it can belong to a previous card
    # and would incorrectly turn a manual "show ambulatory coding" request
    # into a notification.
    notification_description = str(description) if description else ""

    payload = _build_ambulatory_coding_plugin_response(
        patient_id=patient_id,
        encounter_id=encounter_id,
        notification_title=notification_title,
        notification_description=notification_description,
    )
    logger.info(
        "%s canvas_payload_ready patient_context_ready=%s " "description_present=%s render_plugin=%s render_as=canvas",
        _NOTIFICATION_LOG_PREFIX,
        bool(patient_id and encounter_id),
        bool(notification_description),
        payload["pluginName"],
    )
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
        meta={
            "UIPluginRendering": {
                "plugin_title": notification_title,
            }
        },
    )


async def render_ambulatory_coding_notification(
    patient_id: str | None = None,
    encounter_id: str | None = None,
    description: str | None = None,
    notification_title: str | None = None,
    action_context: dict[str, Any] | None = None,
    open_immediately: bool = True,
) -> ToolResult:
    """Render the native Generated Codes notification card without an LLM turn."""
    action_context = action_context or {}
    logger.info(
        "%s notification_requested supplied_patient_id=%s " "supplied_encounter_id=%s supplied_description=%s action_context_keys=%s",
        _NOTIFICATION_MCP_LOG_PREFIX,
        bool(patient_id),
        bool(encounter_id),
        bool(description),
        sorted(action_context.keys()),
    )
    patient_id = patient_id or _first_present_in_event_context(action_context, "patient_id", "patientId")
    encounter_id = encounter_id or _first_present_in_event_context(
        action_context,
        "encounter_id",
        "encounterId",
        "semantic_object_id",
        "semanticObjectId",
    )
    description = description or _first_present_in_event_context(action_context, "description", "notificationDescription")
    notification_title = (
        notification_title or _first_present_in_event_context(action_context, "notificationTitle", "title") or _AMBULATORY_CODING_NOTIFICATION_TITLE
    )
    if not patient_id or not encounter_id:
        logger.warning(
            "%s notification_context_incomplete missing_patient_id=%s " "missing_encounter_id=%s",
            _NOTIFICATION_MCP_LOG_PREFIX,
            not bool(patient_id),
            not bool(encounter_id),
        )

    payload = _build_ambulatory_coding_notification_payload(
        patient_id=patient_id,
        encounter_id=encounter_id,
        description=description,
        notification_title=notification_title,
        open_immediately=open_immediately,
    )
    logger.info(
        "%s notification_payload_ready patient_context_ready=%s "
        "description_present=%s description_length=%s open_immediately=%s "
        "title_overridden=%s render_plugin=%s render_as=card",
        _NOTIFICATION_MCP_LOG_PREFIX,
        bool(patient_id and encounter_id),
        bool(description),
        len(description or ""),
        open_immediately,
        notification_title != _AMBULATORY_CODING_NOTIFICATION_TITLE,
        _AMBULATORY_CODING_NOTIFICATION_PLUGIN_NAME,
    )
    result = ToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
        meta={
            "UIPluginRendering": {
                "plugin_title": str(notification_title),
                "plugin_description": str(description or ""),
            }
        },
    )
    logger.info(
        "%s notification_result_ready_for_agent_panel render_plugin=%s rendered_type=%s " "patient_context_ready=%s open_immediately=%s",
        _NOTIFICATION_MCP_LOG_PREFIX,
        _AMBULATORY_CODING_NOTIFICATION_PLUGIN_NAME,
        payload["renderedType"],
        bool(patient_id and encounter_id),
        open_immediately,
    )
    return result


_attach_fastmcp_meta(render_coding_summary, meta=_AMBULATORY_CODING_TOOL_META)
_attach_fastmcp_meta(
    render_ambulatory_coding_notification,
    meta={"UIPluginRendering": _AMBULATORY_CODING_NOTIFICATION_RENDERING},
)


def _register_fastmcp_tool(
    mcp: "FastMCP",
    func: Callable[..., Any],
    *,
    meta: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    try:
        decorator = mcp.tool(meta=meta) if meta else mcp.tool()
    except TypeError:
        decorator = mcp.tool()
    return decorator(func)


def register_coding_tools(
    mcp: FastMCP,
    get_service: Callable[[], Any] | None = None,
    *,
    include_msdrg_tool: bool = True,
) -> None:
    """Register ambulatory coding tools with a FastMCP server.
    The get_service and include_msdrg_tool parameters are accepted for
    compatibility with the server's existing coding-tool registration call.
    """
    _ = get_service, include_msdrg_tool
    _register_fastmcp_tool(
        mcp,
        render_coding_summary,
        meta=_AMBULATORY_CODING_TOOL_META,
    )
    _register_fastmcp_tool(
        mcp,
        render_ambulatory_coding_notification,
        meta={"UIPluginRendering": _AMBULATORY_CODING_NOTIFICATION_RENDERING},
    )
    logger.info(
        "[profees-mcp] registered ambulatory tools=%s",
        ["render_coding_summary", "render_ambulatory_coding_notification"],
    )
