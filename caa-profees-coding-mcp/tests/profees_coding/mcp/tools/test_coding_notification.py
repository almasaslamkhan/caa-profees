from __future__ import annotations

import asyncio
import inspect

from profees_coding.mcp import aoa_toolkit
from profees_coding.mcp.tools import coding


def _run(coro):
    return asyncio.run(coro)


def test_render_coding_summary_contract_excludes_generation_inputs() -> None:
    parameters = inspect.signature(coding.render_coding_summary).parameters

    assert set(parameters) == {
        "patient_id",
        "encounter_id",
        "description",
        "notification_title",
        "action_context",
    }
    assert "encounter_note" not in parameters
    assert "payload_path" not in parameters
    assert "payload_json" not in parameters
    assert "timeout_seconds" not in parameters


def test_render_coding_summary_uses_canvas_metadata_by_default() -> None:
    result = _run(coding.render_coding_summary())

    assert result.meta == {"UIPluginRendering": {"plugin_title": "Generated codes"}}
    assert result.structured_content == {
        "pluginName": "ambulatorycoding",
        "launchParameters": {
            "renderedType": "canvas",
            "patientId": "",
            "encounterId": "",
            "notificationTitle": "Generated codes",
            "data": {
                "renderedType": "canvas",
                "patientId": "",
                "encounterId": "",
                "notificationTitle": "Generated codes",
            },
        },
    }
    rendering = coding.render_coding_summary.__fastmcp__.meta["UIPluginRendering"]
    assert rendering["plugin_name"] == "a2ui"
    assert rendering["render_as"] == "canvas"
    assert rendering["ui_schema"]["components"][0]["component"]["CaafPluginComponent"] == {
        "pluginName": {"path": "/pluginName"},
        "parameters": {"path": "/launchParameters"},
    }


def test_render_coding_summary_keeps_canvas_contract_when_description_is_explicit() -> None:
    result = _run(
        coding.render_coding_summary(
            description="E/M Code 99202-25,24,22",
            action_context={
                "encounterId": "en-96204d21",
                "notificationTitle": "Generated codes",
                "patientId": "pa-4e81f682",
                "semantic_object_id": "en-96204d21",
            },
        )
    )

    payload = result.structured_content
    assert result.meta == {"UIPluginRendering": {"plugin_title": "Generated codes"}}
    assert payload["launchParameters"]["patientId"] == "pa-4e81f682"
    assert payload["launchParameters"]["encounterId"] == "en-96204d21"
    assert payload["launchParameters"]["description"] == "E/M Code 99202-25,24,22"
    rendering = coding.render_coding_summary.__fastmcp__.meta["UIPluginRendering"]
    assert payload["pluginName"] == "ambulatorycoding"
    assert rendering["plugin_name"] == "a2ui"
    assert rendering["plugin_title"] == "Generated codes"
    assert rendering["plugin_description"] == "Ambulatory coding suggestions"
    assert rendering["render_as"] == "canvas"


def test_render_coding_summary_uses_submitted_title_from_notification_context() -> None:
    result = _run(
        coding.render_coding_summary(
            notification_title="Submitted codes",
            action_context={
                "encounterId": "en-96204d21",
                "patientId": "pa-4e81f682",
            },
        )
    )

    payload = result.structured_content
    assert result.meta == {"UIPluginRendering": {"plugin_title": "Submitted codes"}}
    assert payload["launchParameters"]["notificationTitle"] == "Submitted codes"
    assert payload["launchParameters"]["data"]["notificationTitle"] == "Submitted codes"


def test_render_ambulatory_coding_notification_uses_native_card_contract() -> None:
    result = _run(
        coding.render_ambulatory_coding_notification(
            patient_id="pa-4e81f682",
            encounter_id="en-96204d21",
            description="E/M Code 99202-25",
        )
    )

    assert result.structured_content == {
        "renderedType": "notification",
        "contractVersion": "1.0",
        "title": "Generated codes",
        "description": "E/M Code 99202-25",
        "patientContext": {
            "contextType": "patient-chart",
            "patientId": "pa-4e81f682",
            "encounterId": "en-96204d21",
        },
        "actions": {"openEnabled": True, "openImmediately": True},
    }
    rendering = coding.render_ambulatory_coding_notification.__fastmcp__.meta["UIPluginRendering"]
    assert rendering["plugin_name"] == "ambulatorycodingnotification"
    assert result.meta == {
        "UIPluginRendering": {
            "plugin_title": "Generated codes",
            "plugin_description": "E/M Code 99202-25",
        }
    }


def test_render_ambulatory_coding_notification_uses_nested_event_description() -> None:
    result = _run(
        coding.render_ambulatory_coding_notification(
            action_context={
                "auxiliaryProps": {
                    "patientId": "pa-4e81f682",
                    "encounterId": "en-96204d21",
                    "description": "E/M Code 99213 with 1 more code",
                }
            }
        )
    )

    payload = result.structured_content
    assert payload["description"] == "E/M Code 99213 with 1 more code"
    assert payload["patientContext"] == {
        "contextType": "patient-chart",
        "patientId": "pa-4e81f682",
        "encounterId": "en-96204d21",
    }
    assert result.meta["UIPluginRendering"]["plugin_description"] == ("E/M Code 99213 with 1 more code")


def test_aoa_toolkit_passes_notification_arguments_to_coding_tool() -> None:
    result = _run(
        aoa_toolkit.render_coding_summary(
            patient_id="pa-4e81f682",
            encounter_id="en-96204d21",
            description="E/M Code 99202-25,24,22",
            notification_title="Generated codes",
        )
    )

    assert result.structured_content["launchParameters"]["patientId"] == "pa-4e81f682"
    assert result.structured_content["launchParameters"]["encounterId"] == "en-96204d21"
    assert result.structured_content["launchParameters"]["description"] == "E/M Code 99202-25,24,22"
    assert result.meta == {"UIPluginRendering": {"plugin_title": "Generated codes"}}
