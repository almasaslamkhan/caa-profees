---
name: ambulatory-coding
description: "Launch the CAAD ambulatory coding plugin and run explicit local chat recode requests that save billing codes."
metadata:
  mcpServers:
    - name: profees_coding
      endpoint: http://127.0.0.1:8011/mcp
      auth: none
      includeTools:
        - render_coding_summary
        - render_ambulatory_coding_notification
        - recode_profees_coding
        - recode_profees_coding_ehrm
      observability: true
  eventTriggers:
    - eventType: a2ui.userAction
      actionName: system.updateApplicationContext
      applicationContextGroup: patientassistant/ambulatorycodingnotification
      semanticObject: AmbulatoryCoding
      event: custom.SHOW_AMBULATORY_CODING
      toolCall:
        name: render_ambulatory_coding_notification
        parameterMappings:
          - payloadProperty: context.properties.patientId
            toolArgument: patient_id
          - payloadProperty: context.properties.semantic_object_id
            toolArgument: encounter_id
          - payloadProperty: context.properties.encounterId
            toolArgument: encounter_id
          - payloadProperty: context.properties.description
            toolArgument: description
          - payloadProperty: context.properties.notificationTitle
            toolArgument: notification_title
    - eventType: a2ui.userAction
      actionName: ambulatorycoding.open
      toolCall:
        name: render_coding_summary
        parameterMappings:
          - payloadProperty: patientId
            toolArgument: patient_id
          - payloadProperty: encounterId
            toolArgument: encounter_id
          - payloadProperty: notificationTitle
            toolArgument: notification_title
    - eventType: a2ui.userAction
      actionName: patientassistant.Conversation.ambulatorycodingnotification:ambulatorycoding.open
      toolCall:
        name: render_coding_summary
        parameterMappings:
          - payloadProperty: patientId
            toolArgument: patient_id
          - payloadProperty: encounterId
            toolArgument: encounter_id
          - payloadProperty: notificationTitle
            toolArgument: notification_title
    - eventType: a2ui.userAction
      actionName: aimode.Conversation.ambulatorycodingnotification:ambulatorycoding.open
      toolCall:
        name: render_coding_summary
        parameterMappings:
          - payloadProperty: patientId
            toolArgument: patient_id
          - payloadProperty: encounterId
            toolArgument: encounter_id
          - payloadProperty: notificationTitle
            toolArgument: notification_title
    - eventType: a2ui.userAction
      actionName: system.updateApplicationContext
      applicationContextGroup: patientassistant/ambulatorycoding
      semanticObject: AmbulatoryCoding
      event: custom.codingAgentRecodeCustomEvent
      toolCall:
        name: recode_profees_coding
        parameterMappings:
          - payloadProperty: patientId
            toolArgument: patient_id
          - payloadProperty: semantic_object_id
            toolArgument: encounter_id
          - payloadProperty: encounterId
            toolArgument: encounter_id
          - payloadProperty: requestId
            toolArgument: request_id
          - payloadProperty: document_ref_id
            toolArgument: document_ref_id
          - payloadProperty: docrefId
            toolArgument: document_ref_id
          - payloadProperty: docref_id
            toolArgument: document_ref_id
    - eventType: a2ui.userAction
      actionName: system.updateApplicationContext
      applicationContextGroup: patientchart/documentation
      semanticObject: AmbulatoryCoding
      event: custom.codingAgentRecodeCustomEHRMEvent
      toolCall:
        name: recode_profees_coding_ehrm
        parameterMappings:
          - payloadProperty: patientId
            toolArgument: patient_id
          - payloadProperty: semantic_object_id
            toolArgument: encounter_id
          - payloadProperty: encounterId
            toolArgument: encounter_id
          - payloadProperty: requestId
            toolArgument: request_id
          - payloadProperty: document_ref_id
            toolArgument: document_ref_id
          - payloadProperty: docrefId
            toolArgument: document_ref_id
          - payloadProperty: docref_id
            toolArgument: document_ref_id
    - eventType: a2ui.userAction
      actionName: system.updateApplicationContext
      applicationContextGroup: patientassistant/ambulatorycoding
      semanticObject: AmbulatoryCoding
      event: codingAgentRecodeCustomEvent
      toolCall:
        name: recode_profees_coding
        parameterMappings:
          - payloadProperty: patientId
            toolArgument: patient_id
          - payloadProperty: semantic_object_id
            toolArgument: encounter_id
          - payloadProperty: encounterId
            toolArgument: encounter_id
          - payloadProperty: requestId
            toolArgument: request_id
          - payloadProperty: document_ref_id
            toolArgument: document_ref_id
          - payloadProperty: docrefId
            toolArgument: document_ref_id
          - payloadProperty: docref_id
            toolArgument: document_ref_id
    - eventType: a2ui.userAction
      actionName: system.updateApplicationContext
      applicationContextGroup: codingAgentEvents
      semanticObject: AmbulatoryCoding
      event: codingAgentRecodeCustomEvent
      toolCall:
        name: recode_profees_coding
        parameterMappings:
          - payloadProperty: patientId
            toolArgument: patient_id
          - payloadProperty: semantic_object_id
            toolArgument: encounter_id
          - payloadProperty: encounterId
            toolArgument: encounter_id
          - payloadProperty: requestId
            toolArgument: request_id
          - payloadProperty: document_ref_id
            toolArgument: document_ref_id
          - payloadProperty: docrefId
            toolArgument: document_ref_id
          - payloadProperty: docref_id
            toolArgument: document_ref_id
    - eventType: a2ui.userAction
      actionName: system.updateApplicationContext
      applicationContextGroup: codingAgentEvents
      semanticObject: AmbulatoryCoding
      event: custom.codingAgentRecodeCustomEvent
      toolCall:
        name: recode_profees_coding
        parameterMappings:
          - payloadProperty: patientId
            toolArgument: patient_id
          - payloadProperty: semantic_object_id
            toolArgument: encounter_id
          - payloadProperty: encounterId
            toolArgument: encounter_id
          - payloadProperty: requestId
            toolArgument: request_id
          - payloadProperty: document_ref_id
            toolArgument: document_ref_id
          - payloadProperty: docrefId
            toolArgument: document_ref_id
          - payloadProperty: docref_id
            toolArgument: document_ref_id
---

# Ambulatory Coding

Use this skill only when the user explicitly asks to show ambulatory coding,
show ambulatory codes, show professional fee coding, show generated coding
suggestions, open the generated-codes canvas, or review ambulatory coding
suggestions, or explicitly requests a local chat recode test with patient,
encounter, and document identifiers.

Do not use this skill for medication reconciliation, orders, labs, diagnoses,
notes, generic "show" requests, or unrelated plugins.

For UI-event launches, call the configured ProFees MCP tool with the identifiers
from the event properties. In the physician persona the render tool is exposed as
`profees_coding_render_coding_summary`. Pass `patient_id` from
`properties.patientId`, `encounter_id` from `properties.encounterId` or
`properties.semantic_object_id`, `description` from `properties.description`,
and let the tool provide the notification title. Do not use
`search_available_tools` or `get_tool_details` for this workflow. Do not invent
other names. Do not use chart search, `universal_search_agent`, or note search
for this workflow.

For Recode UI events, call `recode_profees_coding` from the
`codingAgentRecodeCustomEvent` or `custom.codingAgentRecodeCustomEvent` event
trigger. The CAAD shell may publish this under either
`patientassistant/ambulatorycoding` or `codingAgentEvents`; both routes represent
the same Recode request. Pass `patient_id`, `encounter_id`, `request_id`, and
`document_ref_id` from the event properties. Prefer the event payload property
`document_ref_id`; `docrefId` and `docref_id` are accepted only as fallback
aliases. Use the tool response
`status: completed` and `requestId` as the completion correlation payload;
do not use `renderedType` as a completion signal.

For temporary local chat testing, an explicit request to call recode is also
supported when the user supplies patient, encounter, and document identifiers.
Call `profees_coding_recode_profees_coding` with `patient_id`, `encounter_id`,
`document_ref_id`, and `save_billing_codes=true`. Map user-supplied
`doc_ref_id`, `docref_id`, or `docrefId` to `document_ref_id`. Use the host's
configured MCP tool; do not open a separate connection. If any identifier is
missing, ask for it rather than guessing. Do not require a canvas-button event
for this explicit local test. Report the actual tool result and distinguish
tool failure from success. Do not claim recode completed before it returns.
These explicit chat recode requests save predicted billing codes. Ordinary chart questions do not trigger
recode. This temporary chat allowance should be removed after local testing;
the existing Recode UI-event workflow remains unchanged.

The host persona must provide the `profees_coding` MCP server endpoint. Do not
open a new MCP connection from this skill metadata; use the MCP server already
configured by the persona so local overrides and deployed service URLs both
work.

`includeTools` names the raw MCP tools, `render_coding_summary` and
`render_ambulatory_coding_notification`, plus `recode_profees_coding`. The
physician persona exposes the same MCP tools as server-prefixed tools such as
`profees_coding_render_coding_summary`.

If a manual user request does not provide UI-event properties or explicit
patient/encounter identifiers, inspect the current application context before
calling `render_coding_summary`. If the current Ambulatory Coding notification
context contains `notificationTitle`, pass that value as `notification_title`.
This preserves `Submitted codes` when the user opens coding from a submitted
notification. Otherwise, call `render_coding_summary` with no arguments. The
plugin fetches billing codes from its own API using the active patient/encounter
context.

After the coding UI is opened, keep the conversational response short and do
not repeat, enumerate, or infer any CPT, E/M, HCPCS, ICD-10, or modifier codes
from the tool result. Use the Generated-codes wording regardless of the card
title:

- For `Submitted codes`: "Ambulatory coding is open for this encounter. Review
  the suggested codes in the Generated codes card."
- For `Generated codes`: "Ambulatory coding is open for this encounter. Review
  the suggested codes in the Generated codes card, or click Recode to generate
  updated suggestions."

If the coding card or canvas reports an error, is unavailable, or cannot
display details, do not expose any generated code, code count, code
type, or coding suggestion in the conversational response. Do not say that
codes are available. Respond only: "Unable to display generated codes. Please
try again later." Do not offer Recode when the UI is unavailable.

The `render_coding_summary` tool always carries A2UI metadata and opens the
existing `ambulatorycoding` canvas. `render_ambulatory_coding_notification`
is the separate native-card tool for managed UI events. Its static metadata
selects the published `ambulatorycodingnotification` plugin, so it does not
depend on a response `pluginName` to load in SaaS. Clicking that card invokes
`ambulatorycoding.open`, which returns to `render_coding_summary` and opens the
existing coding canvas. Do not call the UI schema renderer separately after a
tool call, especially if the tool call fails.

## UI Rendering Index

```yaml
domain: ambulatory coding
mappings:
  - intent: launch the ambulatory generated-codes plugin
    id: ambulatory_coding_plugin
    pluginName: a2ui
    pluginAgentGroup: ambulatory.coding
    pluginTitle: Generated codes
    pluginDescription: Ambulatory generated codes with editable code selections
    renderAs: canvas
    mcpServer: profees_coding
    tool: render_coding_summary
    ui_schema_path: ambulatory-coding/schemas/ambulatory_coding_canvas.json
    launchesPlugin: ambulatorycoding
  - intent: render generated-codes notification from UI event
    id: ambulatory_coding_notification
    pluginName: ambulatorycodingnotification
    pluginAgentGroup: ambulatory.coding
    pluginTitle: Generated codes
    pluginDescription: Ambulatory generated codes with editable code selections
    renderAs: card
    mcpServer: profees_coding
    tool: render_ambulatory_coding_notification
  - intent: run ambulatory recode and return completion status
    id: ambulatory_coding_recode
    mcpServer: profees_coding
    tool: recode_profees_coding
```
