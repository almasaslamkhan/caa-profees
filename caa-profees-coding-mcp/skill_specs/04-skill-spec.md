# Skill Definition

## Skill source code
`profees_coding_skills/src/profees_ambulatory_coding_skills/skills/ambulatory-coding/SKILL.md`

## Skill name
`ambulatory-coding`

## Short description
The skill enables physicians to launch the Coding plugin, where they can review, recalculate, and submit the proposed professional codes.

## Long description
The skill enables physicians to launch the Coding plugin in A2UI canvas, where they can review, recalculate, and submit the proposed professional codes.

When invoked by the skill, the `render_coding_summary` MCP tool returns a canvas payload. The A2UI schema uses `CaafPluginComponent` to load the `ambulatorycoding` plugin and passes the MCP payload into the plugin as parameters.

## When to use
Use this skill when the physician asks to show or review professional-fees coding suggestions for an ambulatory encounter.

Use it for requests such as:

- "Show ambulatory codes" from chat.
- Opens the **Generated codes** canvas.

## When not to use
Do not use this skill for inpatient facility coding.

## Example user requests

- Show codes.
- Show ambulatory coding.
- Show ambulatory codes.

## Conversational triggers

- User types **show codes**, **show ambulatory coding**, or **show ambulatory codes** in the physician assistant chat.
- The agent routes the request to the ProFees MCP coding tool.
- MCP returns the coding canvas payload and A2UI metadata.
- A2UI opens the **Generated codes** canvas using the `ambulatorycoding` plugin.

## Canvas actions

- **Recode**: Refreshes suggested codes after the visit note or payload has changed.
- **Submit codes**: Submits the selected generated codes from the canvas.

## Conversational trigger schema

| Property | Type | Description | Required | Notes |
|---|---|---|---|---|
| user_utterance | string | User message requesting generated codes. | Yes | Example: `show codes`, `show ambulatory coding`. |
| encounter_id | string | Encounter identifier for the selected visit. | No | Used when available from patient context or payload. |
| patient_id | string | Patient identifier for the generated-codes canvas payload. | No | Used when available from patient context or payload. |

## Data events that trigger the skill

Not Applicable

## Data event schema
Not Applicable

## Outputs
The outcome of running the skill is that the coding plugin launches with professional codes available for physician review.

- `pluginName`: `ambulatorycoding`
- `renderMode`: `canvas`
- `launchParameters`: Patient, encounter
- `result`: UI-ready generated-codes payload.
- `_meta.UIPluginRendering`: A2UI metadata for the **Generated codes** canvas.

Recoverable errors are returned with `success: false`, an error message, and timeout metadata when applicable.

## A2UI schema
`profees_coding_skills/src/profees_ambulatory_coding_skills/skills/ambulatory-coding/schemas/ambulatory_coding_canvas.json`

The schema launches the `ambulatorycoding` CAAD plugin through `CaafPluginComponent` with root component ID `ambulatoryCodingPluginRoot`.

## Dependencies

- ProFees skill distribution: `profees-skills`.
- Mounted Python package: `profees_ambulatory_coding_skills`.
- MCP tool implementation: `cdrc/profees_coding/mcp/tools/coding.py`.
- MCP prompt instructions: `cdrc/profees_coding/mcp/prompts/coding.py`.
- A2UI/CAAD plugin: `ambulatorycoding`.

## Tool exposure

| Field | Value |
|---|---|
| Uses skill-private bundled tools? | No |
| Uses skill-scoped public MCP tools? | Yes |
| Uses hot tools? | Yes, in the physician persona |
| Skill-private bundled tool names | None |
| Skill-scoped public MCP tool names | Raw MCP tool: `render_coding_summary`. |
| Hot tool names | Persona-facing hot tool: `profees_coding_render_coding_summary`. |
