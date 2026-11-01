---
name: profees-medical-coding
description: "Orchestrate professional-fees medical coding for SOAP or encounter inputs by running ICD-10-CM, CPT E/M, CPT procedures, HCPCS, diagnosis association, and modifiers. Use for end-to-end coding or for general coding questions by delegating to the appropriate subskill."
---

# Profees Medical Coding

## Overview
Run the Profees coding workflow by delegating to smaller, independent skills and returning a single coding summary.

**Preferred implementation:** When a coding summary is requested, call the `toolkit_render_coding_summary` tool so the
Profees model API generates the summary. Provide `encounter_note` text (or a `payload_path`) when available. If no
encounter note or payload is provided, call the tool with no arguments so it loads the persona sample payload from
`proactive-pca/personas/physician/src/test_case/payload_1.json`. Do not ask the user to paste encounter documentation
before using the sample payload.

**Guardrail:** Do **not** call `$profees-medical-coding` from within this skill. Only delegate
to subskills (icd-10-cm-coding, cpt-em-coding, cpt-procedures-coding-skill, hcpcs-coding,
icd-10-association, modifiers-coding).

## Input Mode Decision
1. If the user asks for a coding summary, run the coding workflow immediately with `toolkit_render_coding_summary`.
2. If the user provides a SOAP note, encounter context, orders, or structured input, pass it to the tool.
3. If the user does not provide encounter content, call the tool with no arguments and use the persona sample payload.
4. Otherwise, answer as a coding assistant using the most relevant subskill.

## Coding Workflow
Follow this exact order and keep latency low by running parallel tasks when possible.

### Stage 1 (in Parallel)
Run all of the following in parallel using your `task` tool:
1. CPT E/M coding using `cpt-em-coding` skill.
2. CPT procedure coding using `cpt-procedures-coding-skill` skill.
3. HCPCS coding using `hcpcs-coding` skill.
Then combine list of HCPCS codes and list of CPT procedure codes into a single `procedure` list for downstream association and modifiers **and for the final output**.

### Stage 2 (in Parallel)
Run all of the following in parallel:
1. Associate ICD-10-CM diagnoses to E/M and procedure items using `icd-10-association` skill.
2. Apply modifiers to E/M and procedure items using `modifiers-coding` skill. 

### Aggregate
- Merge the final `em`, **combined `procedure`**, and `diagnosis` arrays into the summary object.
- **Dual coding support**: If both outpatient and preventive E/M are supported, ensure the final `em`
  list contains **both** EM items.
- **Association support**: Inject diagnosis links from `icd-10-association` into each E/M and
  procedure item (`linked_diagnosis` field).
- **Modifier support**: Apply modifier results to each E/M and procedure item (`modifiers` field).
- Return the merged summary object.

## Output
Return a JSON object using the tool (if you have one) or conforming to `assets/coding_summary_schema.json`. If you use the tool, don't return the coding summary as separate message.

## General Coding Q and A
When the input is a general coding question, delegate to the relevant subskill and answer using its references.

## References
- `references/schemas.md`
