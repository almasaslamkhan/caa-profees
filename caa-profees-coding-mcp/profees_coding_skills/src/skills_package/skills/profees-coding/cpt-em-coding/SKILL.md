---
name: cpt-em-coding
description: "CPT E/M visit level selection from SOAP or encounter notes using documented MDM or time. Use when choosing an E/M CPT code for outpatient, observation, inpatient, emergency, or preventive encounters."
---

# CPT E/M Coding

## Overview
Choose the correct CPT E/M code for a visit based on encounter type, documented medical decision making (MDM), and/or time.

## Workflow


## Step 1 — Identify Encounter Type and Time

### 1A. Use Structured Metadata First
- If `encounter_metadata.type` is present and unambiguous, use it.
- Select the corresponding encounter-type reference and proceed directly:
  - **Ambulatory (Office/Outpatient)** → `references/encounter_type_ambulatory.md`
  - **Inpatient** → `references/encounter_type_inpatient.md`
  - **Observation** → `references/encounter_type_inpatient.md`
  - **Emergency Department** → `references/encounter_type_ed.md`

For **Ambulatory** service types, the encounter type may be:
- Office/outpatient visit
- Preventive/wellness visit
- **Dual-coded** office/outpatient + preventive (both documented)

Do not perform note-based classification if metadata is clear.

### 1B. Use Note-Based Classification Only If Needed
If metadata is missing or unclear, determine encounter type from documentation.

Key distinctions:
- **Inpatient**: Formal hospital admission (non-observation)
- **Observation**: Observation status (not full admission)
- **Hospital Admit + Discharge (Same Date)**: Admission and discharge documented on same DOS
- **Hospital Discharge**: Discharge management service documented
- **Emergency**: ED visit without hospital admission
- **Outpatient**: Office or ambulatory visit without admission
- **Preventive**: Wellness/preventive service (may be dual-billed with outpatient E/M)
- **Dual ambulatory + preventive**: If both preventive and problem-oriented work are documented,
  you must produce **two E/M items** (preventive + outpatient).

### 1C. Extract Time
- Capture total clinician time when documented.
- Follow `references/time_statement.md`.
- Use time only when permitted by CPT guidelines.

---

## Step 2 — Determine MDM Complexity (When Applicable)

Applies to:
- Office/Outpatient  
- Observation  
- Inpatient  
- Emergency Department  

Does **not** apply to:
- Preventive-only encounters  

### 2A. Evaluate the Three MDM Elements
Assess independently:
- **Problem Complexity**
- **Data Complexity**
- **Risk of Complications and/or Morbidity**

Use:
- `references/mdm_problem_complexity.md`
- `references/mdm_data_complexity.md`
- `references/mdm_risk_complexity.md`

### 2B. Determine Overall MDM Level
Derive overall MDM level (minimal / low / moderate / high) using CPT element rules.

For **office/outpatient encounters**, always document (even when time-based selection is used):
- Individual element levels (Problem/Data/Risk)
- Overall MDM level
- Brief justification summary

---

## Step 3 — Outpatient Audit Validation (Office Only)

For office/outpatient encounters, validate element scoring using:

- `references/audit_problem_complexity.md`
- `references/audit_data_complexity.md`
- `references/audit_risk_complexity.md`

If discrepancies are identified, reconcile before code selection.

---

## Step 4 — Code Selection (Mandatory Tool Use)

- You **must** call the `em_code_selection` tool wrapper.
- The wrapper **must execute** `scripts/em_code_selection.py`.
- If the tool fails, stop and request guidance. Do not guess.
- Do **not** emit HCPCS Medicare AWV codes (G0438/G0439).
- **Dual encounters**: Call `em_code_selection` once per encounter type (preventive and outpatient)
  and return both EMItems in a single `em` list.
- When dual coding is required, pass `dual_encounter: true` to the tool call.

---

## Step 5 — Output Construction

Return JSON formatted per `references/schemas.md`.

When `em_code_selection` returns its flat result object, transform it into an `EMItem`:

- `code` → `code`
- `description` → `code_descriptor`
- `source` → `TIME` or `MDM` (from `selection_method`)
- `time_spent` → `time_minutes`
- `justification` → concatenate `rationale` values into a single string

For dual encounters, repeat the tool call per encounter type and append both EMItems to the list.

### Required Output Structure

```
{
  "em": [EMItem, ...]
}
```

## Additional Requirements for Office/Outpatient Encounters
Include the following fields:
- `mdm_complexity.level` → Overall MDM level (minimal / low / moderate / high)
- `mdm_complexity.justification` → Concise summary of problem, data, and risk rationale
Append the same MDM summary to the `EMItem.justification`.

### Element-Level Detail (Required)
- `pc_complexity.level` → Problem complexity level  
- `pc_complexity.justification` → Problem rationale summary  
- `dc_complexity.level` → Data complexity level  
- `dc_complexity.justification` → Data rationale summary  
- `rc_complexity.level` → Risk complexity level  
- `rc_complexity.justification` → Risk rationale summary  

## Output Rules
- Use `system = "CPT"`.
- Do not output ICD-10-CM codes.
- Do not output HCPCS codes.
- Do not output CPT procedure codes.
- Include a concise `justification` explaining:
  - Encounter type determination  
  - MDM and/or time rationale  
  - Final code selection logic  

## References
- `references/schemas.md`
- `references/encounter_type_ambulatory.md`
- `references/encounter_type_ed.md`
- `references/encounter_type_inpatient.md`
- `references/time_statement.md`
- `references/mdm_problem_complexity.md`
- `references/mdm_data_complexity.md`
- `references/mdm_risk_complexity.md`
- `references/audit_problem_complexity.md`
- `references/audit_data_complexity.md`
- `references/audit_risk_complexity.md`
