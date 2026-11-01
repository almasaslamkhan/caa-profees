---
name: modifiers-coding
description: "Apply CPT or HCPCS modifiers to E/M and procedure items after diagnosis association. Use when modifier rules need to be applied."
---

# Modifiers Coding

## Overview
Apply CPT and HCPCS modifiers after ICD-10 association while preserving linked diagnoses.

## Workflow
1. Read the note and the provided `em`, `procedure`, and `diagnosis` lists.
2. Apply modifier rules from `references/guidelines.md` to the relevant items.
3. Preserve item order and all existing fields, including `linked_diagnosis`.

## Output
Return JSON:
```
{
  "em": [EMItem, ...],
  "procedure": [Procedure, ...],
  "diagnosis": [ICD10CM, ...]
}
```

Rules:
- Do not change codes or descriptions unless a modifier rule explicitly requires it.
- Keep `linked_diagnosis` intact.

## References
- `references/schemas.md`
- `references/guidelines.md`
