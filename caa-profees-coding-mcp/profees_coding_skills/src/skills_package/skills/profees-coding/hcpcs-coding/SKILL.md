---
name: hcpcs-coding
description: "Decide whether to append HCPCS add-on code G2211 for outpatient or preventive visits based on documented longitudinal care and chronic-condition management. Use when evaluating G2211 eligibility."
---

# HCPCS Coding (G2211)

## Overview
Determine whether HCPCS add-on code G2211 should be reported for outpatient office E/M or preventive visits based on documented longitudinal care and chronic-condition management.

## Workflow
Step 1: Follow 'references/g2211_overview.md' to assign G2211 code
- See `references/g2211_overview.md`.

Step 2: Return output
- If eligible, return a Procedure list with code `G2211`, `system = "HCPCS"`, and a brief rule-based justification.
- If not eligible, return an empty Procedure list with justification explaining which requirement was not met.

## Output
Return JSON:
```
{
  "procedure": [Procedure, ...]
}
```

Rules:
- Only output HCPCS code `G2211` for this skill.
- Use `system = "HCPCS"`.
- Include a concise `justification` grounded in the note.

## References
- `references/schemas.md`
- `references/g2211_overview.md`