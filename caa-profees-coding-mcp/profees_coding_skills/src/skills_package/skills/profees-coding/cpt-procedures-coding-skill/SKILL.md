---
name: cpt-procedures-coding-skill
description: "CPT procedure coding from encounter notes or orders. Use when extracting 5-digit CPT procedure codes and descriptions."
---

# CPT Procedure Coding

## Overview
Extract CPT procedure codes that are supported by the note and orders.

## Workflow
### Step 1 
analyze the provided **clinical encounter (SOAP note)** and extract the most appropriate **CPT procedure codes** that are Professional Fee Codes.

Coding Rules
1. **Return only valid CPT procedure codes** — 5-digit numeric codes.
  - ❌ Do NOT include:
    - Evaluation & Management (E/M) CPT codes (e.g., 99212, 99396)
    - Procedures that are not billed for Professional Fees (e.g. Radiology tests, Lab Tests, EKG/ECGs)
    - ICD-10-CM or HCPCS codes
  Additional knowledge:
  - When verifying 36415-venipuncture, it needs to be explicitly mentioned as peformed in office during the encounter. Blood draw or Venipuncture performed in outside labs should **NOT** be verified. Do NOT infer it based on presence of blood tests (and/or their results) that requires blood draw. 
  - To verify Code 93000 - EKG/ECG, it needs to meet **ALL** 3 conditions at the same time: 1. explicited mentioned 'performed today' 2. explicited mentioned 'individually interpreted' 3. results reported in the note.
  - Code 96127 and 96160 **cannot** be verified if the note only mentioned 'Depression/Emotional Screening - reviewed' without explicitly mentioning its ordered and performed today.
  - Imaging tests should not be verified.
  - Lab tests that are performed by technicians in Pathology or dedicated labs should never be billed for professional fees.

  - ✅ Only true CPT *procedure* Professional Fee codes are allowed.
2. **Output schema for each detected item:**
  - `code`: 5-digit CPT code (string)
  - `system`: must always be `"CPT"`
  - `description` : brief human-readable label of the procedure
  - `justification` : 1-sentence concise and no leading reasoning supporting the code
3. **Validation and Deduplication:**
  - Deduplicate procedures by code type.
4. **Conservatism rule:**
  - Be **strict** — if documentation is incomplete or insufficient, **omit the code**.
  - Do **not guess** or infer beyond what is supported in the SOAP note.
5. **Empty output policy:**
  - If the note contains **no supported CPT procedures**, return an **empty list**.

### Step 2
Return structured output that matches `references/schemas.md`.

## Output
Return JSON:
```
{
  "procedure": [Procedure, ...]
}
```

Rules:
- Output CPT procedure codes only and ensure each code is 5 digits.
- Exclude E/M codes, ICD-10-CM codes, HCPCS codes, and Category III codes.
- Use `system = "CPT"`.
- Include a concise `justification`.

## References
- `references/schemas.md`
