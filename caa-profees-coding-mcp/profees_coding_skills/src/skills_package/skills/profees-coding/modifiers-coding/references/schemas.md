# Schemas

## Modifier
Fields:
- code: str
- description: str | null
- justification: str | null

## LinkedDiagnosis
Same fields as ICD10CM.

## ICD10CM
Fields:
- code: str
- is_primary: bool
- system: "ICD-10-CM"
- description: str
- justification: str | null

## EMItem
Fields:
- code: str
- description: str | null
- system: "CPT"
- justification: str | null
- modifiers: list[Modifier]
- linked_diagnosis: list[LinkedDiagnosis]

## Procedure
Fields:
- code: str
- description: str
- system: "CPT" | "HCPCS"
- justification: str | null
- modifiers: list[Modifier]
- linked_diagnosis: list[LinkedDiagnosis]

## Output
Return JSON:
```
{
  "em": [EMItem, ...],
  "procedure": [Procedure, ...],
  "diagnosis": [ICD10CM, ...]
}
```
