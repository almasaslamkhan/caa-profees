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

## Procedure
Fields:
- code: str
- description: str
- system: "HCPCS"
- justification: str | null
- modifiers: list[Modifier]
- linked_diagnosis: list[LinkedDiagnosis]

## Output
Return JSON:
```
{
  "procedure": [Procedure, ...]
}
```
