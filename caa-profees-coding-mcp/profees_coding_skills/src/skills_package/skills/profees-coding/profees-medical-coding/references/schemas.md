# Schemas

## Coding Summary
Fields:
- em: list[EMItem]
- procedure: list[Procedure]
- diagnosis: list[ICD10CM]

## EMItem
Fields:
- code: str
- description: str | null
- system: "CPT"
- source: "MDM" or "Time"
- justification: str | null
- time_spent: int
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

## ICD10CM
Fields:
- code: str
- is_primary: bool
- system: "ICD-10-CM"
- description: str
- justification: str | null

## Modifier
Fields:
- code: str
- description: str | null
- justification: str | null

## LinkedDiagnosis
Same fields as ICD10CM.
