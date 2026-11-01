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
- source: "MDM" or "TIME"
- justification: str | null
- time_spent: int
- mdm_complexity: object | null
- mdm_complexity.level: str | null
- mdm_complexity.justification: str | null
- dc_complexity: object | null
- dc_complexity.level: str | null
- dc_complexity.justification: str | null
- pc_complexity: object | null
- pc_complexity.level: str | null
- pc_complexity.justification: str | null
- rc_complexity: object | null
- rc_complexity.level: str | null
- rc_complexity.justification: str | null
- modifiers: list[Modifier]
- linked_diagnosis: list[LinkedDiagnosis]

## Output
Return JSON:
```
{
  "em": [EMItem, ...]
}
```
