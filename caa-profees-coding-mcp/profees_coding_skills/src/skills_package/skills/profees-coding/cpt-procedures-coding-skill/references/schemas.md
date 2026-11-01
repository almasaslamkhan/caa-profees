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
- system: "CPT"
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

## Professional Fee Procedure Guardrails
- A CPT procedure may be verified only when it was explicitly performed or supervised by the physician during the encounter.
- Extract only 5-digit CPT procedure codes. Exclude E/M, ICD-10-CM, HCPCS, Category II, and Category III/T codes.
- Exclude procedures already present in `known_orders` by exact CPT match or matching order text.
- Procedures completed at outside facilities or labs are not professional-fee procedures for this skill.
- future or conditional actions, such as "will order," "plan to order," or "if symptoms worsen," are insufficient.
- For order-grounded procedures, orders with status "COMPLETED" can support the procedure; orders with status "ORDERED" remain incomplete and do not qualify by themselves.
- Exclude procedures typically billed by labs or technicians, including lab/pathology ranges such as 80000-89999.
- Blood draw or venipuncture performed by outside labs must not be inferred from the presence of lab tests or results.
- Verify 36415 venipuncture only when the note explicitly documents an in-office venipuncture or blood draw during the encounter; do not infer it from blood test results.
- Vaccine administration codes 90471, 90472, and 90480 may be included when vaccine administration is documented; the injection route does not need to be separately stated.
- Exclude EKG/ECG procedures such as 93000 unless the note clearly documents same-day performance, individual interpretation, and results in the note.
- Verify 96127 or 96160 only when the screening was ordered or performed today with supporting score/result context; a reviewed screening alone is insufficient.
- Imaging, lab/pathology, HCPCS, ICD, Category II, Category III, and E/M codes are out of scope for CPT procedure output unless an allowed prompt-specific CPT exception is explicitly supported.
