# Post-Processing Flags

Use these flags only after association is complete. See `scripts/association_postprocess.py` for the exact behavior.

- `cpt_procedure_filtering`: When true, apply CPT eligibility mappings to keep only linked diagnoses allowed for each CPT. If a CPT code is missing from the mapping or has an empty eligibility list, do not restrict its linked diagnoses.
- `hcpcs_procedure_filtering`: When true, apply HCPCS eligibility mappings to keep only linked diagnoses allowed for each HCPCS. If a HCPCS code is missing from the mapping or has an empty eligibility list, do not restrict its linked diagnoses.
- `populate_code_descriptions`: When true, overwrite descriptions using the CPT/HCPCS/ICD-10 description mapping files when a match exists.
- `remove_unassociated_diagnoses`: When true, drop diagnoses that are not linked to any E/M or procedure item.

Script usage:
```
uv run python skills/icd-10-association/scripts/association_postprocess.py --input <in.json> --output <out.json> [--cpt-filtering] [--hcpcs-filtering] [--populate-descriptions] [--remove-unassociated]
```

## Diagnosis Regeneration Rules
- Treat flagged ICD-10-CM diagnosis codes as invalid or non-billable and remove them from the final associations.
- Alternatives Only: replace a flagged diagnosis only with a diagnosis code from the provided valid, billable alternatives for that flagged code.
- Service-Level Support Required: a replacement diagnosis must support the medical necessity of the specific CPT/HCPCS or E/M code to which it is linked.
- Do not invent replacement diagnoses outside the supplied alternatives.
- Choose the most specific valid and billable replacement supported by the documentation.
