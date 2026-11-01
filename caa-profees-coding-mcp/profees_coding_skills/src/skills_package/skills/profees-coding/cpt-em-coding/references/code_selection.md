# Code Selection

Use `scripts/em_code_selection.py` as the single source of truth for mapping MDM, time, and encounter type inputs to CPT E/M codes.

**Guidelines Followed:**
- CPT® (Current Procedural Terminology) standards per AMA
- AAFP (American Academy of Family Physicians) E/M coding guidelines
- CMS Medicare guidelines for E/M code selection (2024-2026)

## Supported Encounter Types

- **outpatient**: Office/outpatient E/M visits (CPT 99202-99215)
- **preventive**: Preventive medicine visits (CPT 99381-99397)
- **observation**: Observation status encounters (CPT 99221-99223, 99231-99233)
- **inpatient**: Inpatient consultation and subsequent care (CPT 99252-99255, 99231-99233)
- **emergency**: Emergency department visits (CPT 99281-99285)
- **hospital_admit_discharge**: Same-day hospital admit/discharge (CPT 99234-99236)
- **hospital_discharge**: Hospital discharge services (CPT 99238-99239)

## Patient Type Classification
- Determine `note_pt`, `decision`, and `final_patient_type` as New, Established, or Unknown.
- Use note evidence first, then combine it with appointment type and any provided patient type.
- Keep evidence source-based and brief.
- Ignore E/M code descriptors when classifying patient type; ordered charge text such as "Office/Outpatient Visit, New" or "Prev Visit, New" is billing leakage, not clinical evidence.
- Ignore Assessment/Plan evidence for new-patient classification when it is only future plans, orders, or charge descriptors.
- When note evidence and appointment type are insufficient or low-confidence, visit history controls `final_patient_type`.
- History-supported Established means same-location encounter history exists in the prior three years; history-supported New means no same-location encounter is found in the prior three years.

## Inputs

### Common Inputs (all encounter types)
- `encounter_type`: `outpatient`, `preventive`, `observation`, `inpatient`, `emergency`,
  `hospital_admit_discharge`, or `hospital_discharge` (required)
- `patient_type`: `new` or `established` (required)

### Encounter-Specific Inputs

**Outpatient encounters:**
- `mdm`: `minimal`, `low`, `moderate`, or `high` (optional)
- `time_minutes`: total clinician time in minutes (optional)
- At least one of `mdm` or `time_minutes` must be provided

**Preventive encounters:**
- `age`: patient age in years (required)

**Observation status encounters:**
- `mdm`: `minimal`, `low`, `moderate`, or `high` (optional)
- `time_minutes`: total clinician time in minutes (optional)
- At least one of `mdm` or `time_minutes` must be provided

**Inpatient consultations:**
- `mdm`: `minimal`, `low`, `moderate`, or `high` (optional)
- `time_minutes`: total clinician time in minutes (optional)
- At least one of `mdm` or `time_minutes` must be provided

**Emergency department encounters:**
- `mdm`: `none`, `minimal`, `low`, `moderate`, or `high` (required)
- `time_minutes`: not used for ED level selection

**Hospital admit/discharge (same-day) encounters:**
- `mdm`: `minimal`, `low`, `moderate`, or `high` (optional)
- `time_minutes`: total clinician time in minutes (optional)
- At least one of `mdm` or `time_minutes` must be provided

**Hospital discharge encounters:**
- `discharge_time`: `leq_30` or `gt_30` (required)

## CPT Code Ranges

| Encounter Type | New/Initial | Established/Subsequent |
|---|---|---|
| Outpatient | 99202-99205 | 99212-99215 |
| Preventive | 99381-99387 (by age) | 99391-99397 (by age) |
| Observation | 99221-99223 | 99231-99233 |
| Inpatient | 99252-99255 | 99231-99233 |
| Emergency | 99281-99285 | N/A |
| Hospital admit/discharge | 99234-99236 | N/A |
| Hospital discharge | 99238-99239 | N/A |

## Output

The script returns JSON:
```
{
  "code": "...",
  "source": "MDM|TIME|PREVENTIVE|ED",
  "notes": "..."
}
```

## Code Selection Logic

**Guideline Reference:** The rules below follow AAFP E/M coding guidelines and CPT standards.

### Outpatient Encounters
- **Rule**: If both MDM and time are provided, the **higher numeric CPT code** takes precedence.
  - Both MDM complexity and time are equally weighted (per AAFP guidelines).
- Example: NEW patient with MDM=minimal (99202) and time=30min (99203) → **99203** (time higher)

### Preventive Encounters
- **Rule**: Determined solely by patient type and age category.
  - No MDM or time input needed/used.
- Example: NEW patient age 30 → **99385**

### Dual Outpatient + Preventive Encounters
- **Rule**: If both a preventive exam and problem-oriented E/M are documented in an
  outpatient setting, return **two codes**: one preventive (99381-99397) and one outpatient
  problem-oriented E/M (99202-99215).
- Example: Annual physical with separate cough workup → **99385** + **99213**

### Observation Status Encounters
- **Rule**: If both MDM and time are provided, the **higher numeric CPT code** takes precedence.
  - Observation follows outpatient-style logic (both equally weighted).
  - **Important**: Observation time thresholds differ from outpatient thresholds.
- Example: NEW patient with MDM=low (99221) and time=55min (99222) → **99222** (time higher)

### Inpatient Consultations
- **GUIDELINE: MDM complexity is the PRIMARY driver** for code selection (differs from outpatient).
  - Per AAFP E/M coding guidelines and CPT standards.
- **Rule**: If both MDM and time are provided:
  - Use MDM code UNLESS time-based code is numerically higher
- Example: NEW patient with MDM=moderate (99253) and time=80min (99255) → **99255** (time higher)
- Example: NEW patient with MDM=moderate (99253) and time=50min (99253) → **99253** (MDM used)
- **Important**: Time-only selection supported (when MDM not available)
- Example: NEW patient with no MDM but time=90min → **99255** (time-only)

### Emergency Department Encounters
- **Rule**: ED code selection is MDM-only; time does not drive the ED level.
- Example: ED patient with MDM=moderate → **99284**
- Example: ED patient with MDM=none (no physician/QHP required) → **99281**

### Hospital Admit/Discharge Same-Day Encounters
- **Rule**: Select 99234-99236 based on MDM or total time when documented.
- Example: Same-day admit/discharge with MDM=moderate → **99235**
- Example: Same-day admit/discharge with time=80 minutes → **99236**

### Hospital Discharge Encounters
- **Rule**: Select 99238 (≤30 minutes) or 99239 (>30 minutes) based on discharge time.
- Example: Discharge time >30 minutes → **99239**

## Examples

### Outpatient Visit
```bash
uv run python scripts/em_code_selection.py \
  --encounter-type outpatient \
  --patient-type new \
  --mdm moderate \
  --time-minutes 50
# Output: code=99204, source=MDM
```

### Observation Status Encounter
```bash
uv run python scripts/em_code_selection.py \
  --encounter-type observation \
  --patient-type new \
  --mdm low
# Output: code=99221, source=MDM
```

### Inpatient Consultation
```bash
uv run python scripts/em_code_selection.py \
  --encounter-type inpatient \
  --patient-type new \
  --mdm moderate
# Output: code=99253, source=MDM
```

### Preventive Visit
```bash
uv run python scripts/em_code_selection.py \
  --encounter-type preventive \
  --patient-type new \
  --age 30
# Output: code=99385, source=PREVENTIVE

### Emergency Department Visit
```bash
uv run python scripts/em_code_selection.py \
  --encounter-type emergency \
  --patient-type new \
  --mdm moderate
# Output: code=99284, source=MDM
```

### Hospital Admit/Discharge Same-Day
```bash
uv run python scripts/em_code_selection.py \
  --encounter-type hospital_admit_discharge \
  --patient-type new \
  --mdm moderate
# Output: code=99235, source=MDM
```

### Hospital Discharge
```bash
uv run python scripts/em_code_selection.py \
  --encounter-type hospital_discharge \
  --patient-type new \
  --discharge-time gt_30
# Output: code=99239, source=TIME
```
```
