# Encounter Type: Inpatient & Observation

## Select Encounter Type
- Select a category only if it is explicitly supported by the SOAP note, especially the Assessment and Plan (A/P).
- Encounter type determines which CPT code range applies (outpatient, observation, inpatient, or preventive).
- If uncertain or not clearly documented, do not select the category.
- Do not mention or justify categories that are not selected.

## Observation Status
Select when the patient is admitted to observation status (also called "obs" or "observation status care") as documented in:
- Admit/discharge section explicitly stating "observation status"
- HPI or A/P noting "admitted for observation"
- Observation billing/acuity level indicated in the record
- Time-stamped observation order in the medical record

Observation status encounters:
- Are typically **shorter duration stays** than inpatient admission (usually < 24-48 hours)
- Represent an intermediate level of care between outpatient and inpatient
- Use the inpatient/observation E/M code family (99221-99223, 99231-99233)
- May transition to inpatient status if the patient is admitted as an inpatient

Do not select Observation Status when:
- The patient is explicitly admitted as an inpatient (use inpatient codes instead)
- The encounter is an outpatient visit or ED visit (no admission)
- The note only mentions potential for observation but no actual admission

## Inpatient Consultation or Admission
Select when the patient is admitted to the hospital as an **inpatient** (non-observation status) as documented in:
- Admit/discharge section explicitly stating "admitted as inpatient" or "hospital admission"
- HPI or A/P noting "admitted to hospital"
- Inpatient bed assignment or inpatient service documented
- Consultation requested for an already-admitted inpatient patient

Inpatient encounter characteristics:
- Patient is under hospital care and not observation status
- Typically **longer duration stays** than observation (usually > 24 hours)
- Use inpatient consultation CPT codes (99252-99255)
- MDM complexity is the primary driver of code selection
- Initial inpatient consultations (new): 99252-99255
- Subsequent inpatient visits use subsequent care codes (99231-99233)

Do not select Inpatient when:
- The patient is in observation status (use observation codes instead)
- The encounter is outpatient or ED-only
- The note refers to a previous inpatient admission but the current encounter is post-discharge/outpatient

## Same-Day Admit/Discharge (Hospital)
Select when documentation shows **admission and discharge on the same date** in the
hospital inpatient or observation setting.

Same-day admit/discharge encounter characteristics:
- Admit and discharge documented on the same service date
- Use CPT codes 99234-99236
- Select by MDM or total clinician time when documented

Do not select Same-Day Admit/Discharge when:
- Admission and discharge occur on different dates
- Discharge-only documentation is present (use hospital discharge codes)

## Hospital Discharge
Select when the encounter documents discharge management time from inpatient/observation.

Hospital discharge encounter characteristics:
- Use CPT codes 99238 (≤30 minutes) or 99239 (>30 minutes)
- Selection based on discharge time category only

Do not select Hospital Discharge when:
- Discharge does not occur on the service date
- No discharge management time is documented

## Encounter Type Priority (for classification)
When determining encounter type, check in this order:
1. **Is this a same-day admit/discharge?** → Use HOSPITAL ADMIT/DISCHARGE
2. **Is this a hospital discharge service?** → Use HOSPITAL DISCHARGE
3. **Is the patient admitted to inpatient status?** → Use INPATIENT
4. **Is the patient in observation status?** → Use OBSERVATION
5. **Is the patient in the emergency department?** → Use EMERGENCY
6. **Is the patient at an office/outpatient visit?** → Use OUTPATIENT
7. **Is this a preventive/wellness encounter?** → Use PREVENTIVE (can combine with outpatient if problem-oriented care also documented)

## Evidence Capture
For each selected encounter type, capture:
- Objective evidence from the Objective/Exam section.
- Assessment/Plan evidence showing the work performed.

For Observation, document the admission to observation status and the duration of stay.
For Inpatient, document the admission to inpatient status and service (initial or subsequent visit).
