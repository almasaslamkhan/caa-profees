# Audit: Data Complexity

Your task is to review and reflect on the generated data complexity assessment to ensure accuracy, adherence to guidelines, and completeness.

# Instructions:
Your role is to double-check the note and the data complexity in the model response and ensure compliance with the additional criteria below.

Apply System 2 Thinking, meaning that for each instruction below, explicitly reason by breaking down your thought process step-by-step before reaching a conclusion.
The reasoning should mention why each instruction (bullet in this case) was either followed, not followed, or not applicable. Ensure that you output the reason for each instruction.

1. If the SOAP note mentioned discussed with another health care provider with specific names or addresses, add discussion of management with external provider (DISC) to the model response.
- Referral to an external provider does not count as DISC.

2. Independent interpretation (Intpr) for radiology/imaging:
- Add 'Intpr' only if the practitioner explicitly documents an independent interpretation of an image/tracing (e.g., x-ray/EKG) that is distinct from the radiology/diagnostic report.
- If CT/MRI/x-ray results are mentioned without a clear statement of independent interpretation by the practitioner in this encounter (e.g., only a pasted radiology impression), do NOT add 'Intpr' (or remove it if present).
- If imaging results are only patient-reported (not reviewed by the practitioner), do NOT add 'Intpr' (or remove it if present).

3. Check the presence of an independent historian (iHx) carefully:
- Add iHx when the history is obtained from someone other than the patient and it is necessary because the patient cannot provide a reliable/complete history (e.g., young child, altered mental status, dementia, language barrier without direct patient history, etc.).
- Do NOT assume iHx based only on age/occupation; it must be supported by documentation (e.g., "mother reports...", "caregiver states...") and necessity.
- Do NOT require physical attendance; iHx may be obtained in-person or by phone/video if documented.
- Do NOT count someone merely participating in treatment-plan discussion (without providing necessary history) as iHx.

4. Check and update (if necessary) the number of tests & documents using AMA 2021+ 'unique test' concepts:
- Physical exams and medication orders should not be counted as tests & documents elements.
- Count each unique test (or unique panel) ordered/reviewed once. Do not double-count the same test as both "ordered" and "result reviewed" for Category 1.
- If test results are provided/reported by the patient (and not reviewed by the practitioner), remove them from the model response.
- If a test order is pending, only discussed, or planned conditionally (e.g., "if symptoms worsen/persist"), remove it from the model response unless it is actually ordered/performed/reviewed today.
- Items reviewed in external records (e.g., ER notes) may be counted if the note documents review; count the external note/document separately and count each distinct test result reviewed as its own unique test element.
- Multiplex/panel tests (e.g., respiratory viral panel; combined COVID/Flu A/B assay) count as ONE unique test element, even if multiple analytes/components are listed in the results.
- Urinalysis, urine microscopy, and urine culture/sensitivity are separate unique tests when individually ordered/reviewed; do not combine them into one element based on specimen source.
- Do not split routine lab panels into subcomponents (e.g., do not split CMP/Chem panels into BUN, bilirubin, LFTs, etc.) unless separately ordered as distinct tests.
- Specimen collection (e.g., capillary blood collection) by itself is not a unique diagnostic test element for MDM data unless an actual test/result is ordered/reviewed.
- All items that appear in Pharmacy orders should be considered medication orders and removed from tests & documents.
- If the encounter includes a preventive visit (e.g., well child check, annual physical, annual wellness): do NOT count routine preventive screening tests/services toward data complexity unless the documentation ties the test to evaluation/management of a separately addressed problem.

Finally return the final list of data complexity items with their types after your proposed changes, if any, and include an overall confidence score between low, medium and high for your data complexity assessment.
