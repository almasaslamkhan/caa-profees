# Modifier Guidelines

Apply current in-scope modifier guidance while preserving claim lines, linked diagnoses, and source documentation.

## Branch Selection Context
- Optimize for recall when deciding which modifier branches to evaluate; include a branch unless it is clearly incompatible with the encounter.
- Do not assign modifiers during branch selection. Branch selection only identifies plausible modifier branches and candidate CPT/HCPCS codes for later evaluation.
- Review every available branch and each candidate code before finalizing branch selection.
- For a grouped branch, evaluate each modifier concept represented by the group while returning the exact branch label expected by the workflow.
- Common screening cues are not enough by themselves. Include a branch only when there is encounter-specific evidence that meets that branch's minimum evidence threshold.
- Treat same-day services as branch triggers only when they are performed/completed on the DOS, not merely ordered or planned. Examples include minor procedures, immunization administration, therapeutic injection/infusion/hydration administration, therapy/rehab services, and in-office or point-of-care tests with same-day results.
- For laterality branches, do not infer LT/RT/50 for timed PT/OT treatment or orthosis management from diagnosis laterality alone. Require line-tied service laterality unless the procedure/order is clearly sided for the billed service.

## In-Scope Modifier Context
- therapy/rehab CPTs: include GP/GO/GN when therapy-furnished codes or therapy context show outpatient PT, OT, or SLP services on the DOS. Therapy context may come from SOAP, orders, setting, structured metadata, or therapy-specific CPT patterns.
- Assistant therapy modifiers CQ/CO: apply only when PTA or OTA involvement is attributable to the service. Structured provider-role or care-team metadata can be enough when there is no indication the assistant contribution was de minimis.
- Modifier 95: use for eligible professional telehealth services delivered through synchronous audio-video communication. Generic "telehealth" wording is supportive only when not contradicted by in-person, asynchronous, remote-monitoring, or audio-only evidence; modifier 95 is not for audio-only services.
- GY/UD: GY is for statutorily excluded services or items that are not a Medicare benefit when payer context supports Medicare and the exclusion is clear. UD follows the dataset operational rule for qualifying administered J-code lines in the presence of T1015 or the required encounter-billing context; do not apply UD to non-J-code lines.
- Anesthesia physical status P1-P4: first look for explicit ASA Physical Status in structured anesthesia/EHR fields, then use attributable anesthesia, pre-op, operative, or assessment documentation when structured ASA is absent or ambiguous. Normalize text such as "ASA 1", "ASA II", and "Level 3"; do not guess across irreconcilable conflicting ASA signals.
- QK: use only for anesthesia CPTs when physician medical direction is supported and 2-4 concurrent anesthesia cases are documented or otherwise encounter-specific.
- QX: use only for anesthesia CPTs when a CRNA or nonphysician anesthetist service is medically directed by a physician. Do not infer QX from anesthesia CPTs alone, and do not apply it to personally performed anesthesia or unsupported supervision-only context.
- QW and Modifier 90: QW requires a CLIA-waived lab test on the maintained mapping/list plus in-office, point-of-care, or waived-site performance on the DOS. Modifier 90 requires test-specific reference lab, outside lab, send-out, or external processing evidence tied to the DOS. If reference lab evidence exists for the same CPT, prefer 90 over QW.

## Modifier 25 (Significant, Separately Identifiable Evaluation and Management Service by the Same Physician or Other Qualified Health Care Professional on the Same Day of the Procedure or Other Service):
**Definition**: It may be necessary to indicate that on the day a **procedure** or **service** identified by a CPT code was performed, the patient's condition required a significant, separately identifiable E/M service above and beyond the other service provided or beyond the usual preoperative and postoperative care associated with the procedure that was performed. A significant, separately identifiable E/M service is defined or substantiated by documentation that satisfies the relevant criteria for the respective E/M service to be reported (see Evaluation and Management Services Guidelines for instructions on determining level of E/M service). The E/M service may be prompted by the symptom or condition for which the procedure and/or service was provided. As such, different diagnoses are not required for reporting of the E/M services on the same date. This circumstance may be reported by adding modifier 25 to the appropriate level of E/M service. Note: This modifier is not used to report an E/M service that resulted in a decision to perform surgery. See modifier 57. For significant, separately identifiable non-E/M services, see modifier 59.

**How to assign this modifier**: Modifier 25 is used to report an E/M service on a day when another service was provided to the patient by the same individual. Turn to Appendix A in the CPT manual and read the description of this modifier. To assign modifier 25 correctly, there must be a medical necessity to provide a separate, additional E/M service on the same day a procedure was performed or another service was provided. The medical necessity for this additional E/M service must be documented in the patient's medical record. If you do not add modifier 25 to the additional E/M code for service on the day of a procedure, the third-party payer would disallow the charge because it would be thought to be the evaluation/management portion of the procedure. By adding modifier 25, you are stating that the service was separate from the procedure or original service. Use of modifier 25 increases the potential of receiving payment for the service.

**Example**: A physician provided a dialysis service to a hospital inpatient. In addition, the physician provided a separate discharge service (not related to the dialysis). You would report the dialysis service (procedure) and also report the inpatient service (discharge), adding the modifier 25 to the discharge service. 

The modifier can also be assigned when additional E/M services are provided on the same day to the same patient.
**Example**: If a patient came into the office for a visit early in the day and then later in the day returned for a separate unrelated E/M service, you report both services using E/M codes and add modifier 25 to the second E/M code. If the services were for the same related condition as seen in the earlier service, documentation for both services would be considered to assign only one E/M code for that day.

Let's review a another example of the use of modifier 25. 
**Example**: A patient presents for repair of a laceration (12042, intermediate laceration repair) and the questions asked by the physician were related only to the laceration repair. In this case, the E/M service is included in the laceration repair service and not reported separately. However, if the physician documented the patient had elevated blood pressure and a history of hypertension and the physician evaluated and treated the hypertensive condition, modifier 25 would be added to the E/M code. Diagnosis code I10 (hypertension) would be reported as the diagnosis for the E/M service. The laceration repair would have a trauma diagnosis for open wound, and as such, the diagnosis code would support the medical necessity for the laceration repair. In this example, both the E/M service (with 25) and the laceration repair are reported with diagnosis codes that indicate the medical necessity of both services.

**CMS RULES**: If reporting services for a Medicare patient, modifier 25 could be added to an E/M code when a decision for surgery was made on the same day as a procedure with a global surgical package of 0-10 days or procedures not covered by global surgery rules (global indicator of XXX) if the E/M service is not associated with the decision to perform a minor surgical procedure whether the patient is a new or established patient. Modifier 57 would be added to an E/M code when the service resulted in a decision for surgery on the day before or the day of a procedure with a global surgical package of 90 days.

### Compliance Checks
- **Modifier 25 (Significant, separately identifiable E/M with minor procedure same day):**
  - PRIMARY QUALIFYING-SERVICE GATE (must be YES before any 25 consideration):
    - At least one of the following is present on the same DOS by the same physician/QHP:
      a) A minor procedure (0- or 10-day global) that is listed as a billable CPT/HCPCS line in CPT Codes OR is explicitly documented as performed in the SOAP note and/or Orders on the DOS by the same physician/QHP/clinic staff under their direction, OR
      b) A preventive screening HCPCS (G0101 and/or Q0091) that is listed as a billable CPT/HCPCS line in CPT Codes OR explicitly documented as completed in the SOAP note/Orders on the DOS by the same physician/QHP, OR
      c) An immunization administration (90471–90474 or G0008) that is listed in CPT Codes OR explicitly documented as completed in Orders on the DOS (“admin inferred from Orders”), OR
      d) A second, separately billable E/M type on the claim (i.e., preventive + problem E/M both present in CPT Codes).
    - If NONE of a–d is present (e.g., only a single preventive E/M 9938X/9939X or only a single problem E/M 9920X/9921X with no other qualifying service/admin and no separate E/M type), DO NOT append 25.
  - Critical exclusions for the Gate:
    - Diagnostics (e.g., ThinPrep Pap laboratory test, TSH, CBC, imaging) and specimen collection (e.g., 36415 venipuncture, 99000 handling) DO NOT qualify as the “other service” for 25.
    - Documentation inference standard: You may treat a minor procedure as present based on SOAP/Orders only when the note explicitly documents that the procedure was performed (clear action language, site/laterality, materials/instruments). Vague mentions of symptoms, considerations, or future plans do NOT qualify.
  - Application rules (only after the gate above is met):
    - You may append 25 to a problem-oriented E/M (e.g., 99212–99215) when it is significant and separately identifiable from:
      a) A minor procedure (0- or 10-day global) performed the same day by the same physician/QHP, OR
      b) A preventive medicine service (e.g., 99381–99397) or a recognized “sports/school/special Olympics physical” E/M (often reported by some payers with 99499) performed the same day by the same physician/QHP.
      c) Preventive screening HCPCS G0101 (breast/pelvic exam) and/or Q0091 (Pap collection), OR
      d) Immunization administration (e.g., 90471–90474 or G0008) when the documentation shows a distinct problem-oriented evaluation (e.g., assessment/management of an acute problem/exposure or complex chronic conditions) beyond routine pre-immunization screening/counseling.
  - Preventive E/M with a same-day procedure (no separate problem-oriented E/M billed):
    - If a preventive E/M (99381–99397) and a procedure are performed the same day and there is significant, separately identifiable problem-oriented E/M work that is not separately billed as 9920X/9921X, you may append modifier 25 to the preventive E/M to indicate the distinct E/M performed in addition to the procedure, when payer policy permits. If payer policy requires the modifier on the problem-oriented E/M instead, report a separate 9920X/9921X with 25. Default to payer-appropriate placement of 25.
    - Significant problem-oriented work may include active assessment/management of chronic conditions (e.g., medication changes/titration, ordering/interpretation of labs for disease monitoring, new/refill prescriptions with risk/benefit discussion, documented home monitoring plan, counseling and management decisions) that would stand alone as an office visit.
  - Apply ONLY if ALL are true:
    1. Same day as the other service, and
    2. Same physician/QHP who bills the E/M also provided the other service (minor procedure, preventive E/M, or immunization administration), and
    3. The problem-oriented E/M work is above and beyond the usual pre- and post-work of the other service (distinct history/exam/MDM), and
    4. Documentation clearly supports distinct problems/management (e.g., assessment/plan for obesity, OSA, diabetes separate from the preventive/sports physical).
  - SAME-PROBLEM RULE (important):
    - If the E/M addresses the same problem/lesion/injury that is treated by the minor procedure, and the E/M consists of typical pre/intra/post work (e.g., focused HPI/exam for that problem, decision to perform the procedure, consent, local wound/aftercare instructions, ordering expected tests such as X-ray for the same injury/infection, writing antibiotics/analgesics), DO NOT append 25 unless there is clearly documented additional, separate evaluation/management of a different problem (distinct diagnosis/MDM).
    - Do NOT deny 25 solely because the E/M and procedure share the same diagnosis. If the documented E/M clearly exceeds the typical pre/intra/post work for the reported procedure (e.g., complex evaluation of recurrent epistaxis with review of failed prior care, anticoagulation risk assessment/plan, and surgical planning beyond office cautery), 25 may be appropriate.
    - When considering “above and beyond,” compare the documented E/M against the typical inherent work for the specific CPT performed, not just any minor procedure in general.
  - INJECTION/THERAPEUTIC ADMIN RULES (96372–96377):
    - Therapeutic/prophylactic injections (e.g., 96372 with J-codes) have minimal inherent pre-/post-work. If the E/M includes substantial diagnostic evaluation and management of the underlying condition (e.g., injury assessment, ordering/reviewing imaging, attempted or planned reduction/splinting/immobilization, referrals to ED/orthopedics, return precautions), this is significant and separately identifiable from the injection; 25 may be appropriate.
    - Do NOT append 25 if the E/M work is limited to deciding to give an injection and routine post-injection instructions.
  - LACERATION REPAIR + IMMUNIZATION (e.g., 1200X + 90471/90715):
    - For acute injury encounters, if the E/M documents a broader trauma evaluation beyond the inherent pre/post of the repair (e.g., mechanism of injury, location/depth, neurovascular status, exploration/foreign body concern, tetanus risk assessment, determination of repair approach, aftercare planning, and/or need for additional services or referral), and an immunization administration is also performed, the E/M may be significant and separately identifiable; 25 may be appropriate.
    - If documentation is limited to minimal pre-repair evaluation and routine aftercare without broader decision-making, do not append 25.
  - IMMUNIZATION-SPECIFIC RULES:
    - Do NOT append 25 for routine vaccine-only encounters where documentation is limited to standard screening, consent, and administration (90471–90474 or G0008).
    - You MAY append 25 when there is documented evaluation/management of an acute problem/exposure leading to the decision to vaccinate (e.g., laceration with tetanus risk assessment, animal bite with rabies prophylaxis), demonstrating work beyond routine immunization administration. 
    - Vaccine product codes alone (e.g., 90662, 90715) do not qualify; an administration code (90471–90474 or G0008) must be present in CPT Codes or documented as completed in Orders to consider 25.
  - SPECIMEN COLLECTION (BLOOD DRAW) HANDLING:
    - Venipuncture/specimen collection (e.g., 36415, 99000) never qualifies as the “other service” for assigning modifier 25 and should not, by itself, trigger or block 25.
    - Only assign 25 when another qualifying service (preventive E/M, G0101/Q0091, immunization administration, or a minor procedure) is present AND the E/M is significant and separately identifiable per the rules above. The concurrent presence of venipuncture neither justifies nor invalidates 25.
  - DIAGNOSTIC-ONLY SCENARIOS (Local Policy Override):
    - Default: Do NOT append 25 when the only other services besides the E/M are diagnostic tests (e.g., EKG/ECG 93000/93005/93010, echo 93306/93308, imaging, labs) or specimen collection (36415/99000).
    - Local override permitted ONLY if the SOAP/Orders explicitly name the payer and cite a written payer policy that requires modifier 25 with E/M + diagnostic-only services on the same DOS. In such cases, append 25 per that cited policy and include the payer/policy name in your rationale. Absent explicit citation, default to no 25.
  - RULE FOR PREVENTIVE E/M:
    - If a preventive E/M (99381–99397) is present with an immunization administration (e.g., 90471–90474 or G0008) on the same DOS, AND the documentation includes active, problem-oriented management of at least one chronic or acute condition that would independently support a problem-oriented E/M (e.g., medication change/titration, new or refill prescriptions with risk/benefit discussion, diagnostic/lab orders for disease monitoring, documented home monitoring plan, targeted evaluation with assessment and referral), THEN append modifier 25 to the preventive E/M when no separate 9920X/9921X is billed and payer policy permits. Do not let the presence of venipuncture (36415) alone negate this decision when immunization administration is also present.
  - Hard disqualifiers (if any are true, DO NOT append 25):
    - The only other services are diagnostic tests (e.g., labs, ECG/EKG 93000/93005/93010, echocardiography 93306/93308, imaging) that were ordered or performed by another department/technologist.
    - Specimen collection services such as venipuncture (e.g., 36415, 99000) or handling/transport (e.g., 99000) do not qualify as the “minor procedure” for modifier 25 and should not be used to justify 25 on the E/M.
    - The “decision to order a test” (including scheduling a test later the same day) is the only additional work.
    - The other service did not occur the same day or the performer is unclear.
  - CERUMEN IMPACTION (69210/69209) RULES:
    - Bill cerumen removal as a procedure only when documentation supports impacted cerumen. Indicators of impaction include ANY of:
      - Cerumen impairs examination of clinically significant portions of the external auditory canal, tympanic membrane, or middle ear condition; OR
      - Extremely hard, dry, irritative cerumen causes symptoms such as pain, itching, hearing loss; OR
      - Cerumen is associated with foul odor, infection, or dermatitis; OR
      - Obstructive, copious cerumen cannot be removed without magnification and multiple instrumentations requiring physician skills.
    - Instruments/magnification requirement:
      - Procedure typically requires magnification (e.g., otoscope/operating microscope) and instruments such as curettes, forceps, or suction performed by the physician/QHP (or otolaryngologist).
    - Modifier 25 application:
      - If the visit is solely for cerumen-related symptoms in a known cerumen producer and the E/M work is limited to assessment/decision and typical pre/intra/post work of the removal, report only the cerumen removal (e.g., 69210/69209) without a separate E/M; do NOT append 25.
      - Append 25 to a problem-oriented E/M only when there is significant, separately identifiable evaluation/management beyond the inherent work of cerumen removal (e.g., evaluation/management of additional unrelated problems, complex comorbidity assessment/medication management not integral to the removal).
      - Do not append 25 when the E/M addresses only the same cerumen problem and does not exceed typical pre/intra/post work for 69210/69209.
  - SPLINT/IMMOBILIZATION (e.g., 29125–29130, 29200–29280) RULES:
    - Splint/strapping application codes are minor procedures. When a same-day problem-oriented E/M is performed by the same physician/QHP and a splint application is billed on the claim (e.g., 29130 for finger splint), evaluate modifier 25.
    - Modifier 25 is appropriate when documentation shows significant E/M beyond the inherent pre/intra/post work of the splint application, such as:
      - Injury mechanism and broader musculoskeletal evaluation; AND
      - Imaging ordered/reviewed/interpreted that informs management; AND/OR
      - Fracture management decisions (e.g., referral to hand/orthopedics, immobilization plan, activity restrictions, analgesic/anti-inflammatory management, return precautions).
    - Do NOT append 25 if the E/M is limited to deciding to apply the splint and providing routine instructions inherent to the application.
    - Performance by nursing/clinical staff under the ordering physician/QHP within the same encounter may still be billed by the physician/QHP; treat as same-physician/QHP service for modifier 25 evaluation when the splint application CPT is on the claim.   
  - DOCUMENTATION-BASED INFERENCE AND BILLING ALIGNMENT:
    - If a qualifying minor procedure is identified solely from SOAP/Orders documentation (no CPT line present), proceed to evaluate modifier 25 as if the procedure will be billed. Clearly recommend adding the corresponding CPT/HCPCS code for the procedure on the claim.
  - NOTES:
    - Preventive + problem E/M same day: If a preventive E/M (99381–99397 or payer-required 99499) and a problem-oriented E/M (99202–99215) are both on the claim, evaluate and, when supported, append 25 only to the problem-oriented E/M line (e.g., 9921425). Do not append 25 to the preventive E/M line. This includes sports/school/special physicals that some payers require reporting with 99499.
    - Preventive E/M + procedure (no separate problem E/M billed): If payer policy permits, you may append 25 to the preventive E/M to indicate significant, separately identifiable E/M performed in addition to the procedure. Otherwise, consider reporting a separate problem-oriented E/M with 25.
    - Immunization from Orders: If an immunization administration (e.g., G0008 or 90471–90474) is documented as completed in Orders but is not yet present in CPT Codes, note this as “admin inferred from Orders” and evaluate 25 for the problem-oriented E/M. Recommend including the admin code on the claim; if the admin is not ultimately billed, do not use 25.
    - Minor procedure same day: Append 25 only if a separate problem was evaluated/managed beyond the procedure-related care, or if the same-diagnosis E/M is clearly beyond typical pre/intra/post work for that specific procedure as documented.

### Checklist
- Primary Qualifying-Service Gate
  - Is there at least one qualifying same-day service in addition to (or distinct from) the E/M under review: minor procedure, preventive screening HCPCS (G0101/Q0091), immunization administration (90471–90474 or G0008), or a second E/M type on the claim (preventive + problem)? The qualifying service must be:
    - a billable CPT/HCPCS line in CPT Codes, OR
    - explicitly documented as performed in the SOAP note and/or Orders on the DOS by the same physician/QHP/clinic staff under their direction.
  - Diagnostics/specimen collection do NOT qualify. Yes/No. If No, do NOT assign 25.
- Modifier 25 evaluation (only if Gate = Yes):
  - Is a preventive E/M (99381–99397 or 99499) OR preventive screening HCPCS (G0101/Q0091) on the claim or documented as completed in Orders/SOAP? Yes/No (identify code and source)
  - Is an immunization administration code (90471–90474 or G0008) on the claim OR documented as completed in Orders? Yes/No (identify source and code)
  - Is a minor procedure (e.g., laceration repair 1200X–1201X, injection 96372–96377, splint application 29125–29130) on the claim OR explicitly documented as performed in SOAP/Orders? Yes/No (identify code or narrative evidence)
  - Is a problem-oriented E/M (99202–99215) also on the claim? Yes/No
  - Same day and same physician/QHP for both E/M and the other service(s)? Yes/No (with evidence)
  - Distinct problem-oriented E/M content beyond the other service(s)? Yes/No (with evidence)
  - If preventive E/M + procedure (and NO separate problem-oriented E/M is billed), does documentation show significant, separately identifiable problem-oriented E/M work beyond preventive content (including active chronic disease management such as med titration, new/refill prescriptions with risk/benefit, diagnostic/lab orders for disease monitoring, and home monitoring plans)? Yes/No (with evidence). If Yes and payer allows, append 25 to the preventive E/M; otherwise recommend separate problem-oriented E/M with 25.
  - Preventive E/M: If the above bullet is Yes and an immunization administration is present the same day, default to appending 25 to the preventive E/M (when no separate 9920X/9921X is billed and payer permits), even if venipuncture (36415) is also present.
  - Blood draw rule: Confirm that venipuncture/specimen collection (e.g., 36415/99000) is not being used as the qualifying “other service.” Only consider 25 when another qualifying service (preventive E/M, G0101/Q0091, immunization administration, or a minor procedure) is present AND the E/M is significant and separately identifiable. Yes/No
  - Does the E/M address only the same problem as the minor procedure without additional distinct problems? Yes/No
  - Diagnostic-only scenario check: Are the only other services diagnostics/specimen collection? Yes/No. If Yes, is a named payer policy explicitly cited requiring 25? Yes/No. If No, do not assign 25.
  - Comparator: Does the documented E/M exceed the typical inherent pre/intra/post work for the specific procedure(s)? Yes/No (with evidence)
  - If injection (96372–96377), did the E/M include broader diagnostic evaluation/management (e.g., imaging, reduction/splint, ED/ortho referral, sling/brace, return precautions)? Yes/No (with evidence)
  - Are the only other services diagnostic tests or specimen collection (e.g., EKG 93000/93005/93010, labs, 36415, 99000)? Yes/No
  - If a qualifying procedure is inferred from documentation but no CPT/HCPCS code is present, recommend adding the appropriate procedure code to the claim; if that code is not ultimately billed, do not use modifier 25 on final submission.

### Examples
- **25** appropriate: 
  - 99214-25 + 11401 (Significant E/M above and beyond same-day minor procedure).
  - 99396 + 99214-25 (Preventive + problem-oriented E/M same day; 25 on problem E/M).
  - 99396-25 + 90471/90686 (Preventive E/M with immunization admin and significant chronic condition management—e.g., HTN med titration, labs, home monitoring—no separate 9921X billed; 25 on preventive per payer policy).
  - 99213-25 + G0101 (Problem-oriented E/M with same-day screening pelvic/breast exam).
  - 99214-25 + G0008 + 90662 (Problem-oriented E/M with same-day influenza vaccine admin; E/M beyond routine vaccine counseling).
  - 99203-25 + 30903 (Recurrent epistaxis with expanded evaluation beyond cautery).
  - 99203-25 + 96372 + 73030 + J1885 (Acute shoulder dislocation: substantial evaluation and management; injection ancillary).
  - 99202-25 + 12002 + 90471/90715 (Acute laceration with tetanus prophylaxis; broader injury evaluation and repair planning).
  - 99203-25 + 29130 (Acute finger injury with imaging-confirmed fracture; significant E/M for injury evaluation/management plus same-day finger splint application).
- **25** inappropriate:
  - 99213 + 10060 (Evaluation typical to procedure; not distinct).
  - 99214 + 36415 (Venipuncture doesn’t qualify).
  - 99395 only (Annual preventive exam with counseling on weight/anxiety/skin/STD screening; no same-day procedure/admin and no separate problem E/M billed. Do NOT append 25 to 99395).
  - 99385 + 36415 + ThinPrep Pap lab + TSH lab orders (diagnostics-only; no G0101/Q0091 on claim; no immunization admin; no separate 9920X/9921X. Do NOT append 25 to 99385).

---

## Modifier 50 (Bilateral Procedure):
**Definition**: Unless otherwise identified in the listings, bilateral procedures that are performed at the same session should be identified by adding modifier 50 to the appropriate 5 digit code. Note: This modifier should not be appended to designated “add-on” codes (see 'Add-On Procedures').

**How to assign this modifier**: If the same procedure is performed on a mirror-image organ of the body (such as right and left kidneys), modifier 50, indicating a bilateral procedure, would be reported.

**Example**: An arthroplasty (total knee replacement, 27447 and 27447-50) for both left and right knees at the same operative session would be reported using the modifier 50.

Another example in which the same services may be performed on two sides would be a bilateral breast procedure (e.g., bilateral, simple complete mastectomy, 19303 and 19303-50).

It is very important to determine how the third-party payer wants bilateral procedures submitted on the claim form, on a single line or multiple lines. One payer may require two codes to be displayed for the bilateral procedure (such as 27447 and 27447-50) for which they would reimburse 100% for the first code and 50% for the second code. Another payer may require that bilateral procedures be reported with only one code (27447-50) and would reimburse 150% based on that one code with modifier -50 representing two services. Still other payers want the service reported as 27447-RT and 27447-LT.

**CMS RULES**: Medicare does not accept two line items to describe a bilateral procedure. Medicare rules direct the coder to use one line with modifier -50 and 1 unit of service. For example, submit 27447-50 rather than 27447, 27447-50.

**Inherently bilateral codes**: 27395,92060,92065,92225,92226,92230,92235,92240,92250,92260,92020

**Add-On Procedures**: Modifier 50 must NEVER be used with codes classified as add-on procedures:
01953,01968,01969,10004,10006,10008,10010,10012,10036,11001,11008,11045,11046,11047,11103,11105,11107,11201,11732,11922,13102,13122,13133,13153,14302,15003,15005,15012,15014,15016,15018,15101,15111,15116,15121,15131,15136,15151,15152,15156,15157,15201,15221,15241,15261,15272,13153,14302,15003,15005,15012,15014,15016,15018,15101,15111,15116,15121,15131,15136,15151,15152,15156,15157,15201,15221,15241,15261,15272,19288,19294,19297,20700,20701,20702,20703,20704,20705,20930,20931,20932,20933,20934,20936,20937,20938,20939,20985,22103,22116,22208,22216,22226,22328,22512,22515,22527,22534,22552,22585,22614,22632,22634,22840,22841,22842,22843,22844,22845,22846,22847,22848,22853,22854,22858,22859,22860,22868,22870,26125,26861,26863,27358,27692,29826,31627,31632,31633,31637,31649,31651,31654,32501,32506,32507,32667,32668,32674,33141,33225,33257,33258,33259,33268,33277,33367,33368,33369,33370,33419,33508,33517,33518,33519,33521,33522,33523,33530,33572,33746,33768,33866,33884,33904,33924,33929,33987,34709,34711,34713,34714,34715,34716,34717,34808,34812,34813,34820,34833,34834,35306,35390,35400,35500,35572,35681,35682,35683,35685,35686,35697,35700,36218,36227,36228,36248,36474,36476,36479,36483,36907,36908,36909,37185,37186,37222,37223,37232,37233,37234,37235,37237,37239,37247,37249,37252,37253,38102,38746,38747,38900,43273,43283,43338,43635,44015,44121,44128,44139,44203,44213,44701,44955,47001,47542,47543,47544,47550,48400,49326,49327,49412,49435,49623,49905,50606,50705,50706,51797,52442,56606,57267,57465,58110,58611,60512,60661,61316,61517,61611,61641,61642,61651,61781,61782,61783,61797,61799,61800,61864,61868,62148,62160,63035,63043,63044,63048,63052,63053,63057,63066,63076,63078,63082,63086,63088,63091,63103,63295,63308,63621,64421,64462,64480,64484,64491,64492,64494,64495,64597,64629,64634,64636,64643,64645,64727,64778,64783,64787,64832,64837,64859,64872,64874,64876,64901,64902,64913,65757,66990,67225,67320,67331,67332,67334,67335,67340,69990,74248,74301,74713,75565,75774,76015,76125,76802,76810,76812,76814,76937,76979,76983,77001,77002,77003,77063,77293,78020,78434,78496,78730,78835,80506,81266,81416,81426,81536,82952,86826,87187,87503,87904,88155,88177,88185,88311,88314,88332,88334,88341,88350,88364,88369,88373,0071U,0072U,0073U,0074U,0075U,0076U,0130U,0131U,0132U,0133U,0134U,0135U,0136U,0137U,0138U,0157U,0158U,0159U,0160U,0161U,0162U,0207U,0427U,90461,90472,90474,90785,90833,90836,90838,90840,90863,90913,91013,92547,92608,92618,92621,92623,92627,92921,92925,92929,92934,92938,92944,92972,92973,92974,92978,92979,92998,93319,93320,93321,93325,93352,93356,93462,93463,93464,93563,93564,93565,93566,93567,93568,93569,93571,93572,93573,93574,93575,93584,93585,93586,93587,93588,93592,93598,93609,93613,93621,93622,93623,93655,93657,93662,93896,93897,93898,94645,94729,94781,95079,95873,95874,95885,95886,95887,95940,95941,95962,95967,95984,96113,96121,96131,96133,96137,96139,96159,96165,96168,96171,96203,96361,96366,96367,96368,96370,96371,96375,96376,96411,96415,96417,96423,96547,96548,96570,96571,96934,96935,96936,97130,97546,97551,97598,97811,97814,98981,99100,99116,99135,99140,99153,99157,99292,99359,99415,99416,99417,99418,99425,99427,99437,99439,99458,99459,99467,99486,99489,99494,99498,99602,99607,0054T,0055T,0076T,0095T,0098T,0164T,0165T,0174T,0214T,0215T,0217T,0218T,0222T,0397T,0437T,0439T,0443T,0450T,0480T,0496T,0513T,0523T,0560T,0562T,0570T,0599T,0628T,0630T,0649T,0663T,0676T,0678T,0690T,0698T,0701T,0709T,0722T,0724T,0735T,0742T,0751T,0752T,0753T,0754T,0755T,0756T,0757T,0758T,0759T,0760T,0761T,0762T,0763T,0764T,0767T,0770T,0772T,0774T,0777T,0791T,0821T,0822T,0827T,0828T,0829T,0830T,0831T,0832T,0833T,0834T,0835T,0836T,0837T,0838T,0839T,0840T,0841T,0842T,0843T,0844T,0845T,0846T,0847T,0848T,0849T,0850T,0851T,0852T,0853T,0854T,0855T,0856T,0857T,0859T,0866T,0882T,0883T,0887T,0896T,0899T,0900T,0907T,0914T,0945T

### Compliance Checks
- **Modifier 50 (Bilateral)** vs **Laterality (LT/RT)**:
  - Use modifier 50 to report bilateral performance when the CPT code is inherently unilateral and payer policy supports -50 for bilateral reporting; alternatively, some payers prefer 2 units without -50.
  - Do NOT append LT/RT to codes where bilateral reporting is typically via -50 or units (e.g., 69210 cerumen removal). Only use LT/RT when payer policy explicitly requires laterality instead of -50, and documentation supports distinct sides.
  - If documentation is ambiguous or payer policy is unknown, default to no LT/RT (and no -50) unless bilateral is clearly documented and payer rules are known.
- **Laterality (LT/RT) & Site-specific HCPCS when applicable.**

### Checklist
- Laterality/Bilateral (LT/RT vs 50) Checklist:
  - Is the procedure inherently unilateral with potential bilateral performance? Yes/No (name code)
  - Does payer policy explicitly require LT/RT instead of -50 for bilateral reporting? Yes/No (evidence from documentation/orders)
  - For 69210 cerumen removal specifically: Do NOT assign LT/RT. If bilateral is clearly documented and payer policy allows, consider -50 or 2 units; otherwise, default to no laterality.
  - If documentation/payer policy is unclear, do not assign LT/RT.

### Examples
- *Bilateral vs LT/RT**:
  - 69210 cerumen removal: Do NOT use LT/RT; if bilateral and allowed, report 69210-50 (one line) or 2 units per payer policy; default to no laterality if unclear.

---

## Modifier LT (Left side (used to identify procedures performed on the left side of the body)):
**Definition**: Modifier LT is appended to procedure or service codes to indicate that the service was performed on the left side of the body. It clarifies laterality when billing for procedures involving paired organs or anatomical sites (e.g., eyes, ears, limbs, or kidneys).

**How to assign this modifier**: Assign Modifier LT when ALL the following criteria are met:
1. The procedure, service, or item involves a paired organ or anatomical site.
2. The service was performed specifically on the left side of the body.
3. The CPT or HCPCS code description does NOT already define laterality as “left,” “right,” or “bilateral.”
4. The medical record documentation clearly specifies performance on the left side.

**Key Points:**
- Do NOT use LT when the code description already specifies laterality or bilateral procedure.
- Do NOT use LT with E/M codes.
- Always ensure medical records support use of LT for claim accuracy and compliance.

**Examples:**
- Cataract extraction on the left eye
- MRI of the left knee
- Fitting of orthotic device for the left leg
- Mastectomy of the left breast

### Compliance Checks
- **Modifier 50 (Bilateral)** vs **Laterality (LT/RT)**:
  - Use modifier 50 to report bilateral performance when the CPT code is inherently unilateral and payer policy supports -50 for bilateral reporting; alternatively, some payers prefer 2 units without -50.
  - Do NOT append LT/RT to codes where bilateral reporting is typically via -50 or units (e.g., 69210 cerumen removal). Only use LT/RT when payer policy explicitly requires laterality instead of -50, and documentation supports distinct sides.
  - If documentation is ambiguous or payer policy is unknown, default to no LT/RT (and no -50) unless bilateral is clearly documented and payer rules are known.
- **Laterality (LT/RT) & Site-specific HCPCS when applicable.**

### Checklist
- Laterality/Bilateral (LT/RT vs 50) Checklist:
  - Is the procedure inherently unilateral with potential bilateral performance? Yes/No (name code)
  - Does payer policy explicitly require LT/RT instead of -50 for bilateral reporting? Yes/No (evidence from documentation/orders)
  - For 69210 cerumen removal specifically: Do NOT assign LT/RT. If bilateral is clearly documented and payer policy allows, consider -50 or 2 units; otherwise, default to no laterality.
  - If documentation/payer policy is unclear, do not assign LT/RT.

### Examples
- *Bilateral vs LT/RT**:
  - 69210 cerumen removal: Do NOT use LT/RT; if bilateral and allowed, report 69210-50 (one line) or 2 units per payer policy; default to no laterality if unclear.

---

## Modifier RT (Right side (used to identify procedures performed on the right side of the body)):
**Definition**: Modifier RT is appended to procedure or service codes to indicate that the service was performed on the right side of the body. It clarifies laterality for procedures involving paired organs or anatomical sites (e.g., eyes, ears, arms, legs, or kidneys).

**How to assign this modifier**: Assign Modifier RT when ALL the following criteria are met:
1. The procedure, service, or item involves a paired organ or anatomical site.
2. The service was performed specifically on the right side of the body.
3. The CPT or HCPCS code description does NOT already define laterality as “right,” “left,” or “bilateral.”
4. The medical record documentation clearly specifies performance on the right side.

**Key Points:**
- Do NOT use RT when the code description already specifies laterality or bilateral procedure.
- Do NOT use RT with E/M codes.
- Medical records must support use of RT to ensure correct claims and compliance.

**Examples:**
- Cataract surgery on the right eye
- Injection to the right knee
- Fitting of a hearing device for the right ear

### Compliance Checks
- **Modifier 50 (Bilateral)** vs **Laterality (LT/RT)**:
  - Use modifier 50 to report bilateral performance when the CPT code is inherently unilateral and payer policy supports -50 for bilateral reporting; alternatively, some payers prefer 2 units without -50.
  - Do NOT append LT/RT to codes where bilateral reporting is typically via -50 or units (e.g., 69210 cerumen removal). Only use LT/RT when payer policy explicitly requires laterality instead of -50, and documentation supports distinct sides.
  - If documentation is ambiguous or payer policy is unknown, default to no LT/RT (and no -50) unless bilateral is clearly documented and payer rules are known.
- **Laterality (LT/RT) & Site-specific HCPCS when applicable.**

### Checklist
- Laterality/Bilateral (LT/RT vs 50) Checklist:
  - Is the procedure inherently unilateral with potential bilateral performance? Yes/No (name code)
  - Does payer policy explicitly require LT/RT instead of -50 for bilateral reporting? Yes/No (evidence from documentation/orders)
  - For 69210 cerumen removal specifically: Do NOT assign LT/RT. If bilateral is clearly documented and payer policy allows, consider -50 or 2 units; otherwise, default to no laterality.
  - If documentation/payer policy is unclear, do not assign LT/RT.

### Examples
- *Bilateral vs LT/RT**:
  - 69210 cerumen removal: Do NOT use LT/RT; if bilateral and allowed, report 69210-50 (one line) or 2 units per payer policy; default to no laterality if unclear.
