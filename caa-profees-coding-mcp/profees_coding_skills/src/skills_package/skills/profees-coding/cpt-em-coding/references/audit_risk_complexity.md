# Audit: Risk of Complications

## CORE DOCUMENTATION RULE (prevents RS errors)
- Count risk ONLY when the provider documents a same-day directive in **Assessment/Plan or Orders** (start/stop/change/continue/no change).
- A medication/device/procedure appearing ONLY in the **med list/history**, charge lines, or templated/carry-forward content is NOT management and does NOT count.
- Orders count only when they clearly reflect today’s management (e.g., named medication prescribed/refilled with directions, or a procedure clearly performed/decided today). Do NOT use procedure/med “orders” that are not supported in the A/P narrative.
- “Consider / will consider / may consider” (without a documented decision today) does NOT count.
- If the note says “medications unchanged/no changes”:
- If tied to **named Rx meds** documented in the same encounter → Rx management present (Moderate).
- If NOT tied to named meds anywhere → “medication without name specified” → Minimal.
- Do NOT hallucinate meds, devices, procedures, or actions.
- Dont double dipping: Stable chronic care during preventive visits does not count as MDM risk unless separately evaluated.

## RISK DRIVERS (keep tight; no duplication)

## Minimal
- Education only; rest; elastic bandage; gargles.
- Diagnostic tests (e.g., labs, EKG, MRI/US, rapid strep, PCR, home sleep test) without therapeutic management.
- Diagnostic procedures (e.g., diagnostic laryngoscopy) without therapeutic management; if separately reportable, do not use it to elevate E/M risk.
- Referrals/follow-up/return precautions/care coordination language **without documented external clinician discussion**.
- Discussion of potential future treatments not initiated/ordered today.
- “Medication” without a name specified.

## Low
- Named OTC medication management (start/continue/stop/defer) and PRN OTC guidance for current symptoms.
- OTC devices (brace/splint) recommended/managed.
- Conditional OTC supportive care tied to testing ordered/performed today (e.g., “if strep negative, use acetaminophen/ibuprofen”).
- Routine immunization administration.
- Decision regarding a minor procedure/surgery **without** documented patient- or procedure-specific risk factors.

## Moderate
- **Prescription drug management**: any start/stop/change/continue/maintain of a **named Rx medication** documented in A/P or Orders.
- Vitamins/supplements/herbals/homeopathic products (e.g., turmeric) do NOT count as prescription drug management unless clearly a regulated prescription drug.
- OTC availability does not matter if it is clearly prescribed/managed as a prescription drug.
- “As prescribed by another provider” counts ONLY if the current provider explicitly directs continuation/change/discontinuation today.
- **Prescribed device management** is Moderate ONLY when the provider initiates/discontinues therapy or changes settings/therapy (e.g., CPAP/BiPAP settings, implanted device programming). Routine counseling/analysis with **no changes** is Low/Minimal.
- Decision regarding a minor procedure/surgery **with** documented patient- or procedure-specific risk factors.
- Decision for elective major surgery **without** patient-specific risk factors.
- Management limited by documented SDoH.

## High
- Drug therapy requiring intensive monitoring for toxicity.
- Decision for emergency major surgery; elective major surgery **with** risk factors.
- Decision for hospitalization/escalation to hospital-level care.
- DNR/de-escalation due to poor prognosis.

# IMPORTANT CLARIFICATIONS (prevents RF errors)
- Referral/consult alone is **Minimal** risk. If the note documents an actual **decision/plan for surgery** or a **decision regarding hospitalization**, classify risk under Moderate/High accordingly.
- “Care coordination” affects **Data** only if there is documented external discussion; it does not raise Risk by itself.
- Do not upcode to High due to long consent/risk counseling for minor procedures.

#######################################################################################################
# OUTPUT RULES
- Base decisions ONLY on encounter note + Orders + Additional Info provided.
- Provide: (1) final risk label, (2) 1-sentence justification within 8 words, (3) confidence: low/medium/high.
- No leading justification and no step-by-step chain-of-thought.
