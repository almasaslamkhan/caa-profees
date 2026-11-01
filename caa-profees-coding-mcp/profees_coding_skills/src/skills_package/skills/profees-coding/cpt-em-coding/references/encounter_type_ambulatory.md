# Encounter Type: Ambulatory & Office/Outpatient

STRICT RULES:
- Select a category ONLY if its criteria are explicitly supported in the SOAP note (especially A/P). If uncertain or not clearly documented, do NOT select it.
- If NOT_SUPPORTED criteria are met, select ONLY NOT_SUPPORTED and do not also select Office/Outpatient or Preventive Medicine.
- Do NOT mention or justify categories you did not select.
- Text must be concise, human-friendly, 1–2 sentence justifications max, and no bad characters (e.g., no <br>).

CATEGORY 1 — NOT_SUPPORTED:
Select ONLY when the note explicitly supports that the same-day E/M service is not separately reportable because it is included in payment for a procedure under NCCI edits or global surgery rules.

Certain NOT_SUPPORTED evidence includes:
- The note states the E/M is supported by NCCI edits or global surgery rules as not separately reportable.
- The note states the procedure has a global period of 000, 010, or 090 days and the same-day service is included in payment for the procedure.
- The note states the decision to perform a minor surgical procedure is included in payment and shall not be reported separately as an E/M service.
- The note states there was not a significant and separately identifiable E/M service unrelated to the problem or decision that led to the procedure.

Do NOT select NOT_SUPPORTED when the note documents a significant, separately identifiable E/M service unrelated to the procedure, or when NCCI/global surgery is mentioned without saying the same-day service is included in procedure payment or not separately reportable.

CATEGORY 2 — Office/Outpatient (Problem-Oriented E/M):
Select ONLY if the A/P shows separate, problem-oriented MDM beyond routine preventive care, tied to a specific problem, such as:
- Problem diagnostics (labs/imaging/tests ordered for symptoms/condition; not routine screening).
- Active medication decision-making: start/stop/change medication, or refill/continue ONLY if A/P includes condition-specific assessment/monitoring/risk discussion that affects management.
- Therapeutic/procedural management (injections, wound care plan, splint/DME orders, etc.).
- Referrals for a problem.
- Condition-specific monitoring/follow-up that reflects ACTIVE management due to instability/symptoms/therapy decisions.
- Counseling directly linked to managing a diagnosed problem.

DO NOT select Office/Outpatient if problems are only listed/history-only, stable/controlled with no active management, reassurance only, trivial/self-limited with no work-up, unrelated past dx, screening result noted without action, or vitals reviewed without condition-based decisions.

CRITICAL STABLE CHRONIC RULE:
If chronic conditions are described as stable/controlled/at goal/asymptomatic/no changes/continue current management AND only routine periodic labs or routine 6–12 month follow-up/lifestyle advice are planned, this is NOT separate problem-oriented MDM → do NOT select Office/Outpatient.

CRITICAL MED REFILL RULE:
A refill/continuation alone (no start/stop/change, no documented management decision about control/risks/side effects) does NOT qualify for Office/Outpatient.

PREVENTIVE OVERRIDE:
If the encounter is clearly a wellness/preventive exam, do NOT add Office/Outpatient merely because stable chronic conditions are listed or long-term meds are continued/refilled. Add Office/Outpatient during a preventive visit ONLY if there is a significant, separately identifiable problem requiring problem-oriented MDM (e.g., med start/stop/change; management change due to worsening/control issues; monitoring orders tied to instability/management decisions; referral for that condition; or a separate condition-specific follow-up plan beyond annual).

CATEGORY 3 — Preventive Medicine (Wellness):
Select ONLY if the A/P explicitly documents a wellness/preventive exam (e.g., annual physical, wellness visit, well-child, routine preventive health assessment; often Z00.xx/Z01.xx/Z12.5, but code not required if exam is explicit).
Do NOT select Preventive if A/P does not clearly document a wellness exam.
Notes: vaccines alone do not make a visit preventive; scheduling a future preventive visit does not count; screening for symptoms/problems is not preventive.

COMBINATION RULE:
If both a documented preventive exam AND separately identifiable problem-oriented MDM are present, select BOTH. Do NOT add Office/Outpatient for trivial/incidental issues.

EVIDENCE REQUIREMENT (for EACH selected category):
- objective_evidence: direct quote(s) from Objective/Exam supporting the category.
- assessment_plan_evidence: direct quote(s) from A/P or note text showing the work performed or the not-supported rule.
For NOT_SUPPORTED, assessment_plan_evidence MUST quote the NCCI/global surgery/payment-included/not-separately-reportable language.
For Preventive Medicine, assessment_plan_evidence MUST explicitly state wellness/preventive exam, otherwise do not select it.
