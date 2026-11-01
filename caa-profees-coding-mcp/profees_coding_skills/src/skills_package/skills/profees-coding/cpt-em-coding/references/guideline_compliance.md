# Guideline Compliance

This document certifies that the inpatient-cpt-em-coding skill follows current industry guidelines for E/M code selection.

## Guidelines Followed

1. **CPT® (Current Procedural Terminology)** per American Medical Association (AMA)
   - Current edition: 2024-2026
   - Reference: https://www.ama-assn.org/practice-management/cpt

2. **AAFP (American Academy of Family Physicians) E/M Coding Guidelines**
   - Reference: https://www.aafp.org/family-physician/practice-and-career/getting-paid/coding/
   - Key principle: MDM complexity is PRIMARY driver for inpatient codes

3. **CMS Medicare E/M Code Selection Guidelines**
   - Based on 1995 and 1997 Documentation Guidelines for E/M Services
   - Reference: https://www.cms.gov/

## Code Selection Rules Compliance

### Outpatient Encounters (CPT 99202-99215)

| Rule | Implementation | Compliance |
|------|-----------------|------------|
| **Decision Rule** | Higher of MDM or TIME wins | ✅ Correctly implemented |
| **MDM Levels** | 4 levels: minimal, low, moderate, high | ✅ All 4 supported |
| **Time Thresholds (NEW)** | 15, 30, 45, 60+ minutes (minimums) | ✅ Per CPT/AAFP |
| **Time Thresholds (EST)** | 10, 20, 30, 40+ minutes (minimums) | ✅ Per CPT/AAFP |
| **Time as Proxy** | Can use time alone if MDM not available | ✅ Supported |

**Guideline Rationale:** Outpatient codes weight both MDM complexity and time equally. Higher numeric code wins to encourage detailed documentation of either dimension.

---

### Observation Status (CPT 99221-99223, 99231-99233)

| Rule | Implementation | Compliance |
|------|-----------------|------------|
| **Decision Rule** | Higher of MDM or TIME wins (like outpatient) | ✅ Correctly implemented |
| **MDM Levels** | 4 levels: minimal, low, moderate, high | ✅ All 4 supported |
| **Time Thresholds (NEW)** | 40, 55, 75+ minutes | ✅ Per CPT spec |
| **Time Thresholds (EST)** | 30, 40, 55+ minutes | ✅ Per CPT spec |
| **Distinct from Outpatient** | Different time thresholds than outpatient | ✅ Correctly differentiated |

**Guideline Rationale:** Observation uses the inpatient/observation code family and time minimums per CPT guidance.

---

### Inpatient Consultations (CPT 99252-99255)

| Rule | Implementation | Compliance |
|------|-----------------|------------|
| **Decision Rule** | **MDM PRIMARY; time SECONDARY** | ✅ Correctly implemented |
| **MDM Levels (NEW)** | 4 levels map to 99252-99255 | ✅ All 4 supported |
| **MDM Levels (EST)** | Subsequent care uses 99231-99233 | ✅ Supported |
| **Code 99255** | Reachable via MDM high or time 80+ min | ✅ Supported |
| **Time Thresholds (NEW)** | 35, 45, 60, 80+ minutes | ✅ Per CPT/AAFP |
| **Time Thresholds (EST)** | 30, 40, 55+ minutes | ✅ Per CPT/AAFP |
| **Time as Proxy** | Can use time alone (no MDM needed) | ✅ Supported for fallback |

**Guideline Rationale (AAFP & CPT):** Inpatient consultations prioritize MDM complexity as the primary driver because:
- Inpatient consultations involve higher clinical stakes and complexity
- MDM better captures the intellectual effort in complex hospital cases
- Time alone may not reflect consultant's actual work (interrupted by other duties)
- Ensures appropriate code selection based on clinical judgment complexity

**Exception Rule:** Time code can override MDM only if numerically higher (e.g., MDM moderate=99253 but time 80+ min=99255).

---

### Preventive Encounters (CPT 99381-99397)

| Rule | Implementation | Compliance |
|------|-----------------|------------|
| **Decision Rule** | Determined by age category only | ✅ Correctly implemented |
| **Age Categories** | 7 categories (infant through senior) | ✅ All 7 supported |
| **MDM Input** | Not used for preventive codes | ✅ Ignored correctly |
| **Time Input** | Not used for preventive codes | ✅ Ignored correctly |

**Guideline Rationale:** Preventive codes are age-based per AMA CPT. MDM and time are not factors for preventive visit codes.

---

### Emergency Department Encounters (CPT 99281-99285)

| Rule | Implementation | Compliance |
|------|-----------------|------------|
| **Decision Rule** | MDM-only (no time selection) | ✅ Correctly implemented |
| **MDM Levels** | none, minimal, low, moderate, high | ✅ All 5 supported |
| **Patient Type** | No new/established distinction | ✅ Correctly implemented |

**Guideline Rationale:** ED level selection is driven by MDM; time is not a determining factor.

---

### Same-Day Hospital Admit/Discharge (CPT 99234-99236)

| Rule | Implementation | Compliance |
|------|-----------------|------------|
| **Decision Rule** | Higher of MDM or TIME wins | ✅ Correctly implemented |
| **MDM Levels** | minimal, low, moderate, high | ✅ All supported |
| **Time Thresholds** | 45, 60, 80+ minutes | ✅ Per CPT spec |

**Guideline Rationale:** Same-day admit/discharge services use combined codes based on MDM or
documented total time.

---

### Hospital Discharge Services (CPT 99238-99239)

| Rule | Implementation | Compliance |
|------|-----------------|------------|
| **Decision Rule** | Time category only | ✅ Correctly implemented |
| **Time Thresholds** | ≤30 minutes (99238), >30 minutes (99239) | ✅ Per CPT spec |

**Guideline Rationale:** Discharge services are selected solely by discharge management time.

---

## Time Definition (Per AAFP & CMS Guidelines)

**Clinician Time** means:
- ✅ **Face-to-face time** with patient (NEW and ESTABLISHED)
- ✅ **Direct care time** for inpatient/observation consultations
- ✅ Time spent on patient's medical record and care coordination directly related to visit
- ❌ NOT administrative work, documentation, or time spent between encounters

**Implementation:** Script accepts `time_minutes` parameter as clinician time in minutes. Caller responsible for accurate time capture per guidelines.

---

## MDM Complexity Assessment (Per AAFP Guidelines)

MDM complexity determined by **three elements** (per AAFP/CMS):
1. **Problem Complexity** - number and severity of problems
2. **Data Complexity** - amount/type of data reviewed/obtained
3. **Risk** - risk of complications/morbidity if untreated

**Implementation:** Script accepts `mdm` parameter (minimal/low/moderate/high). Caller responsible for MDM assessment per documentation guidelines.

**Reference:** See `mdm_problem_complexity.md`, `mdm_data_complexity.md`, `mdm_risk_complexity.md` for detailed MDM assessment rules.

---

## Inpatient vs Observation: Key Distinction

Per AAFP and encounter classification rules:

| Aspect | Inpatient | Observation |
|--------|-----------|------------|
| **Admission Status** | Admitted as inpatient (hospital bed) | Admitted for observation only |
| **CPT Codes** | 99252-99255, 99231-99233 | 99221-99223, 99231-99233 |
| **MDM Rule** | PRIMARY driver | Equal weight with time |
| **Typical Duration** | > 24 hours | < 24-48 hours |
| **Care Level** | Full hospital care | Intermediate level |
| **Code Selection** | MDM complexity (with time exception) | Higher of MDM or time |

**Critical:** Encounter classification (inpatient vs observation) must be determined from admission status documentation BEFORE code selection. This skill assumes correct classification is provided.

---

## Compliance Verification

✅ **All code ranges match current CPT (2024-2026)**
- Outpatient: 99202-99215
- Preventive: 99381-99397  
- Observation: 99221-99223, 99231-99233
- Inpatient: 99252-99255, 99231-99233
- Emergency: 99281-99285
- Same-day admit/discharge: 99234-99236
- Hospital discharge: 99238-99239

✅ **All precedence rules follow AAFP guidelines**
- Outpatient: Higher code wins
- Observation: Higher code wins
- Inpatient: MDM primary (with time override exception)
- Preventive: Age-based only
- Emergency: MDM-only
- Same-day admit/discharge: Higher code wins
- Hospital discharge: Time category only

✅ **All time thresholds match CPT specifications**

✅ **All MDM levels supported (ED adds none)**

✅ **Audit trail**: `source` field tracks whether code came from MDM, TIME, or PREVENTIVE

---

## Testing & Validation

All rules above are tested and validated in `tests/test_em_code_selection.py`:
- Edge cases (missing inputs, boundary times)
- All complexity levels × patient types × encounter types
- Integration tests confirming code ranges are generated per encounter rules
- Precedence rules verification for each encounter type

Run tests: `pytest tests/test_em_code_selection.py -v`

---

## References

1. **AAFP E/M Coding** - https://www.aafp.org/family-physician/practice-and-career/getting-paid/coding/
2. **CPT®** - https://www.ama-assn.org/practice-management/cpt
3. **CMS Documentation Guidelines** - https://www.cms.gov/Medicare/Coding/ICD10/2024-ICD-10-CM
4. **AMA CPT Book (2024)** - Professional Edition
5. **Medicare Learning Network (MLN) Connects** - CMS guidance on E/M coding

**Last Updated:** February 2026
**Guideline Year:** 2024-2026 CPT / AAFP Current Edition
