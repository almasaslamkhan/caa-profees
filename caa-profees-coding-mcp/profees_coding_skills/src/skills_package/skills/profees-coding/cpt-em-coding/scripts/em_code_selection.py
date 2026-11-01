#!/usr/bin/env python3
"""Select CPT E/M codes for outpatient, observation, inpatient, preventive, or ED encounters.

Rules are aligned with:
  - CPT® (Current Procedural Terminology) guidelines per AMA
  - AAFP (American Academy of Family Physicians) E/M coding guidelines
  - CMS Medicare guidelines for E/M code selection
  - cdrc/profees_coding/em/models/utils.py

Supports:
  - Outpatient (office) visits: CPT 99202-99215
  - Preventive visits: CPT 99381-99397
  - Observation/inpatient care: CPT 99221-99223, 99231-99233, 99234-99236, 99238-99239
  - Inpatient consultations: CPT 99252-99255
  - Emergency department visits: CPT 99281-99285

Code Selection Precedence Rules (per current CPT/AAFP guidelines):
  - OUTPATIENT: Higher of MDM or TIME wins (both equally weighted)
  - OBSERVATION: Higher of MDM or TIME wins (both equally weighted)
  - INPATIENT: MDM is PRIMARY driver; time is secondary (overrides only if higher)
  - PREVENTIVE: Age category only (time and MDM not used)
  - EMERGENCY: MDM only; time not used for level selection
"""

from __future__ import annotations

import argparse
import json
from enum import Enum
from typing import Tuple


class PatientType(str, Enum):
    NEW = "new"
    ESTABLISHED = "established"


class EncounterType(str, Enum):
    OUTPATIENT = "outpatient"
    PREVENTIVE = "preventive"
    OBSERVATION = "observation"
    INPATIENT = "inpatient"
    INPATIENT_INITIAL = "inpatient_initial"
    INPATIENT_SUBSEQUENT = "inpatient_subsequent"
    INPATIENT_DISCHARGE = "inpatient_discharge"
    EMERGENCY = "emergency"
    HOSPITAL_ADMIT_DISCHARGE = "hospital_admit_discharge"
    HOSPITAL_DISCHARGE = "hospital_discharge"


class ComplexityLevel(str, Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class EmergencyMdmLevel(str, Enum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class DischargeTimeLevel(str, Enum):
    LEQ_30 = "leq_30"
    GT_30 = "gt_30"


class AgeCategory(str, Enum):
    INFANT = "infant"  # < 1
    EARLY_CHILD = "early_child"  # 1-4
    LATE_CHILD = "late_child"  # 5-11
    ADOLESCENT = "adolescent"  # 12-17
    YOUNG_ADULT = "young_adult"  # 18-39
    ADULT = "adult"  # 40-64
    SENIOR = "senior"  # 65+
    INVALID = "invalid"


def get_age_category(age: int | None) -> AgeCategory:
    """Categorize patient age into preventive medicine age bands.

    Args:
        age: Patient age in years, or None.

    Returns:
        AgeCategory enum value based on age ranges defined by preventive care codes.

    Raises:
        None.

    Example:
        >>> get_age_category(5).value
        'late_child'
        >>> get_age_category(None).value
        'invalid'
    """
    if age is None:
        return AgeCategory.INVALID
    if age < 0:
        return AgeCategory.INVALID
    if age < 1:
        return AgeCategory.INFANT
    if age < 5:
        return AgeCategory.EARLY_CHILD
    if age < 12:
        return AgeCategory.LATE_CHILD
    if age < 18:
        return AgeCategory.ADOLESCENT
    if age < 40:
        return AgeCategory.YOUNG_ADULT
    if age < 65:
        return AgeCategory.ADULT
    return AgeCategory.SENIOR


def get_time_based_em_code(patient_type: PatientType, minutes: int | None) -> str | None:
    """Map outpatient encounter time to CPT code (outpatient only).

    Args:
        patient_type: NEW or ESTABLISHED patient classification.
        minutes: Total clinician time in minutes, or None/invalid.

    Returns:
        CPT outpatient code (99202-99205 or 99212-99215), or None if no match.

    Raises:
        None.

    Example:
        >>> get_time_based_em_code(PatientType.NEW, 50)
        '99204'
    """
    if minutes is None or minutes <= 0:
        return None

    time_table = {
        PatientType.NEW: [
            (15, "99202"),
            (30, "99203"),
            (45, "99204"),
            (60, "99205"),
        ],
        PatientType.ESTABLISHED: [
            (10, "99212"),
            (20, "99213"),
            (30, "99214"),
            (40, "99215"),
        ],
    }

    selected_code = None
    for min_t, code in time_table[patient_type]:
        if minutes >= min_t:
            selected_code = code
    return selected_code


def get_mdm_code(patient_type: PatientType, complexity: ComplexityLevel) -> str:
    """Map outpatient MDM complexity level to CPT code (outpatient only).

    Args:
        patient_type: NEW or ESTABLISHED patient classification.
        complexity: ComplexityLevel enum value (minimal through high).

    Returns:
        CPT outpatient code (99202-99205 or 99212-99215).

    Raises:
        KeyError: If complexity level is not supported for outpatient (e.g., VERY_HIGH).

    Example:
        >>> get_mdm_code(PatientType.NEW, ComplexityLevel.MODERATE)
        '99204'
    """
    code_lookup = {
        PatientType.NEW: {
            ComplexityLevel.MINIMAL: "99202",
            ComplexityLevel.LOW: "99203",
            ComplexityLevel.MODERATE: "99204",
            ComplexityLevel.HIGH: "99205",
        },
        PatientType.ESTABLISHED: {
            ComplexityLevel.MINIMAL: "99212",
            ComplexityLevel.LOW: "99213",
            ComplexityLevel.MODERATE: "99214",
            ComplexityLevel.HIGH: "99215",
        },
    }
    return code_lookup[patient_type][complexity]


def get_preventive_code(patient_type: PatientType, age: int | None) -> Tuple[str | None, str]:
    """Select preventive medicine CPT code by patient age and type.

    Args:
        patient_type: NEW or ESTABLISHED patient classification.
        age: Patient age in years, or None.

    Returns:
        Tuple of (code, note). Code is None if age is invalid; note indicates status.

    Raises:
        None.

    Example:
        >>> get_preventive_code(PatientType.NEW, 25)
        ('99385', 'ok')
    """
    age_category = get_age_category(age)
    if age_category == AgeCategory.INVALID:
        return None, "invalid_age"

    code_lookup = {
        PatientType.NEW: {
            AgeCategory.INFANT: "99381",
            AgeCategory.EARLY_CHILD: "99382",
            AgeCategory.LATE_CHILD: "99383",
            AgeCategory.ADOLESCENT: "99384",
            AgeCategory.YOUNG_ADULT: "99385",
            AgeCategory.ADULT: "99386",
            AgeCategory.SENIOR: "99387",
        },
        PatientType.ESTABLISHED: {
            AgeCategory.INFANT: "99391",
            AgeCategory.EARLY_CHILD: "99392",
            AgeCategory.LATE_CHILD: "99393",
            AgeCategory.ADOLESCENT: "99394",
            AgeCategory.YOUNG_ADULT: "99395",
            AgeCategory.ADULT: "99396",
            AgeCategory.SENIOR: "99397",
        },
    }

    return code_lookup[patient_type][age_category], "ok"


def select_outpatient_code(
    patient_type: PatientType,
    mdm: ComplexityLevel | None,
    time_minutes: int | None,
) -> Tuple[str, str, str]:
    """Select CPT E/M code for outpatient (office) encounter.

    Args:
        patient_type: NEW (new to practice) or ESTABLISHED (returning patient). For outpatient,
            this refers to whether the patient has established care at the practice.
        mdm: Medical Decision Making complexity level (minimal, low, moderate, high),
            or None.
        time_minutes: Total clinician time in minutes, or None.

    Returns:
        Tuple of (code, source, note). Code is empty string if no applicable code.

    Notes:
        - Outpatient codes: 99202-99205 (new patient), 99212-99215 (established patient).
        - If both MDM and time apply, selects the HIGHER numeric CPT code (outpatient precedence).
        - Time thresholds: new 15/30/45/60+ min, established 10/20/30/40+ min.
        - Requires at least one of mdm or time_minutes to be provided.
        - Contrast with INPATIENT where MDM is primary and time is secondary.

    Example:
        >>> select_outpatient_code(PatientType.NEW, ComplexityLevel.MODERATE, None)
        ('99204', 'MDM', 'mdm_only')
        >>> select_outpatient_code(PatientType.NEW, ComplexityLevel.LOW, 50)
        ('99204', 'TIME', 'time_higher')  # 99204 (time) > 99203 (MDM), so time wins
    """
    if mdm is None and (time_minutes is None or time_minutes <= 0):
        return "", "", "missing_mdm_and_time"

    mdm_code = get_mdm_code(patient_type, mdm) if mdm else None
    time_code = get_time_based_em_code(patient_type, time_minutes)

    if mdm_code and time_code:
        # Choose the higher numeric CPT code when both apply
        if int(time_code) > int(mdm_code):
            return time_code, "TIME", "time_higher"
        return mdm_code, "MDM", "mdm_higher_or_equal"

    if mdm_code:
        return mdm_code, "MDM", "mdm_only"
    if time_code:
        return time_code, "TIME", "time_only"

    return "", "", "no_applicable_code"


def get_observation_code(
    patient_type: PatientType,
    mdm: ComplexityLevel | None,
    time_minutes: int | None,
) -> Tuple[str, str, str]:
    """Select CPT E/M code for observation status encounter.

    Args:
        patient_type: NEW (initial observation admission) or ESTABLISHED (subsequent visit on
            same observation admission).
        mdm: Medical Decision Making complexity level (minimal, low, moderate, high),
            or None.
        time_minutes: Total clinician time in minutes, or None.

    Returns:
        Tuple of (code, source, note). Code is empty string if no applicable code.

    Notes:
        - Observation uses the same code family as inpatient/observation care (99221-99223,
          99231-99233) rather than deleted observation codes.
        - If both MDM and time provided, the higher numeric CPT code is selected.
        - Time thresholds: NEW 40/55/75+ minutes; EST 30/40/55+ minutes.
        - Requires at least one of mdm or time_minutes to be provided.

    Example:
        >>> get_observation_code(PatientType.NEW, ComplexityLevel.MODERATE, None)
        ('99222', 'MDM', 'mdm_only')
        >>> get_observation_code(PatientType.NEW, None, 55)
        ('99222', 'TIME', 'time_only')
    """
    if mdm is None and (time_minutes is None or time_minutes <= 0):
        return "", "", "missing_mdm_and_time"

    observation_mdm_table = {
        PatientType.NEW: {
            ComplexityLevel.MINIMAL: "99221",
            ComplexityLevel.LOW: "99221",
            ComplexityLevel.MODERATE: "99222",
            ComplexityLevel.HIGH: "99223",
        },
        PatientType.ESTABLISHED: {
            ComplexityLevel.MINIMAL: "99231",
            ComplexityLevel.LOW: "99231",
            ComplexityLevel.MODERATE: "99232",
            ComplexityLevel.HIGH: "99233",
        },
    }

    observation_time_table = {
        PatientType.NEW: [
            (40, "99221"),
            (55, "99222"),
            (75, "99223"),
        ],
        PatientType.ESTABLISHED: [
            (30, "99231"),
            (40, "99232"),
            (55, "99233"),
        ],
    }

    mdm_code = observation_mdm_table[patient_type][mdm] if mdm else None
    time_code = None
    if time_minutes is not None and time_minutes > 0:
        for min_t, code in observation_time_table[patient_type]:
            if time_minutes >= min_t:
                time_code = code

    if mdm_code and time_code:
        if int(time_code) > int(mdm_code):
            return time_code, "TIME", "time_higher"
        return mdm_code, "MDM", "mdm_higher_or_equal"

    if mdm_code:
        return mdm_code, "MDM", "mdm_only"
    if time_code:
        return time_code, "TIME", "time_only"

    return "", "", "no_applicable_code"


def get_inpatient_code(
    patient_type: PatientType,
    mdm: ComplexityLevel | None,
    time_minutes: int | None,
) -> Tuple[str, str, str]:
    """Select CPT E/M code for inpatient consultation.

    Follows AAFP and CPT guidelines: MDM complexity is the PRIMARY driver for inpatient consults.

    Args:
        patient_type: NEW (initial consultation) or ESTABLISHED (subsequent visit on same
            admission). Note: This refers to whether the consultation is initial or subsequent
            on the current inpatient admission, not whether the patient is new to the practice.
        mdm: Medical Decision Making complexity level (minimal, low, moderate, high),
            or None.
        time_minutes: Total clinician time in minutes, or None.

    Returns:
        Tuple of (code, source, note). Code is empty string if no applicable code.

    Notes:
        - **GUIDELINE**: Inpatient consult MDM complexity is the PRIMARY driver.
        - Initial consultation codes: 99252-99255 (4 MDM levels, one per complexity).
        - Subsequent inpatient consult codes are no longer separately reportable; use
          subsequent inpatient/observation care codes instead.
        - Time-based selection is SECONDARY: only used if (1) no MDM provided, or
          (2) time-based code is numerically higher than MDM code.
        - Time thresholds (total clinician time):
          New: 35/45/60/80+ min → 99252-99255
          Established: 30/40/55+ min → 99231-99233 (fallback to subsequent care codes)
        - Requires at least one of mdm or time_minutes to be provided.

    Example:
        >>> get_inpatient_code(PatientType.NEW, ComplexityLevel.LOW, None)
        ('99252', 'MDM', 'mdm_only')
        >>> get_inpatient_code(PatientType.NEW, None, 90)
        ('99255', 'TIME', 'time_only')
    """
    if mdm is None and (time_minutes is None or time_minutes <= 0):
        return "", "", "missing_mdm_and_time"

    inpatient_mdm_table = {
        PatientType.NEW: {
            ComplexityLevel.MINIMAL: "99252",
            ComplexityLevel.LOW: "99252",
            ComplexityLevel.MODERATE: "99253",
            ComplexityLevel.HIGH: "99255",
        },
        PatientType.ESTABLISHED: {
            ComplexityLevel.MINIMAL: "99231",
            ComplexityLevel.LOW: "99231",
            ComplexityLevel.MODERATE: "99232",
            ComplexityLevel.HIGH: "99233",
        },
    }

    inpatient_time_table = {
        PatientType.NEW: [
            (35, "99252"),
            (45, "99253"),
            (60, "99254"),
            (80, "99255"),
        ],
        PatientType.ESTABLISHED: [
            (30, "99231"),
            (40, "99232"),
            (55, "99233"),
        ],
    }

    mdm_code = inpatient_mdm_table[patient_type][mdm] if mdm else None

    time_code = None
    if time_minutes is not None and time_minutes > 0:
        for min_t, code in inpatient_time_table[patient_type]:
            if time_minutes >= min_t:
                time_code = code

    # GUIDELINE: For inpatient, MDM is PRIMARY driver per AAFP/CPT standards.
    # Time-based code overrides only if numerically HIGHER than MDM code.
    if mdm_code:
        if time_code and int(time_code) > int(mdm_code):
            # Time-based code is higher: exception to MDM primacy (e.g., 99253 MDM but 99255 time)
            return time_code, "TIME", "time_higher"
        # MDM drives code selection (standard rule for inpatient)
        return mdm_code, "MDM", "mdm_used"

    if time_code:
        # No MDM provided: use time-based code as fallback
        return time_code, "TIME", "time_only"

    return "", "", "no_applicable_code"


def get_emergency_department_code(
    mdm: EmergencyMdmLevel | None,
) -> Tuple[str, str, str]:
    """Select CPT E/M code for emergency department encounters.

    Args:
        mdm: EmergencyMdmLevel enum value (none, minimal, low, moderate, high),
            or None.

    Returns:
        Tuple of (code, source, note). Code is empty string if no applicable code.

    Notes:
        - ED codes do not distinguish new vs established patients.
        - ED code selection is based on MDM only; time is not used.
        - 99281 is reserved for encounters that may not require a physician/QHP and has
          no MDM level; select it only when documentation explicitly supports that scenario.

    Example:
        >>> get_emergency_department_code(EmergencyMdmLevel.LOW)
        ('99283', 'MDM', 'mdm_only')
        >>> get_emergency_department_code(EmergencyMdmLevel.NONE)
        ('99281', 'MDM', 'mdm_not_required')
    """
    if mdm is None:
        return "", "", "missing_mdm"

    ed_mdm_table = {
        EmergencyMdmLevel.NONE: "99281",
        EmergencyMdmLevel.MINIMAL: "99282",
        EmergencyMdmLevel.LOW: "99283",
        EmergencyMdmLevel.MODERATE: "99284",
        EmergencyMdmLevel.HIGH: "99285",
    }

    code = ed_mdm_table[mdm]
    note = "mdm_not_required" if mdm == EmergencyMdmLevel.NONE else "mdm_only"
    return code, "MDM", note


def get_hospital_admit_discharge_code(
    mdm: ComplexityLevel | None,
    time_minutes: int | None,
) -> Tuple[str, str, str]:
    """Select CPT code for same-day hospital admit/discharge services.

    Args:
        mdm: Medical Decision Making complexity level (minimal, low, moderate, high),
            or None.
        time_minutes: Total clinician time in minutes, or None.

    Returns:
        Tuple of (code, source, note). Code is empty string if no applicable code.

    Notes:
        - Applies to hospital inpatient/observation admit + discharge on same date.
        - Codes: 99234 (low), 99235 (moderate), 99236 (high).
        - Selection is based on MDM or total time when documented.

    Example:
        >>> get_hospital_admit_discharge_code(ComplexityLevel.MODERATE, None)
        ('99235', 'MDM', 'mdm_only')
    """
    if mdm is None and (time_minutes is None or time_minutes <= 0):
        return "", "", "missing_mdm_and_time"

    mdm_table = {
        ComplexityLevel.MINIMAL: "99234",
        ComplexityLevel.LOW: "99234",
        ComplexityLevel.MODERATE: "99235",
        ComplexityLevel.HIGH: "99236",
    }

    time_table = [
        (45, "99234"),
        (60, "99235"),
        (80, "99236"),
    ]

    mdm_code = mdm_table[mdm] if mdm else None
    time_code = None
    if time_minutes is not None and time_minutes > 0:
        for min_t, code in time_table:
            if time_minutes >= min_t:
                time_code = code

    if mdm_code and time_code:
        if int(time_code) > int(mdm_code):
            return time_code, "TIME", "time_higher"
        return mdm_code, "MDM", "mdm_higher_or_equal"

    if mdm_code:
        return mdm_code, "MDM", "mdm_only"
    if time_code:
        return time_code, "TIME", "time_only"

    return "", "", "no_applicable_code"


def get_hospital_discharge_code(
    discharge_time_level: DischargeTimeLevel | None,
) -> Tuple[str, str, str]:
    """Select CPT code for hospital inpatient/observation discharge services.

    Args:
        discharge_time_level: Discharge time category (leq_30 or gt_30), or None.

    Returns:
        Tuple of (code, source, note). Code is empty string if no applicable code.

    Notes:
        - Codes: 99238 (30 minutes or less), 99239 (more than 30 minutes).

    Example:
        >>> get_hospital_discharge_code(DischargeTimeLevel.GT_30)
        ('99239', 'TIME', 'time_only')
    """
    if discharge_time_level is None:
        return "", "", "missing_discharge_time"

    code_lookup = {
        DischargeTimeLevel.LEQ_30: "99238",
        DischargeTimeLevel.GT_30: "99239",
    }
    return code_lookup[discharge_time_level], "TIME", "time_only"


def main() -> int:
    """CLI entry point for CPT E/M code selection.

    Supported encounter types:
      - outpatient: Office/outpatient E/M visits (CPT 99202-99215)
      - preventive: Preventive medicine visits (CPT 99381-99397)
      - observation: Observation status encounters (CPT 99221-99223, 99231-99233)
      - inpatient: Inpatient consultations (CPT 99252-99255)
      - emergency: Emergency department visits (CPT 99281-99285)
      - hospital_admit_discharge: Same-day hospital admit/discharge (CPT 99234-99236)
      - hospital_discharge: Hospital discharge services (CPT 99238-99239)

    Args:
        None (arguments parsed from sys.argv).

    Returns:
        int: 0 on success.

    Raises:
        SystemExit: On argument parsing error.

    Example:
        python em_code_selection.py --encounter-type inpatient --patient-type new \\
            --mdm moderate
    """
    parser = argparse.ArgumentParser(description="Select CPT E/M codes for various encounter types")
    parser.add_argument(
        "--encounter-type",
        required=True,
        choices=[e.value for e in EncounterType],
        help=(
            "Type of encounter: outpatient, preventive, observation, inpatient, inpatient_initial, "
            "inpatient_subsequent, inpatient_discharge, emergency, hospital_admit_discharge, or "
            "hospital_discharge"
        ),
    )
    parser.add_argument(
        "--patient-type",
        required=True,
        choices=[p.value for p in PatientType],
        help="Patient classification: new or established",
    )
    parser.add_argument("--age", type=int, default=None, help="Patient age in years (required for preventive)")
    parser.add_argument(
        "--time-minutes",
        type=int,
        default=None,
        help="Total clinician time in minutes (for outpatient, observation, or inpatient)",
    )
    parser.add_argument(
        "--mdm",
        choices=[
            EmergencyMdmLevel.NONE.value,
            ComplexityLevel.MINIMAL.value,
            ComplexityLevel.LOW.value,
            ComplexityLevel.MODERATE.value,
            ComplexityLevel.HIGH.value,
        ],
        default=None,
        help=("Medical Decision Making level (for outpatient/observation/inpatient) or ED MDM " "level (use 'none' to select 99281)"),
    )
    parser.add_argument(
        "--discharge-time",
        choices=[t.value for t in DischargeTimeLevel],
        default=None,
        help="Discharge time category (leq_30 or gt_30) for hospital discharge services",
    )

    args = parser.parse_args()

    encounter_type = EncounterType(args.encounter_type)
    if encounter_type in {
        EncounterType.INPATIENT_INITIAL,
        EncounterType.INPATIENT_SUBSEQUENT,
    }:
        encounter_type = EncounterType.INPATIENT
    if encounter_type == EncounterType.INPATIENT_DISCHARGE:
        encounter_type = EncounterType.HOSPITAL_DISCHARGE
    patient_type = PatientType(args.patient_type)
    mdm_level = (
        ComplexityLevel(args.mdm)
        if args.mdm
        in {
            ComplexityLevel.MINIMAL.value,
            ComplexityLevel.LOW.value,
            ComplexityLevel.MODERATE.value,
            ComplexityLevel.HIGH.value,
        }
        else None
    )
    ed_mdm_level = EmergencyMdmLevel(args.mdm) if args.mdm else None
    discharge_time_level = DischargeTimeLevel(args.discharge_time) if args.discharge_time else None

    result = {"code": "", "source": "", "notes": ""}

    if encounter_type == EncounterType.PREVENTIVE:
        code, note = get_preventive_code(patient_type, args.age)
        result["code"] = code or ""
        result["source"] = "PREVENTIVE" if code else ""
        result["notes"] = note
    elif encounter_type == EncounterType.OUTPATIENT:
        code, source, note = select_outpatient_code(patient_type, mdm_level, args.time_minutes)
        result["code"] = code
        result["source"] = source
        result["notes"] = note
    elif encounter_type == EncounterType.OBSERVATION:
        code, source, note = get_observation_code(patient_type, mdm_level, args.time_minutes)
        result["code"] = code
        result["source"] = source
        result["notes"] = note
    elif encounter_type == EncounterType.INPATIENT:
        code, source, note = get_inpatient_code(patient_type, mdm_level, args.time_minutes)
        result["code"] = code
        result["source"] = source
        result["notes"] = note
    elif encounter_type == EncounterType.EMERGENCY:
        code, source, note = get_emergency_department_code(ed_mdm_level)
        result["code"] = code
        result["source"] = source
        result["notes"] = note
    elif encounter_type == EncounterType.HOSPITAL_ADMIT_DISCHARGE:
        code, source, note = get_hospital_admit_discharge_code(mdm_level, args.time_minutes)
        result["code"] = code
        result["source"] = source
        result["notes"] = note
    elif encounter_type == EncounterType.HOSPITAL_DISCHARGE:
        code, source, note = get_hospital_discharge_code(discharge_time_level)
        result["code"] = code
        result["source"] = source
        result["notes"] = note

    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
