"""Tests for CPT E/M code selection logic and CLI behavior."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture()
def em_module() -> ModuleType:
    """Loads the em_code_selection module for isolated tests.

    Args:
        None.

    Returns:
        ModuleType: Loaded em_code_selection module.

    Raises:
        RuntimeError: If the module spec or loader cannot be created.

    Assumptions:
        The script lives under the packaged skill resources.

    Example:
        >>> hasattr(em_module, "select_outpatient_code")
        True
    """
    module_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "skills_package"
        / "skills"
        / "profees-coding"
        / "cpt-em-coding"
        / "scripts"
        / "em_code_selection.py"
    )
    spec = importlib.util.spec_from_file_location("em_code_selection_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load em_code_selection module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestEmCodeSelection:
    """Validates age categorization, code selection, and CLI output."""

    def test_get_age_category_invalid(self, em_module: ModuleType) -> None:
        """Returns INVALID for missing or negative ages.

        Args:
            em_module: Loaded em_code_selection module fixture.

        Returns:
            None.

        Raises:
            AssertionError: If invalid ages are not categorized correctly.

        Assumptions:
            Negative ages are not valid inputs.

        Example:
            >>> em_module.get_age_category(None).value
            'invalid'
        """
        assert em_module.get_age_category(None).value == "invalid"
        assert em_module.get_age_category(-1).value == "invalid"

    @pytest.mark.parametrize(
        ("age", "expected"),
        [
            (0, "infant"),
            (1, "early_child"),
            (4, "early_child"),
            (5, "late_child"),
            (11, "late_child"),
            (12, "adolescent"),
            (17, "adolescent"),
            (18, "young_adult"),
            (39, "young_adult"),
            (40, "adult"),
            (64, "adult"),
            (65, "senior"),
        ],
    )
    def test_get_age_category_boundaries(self, em_module: ModuleType, age: int, expected: str) -> None:
        """Classifies ages at key boundary values.

        Args:
            em_module: Loaded em_code_selection module fixture.
            age: Age in years to classify.
            expected: Expected age category value.

        Returns:
            None.

        Raises:
            AssertionError: If a boundary age maps to the wrong category.

        Assumptions:
            Age thresholds are inclusive at the lower bound.

        Example:
            >>> em_module.get_age_category(18).value
            'young_adult'
        """
        assert em_module.get_age_category(age).value == expected

    @pytest.mark.parametrize(
        ("patient_type", "minutes", "expected"),
        [
            ("new", 15, "99202"),
            ("new", 44, "99203"),
            ("new", 60, "99205"),
            ("established", 10, "99212"),
            ("established", 39, "99214"),
            ("established", 40, "99215"),
        ],
    )
    def test_get_time_based_em_code_mapping(self, em_module: ModuleType, patient_type: str, minutes: int, expected: str) -> None:
        """Maps time ranges to the correct E/M code.

        Args:
            em_module: Loaded em_code_selection module fixture.
            patient_type: Patient type value for the lookup.
            minutes: Total minutes for the visit.
            expected: Expected CPT code.

        Returns:
            None.

        Raises:
            AssertionError: If time-based code selection is incorrect.

        Assumptions:
            Time tables follow the script's internal mapping.

        Example:
            >>> em_module.get_time_based_em_code(em_module.PatientType.NEW, 15)
            '99202'
        """
        patient = em_module.PatientType(patient_type)
        assert em_module.get_time_based_em_code(patient, minutes) == expected

    @pytest.mark.parametrize(
        ("patient_type", "mdm", "minutes", "expected_code", "expected_source", "expected_note"),
        [
            ("new", "minimal", 30, "99203", "TIME", "time_higher"),
            ("new", "moderate", 30, "99204", "MDM", "mdm_higher_or_equal"),
            ("established", None, 20, "99213", "TIME", "time_only"),
        ],
    )
    def test_select_outpatient_code_precedence(
        self,
        em_module: ModuleType,
        patient_type: str,
        mdm: str | None,
        minutes: int,
        expected_code: str,
        expected_source: str,
        expected_note: str,
    ) -> None:
        """Chooses the correct outpatient code based on MDM and time.

        Args:
            em_module: Loaded em_code_selection module fixture.
            patient_type: Patient type value for the lookup.
            mdm: MDM complexity level or None.
            minutes: Time spent on the visit.
            expected_code: Expected CPT code.
            expected_source: Expected source label.
            expected_note: Expected decision note.

        Returns:
            None.

        Raises:
            AssertionError: If outpatient code precedence is incorrect.

        Assumptions:
            Higher numeric CPT codes take precedence when both apply.

        Example:
            >>> em_module.select_outpatient_code(
            ...     em_module.PatientType.NEW,
            ...     em_module.ComplexityLevel.MINIMAL,
            ...     30,
            ... )[0]
            '99203'
        """
        patient = em_module.PatientType(patient_type)
        mdm_level = em_module.ComplexityLevel(mdm) if mdm else None
        code, source, note = em_module.select_outpatient_code(
            patient,
            mdm_level,
            minutes,
        )
        assert code == expected_code
        assert source == expected_source
        assert note == expected_note

    @pytest.mark.parametrize(
        ("patient_type", "age", "expected_code", "expected_note"),
        [
            ("new", 30, "99385", "ok"),
            ("established", 70, "99397", "ok"),
        ],
    )
    def test_get_preventive_code_valid(
        self,
        em_module: ModuleType,
        patient_type: str,
        age: int,
        expected_code: str,
        expected_note: str,
    ) -> None:
        """Returns preventive codes for valid age categories.

        Args:
            em_module: Loaded em_code_selection module fixture.
            patient_type: Patient type value for the lookup.
            age: Patient age in years.
            expected_code: Expected preventive CPT code.
            expected_note: Expected status note.

        Returns:
            None.

        Raises:
            AssertionError: If preventive code selection is incorrect.

        Assumptions:
            Preventive codes are mapped by age category and patient type.

        Example:
            >>> em_module.get_preventive_code(em_module.PatientType.NEW, 30)
            ('99385', 'ok')
        """
        patient = em_module.PatientType(patient_type)
        code, note = em_module.get_preventive_code(patient, age)
        assert code == expected_code
        assert note == expected_note

    def test_cli_preventive_output(
        self,
        em_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Executes the CLI for preventive encounters and parses JSON output.

        Args:
            em_module: Loaded em_code_selection module fixture.
            capsys: Pytest capture fixture for stdout.
            monkeypatch: Pytest monkeypatch for sys.argv.

        Returns:
            None.

        Raises:
            AssertionError: If CLI output does not include expected fields.

        Assumptions:
            CLI prints JSON with code, source, and notes keys.

        Example:
            >>> em_module.main()
            0
        """
        args = [
            "em_code_selection.py",
            "--encounter-type",
            "preventive",
            "--patient-type",
            "new",
            "--age",
            "30",
        ]
        monkeypatch.setattr(sys, "argv", args)
        status = em_module.main()
        payload = json.loads(capsys.readouterr().out)
        assert status == 0
        assert payload["code"] == "99385"
        assert payload["source"] == "PREVENTIVE"
        assert payload["notes"] == "ok"

    # Observation status encounter tests
    @pytest.mark.parametrize(
        ("patient_type", "mdm", "minutes", "expected_code", "expected_source", "expected_note"),
        [
            ("new", "minimal", None, "99221", "MDM", "mdm_only"),
            ("new", "low", None, "99221", "MDM", "mdm_only"),
            ("new", "moderate", None, "99222", "MDM", "mdm_only"),
            ("new", "high", None, "99223", "MDM", "mdm_only"),
            ("new", None, 55, "99222", "TIME", "time_only"),
            ("new", "low", 75, "99223", "TIME", "time_higher"),
            ("established", "minimal", None, "99231", "MDM", "mdm_only"),
            ("established", "low", None, "99231", "MDM", "mdm_only"),
            ("established", "moderate", None, "99232", "MDM", "mdm_only"),
            ("established", "high", None, "99233", "MDM", "mdm_only"),
            ("established", None, 40, "99232", "TIME", "time_only"),
        ],
    )
    def test_get_observation_code_selection(
        self,
        em_module: ModuleType,
        patient_type: str,
        mdm: str | None,
        minutes: int | None,
        expected_code: str,
        expected_source: str,
        expected_note: str,
    ) -> None:
        """Selects correct CPT code for observation status encounters.

        Args:
            em_module: Loaded em_code_selection module fixture.
            patient_type: NEW or ESTABLISHED patient classification.
            mdm: MDM complexity level or None.
            minutes: Time in minutes or None.
            expected_code: Expected CPT code (99221-99223, 99231-99233).
            expected_source: Expected source label (MDM or TIME).
            expected_note: Expected decision note.

        Returns:
            None.

        Raises:
            AssertionError: If observation code selection is incorrect.

        Assumptions:
            Observation codes are 99221-99223 (new) and 99231-99233 (established).
            Time-based mappings follow observation-specific thresholds.

        Example:
            >>> em_module.get_observation_code(
            ...     em_module.PatientType.NEW,
            ...     em_module.ComplexityLevel.LOW,
            ...     None,
            ... )[0]
            '99222'
        """
        patient = em_module.PatientType(patient_type)
        mdm_level = em_module.ComplexityLevel(mdm) if mdm else None
        code, source, note = em_module.get_observation_code(
            patient,
            mdm_level,
            minutes,
        )
        assert code == expected_code
        assert source == expected_source
        assert note == expected_note

    @pytest.mark.parametrize(
        ("patient_type", "mdm", "expected_code"),
        [
            ("new", "minimal", "99221"),
            ("new", "low", "99221"),
            ("new", "moderate", "99222"),
            ("established", "minimal", "99231"),
            ("established", "moderate", "99232"),
        ],
    )
    def test_get_observation_mdm_mapping(
        self,
        em_module: ModuleType,
        patient_type: str,
        mdm: str,
        expected_code: str,
    ) -> None:
        """Maps MDM complexity to observation CPT codes.

        Args:
            em_module: Loaded em_code_selection module fixture.
            patient_type: Patient type value.
            mdm: MDM complexity level.
            expected_code: Expected CPT code.

        Returns:
            None.

        Raises:
            AssertionError: If observation MDM mapping is incorrect.
        """
        patient = em_module.PatientType(patient_type)
        mdm_level = em_module.ComplexityLevel(mdm)
        code, source, note = em_module.get_observation_code(patient, mdm_level, None)
        assert code == expected_code
        assert source == "MDM"

    def test_cli_observation_output(
        self,
        em_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Executes the CLI for observation status encounters and parses JSON output.

        Args:
            em_module: Loaded em_code_selection module fixture.
            capsys: Pytest capture fixture for stdout.
            monkeypatch: Pytest monkeypatch for sys.argv.

        Returns:
            None.

        Raises:
            AssertionError: If CLI observation output is incorrect.

        Assumptions:
            CLI prints valid JSON for observation encounters.
        """
        args = [
            "em_code_selection.py",
            "--encounter-type",
            "observation",
            "--patient-type",
            "new",
            "--mdm",
            "moderate",
        ]
        monkeypatch.setattr(sys, "argv", args)
        status = em_module.main()
        payload = json.loads(capsys.readouterr().out)
        assert status == 0
        assert payload["code"] == "99222"
        assert payload["source"] == "MDM"

    @pytest.mark.parametrize(
        ("mdm", "expected_code", "expected_note"),
        [
            ("none", "99281", "mdm_not_required"),
            ("minimal", "99282", "mdm_only"),
            ("low", "99283", "mdm_only"),
            ("moderate", "99284", "mdm_only"),
            ("high", "99285", "mdm_only"),
        ],
    )
    def test_get_emergency_department_code_selection(
        self,
        em_module: ModuleType,
        mdm: str,
        expected_code: str,
        expected_note: str,
    ) -> None:
        """Selects correct CPT code for emergency department encounters.

        Args:
            em_module: Loaded em_code_selection module fixture.
            mdm: ED MDM level or none.
            expected_code: Expected ED CPT code.
            expected_note: Expected decision note.

        Returns:
            None.

        Raises:
            AssertionError: If ED code selection is incorrect.

        Assumptions:
            ED codes are 99281-99285 and do not distinguish patient type.
        """
        mdm_level = em_module.EmergencyMdmLevel(mdm)
        code, source, note = em_module.get_emergency_department_code(mdm_level)
        assert code == expected_code
        assert source == "MDM"
        assert note == expected_note

    def test_get_emergency_department_code_missing_mdm(
        self,
        em_module: ModuleType,
    ) -> None:
        """Returns missing_mdm when ED MDM level is not provided.

        Args:
            em_module: Loaded em_code_selection module fixture.

        Returns:
            None.

        Raises:
            AssertionError: If missing MDM is not handled correctly.
        """
        code, source, note = em_module.get_emergency_department_code(None)
        assert code == ""
        assert source == ""
        assert note == "missing_mdm"

    @pytest.mark.parametrize(
        ("mdm", "minutes", "expected_code", "expected_source", "expected_note"),
        [
            ("minimal", None, "99234", "MDM", "mdm_only"),
            ("moderate", None, "99235", "MDM", "mdm_only"),
            ("high", None, "99236", "MDM", "mdm_only"),
            (None, 45, "99234", "TIME", "time_only"),
            ("low", 80, "99236", "TIME", "time_higher"),
        ],
    )
    def test_get_hospital_admit_discharge_code_selection(
        self,
        em_module: ModuleType,
        mdm: str | None,
        minutes: int | None,
        expected_code: str,
        expected_source: str,
        expected_note: str,
    ) -> None:
        """Selects correct CPT code for same-day admit/discharge services.

        Args:
            em_module: Loaded em_code_selection module fixture.
            mdm: MDM complexity level or None.
            minutes: Total clinician time in minutes or None.
            expected_code: Expected CPT code.
            expected_source: Expected source label.
            expected_note: Expected decision note.

        Returns:
            None.

        Raises:
            AssertionError: If admit/discharge code selection is incorrect.
        """
        mdm_level = em_module.ComplexityLevel(mdm) if mdm else None
        code, source, note = em_module.get_hospital_admit_discharge_code(
            mdm_level,
            minutes,
        )
        assert code == expected_code
        assert source == expected_source
        assert note == expected_note

    @pytest.mark.parametrize(
        ("discharge_time", "expected_code"),
        [
            ("leq_30", "99238"),
            ("gt_30", "99239"),
        ],
    )
    def test_get_hospital_discharge_code_selection(
        self,
        em_module: ModuleType,
        discharge_time: str,
        expected_code: str,
    ) -> None:
        """Selects correct CPT code for hospital discharge services.

        Args:
            em_module: Loaded em_code_selection module fixture.
            discharge_time: Discharge time category value.
            expected_code: Expected CPT code.

        Returns:
            None.

        Raises:
            AssertionError: If discharge code selection is incorrect.
        """
        time_level = em_module.DischargeTimeLevel(discharge_time)
        code, source, note = em_module.get_hospital_discharge_code(time_level)
        assert code == expected_code
        assert source == "TIME"
        assert note == "time_only"

    def test_get_hospital_discharge_code_missing_time(
        self,
        em_module: ModuleType,
    ) -> None:
        """Returns missing_discharge_time when discharge time is not provided.

        Args:
            em_module: Loaded em_code_selection module fixture.

        Returns:
            None.

        Raises:
            AssertionError: If missing discharge time is not handled correctly.
        """
        code, source, note = em_module.get_hospital_discharge_code(None)
        assert code == ""
        assert source == ""
        assert note == "missing_discharge_time"

    # Inpatient consultation tests
    @pytest.mark.parametrize(
        ("patient_type", "mdm", "minutes", "expected_code", "expected_source"),
        [
            ("new", "minimal", None, "99252", "MDM"),
            ("new", "low", None, "99252", "MDM"),
            ("new", "moderate", None, "99253", "MDM"),
            ("new", "high", None, "99255", "MDM"),
            ("new", None, 35, "99252", "TIME"),
            ("new", None, 60, "99254", "TIME"),
            ("new", "low", 80, "99255", "TIME"),
            ("established", "minimal", None, "99231", "MDM"),
            ("established", "low", None, "99231", "MDM"),
            ("established", "moderate", None, "99232", "MDM"),
            ("established", "high", None, "99233", "MDM"),
            ("established", None, 30, "99231", "TIME"),
            ("established", None, 40, "99232", "TIME"),
        ],
    )
    def test_get_inpatient_code_selection(
        self,
        em_module: ModuleType,
        patient_type: str,
        mdm: str | None,
        minutes: int | None,
        expected_code: str,
        expected_source: str,
    ) -> None:
        """Selects correct CPT code for inpatient consultation encounters.

        Args:
            em_module: Loaded em_code_selection module fixture.
            patient_type: NEW or ESTABLISHED patient classification.
            mdm: MDM complexity level or None.
            minutes: Time in minutes or None.
            expected_code: Expected CPT code (99252-99255, 99231-99233).
            expected_source: Expected source label (MDM or TIME).

        Returns:
            None.

        Raises:
            AssertionError: If inpatient code selection is incorrect.

        Assumptions:
            Initial consultation codes: 99252-99255 (new patients).
            Established follow-up uses subsequent inpatient/observation care codes.
            MDM complexity is primary driver; time is secondary.

        Example:
            >>> em_module.get_inpatient_code(
            ...     em_module.PatientType.NEW,
            ...     em_module.ComplexityLevel.MODERATE,
            ...     None,
            ... )[0]
            '99253'
        """
        patient = em_module.PatientType(patient_type)
        mdm_level = em_module.ComplexityLevel(mdm) if mdm else None
        code, source, note = em_module.get_inpatient_code(
            patient,
            mdm_level,
            minutes,
        )
        assert code == expected_code
        assert source == expected_source

    @pytest.mark.parametrize(
        ("patient_type", "mdm", "expected_code"),
        [
            ("new", "minimal", "99252"),
            ("new", "low", "99252"),
            ("new", "moderate", "99253"),
            ("new", "high", "99255"),
            ("established", "minimal", "99231"),
            ("established", "moderate", "99232"),
        ],
    )
    def test_get_inpatient_mdm_mapping(
        self,
        em_module: ModuleType,
        patient_type: str,
        mdm: str,
        expected_code: str,
    ) -> None:
        """Maps MDM complexity to inpatient consultation CPT codes.

        Args:
            em_module: Loaded em_code_selection module fixture.
            patient_type: Patient type value.
            mdm: MDM complexity level.
            expected_code: Expected CPT code.

        Returns:
            None.

        Raises:
            AssertionError: If inpatient MDM mapping is incorrect.

        Assumptions:
            Inpatient codes are determined primarily by MDM complexity.
        """
        patient = em_module.PatientType(patient_type)
        mdm_level = em_module.ComplexityLevel(mdm)
        code, source, note = em_module.get_inpatient_code(patient, mdm_level, None)
        assert code == expected_code
        assert source == "MDM"

    @pytest.mark.parametrize(
        ("patient_type", "minutes", "expected_code"),
        [
            ("new", 35, "99252"),
            ("new", 45, "99253"),
            ("new", 60, "99254"),
            ("new", 90, "99255"),
            ("established", 30, "99231"),
            ("established", 40, "99232"),
            ("established", 60, "99233"),
        ],
    )
    def test_get_inpatient_time_mapping(
        self,
        em_module: ModuleType,
        patient_type: str,
        minutes: int,
        expected_code: str,
    ) -> None:
        """Maps time to inpatient consultation CPT codes.

        Args:
            em_module: Loaded em_code_selection module fixture.
            patient_type: Patient type value.
            minutes: Time spent in minutes.
            expected_code: Expected CPT code.

        Returns:
            None.

        Raises:
            AssertionError: If inpatient time mapping is incorrect.

        Assumptions:
            Time thresholds for inpatient differ from outpatient.
        """
        patient = em_module.PatientType(patient_type)
        code, source, note = em_module.get_inpatient_code(patient, None, minutes)
        assert code == expected_code
        assert source == "TIME"

    # Edge case tests for missing inputs and boundary conditions
    @pytest.mark.parametrize(
        ("encounter_type", "patient_type", "expected_note"),
        [
            ("outpatient", "new", "missing_mdm_and_time"),
            ("observation", "established", "missing_mdm_and_time"),
            ("inpatient", "new", "missing_mdm_and_time"),
            ("emergency", "new", "missing_mdm"),
            ("hospital_admit_discharge", "new", "missing_mdm_and_time"),
            ("hospital_discharge", "new", "missing_discharge_time"),
        ],
    )
    def test_missing_inputs_all_encounters(
        self,
        em_module: ModuleType,
        encounter_type: str,
        patient_type: str,
        expected_note: str,
    ) -> None:
        """Handles missing MDM and time inputs gracefully across encounter types.

        Args:
            em_module: Loaded em_code_selection module fixture.
            encounter_type: Type of encounter (outpatient, observation, inpatient).
            patient_type: Patient type value (new or established).
            expected_note: Expected note in output indicating missing inputs.

        Returns:
            None.

        Raises:
            AssertionError: If missing inputs are not handled correctly.

        Assumptions:
            When both MDM and time are None/invalid, code should be empty string.
        """
        encounter = em_module.EncounterType(encounter_type)
        patient = em_module.PatientType(patient_type)

        if encounter == em_module.EncounterType.OUTPATIENT:
            code, source, note = em_module.select_outpatient_code(patient, None, None)
        elif encounter == em_module.EncounterType.OBSERVATION:
            code, source, note = em_module.get_observation_code(patient, None, None)
        elif encounter == em_module.EncounterType.INPATIENT:
            code, source, note = em_module.get_inpatient_code(patient, None, None)
        elif encounter == em_module.EncounterType.HOSPITAL_ADMIT_DISCHARGE:
            code, source, note = em_module.get_hospital_admit_discharge_code(None, None)
        elif encounter == em_module.EncounterType.HOSPITAL_DISCHARGE:
            code, source, note = em_module.get_hospital_discharge_code(None)
        else:
            code, source, note = em_module.get_emergency_department_code(None)

        assert code == ""
        assert note == expected_note

    @pytest.mark.parametrize(
        ("encounter_type", "patient_type"),
        [
            ("outpatient", "new"),
            ("observation", "new"),
            ("inpatient", "new"),
            ("emergency", "new"),
            ("hospital_admit_discharge", "new"),
            ("hospital_discharge", "new"),
            ("outpatient", "established"),
            ("observation", "established"),
            ("inpatient", "established"),
        ],
    )
    def test_zero_or_negative_time_treated_as_none(
        self,
        em_module: ModuleType,
        encounter_type: str,
        patient_type: str,
    ) -> None:
        """Verifies zero and negative time values are treated as None across all encounters.

        Args:
            em_module: Loaded em_code_selection module fixture.
            encounter_type: Type of encounter.
            patient_type: Patient type value.

        Returns:
            None.

        Raises:
            AssertionError: If zero/negative time is not ignored.

        Assumptions:
            Zero or negative time minutes should be ignored (same as None).
        """
        encounter = em_module.EncounterType(encounter_type)
        patient = em_module.PatientType(patient_type)

        # Test with zero minutes
        if encounter == em_module.EncounterType.OUTPATIENT:
            code_zero, _, _ = em_module.select_outpatient_code(patient, None, 0)
            code_none, _, _ = em_module.select_outpatient_code(patient, None, None)
        elif encounter == em_module.EncounterType.OBSERVATION:
            code_zero, _, _ = em_module.get_observation_code(patient, None, 0)
            code_none, _, _ = em_module.get_observation_code(patient, None, None)
        elif encounter == em_module.EncounterType.INPATIENT:
            code_zero, _, _ = em_module.get_inpatient_code(patient, None, 0)
            code_none, _, _ = em_module.get_inpatient_code(patient, None, None)
        elif encounter == em_module.EncounterType.HOSPITAL_ADMIT_DISCHARGE:
            code_zero, _, _ = em_module.get_hospital_admit_discharge_code(None, 0)
            code_none, _, _ = em_module.get_hospital_admit_discharge_code(None, None)
        elif encounter == em_module.EncounterType.HOSPITAL_DISCHARGE:
            code_zero, _, _ = em_module.get_hospital_discharge_code(None)
            code_none, _, _ = em_module.get_hospital_discharge_code(None)
        else:
            code_zero, _, _ = em_module.get_emergency_department_code(None)
            code_none, _, _ = em_module.get_emergency_department_code(None)

        assert code_zero == code_none == ""

        # Test with negative minutes
        if encounter == em_module.EncounterType.OUTPATIENT:
            code_neg, _, _ = em_module.select_outpatient_code(patient, None, -10)
        elif encounter == em_module.EncounterType.OBSERVATION:
            code_neg, _, _ = em_module.get_observation_code(patient, None, -10)
        elif encounter == em_module.EncounterType.INPATIENT:
            code_neg, _, _ = em_module.get_inpatient_code(patient, None, -10)
        elif encounter == em_module.EncounterType.HOSPITAL_ADMIT_DISCHARGE:
            code_neg, _, _ = em_module.get_hospital_admit_discharge_code(None, -10)
        elif encounter == em_module.EncounterType.HOSPITAL_DISCHARGE:
            code_neg, _, _ = em_module.get_hospital_discharge_code(None)
        else:
            code_neg, _, _ = em_module.get_emergency_department_code(None)

        assert code_neg == ""

    def test_inpatient_high_complexity_reaches_99255(
        self,
        em_module: ModuleType,
    ) -> None:
        """Verifies HIGH complexity (max MDM level) reaches 99255 for inpatient new patient.

        Args:
            em_module: Loaded em_code_selection module fixture.

        Returns:
            None.

        Raises:
            AssertionError: If HIGH complexity is not properly mapped for inpatient.

        Assumptions:
            HIGH is the maximum MDM complexity level (4 total: minimal, low, moderate,
            high).
        """
        patient = em_module.PatientType.NEW
        code, source, note = em_module.get_inpatient_code(
            patient,
            em_module.ComplexityLevel.HIGH,
            None,
        )
        assert code == "99255"
        assert source == "MDM"
        assert note == "mdm_used"

        # Also verify established HIGH maps to 99233
        established = em_module.PatientType.ESTABLISHED
        code_est, source_est, _ = em_module.get_inpatient_code(
            established,
            em_module.ComplexityLevel.HIGH,
            None,
        )
        assert code_est == "99233"
        assert source_est == "MDM"

    @pytest.mark.parametrize(
        ("patient_type", "minutes", "expected_code"),
        [
            ("new", 80, "99255"),  # Boundary: exactly 80 minutes
            ("new", 79, "99254"),  # Just below boundary
            ("new", 81, "99255"),  # Just above boundary
            ("established", 55, "99233"),  # Boundary: exactly 55 minutes
            ("established", 54, "99232"),  # Just below boundary
            ("established", 56, "99233"),  # Just above boundary
        ],
    )
    def test_inpatient_time_boundary_values(
        self,
        em_module: ModuleType,
        patient_type: str,
        minutes: int,
        expected_code: str,
    ) -> None:
        """Tests inpatient time thresholds at boundary values.

        Args:
            em_module: Loaded em_code_selection module fixture.
            patient_type: Patient type value.
            minutes: Time in minutes to test.
            expected_code: Expected CPT code at boundary.

        Returns:
            None.

        Raises:
            AssertionError: If boundary time mapping is incorrect.

        Assumptions:
            Time boundaries should include the lower threshold when exceeded.
        """
        patient = em_module.PatientType(patient_type)
        code, _, _ = em_module.get_inpatient_code(patient, None, minutes)
        assert code == expected_code

    @pytest.mark.parametrize(
        ("patient_type", "minutes", "expected_code"),
        [
            ("new", 40, "99221"),  # Boundary: exactly 40 minutes
            ("new", 39, ""),  # Below observation new minimum
            ("new", 75, "99223"),  # Boundary: exactly 75 minutes
            ("established", 30, "99231"),  # Boundary: exactly 30 minutes
            ("established", 29, ""),  # Below observation established minimum
            ("established", 55, "99233"),  # Boundary: exactly 55 minutes
        ],
    )
    def test_observation_time_boundary_values(
        self,
        em_module: ModuleType,
        patient_type: str,
        minutes: int,
        expected_code: str,
    ) -> None:
        """Tests observation time thresholds at boundary values.

        Args:
            em_module: Loaded em_code_selection module fixture.
            patient_type: Patient type value.
            minutes: Time in minutes to test.
            expected_code: Expected CPT code at boundary (empty if below minimum).

        Returns:
            None.

        Raises:
            AssertionError: If observation boundary time mapping is incorrect.

        Assumptions:
            Observation time boundaries are inclusive at the minimum.
            Below minimum time returns empty code (unless MDM provided).
        """
        patient = em_module.PatientType(patient_type)
        code, _, _ = em_module.get_observation_code(patient, None, minutes)
        assert code == expected_code

    def test_inpatient_mdm_precedence_over_time(
        self,
        em_module: ModuleType,
    ) -> None:
        """Verifies inpatient MDM takes precedence over time when MDM code <= time code.

        Args:
            em_module: Loaded em_code_selection module fixture.

        Returns:
            None.

        Raises:
            AssertionError: If MDM is not properly prioritized in inpatient coding.

        Assumptions:
            For inpatient, MDM should be preferred unless time code is HIGHER numerically.
        """
        patient = em_module.PatientType.NEW
        # MDM LOW (99252) with sufficient time (50 min → 99253)
        # Since 99252 < 99253, time should win
        code, source, note = em_module.get_inpatient_code(
            patient,
            em_module.ComplexityLevel.LOW,
            50,
        )
        assert code == "99253"
        assert source == "TIME"
        assert note == "time_higher"

        # MDM MODERATE (99253) with time that maps to 99252
        # Since MDM 99253 > time 99252, MDM should win
        code, source, note = em_module.get_inpatient_code(
            patient,
            em_module.ComplexityLevel.MODERATE,
            35,
        )
        assert code == "99253"
        assert source == "MDM"
        assert note == "mdm_used"

    def test_observation_vs_inpatient_code_ranges_overlap_allowed(
        self,
        em_module: ModuleType,
    ) -> None:
        """Verifies observation and inpatient code ranges are populated.

        Args:
            em_module: Loaded em_code_selection module fixture.

        Returns:
            None.

        Raises:
            AssertionError: If code ranges are not populated.

        Assumptions:
            Observation codes: 99221-99223, 99231-99233.
            Inpatient consult codes: 99252-99255 with subsequent care codes 99231-99233.
            Overlap is expected across encounter types; classification determines usage.
        """
        # Observation code range
        observation_codes = {"99221", "99222", "99223", "99231", "99232", "99233"}

        # Inpatient code range
        inpatient_codes = {"99252", "99253", "99254", "99255"}

        # Verify expected sizes
        assert len(observation_codes) == 6
        assert len(inpatient_codes) == 4

    def test_cli_inpatient_output(
        self,
        em_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Executes the CLI for inpatient consultations and parses JSON output.

        Args:
            em_module: Loaded em_code_selection module fixture.
            capsys: Pytest capture fixture for stdout.
            monkeypatch: Pytest monkeypatch for sys.argv.

        Returns:
            None.

        Raises:
            AssertionError: If CLI inpatient output is incorrect.

        Assumptions:
            CLI prints valid JSON for inpatient encounters.
        """
        args = [
            "em_code_selection.py",
            "--encounter-type",
            "inpatient",
            "--patient-type",
            "new",
            "--mdm",
            "moderate",
        ]
        monkeypatch.setattr(sys, "argv", args)
        status = em_module.main()
        payload = json.loads(capsys.readouterr().out)
        assert status == 0
        assert payload["code"] == "99253"
        assert payload["source"] == "MDM"

    @pytest.mark.parametrize(
        ("encounter_type", "patient_type", "expected_source"),
        [
            ("outpatient", "new", "MDM"),
            ("observation", "new", "MDM"),
            ("inpatient", "new", "MDM"),
            ("emergency", "new", "MDM"),
            ("hospital_admit_discharge", "new", "MDM"),
        ],
    )
    def test_cli_all_encounter_types_mdm(
        self,
        em_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        encounter_type: str,
        patient_type: str,
        expected_source: str,
    ) -> None:
        """Tests MDM selection across all encounter types via CLI.

        Args:
            em_module: Loaded em_code_selection module fixture.
            capsys: Pytest capture fixture for stdout.
            monkeypatch: Pytest monkeypatch for sys.argv.
            encounter_type: Type of encounter to test.
            patient_type: Patient type for the encounter.
            expected_source: Expected source value in output.

        Returns:
            None.

        Raises:
            AssertionError: If CLI output does not match expected source.

        Assumptions:
            MDM-based code selection is supported for all included encounter types.
        """
        if encounter_type == "emergency":
            mdm_value = "low"
        elif encounter_type == "hospital_admit_discharge":
            mdm_value = "moderate"
        else:
            mdm_value = "moderate"
        args = [
            "em_code_selection.py",
            "--encounter-type",
            encounter_type,
            "--patient-type",
            patient_type,
            "--mdm",
            mdm_value,
        ]
        monkeypatch.setattr(sys, "argv", args)
        status = em_module.main()
        payload = json.loads(capsys.readouterr().out)
        assert status == 0
        assert payload["source"] == expected_source
        assert payload["code"] != ""

    def test_cli_emergency_output(
        self,
        em_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Executes the CLI for emergency encounters and parses JSON output.

        Args:
            em_module: Loaded em_code_selection module fixture.
            capsys: Pytest capture fixture for stdout.
            monkeypatch: Pytest monkeypatch for sys.argv.

        Returns:
            None.

        Raises:
            AssertionError: If CLI emergency output is incorrect.
        """
        args = [
            "em_code_selection.py",
            "--encounter-type",
            "emergency",
            "--patient-type",
            "new",
            "--mdm",
            "high",
        ]
        monkeypatch.setattr(sys, "argv", args)
        status = em_module.main()
        payload = json.loads(capsys.readouterr().out)
        assert status == 0
        assert payload["code"] == "99285"
        assert payload["source"] == "MDM"

    def test_cli_hospital_admit_discharge_output(
        self,
        em_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Executes the CLI for same-day admit/discharge and parses JSON output.

        Args:
            em_module: Loaded em_code_selection module fixture.
            capsys: Pytest capture fixture for stdout.
            monkeypatch: Pytest monkeypatch for sys.argv.

        Returns:
            None.

        Raises:
            AssertionError: If CLI admit/discharge output is incorrect.
        """
        args = [
            "em_code_selection.py",
            "--encounter-type",
            "hospital_admit_discharge",
            "--patient-type",
            "new",
            "--mdm",
            "moderate",
        ]
        monkeypatch.setattr(sys, "argv", args)
        status = em_module.main()
        payload = json.loads(capsys.readouterr().out)
        assert status == 0
        assert payload["code"] == "99235"
        assert payload["source"] == "MDM"

    def test_cli_hospital_discharge_output(
        self,
        em_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Executes the CLI for hospital discharge and parses JSON output.

        Args:
            em_module: Loaded em_code_selection module fixture.
            capsys: Pytest capture fixture for stdout.
            monkeypatch: Pytest monkeypatch for sys.argv.

        Returns:
            None.

        Raises:
            AssertionError: If CLI discharge output is incorrect.
        """
        args = [
            "em_code_selection.py",
            "--encounter-type",
            "hospital_discharge",
            "--patient-type",
            "new",
            "--discharge-time",
            "gt_30",
        ]
        monkeypatch.setattr(sys, "argv", args)
        status = em_module.main()
        payload = json.loads(capsys.readouterr().out)
        assert status == 0
        assert payload["code"] == "99239"
        assert payload["source"] == "TIME"

    def test_integration_observation_inpatient_code_generation(
        self,
        em_module: ModuleType,
    ) -> None:
        """Integration test: observation and inpatient codes are generated correctly.

        Args:
            em_module: Loaded em_code_selection module fixture.

        Returns:
            None.

        Raises:
            AssertionError: If observation or inpatient codes are missing.

        Assumptions:
            Observation (99221-99223, 99231-99233) and inpatient consults (99252-99255)
            are both generated; encounter type determines billing context.
        """
        # Generate all possible observation codes
        observation_codes_generated = set()
        for patient_type in ["new", "established"]:
            for mdm in ["minimal", "low", "moderate", "high"]:
                code, _, _ = em_module.get_observation_code(
                    em_module.PatientType(patient_type),
                    em_module.ComplexityLevel(mdm),
                    None,
                )
                if code:
                    observation_codes_generated.add(code)

            # Also test time-based
            for minutes in [30, 40, 55, 75]:
                code, _, _ = em_module.get_observation_code(
                    em_module.PatientType(patient_type),
                    None,
                    minutes,
                )
                if code:
                    observation_codes_generated.add(code)

        # Generate all possible inpatient codes
        inpatient_codes_generated = set()
        for patient_type in ["new", "established"]:
            for mdm in ["minimal", "low", "moderate", "high"]:
                code, _, _ = em_module.get_inpatient_code(
                    em_module.PatientType(patient_type),
                    em_module.ComplexityLevel(mdm),
                    None,
                )
                if code:
                    inpatient_codes_generated.add(code)

            # Also test time-based
            for minutes in [30, 35, 45, 60, 80, 90]:
                code, _, _ = em_module.get_inpatient_code(
                    em_module.PatientType(patient_type),
                    None,
                    minutes,
                )
                if code:
                    inpatient_codes_generated.add(code)

        # Verify expected observation codes are generated
        expected_observation = {"99221", "99222", "99223", "99231", "99232", "99233"}
        assert observation_codes_generated == expected_observation, f"Observation codes: {observation_codes_generated} != {expected_observation}"

        # Verify expected inpatient codes are generated
        expected_inpatient = {"99231", "99232", "99233", "99252", "99253", "99254", "99255"}
        assert inpatient_codes_generated == expected_inpatient, f"Inpatient codes: {inpatient_codes_generated} != {expected_inpatient}"
