"""Tests for association post-processing rules and CLI wiring."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture()
def association_module() -> ModuleType:
    """Loads the association_postprocess module for isolated tests.

    Args:
        None.

    Returns:
        ModuleType: Loaded association_postprocess module.

    Raises:
        RuntimeError: If the module spec or loader cannot be created.

    Assumptions:
        The script lives under the packaged skill resources.

    Example:
        >>> hasattr(association_module, "_filter_linked_diagnoses")
        True
    """
    module_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "skills_package"
        / "skills"
        / "profees-coding"
        / "icd-10-association"
        / "scripts"
        / "association_postprocess.py"
    )
    spec = importlib.util.spec_from_file_location("association_postprocess_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load association_postprocess module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestAssociationPostprocess:
    """Validates filtering, enrichment, cleanup, and CLI flow."""

    def test_filter_linked_diagnoses_cpt(self, association_module: ModuleType) -> None:
        """Filters CPT-linked diagnoses using the eligibility mapping.

        Args:
            association_module: Loaded association_postprocess module fixture.

        Returns:
            None.

        Raises:
            AssertionError: If filtering does not retain only eligible diagnoses.

        Assumptions:
            Only CPT items are filtered for CPT mappings.

        Example:
            >>> association_module._filter_linked_diagnoses([], {}, "CPT")
            []
        """
        items = [
            {
                "system": "CPT",
                "code": "1234",
                "linked_diagnosis": [{"code": "A00.1"}, {"code": "B00.1"}],
            },
            {
                "system": "HCPCS",
                "code": "H001",
                "linked_diagnosis": [{"code": "A00.1"}],
            },
        ]
        mapping = {"1234": {"A00.1"}}
        updated = association_module._filter_linked_diagnoses(items, mapping, "CPT")
        assert updated[0]["linked_diagnosis"] == [{"code": "A00.1"}]
        assert updated[1]["linked_diagnosis"] == [{"code": "A00.1"}]

    def test_populate_descriptions(self, association_module: ModuleType) -> None:
        """Populates procedure, E/M, and diagnosis descriptions.

        Args:
            association_module: Loaded association_postprocess module fixture.

        Returns:
            None.

        Raises:
            AssertionError: If description fields are not populated as expected.

        Assumptions:
            Description lookups are code-keyed dictionaries.

        Example:
            >>> state = {"procedure": [{"code": "1234"}]}
            >>> association_module._populate_descriptions(state, {"1234": "X"}, {}, {})
        """
        state = {
            "procedure": [{"code": "1234"}],
            "em": [{"code": "99213"}],
            "diagnosis": [{"code": "A00.1"}],
        }
        cpt_desc = {"1234": "Procedure desc", "99213": "E/M desc"}
        hcpcs_desc = {"H001": "HCPCS desc"}
        icd10_desc = {"A00.1": "Cholera"}
        association_module._populate_descriptions(state, cpt_desc, hcpcs_desc, icd10_desc)
        assert state["procedure"][0]["description"] == "Procedure desc"
        assert state["em"][0]["description"] == "E/M desc"
        assert state["diagnosis"][0]["description"] == "Cholera"

    def test_remove_unassociated_diagnoses_with_links(self, association_module: ModuleType) -> None:
        """Retains only diagnoses linked to charges when associations exist.

        Args:
            association_module: Loaded association_postprocess module fixture.

        Returns:
            None.

        Raises:
            AssertionError: If unrelated diagnoses are not removed.

        Assumptions:
            Linked diagnoses appear under `linked_diagnosis` on charges.

        Example:
            >>> state = {"procedure": [{"linked_diagnosis": [{"code": "A00.1"}]}]}
            >>> association_module._remove_unassociated_diagnoses(state)
        """
        state = {
            "procedure": [{"linked_diagnosis": [{"code": "A00.1"}]}],
            "em": [],
            "diagnosis": [{"code": "A00.1"}, {"code": "B00.1"}],
        }
        association_module._remove_unassociated_diagnoses(state)
        assert state["diagnosis"] == [{"code": "A00.1"}]

    def test_remove_unassociated_diagnoses_without_links(self, association_module: ModuleType) -> None:
        """Clears diagnoses when no charge links exist.

        Args:
            association_module: Loaded association_postprocess module fixture.

        Returns:
            None.

        Raises:
            AssertionError: If diagnoses are not cleared when no links exist.

        Assumptions:
            Empty associations mean no diagnoses should remain.

        Example:
            >>> state = {"diagnosis": [{"code": "A00.1"}]}
            >>> association_module._remove_unassociated_diagnoses(state)
        """
        state = {"procedure": [], "em": [], "diagnosis": [{"code": "A00.1"}]}
        association_module._remove_unassociated_diagnoses(state)
        assert state["diagnosis"] == []

    def test_main_end_to_end(
        self,
        association_module: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Runs the CLI pipeline with filtering, enrichment, and cleanup.

        Args:
            association_module: Loaded association_postprocess module fixture.
            tmp_path: Temporary directory provided by pytest for file fixtures.
            monkeypatch: Pytest monkeypatch for sys.argv.

        Returns:
            None.

        Raises:
            AssertionError: If the CLI output does not match expectations.

        Assumptions:
            File inputs and outputs are JSON encoded.

        Example:
            >>> association_module.main()
        """
        input_state = {
            "procedure": [
                {
                    "system": "CPT",
                    "code": "1234",
                    "linked_diagnosis": [{"code": "A00.1"}, {"code": "B00.1"}],
                }
            ],
            "em": [{"code": "99213", "linked_diagnosis": [{"code": "A00.1"}]}],
            "diagnosis": [{"code": "A00.1"}, {"code": "B00.1"}],
        }
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(input_state), encoding="utf-8")

        cpt_mapping_path = tmp_path / "cpt_map.json"
        cpt_mapping_path.write_text(json.dumps({"1234": ["A00.1"]}), encoding="utf-8")
        cpt_desc_path = tmp_path / "cpt_desc.json"
        cpt_desc_path.write_text(
            json.dumps({"1234": "Procedure desc", "99213": "E/M desc"}),
            encoding="utf-8",
        )
        hcpcs_desc_path = tmp_path / "hcpcs_desc.json"
        hcpcs_desc_path.write_text(json.dumps({}), encoding="utf-8")
        icd10_desc_path = tmp_path / "icd10_desc.json"
        icd10_desc_path.write_text(json.dumps({"A00.1": "Cholera"}), encoding="utf-8")

        output_path = tmp_path / "output.json"
        args = [
            "association_postprocess.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--cpt-filtering",
            "--populate-descriptions",
            "--remove-unassociated",
            "--cpt-icd-mapping-path",
            str(cpt_mapping_path),
            "--cpt-description-path",
            str(cpt_desc_path),
            "--hcpcs-description-path",
            str(hcpcs_desc_path),
            "--icd10-description-path",
            str(icd10_desc_path),
        ]
        monkeypatch.setattr(sys, "argv", args)
        association_module.main()

        output_state = json.loads(output_path.read_text(encoding="utf-8"))
        assert output_state["procedure"][0]["linked_diagnosis"] == [{"code": "A00.1"}]
        assert output_state["procedure"][0]["description"] == "Procedure desc"
        assert output_state["em"][0]["description"] == "E/M desc"
        assert output_state["diagnosis"][0]["code"] == "A00.1"
        assert output_state["diagnosis"][0]["description"] == "Cholera"
