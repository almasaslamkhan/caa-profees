import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "src" / "skills_package" / "skills" / "profees-coding"
AMBULATORY_SKILLS_ROOT = ROOT / "src" / "profees_ambulatory_coding_skills" / "skills"

ROOT_SKILL = SKILLS_ROOT / "SKILL.md"
MEDICAL_CODING_SKILL = SKILLS_ROOT / "profees-medical-coding" / "SKILL.md"

EXPECTED_SKILL_NAMES = {
    "cpt-em-coding": SKILLS_ROOT / "cpt-em-coding" / "SKILL.md",
    "cpt-procedures-coding-skill": SKILLS_ROOT / "cpt-procedures-coding-skill" / "SKILL.md",
    "hcpcs-coding": SKILLS_ROOT / "hcpcs-coding" / "SKILL.md",
    "icd-10-association": SKILLS_ROOT / "icd-10-association" / "SKILL.md",
    "modifiers-coding": SKILLS_ROOT / "modifiers-coding" / "SKILL.md",
    "profees-medical-coding": MEDICAL_CODING_SKILL,
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_root_skill_routes_coding_summary_to_mcp_tool() -> None:
    text = _text(ROOT_SKILL)

    assert "name: profees-medical-coding" in text
    assert "toolkit_render_coding_summary" in text
    assert "call the tool with no arguments" in text


def test_package_config_publishes_only_ambulatory_skill_package() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    package_find = project["tool"]["setuptools"]["packages"]["find"]
    assert package_find["where"] == ["src"]
    assert package_find["include"] == ["profees_ambulatory_coding_skills*"]
    assert package_find["exclude"] == ["skills_package", "skills_package.*"]

    package_data = project["tool"]["setuptools"]["package-data"]
    assert package_data == {"profees_ambulatory_coding_skills": ["skills/**"]}


def test_legacy_skill_source_is_retained_but_not_declared_for_packaging() -> None:
    assert (SKILLS_ROOT / "SKILL.md").is_file()


def test_lean_ambulatory_skill_package_contains_only_canvas_resources() -> None:
    package_init = ROOT / "src" / "profees_ambulatory_coding_skills" / "__init__.py"
    skill_path = AMBULATORY_SKILLS_ROOT / "ambulatory-coding" / "SKILL.md"
    schema_path = AMBULATORY_SKILLS_ROOT / "ambulatory-coding" / "schemas" / "ambulatory_coding_canvas.json"

    assert package_init.is_file()
    assert skill_path.is_file()
    assert schema_path.is_file()

    # The skill is prose, so line wrapping must not make its contract brittle.
    text = " ".join(_text(skill_path).split())
    assert "name: ambulatory-coding" in text
    assert "mcpServers:" in text
    assert "name: profees_coding" in text
    assert any(
        endpoint in text
        for endpoint in (
            "endpoint: http://caa-profees-coding-mcp.caa-profees-coding-mcp.svc.cluster.local:80/mcp",
            "endpoint: http://127.0.0.1:9000/mcp",
        )
    )
    assert "auth: none" in text
    assert "- render_coding_summary" in text
    assert "- recode_profees_coding" in text
    assert "- recode_profees_coding_ehrm" in text
    assert "`profees_coding_render_coding_summary`" in text
    assert text.count("eventType: a2ui.userAction") == 9
    assert "actionName: system.updateApplicationContext" in text
    assert "applicationContextGroup: patientassistant/ambulatorycoding" in text
    assert "applicationContextGroup: patientchart/documentation" in text
    assert "applicationContextGroup: codingAgentEvents" in text
    assert text.count("semanticObject: AmbulatoryCoding") == 6
    assert "event: custom.SHOW_AMBULATORY_CODING" in text
    assert "event: codingAgentRecodeCustomEvent" in text
    assert "event: custom.codingAgentRecodeCustomEvent" in text
    assert "event: custom.codingAgentRecodeCustomEHRMEvent" in text
    assert "system.updateApplicationContextCommand" not in text
    assert "payloadProperty: patientId" in text
    assert "toolArgument: patient_id" in text
    assert "payloadProperty: encounterId" in text
    assert "toolArgument: encounter_id" in text
    assert "payloadProperty: context.properties.description" in text
    assert "toolArgument: description" in text
    assert "payloadProperty: requestId" in text
    assert "toolArgument: request_id" in text
    assert "payloadProperty: docrefId" in text
    assert "payloadProperty: document_ref_id" in text
    assert "toolArgument: document_ref_id" in text
    assert "Use this skill only when the user explicitly asks" in text
    assert "Do not use this skill for medication reconciliation" in text
    assert "`includeTools` names the raw MCP tools, `render_coding_summary` and" in text
    assert "`recode_profees_coding`" in text
    assert "server-prefixed tool" in text
    assert "let the tool provide the notification title" in text
    assert "The host persona must provide the `profees_coding` MCP server endpoint" in text
    assert "do not use `renderedType` as a completion signal" in text
    assert "Never call `recode_profees_coding` for a conversational request" in text
    assert "do not render a card or canvas" in text
    assert "Recode is not supported through Chat" in text
    assert "click Recode to generate updated suggestions" in text
    assert "do not repeat, enumerate, or infer any CPT, E/M, HCPCS, ICD-10, or" in text
    assert "notificationTitle" in text
    assert "Review the suggested codes in the Generated codes card." in text
    assert "Review the suggested codes in the Generated codes card" in text
    assert "do not expose any generated code, code count, code" in text
    assert "Do not say that codes are available" in text
    assert "Unable to display generated codes. Please" in text
    assert "pluginName: a2ui" in text
    assert "pluginName: ambulatorycoding" in text
    assert "launchesPlugin: ambulatorycoding" in text
    assert "tool: recode_profees_coding" in text
    assert "ui_schema_path: ambulatory-coding/schemas/ambulatory_coding_canvas.json" in text
    assert "profees-coding/ambulatory-coding/schemas" not in text

    packaged_skill_dirs = [path.name for path in AMBULATORY_SKILLS_ROOT.iterdir() if path.is_dir() and path.name != "__pycache__"]
    assert packaged_skill_dirs == ["ambulatory-coding"]


def test_orchestrator_skill_delegates_to_current_profees_subskills() -> None:
    text = _text(MEDICAL_CODING_SKILL)

    for skill_name in EXPECTED_SKILL_NAMES:
        assert skill_name in text

    assert "toolkit_render_coding_summary" in text
    assert "coding_summary_schema.json" in text
    assert "$profees-medical-coding" in text


def test_all_declared_skill_names_match_current_directories() -> None:
    for expected_name, skill_path in EXPECTED_SKILL_NAMES.items():
        assert skill_path.is_file()
        assert f"name: {expected_name}" in _text(skill_path)


def test_cpt_em_skill_requires_wrapper_and_current_references() -> None:
    text = _text(SKILLS_ROOT / "cpt-em-coding" / "SKILL.md")

    expected = {
        "em_code_selection",
        "scripts/em_code_selection.py",
        "references/encounter_type_ambulatory.md",
        "references/encounter_type_inpatient.md",
        "references/encounter_type_ed.md",
        "references/time_statement.md",
        "references/mdm_problem_complexity.md",
        "references/mdm_data_complexity.md",
        "references/mdm_risk_complexity.md",
        "Dual encounters",
    }

    for value in expected:
        assert value in text


def test_runtime_prompt_context_is_recorded_in_matching_references() -> None:
    expected_by_reference = {
        "cpt-em-coding/references/encounter_type_ambulatory.md": {
            "NOT_SUPPORTED",
            "NCCI edits or global surgery rules",
            "included in payment for a procedure",
        },
        "cpt-em-coding/references/encounter_type_ed.md": {
            "ED disposition",
            "No new vs established patient distinction",
        },
        "cpt-em-coding/references/encounter_type_inpatient.md": {
            "Same-Day Admit/Discharge",
            "Hospital Discharge",
        },
        "cpt-em-coding/references/time_statement.md": {
            "E/M billing codes or descriptors",
            "de-identified",
        },
        "cpt-em-coding/references/code_selection.md": {
            "final_patient_type",
            "Ignore E/M code descriptors",
            "visit history controls",
            "same-location encounter",
        },
        "cpt-em-coding/references/mdm_problem_complexity.md": {
            "Risk does not upgrade Problem Complexity by itself",
            "HPI and objective data may be used",
        },
        "cpt-em-coding/references/mdm_data_complexity.md": {
            "Laboratory panels",
            "single test/order",
            "de-identified placeholder",
            "Independent historian",
        },
        "cpt-em-coding/references/mdm_risk_complexity.md": {
            "management actions documented",
            "same-day directive",
        },
        "cpt-em-coding/references/audit_problem_complexity.md": {
            "Risk",
            "does NOT upgrade Problem Complexity",
        },
        "cpt-em-coding/references/audit_data_complexity.md": {
            "external provider",
            "DISC",
        },
        "cpt-em-coding/references/audit_risk_complexity.md": {
            "same-day directive",
            "med list/history",
        },
        "cpt-procedures-coding-skill/references/schemas.md": {
            "explicitly performed or supervised by the physician",
            "outside facilities or labs",
            "future or conditional",
            "known_orders",
            'orders with status "COMPLETED"',
            "80000-89999",
            "90471",
            "90472",
            "90480",
            "93000",
            "96127",
            "96160",
        },
        "hcpcs-coding/references/g2211_overview.md": {
            "SOAP note and orders together",
            "retrieval examples",
            "G2211 is never stand-alone",
            "same-date qualifying",
            "Welcome to Medicare",
            "Medicare Part B",
            "hard_blocks",
            "g0402_eligible",
            "G0439, then G0438, then G0402",
            "oncology professional-fee",
        },
        "icd-10-association/references/documentation_standards.md": {
            "No Uncertain Diagnoses",
            "Main Reason for Visit",
            "No Medication-Only Coding",
            "Highest Specificity",
            "Supported Z-Codes Only",
            "professional-fee association coding",
            "candidate diagnoses",
            "Avoid shotgun association",
            "empty linked diagnosis list",
            "strongest diagnosis as primary",
        },
        "icd-10-association/references/postprocess_flags.md": {
            "invalid or non-billable",
            "Alternatives Only",
            "Service-Level Support Required",
        },
        "modifiers-coding/references/guidelines.md": {
            "Optimize for recall",
            "Do not assign modifiers during branch selection",
            "grouped branch",
            "performed/completed on the DOS",
            "therapy/rehab CPTs",
            "GP/GO/GN",
            "CQ/CO",
            "Modifier 95",
            "synchronous audio-video",
            "not for audio-only",
            "statutorily excluded",
            "UD",
            "ASA Physical Status",
            "P1-P4",
            "QK",
            "QX",
            "CLIA-waived",
            "reference lab",
            "Modifier 90",
        },
    }

    for reference_path, expected_values in expected_by_reference.items():
        text = _text(SKILLS_ROOT / reference_path)
        for expected_value in expected_values:
            assert expected_value in text


def test_no_standalone_prompt_version_reference_files_exist() -> None:
    assert not list(SKILLS_ROOT.glob("*/references/prompt_versions.md"))


def test_reference_content_does_not_use_runtime_prompt_source_metadata() -> None:
    for path in SKILLS_ROOT.glob("*/references/*.md"):
        assert "Runtime prompt source" not in _text(path)


def test_procedure_and_hcpcs_skills_keep_procedure_outputs_separate() -> None:
    procedure_text = _text(SKILLS_ROOT / "cpt-procedures-coding-skill" / "SKILL.md")
    hcpcs_text = _text(SKILLS_ROOT / "hcpcs-coding" / "SKILL.md")

    assert "5-digit CPT procedure codes" in procedure_text
    assert "Exclude E/M codes, ICD-10-CM codes, HCPCS codes" in procedure_text
    assert '"procedure": [Procedure, ...]' in procedure_text

    assert "G2211" in hcpcs_text
    assert 'system = "HCPCS"' in hcpcs_text
    assert '"procedure": [Procedure, ...]' in hcpcs_text


def test_association_and_modifier_skills_preserve_current_code_lists() -> None:
    association_text = _text(SKILLS_ROOT / "icd-10-association" / "SKILL.md")
    modifier_text = _text(SKILLS_ROOT / "modifiers-coding" / "SKILL.md")

    assert "do not introduce, modify, or omit any codes" in association_text
    assert "linked diagnosis lists" in association_text
    assert '"diagnosis": [ICD10CM, ...]' in association_text

    assert "Preserve item order" in modifier_text
    assert "Keep `linked_diagnosis` intact" in modifier_text
    assert '"em": [EMItem, ...]' in modifier_text


def test_coding_summary_schema_matches_tool_output_shape() -> None:
    schema_path = SKILLS_ROOT / "profees-medical-coding" / "assets" / "coding_summary_schema.json"
    schema = json.loads(_text(schema_path))

    assert schema["title"] == "ProfeesCodingSummary"
    assert schema["required"] == ["em", "procedure", "diagnosis"]
    assert schema["properties"]["em"]["items"]["$ref"] == "#/$defs/EMItem"
    assert schema["properties"]["procedure"]["items"]["$ref"] == "#/$defs/Procedure"
    assert schema["properties"]["diagnosis"]["items"]["$ref"] == "#/$defs/ICD10CM"


def test_no_stale_inpatient_skill_package_paths_remain() -> None:
    scanned_files = [
        ROOT_SKILL,
        MEDICAL_CODING_SKILL,
        *EXPECTED_SKILL_NAMES.values(),
    ]
    stale_terms = {
        "profees_skills",
        "inpatient-encounter-coding",
        "icd10cm-encoder",
        "icd10pcs-encoder",
        "run_inpatient_coding",
    }

    for path in scanned_files:
        text = _text(path)
        for term in stale_terms:
            assert term not in text
