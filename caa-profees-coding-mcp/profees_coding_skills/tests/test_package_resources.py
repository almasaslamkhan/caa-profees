from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT / "src" / "skills_package"


SKILL_FILES = [
    "skills/profees-coding/SKILL.md",
    "skills/profees-coding/cpt-em-coding/SKILL.md",
    "skills/profees-coding/cpt-procedures-coding-skill/SKILL.md",
    "skills/profees-coding/hcpcs-coding/SKILL.md",
    "skills/profees-coding/icd-10-association/SKILL.md",
    "skills/profees-coding/modifiers-coding/SKILL.md",
    "skills/profees-coding/profees-medical-coding/SKILL.md",
]

REFERENCE_FILES = [
    "skills/profees-coding/cpt-em-coding/references/code_selection.md",
    "skills/profees-coding/cpt-em-coding/references/encounter_type_ambulatory.md",
    "skills/profees-coding/cpt-em-coding/references/mdm_problem_complexity.md",
    "skills/profees-coding/cpt-em-coding/scripts/em_code_selection.py",
    "skills/profees-coding/cpt-procedures-coding-skill/assets/cpt_long_descriptions.json",
    "skills/profees-coding/hcpcs-coding/references/g2211_overview.md",
    "skills/profees-coding/icd-10-association/assets/cpt_icd10_eligibility.json",
    "skills/profees-coding/icd-10-association/scripts/association_postprocess.py",
    "skills/profees-coding/modifiers-coding/references/guidelines.md",
    "skills/profees-coding/profees-medical-coding/assets/coding_summary_schema.json",
]


def test_legacy_skill_source_is_retained() -> None:
    for relative_path in SKILL_FILES:
        assert LEGACY_ROOT.joinpath(relative_path).is_file()


def test_legacy_supporting_resource_source_is_retained() -> None:
    for relative_path in REFERENCE_FILES:
        assert LEGACY_ROOT.joinpath(relative_path).is_file()
