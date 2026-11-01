from dataclasses import dataclass


@dataclass(frozen=True)
class ICD10CMDataPaths:
    year: str
    order_file: str
    tabular_file: str
    search_index_file: str


SUPPORTED_ICD10_CM_YEARS = ("2025", "2026")
DEFAULT_ICD10_CM_YEAR = "2026"


def get_icd10_cm_data_paths(year: str = DEFAULT_ICD10_CM_YEAR) -> ICD10CMDataPaths:
    if year not in SUPPORTED_ICD10_CM_YEARS:
        raise ValueError(f"Unsupported ICD-10-CM year: {year}")

    source_dir = f"icd10-cm-{year}"
    return ICD10CMDataPaths(
        year=year,
        order_file=f"{source_dir}/icd10cm_order_{year}.txt",
        tabular_file=f"{source_dir}/icd10cm_tabular_{year}.xml",
        search_index_file=f"indices/{source_dir}/cm_tables/bm25s_index.pkl",
    )


DEFAULT_ICD10_CM_DATA_PATHS = get_icd10_cm_data_paths()
