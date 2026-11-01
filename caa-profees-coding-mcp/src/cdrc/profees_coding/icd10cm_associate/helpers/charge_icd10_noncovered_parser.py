"""
Builds CPT/HCPCS -> explicitly noncovered ICD-10 mappings from CMS MCD Article CSV files.
"""

import collections
import csv
import datetime
import json
import os
import re


def main():

    # Hyperparameters for the input/output directories and export file names.
    year = "2026"
    input_directory = f"clinical-rev-cycle/cdrc/profees_coding/icd10cm_associate/data/{year}/"
    output_directory = f"clinical-rev-cycle/cdrc/profees_coding/icd10cm_associate/data/{year}/"
    hcpcs_table_file_name = "article_x_hcpc_code.csv"
    code_table_file_name = "article_x_code_table.csv"
    icd10_table_file_name = "article_x_icd10_noncovered.csv"
    cpt_output_file_name = f"cpt_icd10_noncovered_{year}.json"
    hcpcs_output_file_name = f"hcpcs_icd10_noncovered_{year}.json"

    # Resolving configured paths relative to the project root.
    project_root = find_project_root()
    input_path = os.path.join(project_root, input_directory)
    output_path = os.path.join(project_root, output_directory)
    cpt_output_path = os.path.join(output_path, cpt_output_file_name)
    hcpcs_output_path = os.path.join(output_path, hcpcs_output_file_name)

    # Validating that the output directory exists before writing files.
    os.makedirs(output_path, exist_ok=True)

    # Loading article-level CPT/HCPCS charge codes.
    charge_codes_by_article_group = load_charge_codes_by_article(
        os.path.join(input_path, hcpcs_table_file_name),
        os.path.join(input_path, code_table_file_name),
    )

    # Loading article-level noncovered ICD-10 code groups.
    icd10_codes_by_article_group = load_icd10_codes_by_article(
        os.path.join(input_path, icd10_table_file_name),
        "icd10_noncovered_group",
    )

    # Building the final charge -> explicitly noncovered ICD-10 mapping.
    noncovered_mapping = build_noncovered_mapping(
        charge_codes_by_article_group,
        icd10_codes_by_article_group,
    )

    # Splitting the combined mapping into CPT and HCPCS output files.
    cpt_mapping, hcpcs_mapping = split_mappings_by_code_system(noncovered_mapping)

    # Writing the split mappings to disk.
    write_mapping(cpt_output_path, cpt_mapping)
    write_mapping(hcpcs_output_path, hcpcs_mapping)

    print_split_summary(cpt_output_path, cpt_mapping, hcpcs_output_path, hcpcs_mapping)


def find_project_root():

    # Walking upward until we find the repository root that contains the package directory.
    current_path = os.path.abspath(os.path.dirname(__file__))

    while True:
        if os.path.isdir(os.path.join(current_path, "clinical-rev-cycle")):
            return current_path

        parent_path = os.path.dirname(current_path)

        if parent_path == current_path:
            raise FileNotFoundError("Could not locate the project root. Expected an ancestor directory " "containing 'clinical-rev-cycle'.")

        current_path = parent_path


def load_charge_codes_by_article(hcpcs_table_path, code_table_path):

    # Tracking which CPT/HCPCS codes belong to each article/version and article group.
    charge_codes_by_article_group = collections.defaultdict(set)

    for row in iterate_csv_rows(hcpcs_table_path, required=True):
        article_key = get_article_key(row, hcpcs_table_path)
        article_group = str(row.get("hcpc_code_group", "")).strip()
        charge_code = normalize_code(row.get("hcpc_code_id"))

        if is_supported_charge_code(charge_code):
            charge_codes_by_article_group[(article_key, article_group)].add(charge_code)

    for row in iterate_csv_rows(code_table_path, required=False):
        article_key = get_article_key(row, code_table_path)

        if not is_active_code_table_row(row):
            continue

        charge_code = normalize_code(row.get("hcpc_code_id"))

        if is_supported_charge_code(charge_code):
            charge_codes_by_article_group[(article_key, "")].add(charge_code)

    return charge_codes_by_article_group


def load_icd10_codes_by_article(icd10_table_path, group_column_name):

    # Tracking which ICD-10 codes belong to each article/version and ICD-10 group.
    icd10_codes_by_article_group = collections.defaultdict(set)

    for row in iterate_csv_rows(icd10_table_path, required=True):
        article_key = get_article_key(row, icd10_table_path)
        article_group = str(row.get(group_column_name, "")).strip()
        icd10_code = normalize_code(row.get("icd10_code_id"))

        if icd10_code and icd10_code != "XX000":
            icd10_codes_by_article_group[(article_key, article_group)].add(icd10_code)

    return icd10_codes_by_article_group


def iterate_csv_rows(csv_file_path, required):

    if not os.path.exists(csv_file_path):
        if required:
            raise FileNotFoundError(f"Could not find required table: {csv_file_path}")
        return

    last_error = None

    for encoding_name in ("utf-8-sig", "latin1"):
        try:
            with open(csv_file_path, "r", encoding=encoding_name, newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    normalized_row = {}

                    for column_name, value in row.items():
                        normalized_row[normalize_name(column_name)] = str(value or "").strip()

                    yield normalized_row
            return
        except UnicodeDecodeError as error:
            last_error = error

    raise last_error


def get_article_key(row, table_name):

    # Accessing the article id from the CMS row.
    article_id = get_required_column_value(row, "article_id", table_name)

    # Accessing the article version from the CMS row.
    article_version = get_required_column_value(row, "article_version", table_name)

    return article_id, article_version


def get_required_column_value(row, column_name, table_name):

    # Accessing a required value from a normalized CSV row.
    column_value = str(row.get(column_name, "")).strip()

    if not column_value:
        raise ValueError(f"Missing required column '{column_name}' in table '{table_name}'. Problem row: {row}")

    return column_value


def is_active_code_table_row(row):

    # Treating blank end dates as currently active, and skipping rows that have already ended.
    end_date = str(row.get("end_date", "")).strip()

    if not end_date:
        return True

    end_date_value = parse_iso_date_prefix(end_date)

    if end_date_value is None:
        return True

    return end_date_value >= datetime.date.today()


def parse_iso_date_prefix(value):

    # Parsing the YYYY-MM-DD portion of a CMS timestamp when available.
    value = str(value or "").strip()

    if len(value) < 10:
        return None

    try:
        return datetime.date.fromisoformat(value[:10])
    except ValueError:
        return None


def normalize_name(value):

    # Normalizing column names for case-insensitive CSV handling.
    value = str(value or "").lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)

    return value.strip("_")


def normalize_code(value):

    # Normalizing CPT/HCPCS and ICD-10 codes before joining or exporting.
    return str(value or "").strip().upper().replace(" ", "")


def is_supported_charge_code(code):

    # Skipping placeholder procedure codes that should not appear in the final mapping.
    return bool(code) and code != "XX000"


def build_noncovered_mapping(charge_codes_by_article_group, icd10_codes_by_article_group):

    # Tracking every CPT/HCPCS code found in the CMS article tables.
    all_charge_codes = set()

    # Tracking explicit ICD-10 noncovered lists for each CPT/HCPCS code.
    noncovered_icd10_codes_by_charge = collections.defaultdict(set)

    # Grouping noncovered information by article/version so we can match diagnosis groups to code groups.
    icd10_groups_by_article = collections.defaultdict(dict)

    for (article_key, article_group), icd10_codes in icd10_codes_by_article_group.items():
        icd10_groups_by_article[article_key][article_group] = icd10_codes

    for (article_key, article_group), charge_codes in charge_codes_by_article_group.items():
        article_groups = icd10_groups_by_article.get(article_key, {})
        noncovered_icd10_codes = resolve_icd10_codes_for_article_group(article_groups, article_group)

        for charge_code in charge_codes:
            all_charge_codes.add(charge_code)

            if len(noncovered_icd10_codes) > 0:
                noncovered_icd10_codes_by_charge[charge_code].update(noncovered_icd10_codes)

    # Building the final charge -> noncovered ICD-10 mapping.
    noncovered_mapping = {}

    for charge_code in sorted(all_charge_codes, key=natural_sort_key):
        noncovered_icd10_codes = noncovered_icd10_codes_by_charge.get(charge_code, set())
        noncovered_mapping[charge_code] = sorted(noncovered_icd10_codes, key=natural_sort_key)

    return noncovered_mapping


def resolve_icd10_codes_for_article_group(icd10_codes_by_group, article_group):

    # Matching grouped ICD-10 lists directly when possible.
    if article_group in icd10_codes_by_group:
        return icd10_codes_by_group[article_group]

    # If the article stores only ungrouped ICD-10 rows, use that flat article-level list.
    nonempty_groups = {group for group, codes in icd10_codes_by_group.items() if len(codes) > 0}

    if nonempty_groups == {""}:
        return icd10_codes_by_group.get("", set())

    # Otherwise, do not cross-apply a different numbered ICD group to this procedure-code group.
    return set()


def natural_sort_key(code):

    # Sorting codes in a more readable way, so A2 comes before A10.
    code_parts = re.split(r"(\d+)", code)

    return [int(part) if part.isdigit() else part for part in code_parts]


def split_mappings_by_code_system(mapping):

    # Separating numeric and CPT-suffix codes from level-II HCPCS alpha-numeric codes.
    cpt_mapping = {}
    hcpcs_mapping = {}

    for code, icd10_codes in mapping.items():
        if is_hcpcs_level_ii_code(code):
            hcpcs_mapping[code] = icd10_codes
        else:
            cpt_mapping[code] = icd10_codes

    return cpt_mapping, hcpcs_mapping


def is_hcpcs_level_ii_code(code):

    # CMS Level II HCPCS codes use one letter followed by four digits.
    return re.fullmatch(r"[A-Z]\d{4}", code) is not None


def write_mapping(output_path, mapping):

    # Writing deterministic pretty-printed JSON for easier review in diffs.
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=4, sort_keys=True)


def print_summary(output_path, mapping):

    # Counting total charge codes exported.
    total_charge_code_count = len(mapping)

    # Counting charge codes that have at least one explicit noncovered ICD-10 code.
    flagged_charge_code_count = sum(1 for icd10_codes in mapping.values() if len(icd10_codes) > 0)
    empty_charge_code_count = total_charge_code_count - flagged_charge_code_count

    print(f"Wrote: {output_path}")
    print(f"Total charge codes: {total_charge_code_count:,}")
    print(f"Charge codes with explicit noncovered ICD-10 lists: {flagged_charge_code_count:,}")
    print(f"Charge codes with [] no explicit noncovered list: {empty_charge_code_count:,}")


def print_split_summary(cpt_output_path, cpt_mapping, hcpcs_output_path, hcpcs_mapping):

    # Reusing the existing summary format for each exported code system.
    print_summary(cpt_output_path, cpt_mapping)
    print_summary(hcpcs_output_path, hcpcs_mapping)


if __name__ == "__main__":
    main()
