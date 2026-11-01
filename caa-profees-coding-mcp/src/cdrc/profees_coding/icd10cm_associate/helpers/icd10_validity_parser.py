"""
Builds an ICD-10-CM code -> validity mapping from the 2026 ICD-10-CM order file.
"""

import json
import os


def main():

    # Hyperparameters for the input/output directories and export file names.
    year = "2026"
    input_directory = f"clinical-rev-cycle/cdrc/profees_coding/icd10cm_associate/data/{year}/"
    output_directory = f"clinical-rev-cycle/cdrc/profees_coding/icd10cm_associate/data/{year}/"
    output_file_name = f"icd10_validity_{year}.json"

    # Resolving configured paths relative to the project root.
    project_root = find_project_root()
    input_path = find_first_existing_file(
        project_root,
        input_directory,
        [
            f"icd10cm_order_{year}.txt",
            f"icd10cm-order-{year}.txt",
        ],
    )
    output_path = os.path.join(project_root, output_directory, output_file_name)

    # Validating that the output directory exists before writing files.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Loading the full valid ICD-10-CM code universe from the order file.
    validity_by_code = load_validity_from_order_file(input_path)

    # Writing deterministic pretty-printed JSON for easier review in diffs.
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(validity_by_code, file, indent=4, sort_keys=True)

    print(f"Wrote: {output_path}")
    print(f"Total valid ICD-10-CM codes: {len(validity_by_code):,}")


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


def find_first_existing_file(project_root, directory_path, file_names):

    # Returning the first matching file so minor yearly naming differences do not break the script.
    for file_name in file_names:
        candidate_path = os.path.join(project_root, directory_path, file_name)

        if os.path.exists(candidate_path):
            return candidate_path

    raise FileNotFoundError(f"Could not find any of the expected files under '{os.path.join(project_root, directory_path)}': " f"{file_names}")


def load_validity_from_order_file(order_file_path):

    # Marking every official ICD-10-CM code in the order file as valid.
    validity_by_code = {}

    with open(order_file_path, "r", encoding="utf-8") as file:
        for line in file:
            raw_code = line[6:13].strip()
            code = normalize_icd10_code(raw_code)

            if code:
                validity_by_code[code] = 1

    return validity_by_code


def normalize_icd10_code(value):

    # Converting raw ICD-10-CM codes to the standard dotted uppercase representation.
    code = str(value or "").strip().upper().replace(".", "").replace(" ", "")

    if not code or code == "XX000":
        return ""

    if len(code) <= 3:
        return code

    return f"{code[:3]}.{code[3:]}"


if __name__ == "__main__":
    main()
