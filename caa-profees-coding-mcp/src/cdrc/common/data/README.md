# Filtering Data in CDRC Evaluation Framework

## Overview

This README explains how to use and extend the data filtering system in the CDRC evaluation framework. The filtering system allows you to select specific subsets of your evaluation data based on various criteria such as example IDs, text content, encounter types, and complexity levels.

## Available Filters

The framework provides two categories of filters:

1. **Common Filters** (defined in `cdrc/common/data/filters.py`):
   - `example_id`: Filter by specific example IDs
   - `text_contains`: Filter by text patterns in specified fields
   - `composite`: Apply multiple filters in sequence

2. **Domain-Specific Filters** (e.g., CPT EM filters in `cdrc/profees_coding/em/data/filters.py`):
   - `encounter_type`: Filter by encounter types
   - `complexity_level`: Filter by complexity levels

## Using Filters in Evaluation Scripts

Filters can be applied in your evaluation scripts by specifying two parameters:

1. `--filter_name`: The name of the filter to apply
2. `--filter_kwargs`: A Python dictionary as a string containing filter arguments

### Example Usage

Here's how to use different filters in your evaluation scripts:

```bash
# Example 1: Filter by example IDs from a file
FILTER_NAME="example_id"
FILTER_KWARGS="{'allowed_ids_file': '/path/to/ids.txt'}"

# Example 2: Filter by example IDs directly
FILTER_NAME="example_id"
FILTER_KWARGS="{'allowed_ids': ['id1', 'id2', 'id3']}"

# Example 3: Filter by encounter type
FILTER_NAME="encounter_type"
FILTER_KWARGS="{'encounter_types': ['Office or other outpatient visit', 'Hospital observation care']}"

# Example 4: Filter by text content in a nested field
FILTER_NAME="text_contains"
FILTER_KWARGS="{'patterns': ['diabetes', 'hypertension'], 'field': 'soap.text'}"  # dot notation to access nested fields

# Example 5: Filter by complexity level
FILTER_NAME="complexity_level"
FILTER_KWARGS="{'levels': ['high', 'medium'], 'complexity_type': 'mdm_complexity'}"
```

### Composite Filters

You can combine multiple filters using the `composite` filter:

```bash
# Example: Filter for high complexity office visits
FILTER_NAME="composite"
FILTER_KWARGS="{'filters': [
    {'name': 'encounter_type', 'args': {'encounter_types': ['Office or other outpatient visit']}},
    {'name': 'complexity_level', 'args': {'levels': ['high']}}
]}"
```

## Adding Custom Filters

You can add custom filters for your module.
Refer to `cdrc/profees_coding/em/data/filters.py` for examples.


## Implementation Details

The filtering system is implemented in several key components:

1. **Filter Definitions**: Domain-specific filters are defined in domain modules (e.g., `cdrc/profees_coding/em/data/filters.py`), while common filters are defined in `cdrc/common/data/filters.py`.

2. **Data Reader**: The filtering is applied in the `read_data` method of `JsonDataReader` in `cdrc/common/data/reader.py`.

3. **Evaluator**: Filters are passed to the `run` method of `BaseEvaluator` in `cdrc/common/evaluator/runner.py`.

## Example Evaluation Script

Here's a complete example of using filters in an evaluation script:

```bash
python cdrc/profees_coding/em/eval_runner.py \
    --data_path data.json \
    --design medprompt_detailed \
    --config_path ./cdrc/profees_coding/em/models/medprompt_detailed/config/runner.yaml \
    --filter_name "example_id" \
    --filter_kwargs "{'allowed_ids_file': '/path/to/ids.txt'}" \
    # ... other arguments ...
```

## Passing Filters in Bash Scripts

When setting up filters in a bash script, use the following pattern to ensure proper parsing:

```bash
FILTER_NAME="example_id"
FILTER_KWARGS="{'allowed_ids_file': '/path/to/ids.txt'}"

# Create an array for optional arguments
OPTIONAL_ARGS=()
[ -n "${FILTER_NAME}" ] && OPTIONAL_ARGS+=(--filter_name "${FILTER_NAME}")
[ -n "${FILTER_KWARGS}" ] && OPTIONAL_ARGS+=(--filter_kwargs "${FILTER_KWARGS}")

# Then include the optional args in your command
python cdrc/profees_coding/em/eval_runner.py \
    # ... other arguments ... \
    "${OPTIONAL_ARGS[@]}" \
    # ... remaining arguments ...
```

This approach ensures that the string arguments are properly passed to the Python script and can be evaluated correctly.
