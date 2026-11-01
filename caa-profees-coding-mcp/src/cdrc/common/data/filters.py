import os
import re
from typing import Any, Dict, List, Optional

# Dictionary of available filters
AVAILABLE_FILTERS = {}


def example_id_filter(
    data: List[Dict[str, Any]],
    allowed_ids_file: str = None,
    allowed_ids: List[str] = None,
    allow_prefix_match: bool = True,
) -> List[Dict[str, Any]]:
    """Filter data items based on example_id.

    Args:
        data: List of data items to filter.
        allowed_ids_file: Path to a file containing allowed example IDs (one per line).
        allowed_ids: List of allowed example IDs.
        allow_prefix_match: Match prefixed example IDs (e.g., per-day/per-specialty slices).

    Returns:
        List[Dict[str, Any]]: Filtered list of data items.
    """
    if allowed_ids is None:
        allowed_ids = []

    # Load IDs from file if provided - only once per filtering operation
    if allowed_ids_file and os.path.exists(allowed_ids_file):
        with open(allowed_ids_file, "r") as f:
            file_ids = {line.strip() for line in f if line.strip()}
            allowed_ids.extend(file_ids)

    allowed_ids_set = set(allowed_ids)

    if not allow_prefix_match:
        return [item for item in data if item.get("example_id") in allowed_ids_set]

    def _matches(value: Optional[str]) -> bool:
        if value in allowed_ids_set:
            return True
        if not value:
            return False
        prefix = value.split("-", 1)[0]
        return prefix in allowed_ids_set

    return [item for item in data if _matches(item.get("example_id"))]


def text_contains_filter(data: List[Dict[str, Any]], patterns: List[str] = None, field: str = None) -> List[Dict[str, Any]]:
    """Filter data items based on whether the specified field contains any of the patterns.

    Args:
        data: List of data items to filter.
        patterns: List of regex patterns to search for.
        field: The field to search in, can use dot notation for nested fields (e.g., "gold.cpt.description").
               This parameter is required.

    Returns:
        List[Dict[str, Any]]: Filtered list of data items.

    Raises:
        ValueError: If field is not specified.
    """
    if field is None:
        raise ValueError("The 'field' parameter is required for text_contains_filter.")

    if not patterns:
        return data

    # Compile patterns once for efficiency
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    result = []

    # Function to get nested field value using dot notation
    def get_nested_field(item, field_path):
        if not field_path or not item:
            return None

        parts = field_path.split(".")
        value = item

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                # Handle list indexing in dot notation (e.g., "gold.cpt.0.code")
                value = value[int(part)]
            else:
                return None

        return value

    for item in data:
        # Get field value, handling nested fields
        field_value = get_nested_field(item, field)

        # Skip items without the specified field
        if field_value is None:
            continue

        # Convert to string if not already
        if not isinstance(field_value, str):
            field_value = str(field_value)

        # Match against all patterns
        if any(pattern.search(field_value) for pattern in compiled_patterns):
            result.append(item)

    return result


def composite_filter(data: List[Dict[str, Any]], filters: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Apply multiple filters in sequence.

    Args:
        data: List of data items to filter.
        filters: List of filter configurations, each with 'name' and 'args' keys.

    Returns:
        List[Dict[str, Any]]: Filtered list of data items.
    """
    if not filters:
        return data

    filtered_data = data
    for filter_config in filters:
        if isinstance(filter_config, dict) and "name" in filter_config:
            filter_name = filter_config["name"]
            filter_args = filter_config.get("args", {})

            if filter_name in AVAILABLE_FILTERS:
                filter_func = AVAILABLE_FILTERS[filter_name]
                # Pass the current filtered_data as input to the next filter
                filtered_data = filter_func(filtered_data, **filter_args)
            else:
                print(f"Warning: Filter '{filter_name}' not found in AVAILABLE_FILTERS")

    return filtered_data


def client_filter(data: List[Dict[str, Any]], client: List[str] = None) -> List[Dict[str, Any]]:
    """Filter data items based on client.

    Args:
        data: List of data items to filter.
        client: List of allowed clients.

    Returns:
        List[Dict[str, Any]]: Filtered list of data items.
    """
    if not client:
        return data

    return [item for item in data if item.get("input", {}).get("source") in client]


# Dictionary of available filters
AVAILABLE_FILTERS = {
    "example_id": example_id_filter,
    "text_contains": text_contains_filter,
    "composite": composite_filter,
    "client": client_filter,
}
