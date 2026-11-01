import json
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional, Tuple, Type

try:  # pragma: no cover - optional dependency for environments without loguru
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from pydantic import BaseModel


class BaseDataReader(ABC):
    """Abstract base class for data readers."""

    def __init__(self, input_schema: Type[BaseModel], output_schema: Optional[Type[BaseModel]] = None, n_instances: int = -1):
        """Initialize with input/output schemas.

        Args:
            input_schema (Type[BaseModel]): The Pydantic model for the input data.
            output_schema (Optional[Type[BaseModel]]): The Pydantic model for the output data.
        """
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.n_instances = n_instances

    @abstractmethod
    def read_data(
        self, file_path: str, filter_function: Optional[Callable] = None, **filter_args
    ) -> Tuple[list[BaseModel], Optional[list[BaseModel]]]:
        """Abstract method to read data from a file path with optional filtering.

        Args:
            file_path (str): The path to the data file.
            filter_function (Optional[Callable]): Function to apply to the data.
            **filter_args: Arguments to pass to the filter function.

        Returns:
            Tuple[list[BaseModel], Optional[list[BaseModel]]]: The input data and optional output data.
        """
        pass


class JsonDataReader(BaseDataReader):
    """Concrete data reader that reads data from a JSON file."""

    def read_data(
        self, file_path: str, filter_function: Optional[Callable] = None, **filter_args
    ) -> Tuple[list[BaseModel], Optional[list[BaseModel]]]:
        """Read data from a file path and separate input and output data.

        Args:
            file_path (str): The path to the data file.
            filter_function (Optional[Callable]): Function to apply to the data.
            **filter_args: Arguments to pass to the filter function.

        Returns:
            Tuple[list[BaseModel], Optional[list[BaseModel]]]: The input data and optional output data.
        """
        with open(file_path, "r") as f:
            if file_path.endswith(".json"):
                data = json.load(f)
            elif file_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                raise ValueError("Unsupported file format.")

        if len(data) > 0 and self.n_instances > 0:
            data = data[: self.n_instances]
            print(f"Only keep the first {len(data)} instances in the dataset.")

        # Apply filter if specified
        if filter_function is not None:
            original_count = len(data)
            data = filter_function(data, **filter_args)
            filtered_count = len(data)
            print(f"Filtered data: {filtered_count} items remaining from {original_count} (removed {original_count - filtered_count})")

        input_data = []
        output_data = [] if self.output_schema else None

        def _merge_field(value, gold_value):
            if isinstance(value, list) or isinstance(gold_value, list):
                base_list = list(value or [])
                gold_list = list(gold_value or [])
                return base_list + gold_list if gold_list else base_list
            return gold_value if value is None else value

        for item in data:
            # Extract input data based on the input schema
            input_item = item.get("input", item)
            gold_item = item.get("gold", {})
            input_fields = {field: _merge_field(input_item.get(field), gold_item.get(field)) for field in self.input_schema.model_fields.keys()}
            input_data.append(self.input_schema(**input_fields))

            # Extract output data based on the output schema and 'gold' key
            if self.output_schema:
                output_fields = {field: gold_item.get(field) for field in self.output_schema.model_fields.keys()}
                output_data.append(self.output_schema(**output_fields))

        return input_data, output_data

    def read_in_batches(
        self, file_path: str, batch_size: int = 32, filter_function: Optional[Callable] = None, **filter_args
    ) -> Iterator[Tuple[list[BaseModel], Optional[list[BaseModel]]]]:
        """Read data in batches from a file.

        Args:
            file_path (str): The path to the data file.
            batch_size (int): The number of samples per batch.
            filter_function (Optional[Callable]): Function to apply to the data.
            **filter_args: Arguments to pass to the filter function.

        Yields:
            Iterator[Tuple[list[BaseModel], Optional[list[BaseModel]]]]: Batches of input and optional output data.
        """
        with open(file_path, "r") as f:
            if file_path.endswith(".json"):
                data = json.load(f)
            elif file_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                raise ValueError("Unsupported file format.")

        # Apply filter if specified
        if filter_function is not None:
            original_count = len(data)
            data = filter_function(data, **filter_args)
            filtered_count = len(data)
            print(f"Filtered data: {filtered_count} items remaining from {original_count} (removed {original_count - filtered_count})")

        def _merge_field(value, gold_value):
            if isinstance(value, list) or isinstance(gold_value, list):
                base_list = list(value or [])
                gold_list = list(gold_value or [])
                return base_list + gold_list if gold_list else base_list
            return gold_value if value is None else value

        for i in range(0, len(data), batch_size):
            batch_data = data[i : i + batch_size]
            input_batch = []
            output_batch = [] if self.output_schema else None

            for item in batch_data:
                input_item = item.get("input", {})
                gold_item = item.get("gold", {})
                input_fields = {field: _merge_field(input_item.get(field), gold_item.get(field)) for field in self.input_schema.model_fields.keys()}
                input_batch.append(self.input_schema(**input_fields))

                if self.output_schema:
                    output_fields = {field: gold_item.get(field) for field in self.output_schema.model_fields.keys()}
                    output_batch.append(self.output_schema(**output_fields))

            yield input_batch, output_batch


class ConcurrentDataReader(BaseDataReader):
    """Shared concurrent reader that expands encounters into per-day slices."""

    def __init__(
        self,
        input_schema: Type[BaseModel],
        output_schema: Optional[Type[BaseModel]] = None,
        n_instances: int = -1,
        *,
        use_specialty_slices: bool = False,
        postprocess_slice: Optional[Callable[[BaseModel, Optional[BaseModel]], None]] = None,
    ):
        super().__init__(input_schema=input_schema, output_schema=output_schema, n_instances=n_instances)
        self.use_specialty_slices = use_specialty_slices
        self.postprocess_slice = postprocess_slice

    def read_data(  # type: ignore[override]
        self,
        file_path: str,
        filter_function: Optional[Callable] = None,
        **filter_args,
    ) -> Tuple[list[BaseModel], Optional[list[BaseModel]]]:
        from cdrc.common.utils import split_concurrent_input_by_service_date, split_concurrent_input_by_service_date_and_specialty

        with open(file_path, "r") as f:
            if file_path.endswith(".json"):
                data = json.load(f)
            elif file_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                raise ValueError("Unsupported file format.")

        if len(data) > 0 and self.n_instances > 0:
            data = data[: self.n_instances]
            logger.info("Only keep the first {} instances in the dataset.", len(data))

        if filter_function is not None:
            original_count = len(data)
            data = filter_function(data, **filter_args)
            filtered_count = len(data)
            logger.info(
                "Filtered data: {} items remaining from {} (removed {})",
                filtered_count,
                original_count,
                original_count - filtered_count,
            )

        input_slices: list[BaseModel] = []
        output_slices: Optional[list[BaseModel]] = [] if self.output_schema else None

        for encounter in data:
            raw_input = encounter.get("input", encounter)
            base_input = self.input_schema(**raw_input)

            if self.use_specialty_slices:
                slices = split_concurrent_input_by_service_date_and_specialty(base_input)
            else:
                slices = split_concurrent_input_by_service_date(base_input)

            gold_by_date = {}
            if self.output_schema:
                for item in encounter.get("gold", []) or []:
                    service_date = item.get("service_date")
                    if service_date:
                        gold_by_date[service_date] = item

            for slice_item in slices:
                day_input = slice_item.day_input
                base_example_id = day_input.example_id or raw_input.get("example_id")
                if base_example_id:
                    suffix = slice_item.service_date
                    if slice_item.specialty:
                        suffix = f"{suffix}-{slice_item.specialty}"
                    day_input.example_id = f"{base_example_id}-{suffix}"

                input_slices.append(day_input)

                output_model = None
                if output_slices is not None:
                    gold_payload = gold_by_date.get(slice_item.service_date)
                    if gold_payload is None:
                        raise ValueError(f"Missing gold labels for service date {slice_item.service_date} in example {base_example_id}.")
                    output_model = self.output_schema(**gold_payload)
                    output_slices.append(output_model)

                if self.postprocess_slice is not None:
                    self.postprocess_slice(day_input, output_model)

        return input_slices, output_slices
