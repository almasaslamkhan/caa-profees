from cdrc.profees_coding.icd10cm.data.reader import ConcurrentICD10CMReader
from cdrc.profees_coding.icd10cm.data.schema import InputSchema, OutputSchema


def get_default_reader(n_instances: int = -1) -> ConcurrentICD10CMReader:
    return ConcurrentICD10CMReader(input_schema=InputSchema, output_schema=OutputSchema, n_instances=n_instances)


__all__ = ["get_default_reader", "ConcurrentICD10CMReader", "InputSchema", "OutputSchema"]
