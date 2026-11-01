from cdrc.profees_coding.em.data.reader import ConcurrentEMDataReader
from cdrc.profees_coding.em.data.schema import InputSchema, OutputSchema


def get_default_reader() -> ConcurrentEMDataReader:
    return ConcurrentEMDataReader(input_schema=InputSchema, output_schema=OutputSchema)


__all__ = ["get_default_reader", "ConcurrentEMDataReader", "InputSchema", "OutputSchema"]
