from cdrc.profees_coding.hcpcs_codes.data.reader import ConcurrentHCPCSReader
from cdrc.profees_coding.hcpcs_codes.data.schema import InputSchema, OutputSchema


def get_default_reader() -> ConcurrentHCPCSReader:
    return ConcurrentHCPCSReader(input_schema=InputSchema, output_schema=OutputSchema)


__all__ = ["get_default_reader", "ConcurrentHCPCSReader", "InputSchema", "OutputSchema"]
