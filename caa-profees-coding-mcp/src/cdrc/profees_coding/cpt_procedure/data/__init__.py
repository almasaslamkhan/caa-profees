from cdrc.profees_coding.cpt_procedure.data.reader import ConcurrentCPTProcedureReader
from cdrc.profees_coding.cpt_procedure.data.schema import InputSchema, OutputSchema


def get_default_reader() -> ConcurrentCPTProcedureReader:
    return ConcurrentCPTProcedureReader(input_schema=InputSchema, output_schema=OutputSchema)


__all__ = ["get_default_reader", "ConcurrentCPTProcedureReader", "InputSchema", "OutputSchema"]
