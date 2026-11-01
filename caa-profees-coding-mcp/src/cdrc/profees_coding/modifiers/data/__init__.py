from cdrc.profees_coding.modifiers.data.reader import ConcurrentModifiersReader
from cdrc.profees_coding.modifiers.data.schema import InputSchema, OutputSchema


def get_default_reader() -> ConcurrentModifiersReader:
    return ConcurrentModifiersReader(input_schema=InputSchema, output_schema=OutputSchema)


__all__ = ["get_default_reader", "ConcurrentModifiersReader", "InputSchema", "OutputSchema"]
