from cdrc.profees_coding.icd10cm_associate.data.reader import ConcurrentICD10CMAssociateReader
from cdrc.profees_coding.icd10cm_associate.data.schema import InputSchema, OutputSchema


def get_default_reader(n_instances: int = -1) -> ConcurrentICD10CMAssociateReader:
    return ConcurrentICD10CMAssociateReader(
        input_schema=InputSchema,
        output_schema=OutputSchema,
        n_instances=n_instances,
    )


__all__ = ["get_default_reader", "ConcurrentICD10CMAssociateReader", "InputSchema", "OutputSchema"]
