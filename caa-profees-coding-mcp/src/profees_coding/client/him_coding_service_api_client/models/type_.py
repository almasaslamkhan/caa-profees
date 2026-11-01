from enum import Enum


class Type(str, Enum):
    DIAGNOSIS = "DIAGNOSIS"
    EM = "EM"
    PREVENTIVE = "PREVENTIVE"
    PROCEDURE = "PROCEDURE"

    def __str__(self) -> str:
        return str(self.value)
