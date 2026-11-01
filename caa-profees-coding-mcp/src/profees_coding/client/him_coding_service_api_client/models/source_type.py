from enum import Enum


class SourceType(str, Enum):
    DOCUMENT_REFERENCE = "DOCUMENT_REFERENCE"

    def __str__(self) -> str:
        return str(self.value)
