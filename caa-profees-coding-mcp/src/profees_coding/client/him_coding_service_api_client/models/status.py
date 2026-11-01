from enum import Enum


class Status(str, Enum):
    ACCEPTED = "ACCEPTED"
    PROPOSED = "PROPOSED"
    PROPOSED_ACCEPTED = "PROPOSED_ACCEPTED"
    PROPOSED_SAVED = "PROPOSED_SAVED"
    PROPOSED_SUBMITTED = "PROPOSED_SUBMITTED"
    SAVED = "SAVED"
    SUBMITTED = "SUBMITTED"

    def __str__(self) -> str:
        return str(self.value)
