from typing import Annotated, Iterator, TypeVar

from typing_extensions import override

from cdrc.common.icd10_cm_tools.src.tools.searcher.base import Document, SearchCorpus
from cdrc.common.icd10_cm_tools.src.tools.searcher.icd10cm import CMCode
from cdrc.common.icd10_cm_tools.src.tools.validator.cm_code_validator import get_description

T = TypeVar("T")


class CMMany2OneCorpus(SearchCorpus[CMCode]):
    name: str = "corpus-icd10cm-M-1"

    @override
    def code_to_documents(self, code: CMCode, idx: int) -> Iterator[Document]:
        yield Document(
            id=f"{code.code}__LongDesc",
            page_content=code.description,
            metadata={"code_idx": idx},
        )
        if get_description(code.code).lower() != code.description.lower():
            yield Document(
                id=f"{code.code}__LongDesc_Tabular",
                page_content=get_description(code.code),
                metadata={"code_idx": idx},
            )
        if code.short_description:
            yield Document(
                id=f"{code.code}__ShortDesc",
                page_content=code.short_description,
                metadata={"code_idx": idx},
            )
        if code.inclusion_term:
            for i, term in enumerate(code.inclusion_term):
                yield Document(
                    id=f"{code.code}__IncTerm_{i}",
                    page_content=term,
                    metadata={"code_idx": idx},
                )
