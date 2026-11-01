from abc import ABC, abstractmethod
from typing import Any, Generic, Iterator, TypeVar

from pydantic import BaseModel, Field, model_validator


class Document(BaseModel):
    """A simple document model to hold the content and metadata."""

    id: str = Field(..., description="Unique identifier for the document")
    page_content: str = Field(..., description="Content of the document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the document")


class ICD10Code(BaseModel):
    code: str | None = Field(default=None, min_length=3, max_length=8, frozen=True)
    is_valid_code: bool | None = Field(default=None, frozen=True)
    description: str | None = Field(default=None, frozen=False)  # Updated from order file for PCS

    # Fields from order file for PCS
    short_description: str | None = Field(default=None, frozen=False)
    order_number: int | None = Field(default=None, ge=1, frozen=False)

    @model_validator(mode="after")
    def validate_non_empty(self) -> "ICD10Code":
        assert self.code is not None, "Code must not be None"
        assert self.description is not None, "Description must not be None"
        assert self.is_valid_code is not None, "is_valid_code must not be None"
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.code}]({self.description})"


ICD10CodeType = TypeVar("ICD10CodeType", bound=ICD10Code)


class ICD10IteratorNode(ABC, Generic[ICD10CodeType]):
    @abstractmethod
    def list_codes(self) -> Iterator[ICD10CodeType]:
        """Yield ICD10 codes in the node."""
        pass


class SearchResult(BaseModel, Generic[ICD10CodeType]):
    code: ICD10CodeType | str = Field(..., frozen=True)
    score: float = Field(..., frozen=True, ge=0.0)


class SearchCorpus(ABC, BaseModel, Generic[ICD10CodeType]):
    """
    Converts ICD10 codes (PCS and CM) to documents.
    One code may generate multiple documents, each from description, inclusion terms, synonyms.
    """

    name: str | None = Field(default="NotSet", description="Name of dataset")
    codes: list[ICD10CodeType] = Field(..., description="List of codes")

    @model_validator(mode="after")
    def validate_non_empty(self) -> "SearchCorpus":
        assert self.name is not None, "Corpus name must not be None"
        return self

    def __getitem__(self, idx: int) -> ICD10CodeType:
        return self.codes[idx]

    def __repr__(self) -> str:
        return f"{self.name}"

    def __len__(self) -> int:
        return len(self.codes)

    @abstractmethod
    def code_to_documents(self, code: ICD10Code, idx: int) -> Iterator[Document]:
        """Convert a code to one or more documents."""
        pass

    def get_documents(self) -> Iterator[Document]:
        """Yield all documents from all codes."""
        for idx, code in enumerate(self.codes):
            yield from self.code_to_documents(code, idx)

    def __call__(self) -> Iterator[Document]:
        yield from self.get_documents()


class BaseICD10Search(ABC):
    """
    Abstract base class for various types of ICD10-CM searches.
    Each subclass should implement the 'search' method.
    """

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> Iterator[SearchResult]:
        """
        Perform a search and return the top_k results as an iterator of SearchResult.
        """
        pass
