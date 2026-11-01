from typing import Any, Optional, Sequence

from langchain_classic.retrievers.document_compressors.listwise_rerank import LLMListwiseRerank, _get_prompt_input
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

_reasoning_system_tmpl = """{context}
Re-rank candidates by:
- Relevance to the SOAP note entities
- Support from Proposed Actions (if aligned)
- Physician ordering patterns (if available)

Apply constraints:

- Only include items supported by SOAP note context
- Remove invalid/duplicate orders (including already signed or in PA list)
- Ensure medical coding compliance (map to valid chargemaster CPT/HCPCS/LOINC codes)

Output:
- Top-K ranked order catalog list
- For each item: rationale, evidence (entity match, source), and confidence score"""


_DEFAULT_REASONING_PROMPT: BasePromptTemplate = ChatPromptTemplate.from_messages(
    [("system", _reasoning_system_tmpl), ("human", "SOAP Note: {query}")]
)


class _RankDocsWithReasoning(BaseModel):
    """Rank docs and give reasoning per doc."""

    ranked_document_ids: list[int] = Field(..., description="Doc IDs sorted from most to least relevant")
    reasoning: list[str] = Field(
        ...,
        description="Concise reasoning for each ranked doc as to why they are ranked at particular positions, "
        "or why they are less relevant to those higher in the rank.",
    )


def _parse(result: dict) -> list[Document]:
    ranking: _RankDocsWithReasoning = result["ranking"]
    docs: Sequence[Document] = result["documents"]

    ordered = [docs[i] for i in ranking.ranked_document_ids]
    for doc, reason in zip(ordered, ranking.reasoning):
        doc.metadata["rerank_reason"] = reason
    return ordered


class LLMListwiseRerankWithReasoning(LLMListwiseRerank):

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> "LLMListwiseRerankWithReasoning":

        if llm.with_structured_output == BaseLanguageModel.with_structured_output:
            raise ValueError(f"llm of type {type(llm)} does not implement `with_structured_output`.")

        _prompt = prompt or _DEFAULT_REASONING_PROMPT

        reranker = RunnablePassthrough.assign(
            ranking=RunnableLambda(_get_prompt_input) | _prompt | llm.with_structured_output(_RankDocsWithReasoning)
        ) | RunnableLambda(_parse)
        return cls(reranker=reranker, **kwargs)
