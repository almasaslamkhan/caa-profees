from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

from langchain_core.documents import Document
from loguru import logger

from cdrc.profees_coding.cpt_procedure.rag.data_classes import CONFIG, METADATA_KEYS
from cdrc.profees_coding.cpt_procedure.rag.query.selector import QueryPartSelector
from cdrc.profees_coding.cpt_procedure.rag.reranker.factory import RerankerFactory
from cdrc.profees_coding.cpt_procedure.rag.retriever.factory import RetrieverFactory


class BaseRetrievalEngine(ABC):
    @abstractmethod
    def run(self, query: Union[str, Dict[str, str]], **kwargs) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Args:
            query: can be either a string or a dictionary of different query parts
        Returns:
            either a list of Documents, or a list of tuples of (Document, similarity score)
        """


class RetrievalEngine(BaseRetrievalEngine):
    def __init__(self, query_config: Dict[str, Any], retriever_config: Dict[str, Any], reranker_config: Dict[str, Any] = None):
        self.query_part_selector = QueryPartSelector(query_config)
        self.retriever = RetrieverFactory().build(retriever_config)
        self.reranker = RerankerFactory().build(reranker_config) if reranker_config is not None else None

    def run(
        self, query: Union[str, Dict[str, Any]], context: Union[str, Dict[str, Any]], **kwargs
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        retrieval_query = self.query_part_selector.select(query, stage=CONFIG.RETRIEVAL)
        docs = self.retriever.invoke(retrieval_query)
        # if self.reranker is None:
        #     logger.debug(
        #         f"Retrieved example list for encounter {query}:\n\n"
        #         f"\tRetrieved example ids: {[doc.metadata[METADATA_KEYS.DESCRIPTION] for doc in docs]}\n\n"
        #         "----------------------------------------------------\n\n"
        #     )
        #     return docs
        # reranked_docs = self.reranker.compress_documents(docs, context)
        # _doc_info = [
        #     (
        #         (doc.metadata[METADATA_KEYS.DESCRIPTION], doc.metadata["rerank_reason"])
        #         if "rerank_reason" in doc.metadata
        #         else doc.metadata[METADATA_KEYS.DESCRIPTION]
        #     )
        #     for doc in reranked_docs
        # ]
        # logger.debug(
        #     f"Retrieved example list for encounter {query}"
        #     f"\tPrior-reranking example ids: {[doc.metadata[METADATA_KEYS.DESCRIPTION] for doc in docs]}\n\n"
        #     f"\tPost-reranking example ids: "
        #     f"{_doc_info}\n\n"
        #     "----------------------------------------------------\n\n"
        # )
        return docs
