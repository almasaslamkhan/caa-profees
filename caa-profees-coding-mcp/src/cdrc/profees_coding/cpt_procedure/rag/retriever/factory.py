from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, Union

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from cdrc.profees_coding.cpt_procedure.rag.data_classes import CONFIG
from cdrc.profees_coding.cpt_procedure.rag.vector_store_utils import get_vector_stores


class BaseRetrieverFactory(ABC):
    @abstractmethod
    def build(self, config: Dict[str, Any]) -> Union[BaseRetriever, Collection[BaseRetriever]]:
        """
        Args:
            config: a config file for building retrievers
        Returns:
            a collection of retrievers built
        """


class RetrieverFactory(BaseRetrieverFactory):
    def build(self, config: Dict[str, Any]) -> BaseRetriever:
        method = config["method"]
        if method in ["lexical", "semantic"]:
            vector_store = get_vector_stores({**config[CONFIG.VDB_KWARGS]})
            if isinstance(vector_store, VectorStore):
                retriever = vector_store.as_retriever(
                    search_type=config.get("search_type", "similarity"), search_kwargs=config.get(CONFIG.SEARCH_KWARGS, {})
                )
            elif isinstance(vector_store, BaseRetriever):
                # this is for cases like bm25, tf-idf where the langchain retrievers are merely wrappers of third-party packages.
                retriever = vector_store
            else:
                raise ValueError("Invalid vector store class.")

        elif method == "hybrid":
            # Combine parent retriever search kwargs with children's search kwargs.
            retriever = EnsembleRetriever(
                retrievers=[self.build({**c, CONFIG.SEARCH_KWARGS: {**c[CONFIG.SEARCH_KWARGS]}}) for c in config["sub_retrievers"].values()],
                weights=config.get("weights", [1] * len(config["sub_retrievers"])),
            )
        else:
            raise NotImplementedError(f"Retrieval method {method} is currently not supported.")
        return retriever
