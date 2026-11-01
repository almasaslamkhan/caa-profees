from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, Union

from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.prompts import ChatPromptTemplate

from cdrc.common.llm_utils import get_llm
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.cpt_procedure.rag.data_classes import CONFIG
from cdrc.profees_coding.cpt_procedure.rag.reranker.custom_models import LLMListwiseRerankWithReasoning


class BaseRerankerFactory(ABC):
    @abstractmethod
    def build(self, config: Dict[str, Any]) -> Union[BaseDocumentCompressor, Collection[BaseDocumentCompressor]]:
        """
        Args:
            config: a config file for building rerankers
        Returns:
            a collection of rerankers built
        """


class RerankerFactory(BaseRerankerFactory):
    def build(self, config: Dict[str, Any]) -> BaseDocumentCompressor:
        method = config["method"]
        if method in ["llm_listwise", "llm_listwise_with_reason"]:
            _LLM_LISTWISE_RERANKER_REGISTRY = {"llm_listwise": LLMListwiseRerank, "llm_listwise_with_reason": LLMListwiseRerankWithReasoning}
            llm_reranker = get_llm()[config[CONFIG.MODEL_NAME]](**config[CONFIG.MODEL_KWARGS])
            prompt_version = config[CONFIG.RERANK_PROMPT_VERSION]
            rerank_prompt_str = load_from_yaml(config["prompt_path"])[prompt_version]
            rerank_prompt = ChatPromptTemplate.from_template(rerank_prompt_str)
            return _LLM_LISTWISE_RERANKER_REGISTRY[method].from_llm(llm_reranker, top_n=config["top_n"], prompt=rerank_prompt)
        else:
            raise NotImplementedError("Only llm listwise rerankers have been implemented.")
