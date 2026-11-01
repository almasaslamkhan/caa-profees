from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from bm25s.tokenization import Tokenizer
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from nltk.corpus import stopwords
from pydantic import ConfigDict, Field
from Stemmer import Stemmer

stopwords = set(stopwords.words("english"))
stemmer = Stemmer("english")
tokenizer = Tokenizer(stemmer=stemmer, stopwords=stopwords)


def preprocess_func(texts: Union[str, Tuple[str, ...]]) -> Union[str, List[str]]:
    if isinstance(texts, str):
        texts = texts.lower()
        texts = [token for token in texts.split() if any(ch.isalpha() for ch in token) and not token.isnumeric()]
        texts = " ".join(texts)
        return texts

    elif isinstance(texts, tuple) or isinstance(texts, list):
        processed_texts = []
        for text in texts:
            text = text.lower()
            tokens = [token for token in text.split() if any(ch.isalpha() for ch in token) and not token.isnumeric()]
            text = " ".join(tokens)
            processed_texts.append(text)
        return processed_texts


class BM25Retriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any = None
    """ BM25 vectorizer."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
    k: int = 4
    """ Number of documents to return."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[Iterable[str]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BM25Retriever:
        """
        Create a BM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            ids: A list of ids to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        try:
            from bm25s import BM25
        except ImportError:
            raise ImportError("Could not import bm25s, please install with `pip install " "bm25s`.")

        texts_processed = tokenizer.tokenize(preprocess_func(texts))
        bm25_params = bm25_params or {}
        vectorizer = BM25(**bm25_params)
        vectorizer.index(texts_processed)

        metadatas = metadatas or ({} for _ in texts)
        if ids:
            docs = [Document(page_content=t, metadata=m, id=i) for t, m, i in zip(texts, metadatas, ids)]
        else:
            docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(vectorizer=vectorizer, docs=docs, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BM25Retriever:
        """
        Create a BM25Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        texts, metadatas, ids = zip(*((d.page_content, d.metadata, d.id) for d in documents))
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

        processed_query = tokenizer.tokenize([preprocess_func(query)])
        return_docs, scores = self.vectorizer.retrieve(processed_query, self.docs, k=self.k)

        return return_docs[0]


if __name__ == "__main__":
    # Test the BM25Retriever
    docs = [
        Document(page_content="The cat sat on the mat.", metadata={"id": 1}),
        Document(page_content="The dog barked at the cat.", metadata={"id": 2}),
        Document(page_content="Cats and dogs are great pets.", metadata={"id": 3}),
    ]
    retriever = BM25Retriever.from_documents(docs, k=2)
    results = retriever.invoke("cat pets")
    print(results)
    for doc in results:
        print(doc.page_content)
        print(doc.metadata)
