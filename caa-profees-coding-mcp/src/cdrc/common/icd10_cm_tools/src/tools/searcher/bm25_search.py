import os
import pickle
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import bm25s
import Stemmer
from loguru import logger

from cdrc.common.icd10_cm_tools.src.tools.searcher.base import BaseICD10Search, SearchCorpus, SearchResult

_stemmer = Stemmer.Stemmer("english")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _resolve_index_path(index_path: str | Path) -> Path:
    path = Path(str(index_path))
    if path.is_absolute():
        return path
    return Path(str(files("cdrc.common.icd10_cm_tools.data.source").joinpath(str(path))))


class BM25sICD10Search(BaseICD10Search):
    def __init__(
        self,
        corpus: SearchCorpus | None = None,
        index_path: Path | None = None,
        **kwargs: Any,
    ):
        if index_path is None:
            assert corpus is not None, "Provide either index_path or corpus"
            index_root = files("cdrc.common.icd10_cm_tools.data.source").joinpath(f"indices/{corpus.name}")
            index_root.mkdir(parents=True, exist_ok=True)
            self.index_path = Path(str(index_root / "bm25s_index.pkl"))
            try:
                self.load_index()
            except Exception as ex:
                print(f"Error reading index: {ex}")
                self.corpus = corpus
                self.create_index_from_data(**kwargs)
        else:
            self.index_path = _resolve_index_path(index_path)
            if corpus is not None:
                try:
                    self.load_index()
                except Exception as ex:
                    print(f"Error reading index: {ex}")
                    self.corpus = corpus
                    self.create_index_from_data(**kwargs)
            else:
                self.load_index()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"#codes={len(self.corpus.codes)}, " f"#docs={len(self.index)}, " f"index-path={self.index_path})"

    def tokenize(self, texts: list[str]) -> list[list[str]]:
        ntexts = [normalize_text(text) for text in texts]
        return bm25s.tokenize(
            ntexts,
            stopwords="en",
            stemmer=_stemmer,
            show_progress=False,
            return_ids=False,
        )

    def create_index_from_data(self, **kwargs: Any):
        docs = list(self.corpus.get_documents())
        self.bm25 = bm25s.BM25()
        print(f"BM25 indexing {len(docs)} docs")
        self.bm25.index(self.tokenize([doc.page_content for doc in docs]))
        self.index = [doc.metadata["code_idx"] for doc in docs]
        self.save_index()

    def save_index(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump((self.index, self.corpus, self.bm25), f)
        print(f"Saved index to {self.index_path}")

    def load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                self.index, self.corpus, self.bm25 = pickle.load(f)
            logger.debug(f"Loaded index from {self.index_path}")
            return True
        raise Exception(f"Index at {self.index_path} doesn't exist")

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        tokenized_query = self.tokenize([query])
        results, scores = self.bm25.retrieve(tokenized_query, k=min(top_k, len(self.index)), show_progress=False)
        code_to_best_score = {}

        for i in range(results.shape[1]):
            code_idx, score = self.index[results[0, i]], scores[0, i]
            if code_idx not in code_to_best_score or score > code_to_best_score[code_idx]:
                code_to_best_score[code_idx] = score

        return [
            SearchResult(code=self.corpus[code_idx], score=score)
            for code_idx, score in sorted(code_to_best_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]
