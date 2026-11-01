import pickle
from typing import Dict, List

import nltk
from langchain_community.vectorstores import FAISS
from loguru import logger

from cdrc.common.embeddings_utils import get_embedding_models
from cdrc.profees_coding.cpt_procedure.rag.data_classes import CONFIG  # This needs to be generalized to some common config keys.

nltk_data_dir = "cdrc/profees_coding/cpt_procedure/data/nltk_data"

if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)


def _load_faiss_obj(path, emb):
    faiss_vdb = FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
    logger.debug(f"Faiss vdb has been loaded from {path}.")
    return faiss_vdb


def _load_bm25_index_obj(path: str):
    with open(path, "rb") as f:
        bm25_obj = pickle.load(f)
    logger.debug(f"BM25 index has been loaded from {path}.")
    return bm25_obj


def get_vector_stores(config: Dict[str, str]):

    store_type = config["type"].lower()
    store_path = config["path"]

    try:

        if store_type == "faiss":
            emb_model_name = config[CONFIG.EMBEDDING_MODEL]
            emb_model = get_embedding_models()[emb_model_name]()
            return _load_faiss_obj(store_path, emb_model)

        elif store_type == "bm25":
            store_path = store_path.replace("{" + CONFIG.RETRIEVAL_FILTER + "}", config[CONFIG.RETRIEVAL_FILTER])
            # this is actually loading a bm25 retriever as langchain bm25 retriever is merely a wrapper of BM25Okapi instance
            return _load_bm25_index_obj(store_path)

        else:
            raise NotImplementedError(f"Vector store {store_type} is currently not supported.")

    except FileNotFoundError:

        raise FileNotFoundError(f"Vector store file for {store_type} not found in path: {store_path}")
