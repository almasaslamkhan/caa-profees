import json
import os
import pickle
from pathlib import Path
from typing import List, Literal
from uuid import uuid4

import faiss
import nltk
import pandas as pd
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from cdrc.common.embeddings_utils import get_embedding_models
from cdrc.profees_coding.cpt_procedure.rag.retriever.bm25_retriever import BM25Retriever

nltk_data_dir = "cdrc/profees_coding/cpt_procedure/data/nltk_data"

if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)


def save_documents(documents: List[Document], vdb_dir_path: str):
    df = pd.DataFrame([doc.metadata for doc in documents])
    df["indexable_text"] = [doc.page_content for doc in documents]
    path = Path(vdb_dir_path) / "indexed_data.csv"
    df.to_csv(path, index=False)


def create_documents(knowledge_base_file: str):
    with open(knowledge_base_file, "r") as f:
        knowledge_base = json.load(f)

    documents = []
    for code in knowledge_base:
        documents.append(
            Document(
                page_content=code["description"],
                metadata={
                    "description": code["description"],
                    "code": code["code"],
                },
            )
        )

    if not documents:
        raise ValueError("No documents found to build the vector database.")

    return documents


def build_vdb(
    knowledge_base_file: str,
    model_name: str,
    vdb_dir_root: str,
    vdb_type: Literal["faiss", "chroma"] = "faiss",
    collection_name: str = "default_collection",
):

    embeddings = get_embedding_models()[model_name]()
    Path(vdb_dir_root).mkdir(parents=True, exist_ok=True)
    vdb_dir_path = str(Path(vdb_dir_root) / f"vdb_{Path(knowledge_base_file).stem}_{model_name}")

    if vdb_type == "faiss":
        index = faiss.IndexFlatIP(len(embeddings.embed_query("hello world")))  # dummy query to get the embedding size
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy="COSINE",
            normalize_L2=True,
        )
    elif vdb_type == "chroma":
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=vdb_dir_path,
        )
    else:
        raise NotImplementedError(f"Backend {vdb_type} has not been implemented.")

    documents = create_documents(knowledge_base_file)

    print(f"Adding {len(documents)} documents to the vector database.")
    batch_size = 20
    for i in tqdm(range(0, len(documents), batch_size)):
        vector_store.add_documents(documents[i : i + batch_size], ids=[str(uuid4()) for _ in documents[i : i + batch_size]])
    print("Documents added to the vector database.")
    if vdb_type == "faiss":
        vector_store.save_local(vdb_dir_path)
    else:  # chromadb
        vector_store.persist()
    save_documents(documents, vdb_dir_path)


def build_bm25_index(knowledge_base_file: str, vdb_dir_root: str):

    documents = create_documents(knowledge_base_file)
    vdb_dir_path = str(Path(vdb_dir_root) / f"vdb_{Path(knowledge_base_file).stem}_bm25")
    os.makedirs(vdb_dir_path, exist_ok=True)
    bm25 = BM25Retriever.from_documents(documents)
    with open(f"{vdb_dir_path}/vdb.pickle", "wb") as f:
        pickle.dump(bm25, f)

    return "bm25 vector store(s) created successully"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vdb_type", type=str, required=True)
    parser.add_argument("--knowledge_base_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--vdb_dir_root", type=str, required=True)
    parser.add_argument("--nltk_data_path", type=str, required=False, default=str(Path.home() / "nltk_data"))
    args = parser.parse_args()
    if args.vdb_type == "bm25":
        build_bm25_index(args.knowledge_base_file, args.vdb_dir_root)
    elif args.vdb_type in ["faiss", "chroma"]:
        build_vdb(args.knowledge_base_file, args.model_name, args.vdb_dir_root, args.vdb_type)
    else:
        NotImplementedError(f"VDB Type {args.vdb_type} is not implemented yet.")


if __name__ == "__main__":
    main()
