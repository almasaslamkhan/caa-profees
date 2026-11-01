import json
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import faiss
import pandas as pd
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from cdrc.common.embeddings_utils import get_embedding_models


def save_documents(documents: List[Document], vdb_dir_path: str):
    df = pd.DataFrame([doc.metadata for doc in documents])
    df["indexable_text"] = [doc.page_content for doc in documents]
    path = Path(vdb_dir_path) / "indexed_data.csv"
    df.to_csv(path, index=False)


def extract_indexable_text(example: Dict) -> str:
    return "\n\n".join([(elem["gold_justification"] + "\n" + elem["model_justification"]) for elem in example.values()])


def build_vdb(knowledge_base_file: str, model_name: str, vdb_dir_root: str):
    embeddings = get_embedding_models()[model_name]()
    index = faiss.IndexFlatIP(len(embeddings.embed_query("hello world")))  # dummy query to get the embedding size

    vector_store = FAISS(
        embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={}, distance_strategy="MAX_INNER_PRODUCT"
    )

    with open(knowledge_base_file, "r") as f:
        knowledge_base = json.load(f)

    kb_by_example_id = {}
    for category in knowledge_base:
        for example in knowledge_base[category]:
            if example["example_id"] not in kb_by_example_id:
                kb_by_example_id[example["example_id"]] = {}
            kb_by_example_id[example["example_id"]][category] = {k: v for k, v in example.items() if k != "example_id"}

    documents = []
    for example_id, example in kb_by_example_id.items():
        documents.append(
            Document(
                page_content=extract_indexable_text(example),
                metadata={"example_id": example_id, **example},
            )
        )

    if not documents:
        raise ValueError("No documents found to build the vector database.")
    print(f"Adding {len(documents)} documents to the vector database.")
    batch_size = 20
    for i in range(0, len(documents), batch_size):
        vector_store.add_documents(documents[i : i + batch_size], ids=[str(uuid4()) for _ in documents[i : i + batch_size]])
    print("Documents added to the vector database.")
    Path(vdb_dir_root).mkdir(parents=True, exist_ok=True)
    vdb_dir_path = str(Path(vdb_dir_root) / f"vdb_{Path(knowledge_base_file).stem}_{model_name}")
    vector_store.save_local(vdb_dir_path)
    save_documents(documents, vdb_dir_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge_base_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--vdb_dir_root", type=str, required=True)

    args = parser.parse_args()

    build_vdb(args.knowledge_base_file, args.model_name, args.vdb_dir_root)


if __name__ == "__main__":
    main()
