import warnings
from pathlib import Path
from typing import Dict, List, Literal, TypedDict
from uuid import uuid4

import faiss
import pandas as pd
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from cdrc.common.embeddings_utils import get_embedding_models
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import check_and_reload_env, get_llm
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.em.data import get_default_reader
from cdrc.profees_coding.em.data.schema import ComplexityDetails, ComplexityDetailsOld, InputSchema, OutputSchema
from cdrc.profees_coding.em.models.medprompt_detailed_multi.data_classes import CONFIG, METADATA_KEYS, PROMPTS
from cdrc.profees_coding.em.models.utils import get_complexity_from_code, merge_config

check_and_reload_env()


class MDMResultOld(BaseModel):
    problem_complexity: ComplexityDetailsOld
    data_complexity: ComplexityDetailsOld
    risk_of_complications: ComplexityDetailsOld


class MDMResult(BaseModel):
    problem_complexity: ComplexityDetails
    data_complexity: ComplexityDetails
    risk_of_complications: ComplexityDetails


# Define the state for the graph
class State(TypedDict):
    text: List[str]
    problems: List[List[str]]
    results: List[MDMResult]
    gold_elements: List[MDMResult]
    medications: List[List[str]]
    orders_display: List[str]


# Define the logic for each node
def get_mdm_and_cot(state: State, config: Dict) -> State:
    model = get_llm()[config["configurable"][CONFIG.MODEL_NAME]](**config["configurable"][CONFIG.MODEL_KWARGS])
    prompt_key = PROMPTS.MEDPROMPT_OURS
    parser = PydanticOutputParserWithRetries(pydantic_object=MDMResult, correction_model=model, num_retries=3)
    system_prompt = SystemMessagePromptTemplate.from_template(config["configurable"][CONFIG.PROMPTS][prompt_key]["system"], template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(config["configurable"][CONFIG.PROMPTS][prompt_key]["user"], template_format="jinja2")
    chain = model | parser
    messages = []
    for text, problems, gold_elements, medications, orders_display in zip(
        state["text"], state["problems"], state["gold_elements"], state["medications"], state["orders_display"]
    ):
        additional_info = ""
        if problems:
            additional_info = "Problems addressed during the encounter:\n" + "\n".join(
                [f"- {problem}".replace("type=None", "") for problem in problems]
            )
            additional_info += "\n\n"
        # if [elem for elem in medications if elem.strip()] and not orders_display.strip():
        #     additional_info += "Medications the patient is on (not neccessarily related to this encounter):\n" + "\n".join(
        #         [f"- {medication}" for medication in medications]
        #     )
        #     additional_info += "\n\n"
        if orders_display.strip():
            additional_info += "Orders for the encounter:\n" + orders_display
            additional_info += "\n\n"

        user_prompt_kwargs = {
            "text": text,
            "additional_info": additional_info,
            "format_instructions": parser.get_format_instructions(),
            "mdm_elements_and_complexities": "\n".join([f"{k} : {v['level'].value}" for k, v in gold_elements.model_dump().items()]),
        }
        messages.append(
            [
                system_prompt.format(few_shot_examples=[]),
                user_prompt.format(**user_prompt_kwargs),
            ]
        )
    print("Initiating batch inference for CoT extraction.")
    results = chain.batch(messages, return_exceptions=True)
    print("Batch inference completed")
    results = [result if isinstance(result, MDMResult) else {} for result in results]
    return {"results": results}


def extract_indexable_text(text: str) -> str:
    return text


def save_documents(documents: List[Document], config: Dict):
    df = pd.DataFrame([doc.metadata for doc in documents])
    df["indexable_text"] = [doc.page_content for doc in documents]
    path = Path(config["configurable"][CONFIG.VDB_INDEX_PATH]) / "indexed_data.csv"
    df.to_csv(path, index=False)


def build_vdb(state: State, config: Dict) -> State:
    embeddings = get_embedding_models()[config["configurable"][CONFIG.EMBEDDING_MODEL]]()
    index = faiss.IndexFlatIP(len(embeddings.embed_query("hello world")))  # dummy query to get the embedding size

    vector_store = FAISS(
        embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={}, distance_strategy="MAX_INNER_PRODUCT"
    )

    documents = []
    for text, result, gold in zip(state["text"], state["results"], state["gold_elements"]):
        if result:
            documents.append(
                Document(
                    page_content=extract_indexable_text(text),
                    metadata={
                        METADATA_KEYS.MDM_COMPLEXITY: gold.model_dump_json(),
                        METADATA_KEYS.SOAP_NOTE: text,
                        METADATA_KEYS.CHAIN_OF_THOUGHT_REASONING: result.model_dump_json(),
                    },
                )
            )

    if not documents:
        raise ValueError("No documents found to build the vector database.")
    print(f"Adding {len(documents)} documents to the vector database.")
    batch_size = 20
    for i in range(0, len(documents), batch_size):
        vector_store.add_documents(documents[i : i + batch_size], ids=[str(uuid4()) for _ in documents[i : i + batch_size]])
    print("Documents added to the vector database.")
    vector_store.save_local(config["configurable"][CONFIG.VDB_INDEX_PATH])
    save_documents(documents, config)

    return state


def get_graph():
    # Create the graph
    graph = StateGraph(State)

    # Add nodes to the graph
    graph.add_node("get_mdm_and_cot", get_mdm_and_cot)
    graph.add_node("build_vdb", build_vdb)
    # Define the edges between nodes
    graph.add_edge(START, "get_mdm_and_cot")
    graph.add_edge("get_mdm_and_cot", "build_vdb")
    graph.add_edge("build_vdb", END)

    return graph.compile()


def format_intermediate_complexities(gold_output: OutputSchema) -> str:
    return (
        f"*problem_complexity*: {gold_output.problem_complexity.level.value}\n"
        f"*data_complexity*: {gold_output.data_complexity.level.value}\n"
        f"*risk_of_complications*: {gold_output.risk_of_complications.level.value}\n"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, default=str(Path(__file__).parent / "config/prompts.yaml"))
    parser.add_argument("--config_path", type=str, default=str(Path(__file__).parent / "config/runner.yaml"))

    args, unknown_args = parser.parse_known_args()

    config = load_from_yaml(args.config_path)
    args = merge_config(args, unknown_args, config)
    data_reader = get_default_reader()
    input_data, output_data = data_reader.read_data(args.input_path)
    print("Preprocessing args:", args)
    prompts = load_from_yaml(args.prompt_path)

    config = {"configurable": {**vars(args), "prompts": prompts}}

    app = get_graph()

    invokation_data = {"text": [], "problems": [], "gold_elements": [], "medications": [], "orders_display": []}
    for input, output in zip(input_data, output_data):
        if complexity := MDMResult(
            problem_complexity=output.problem_complexity, data_complexity=output.data_complexity, risk_of_complications=output.risk_of_complications
        ):
            invokation_data["text"].append(input.text)
            invokation_data["problems"].append(input.all_patient_problems)
            invokation_data["medications"].append(input.medications)
            invokation_data["orders_display"].append(input.orders_display)
            invokation_data["gold_elements"].append(complexity)
        else:
            warnings.warn(f"MDM complexity can't be retrieved for {output.cpt.code}")

    output_state = app.invoke(invokation_data, config=config)
    print(output_state)


if __name__ == "__main__":
    main()
