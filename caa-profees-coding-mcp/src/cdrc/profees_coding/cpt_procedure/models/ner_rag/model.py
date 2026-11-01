import json
from argparse import Namespace
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.cpt_procedure.data.schema import Order, Problem, Procedure
from cdrc.profees_coding.cpt_procedure.models.ner_rag.data_classes import CONFIG, PROMPTS
from cdrc.profees_coding.cpt_procedure.models.utils import is_category_2_code, is_ekg_code, is_imaging_code, is_lab_code, is_vaccine_product
from cdrc.profees_coding.cpt_procedure.rag.retrieval_engine import RetrievalEngine

CODE_DESCRIPTIONS = json.load(open("cdrc/profees_coding/cpt_procedure/rag/vdb/cpt_long_descriptions.json"))
CODE_LOOKUP = {i["code"]: i["description"] for i in CODE_DESCRIPTIONS}


# ------------------------ defining the graph state ---------------------------
class ExtractedProcedure(BaseModel):
    code: str = Field(..., description="The extracted procedure code")
    descriptions: str = Field(..., description="The descriptions of the extracted procedures")
    justifications: str = Field(..., description="Model justification for why these procedures were extracted")


class ExtractedProcedureList(BaseModel):
    procedures: List[ExtractedProcedure] = Field(..., description="List of extracted procedures")


class CPTProcedureInputGraphState(TypedDict):
    example_id: Union[int, str, None]
    text: str
    patient_sex: str
    patient_age: int
    soap: Dict[str, Any]
    orders: Optional[List[Order]] = Field(default_factory=list, description="The list of orders for the encounter.")
    orders_display: Optional[str] = Field(description="The list of orders for the encounter displayed as a single string for convenience.")
    all_patient_problems: Optional[List[Problem]] = Field(None, description="The expanding list of problems of the patient.")


class CPTProcedureOutputGraphState(TypedDict):
    extracted_procedures: ExtractedProcedureList
    knowledge_enhancement_results: ExtractedProcedureList
    vdb_retrieval_results: Dict[str, List[Dict[str, Any]]]
    llm_retrieval_results: Dict[str, List[Dict[str, Any]]]
    code_retrieval_results: Dict[str, List[Dict[str, Any]]]
    code_verification_results: List[Dict[str, Any]]
    procedure: List[Procedure] = Field(default_factory=list, description="List of CPT Procedure codes")


class CPTProcedureGraphState(CPTProcedureInputGraphState, CPTProcedureOutputGraphState):
    pass


class VerificationResult(BaseModel):
    code: str = Field(..., description="The CPT procedure code that can be verified")
    descriptions: str = Field(..., description="The description of CPT procedure code that can be verified")
    justifications: str = Field(..., description="Model justification for why this procedure can or cannot be verified")
    verified: bool = Field(..., description="Whether the code can be verified")


class VerificationResultList(BaseModel):
    results: List[VerificationResult] = Field(..., description="List of verification results")


def format_procedure(verification_results: dict) -> Procedure:

    code = verification_results.get("code", "")
    description = CODE_LOOKUP.get(code, "")
    justifications = verification_results.get("justifications", "")

    return Procedure(code=code, description=description, system="CPT", justification=justifications)


def verification_post_processing(procedures: List[Procedure]):
    filtered_procedures = {p.code: p for p in procedures}
    results = []

    for code in filtered_procedures:
        if (
            not is_ekg_code(code)
            and not is_imaging_code(code)
            and not is_lab_code(code)
            and not is_category_2_code(code)
            and not is_vaccine_product(code)
        ):
            results.append(filtered_procedures[code])

    return results


def candidate_post_processing(candidate_procedures: List[dict]):
    candidate_procedures_dict = {p["code"]: p for p in candidate_procedures}
    results = []

    for code in candidate_procedures_dict:
        if not is_imaging_code(code) and not is_lab_code(code) and not is_category_2_code(code) and not is_vaccine_product(code):
            results.append(candidate_procedures_dict[code])

    return results


# ------------------------ defining the graph nodes ---------------------------
@track_tokens("model_usage")
def extraction_node(state: CPTProcedureGraphState, config: RunnableConfig) -> dict:
    model_name = config["configurable"][CONFIG.MODEL_NAME]
    model_kwargs = config["configurable"][CONFIG.MODEL_KWARGS]
    model = get_llm()[model_name](**model_kwargs)

    output_parser = PydanticOutputParserWithRetries(pydantic_object=ExtractedProcedureList, correction_model=model, num_retries=3)

    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.EXTRACTION]
    system_prompt = SystemMessagePromptTemplate.from_template(prompts["system"], template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(prompts["user"], template_format="jinja2")

    messages = [
        system_prompt.format(),
        user_prompt.format(
            soap_note=json.dumps(state["text"], indent=2),
            patient_age=state.get("patient_age", ""),
            patient_sex=state.get("patient_sex", ""),
            known_orders=state.get("orders_display", ""),
            format_instructions=output_parser.get_format_instructions(),
        ),
    ]

    chain = model | output_parser

    response = chain.invoke(messages)
    extraction_results = [p.model_dump() for p in response.procedures]
    logger.debug(f"Extraction results: {extraction_results}")

    # print("Extraction Results: ")
    # print(extraction_results)

    return {"extracted_procedures": extraction_results}


@track_tokens("model_usage")
def knowledge_enhancement_node(state: CPTProcedureGraphState, config: RunnableConfig) -> dict:
    model_name = config["configurable"][CONFIG.MODEL_NAME]
    model_kwargs = config["configurable"][CONFIG.MODEL_KWARGS]
    model = get_llm()[model_name](**model_kwargs)

    output_parser = PydanticOutputParserWithRetries(pydantic_object=ExtractedProcedureList, correction_model=model, num_retries=3)

    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.KNOWLEDGE_ENHANCEMENT]
    system_prompt = SystemMessagePromptTemplate.from_template(prompts["system"], template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(prompts["user"], template_format="jinja2")

    messages = [
        system_prompt.format(),
        user_prompt.format(
            soap_note=json.dumps(state["text"], indent=2),
            known_orders=state.get("orders_display", ""),
            format_instructions=output_parser.get_format_instructions(),
        ),
    ]

    chain = model | output_parser

    response = chain.invoke(messages)
    extraction_results = [p.model_dump() for p in response.procedures]
    logger.debug(f"Knowledge enhancement results: {extraction_results}")

    # print("Enhancement Results: ")
    # print(extraction_results)

    return {"knowledge_enhancement_results": extraction_results}


@track_tokens("model_usage")
def vdb_retrieval_node(state: CPTProcedureGraphState, config: RunnableConfig) -> dict:
    retrieval_engine = RetrievalEngine(
        retriever_config=config["configurable"][CONFIG.RETRIEVAL],
        query_config=config["configurable"][CONFIG.QUERY],
        reranker_config=config["configurable"][CONFIG.RERANK],
    )

    retrieval_results = {}
    for extracted_procedure in state["extracted_procedures"]:
        descriptions = extracted_procedure["descriptions"]
        # Prefer signed notes for context; gracefully fall back to draft text to avoid KeyErrors when missing.
        base_context = state.get("text", "") or state.get("soap")
        orders_context = state.get("orders_display", "")
        context = base_context + ("\n" + orders_context if orders_context else "")
        retrieval_results[descriptions] = [doc.metadata for doc in retrieval_engine.run(descriptions, context)]
        for item in retrieval_results[descriptions]:
            if "descriptions" not in item:
                item["descriptions"] = item.pop("description", "")

    # print("VDB Retrieval Results:", retrieval_results)

    return {"vdb_retrieval_results": retrieval_results}


@track_tokens("model_usage")
def llm_retrieval_node(state: CPTProcedureGraphState, config: RunnableConfig) -> dict:
    model_name = config["configurable"][CONFIG.MODEL_NAME]
    model_kwargs = config["configurable"][CONFIG.MODEL_KWARGS]
    model = get_llm()[model_name](**model_kwargs)

    output_parser = PydanticOutputParserWithRetries(pydantic_object=ExtractedProcedureList, correction_model=model, num_retries=3)

    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.RETRIEVAL]
    system_prompt = SystemMessagePromptTemplate.from_template(prompts["system"], template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(prompts["user"], template_format="jinja2")

    messages = []
    retrieval_results = {}

    for extracted_procedure in state["extracted_procedures"]:
        msg = [
            system_prompt.format(),
            user_prompt.format(
                code=extracted_procedure["code"],
                description=extracted_procedure["descriptions"],
                format_instructions=output_parser.get_format_instructions(),
            ),
        ]
        messages.append(msg)

    chain = model | output_parser
    responses = chain.batch(messages)

    for extracted_procedure, response in zip(state["extracted_procedures"], responses):
        retrieval_results[extracted_procedure["descriptions"]] = [doc.model_dump() for doc in response.procedures]

    # print("LLM Retrieval Results:", retrieval_results)

    return {"llm_retrieval_results": retrieval_results}


def retrieval_merging_node(state: CPTProcedureGraphState, config: RunnableConfig) -> dict:

    merged_results = {}
    for extracted_procedure in state["extracted_procedures"]:
        descriptions = extracted_procedure["descriptions"]
        vdb_results = state["vdb_retrieval_results"].get(descriptions, []) if state.get("vdb_retrieval_results") else []
        llm_results = state["llm_retrieval_results"].get(descriptions, []) if state.get("llm_retrieval_results") else []

        # Merge and deduplicate results
        combined = {item["code"]: item for item in vdb_results + llm_results}
        merged_results[descriptions] = list(combined.values())

        # Normalize code descriptions
        for item in merged_results[descriptions]:
            item["descriptions"] = CODE_LOOKUP.get(item["code"], item["descriptions"])

    # print("Merged Results: ", merged_results)

    return {"code_retrieval_results": merged_results}


@track_tokens("model_usage")
def verification_node(state: CPTProcedureGraphState, config: RunnableConfig) -> dict:
    model_name = config["configurable"][CONFIG.MODEL_NAME]
    model_kwargs = config["configurable"][CONFIG.MODEL_KWARGS]
    model = get_llm()[model_name](**model_kwargs)

    output_parser = PydanticOutputParserWithRetries(pydantic_object=VerificationResultList, correction_model=model, num_retries=3)

    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.VERIFICATION]
    system_prompt = SystemMessagePromptTemplate.from_template(prompts["system"], template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(prompts["user"], template_format="jinja2")

    candidate_code_dict = {}
    for model_code in state["extracted_procedures"]:
        rag_candidates = state["code_retrieval_results"][model_code["descriptions"]]

        if model_code["code"] not in candidate_code_dict:
            candidate_code_dict[model_code["code"]] = model_code

        for candidate in rag_candidates:
            if candidate["code"] not in candidate_code_dict:
                candidate_code_dict[candidate["code"]] = candidate

    for procedure in state["knowledge_enhancement_results"]:
        if procedure["code"] not in candidate_code_dict:
            candidate_code_dict[procedure["code"]] = procedure

    candidate_code_list = list(candidate_code_dict.values())
    candidate_code_list = candidate_post_processing(candidate_code_list)

    if candidate_code_list:
        messages = [
            system_prompt.format(),
            user_prompt.format(
                model_code=json.dumps(candidate_code_list, indent=2),
                patient_age=state.get("patient_age", ""),
                patient_sex=state.get("patient_sex", ""),
                # known_orders=state.get("orders_display", ""),
                soap_note=json.dumps(state["text"], indent=2),
                format_instructions=output_parser.get_format_instructions(),
            ),
        ]

        chain = model | output_parser
        response = chain.invoke(messages)

        verification_results = [result.model_dump() for result in response.results]

        logger.debug(f"Verification results: {verification_results}")
        verified_procedures = [procedure for procedure in verification_results if procedure["verified"]]
        verified_procedures = [procedure for procedure in verified_procedures if procedure.get("code", "") in CODE_LOOKUP]
        verified_procedures = list(map(format_procedure, verified_procedures))

        # deduplicate verified procedures
        unique_verified_procedures = verification_post_processing(verified_procedures)
    else:
        verification_results = []
        unique_verified_procedures = []

    return {"code_verification_results": verification_results, "procedure": unique_verified_procedures}


def build_ner_llm_rag_graph(args: Namespace):

    # set the prompts and model config
    prompts = load_from_yaml(args.prompt_path)[args.prompt_version]
    model_config = {
        "model_name": args.model_name,
        "model_kwargs": args.model_kwargs,
        "configurable": {
            "prompts": prompts,
            "model_name": args.model_name,
            "model_kwargs": args.model_kwargs,
            "additional_info": "",
            "full_features": "",
            "query": args.query if hasattr(args, "query") else None,
            "retrieval": args.retrieval if hasattr(args, "retrieval") else None,
            "rerank": args.rerank if hasattr(args, "rerank") else None,
        },
    }

    workflow = StateGraph(CPTProcedureGraphState)
    workflow.add_node("extractor", extraction_node)
    workflow.add_node("vdb_retriever", vdb_retrieval_node)
    workflow.add_node("llm_retriever", llm_retrieval_node)
    workflow.add_node("retrieval_merging", retrieval_merging_node)
    workflow.add_node("verifier", verification_node)
    workflow.add_node("knowledge_enhancer", knowledge_enhancement_node)

    workflow.add_edge(START, "extractor")
    workflow.add_edge(START, "knowledge_enhancer")
    workflow.add_edge("extractor", "vdb_retriever")
    workflow.add_edge("extractor", "llm_retriever")
    workflow.add_edge("knowledge_enhancer", "vdb_retriever")
    workflow.add_edge("knowledge_enhancer", "llm_retriever")
    workflow.add_edge("vdb_retriever", "retrieval_merging")
    workflow.add_edge("llm_retriever", "retrieval_merging")
    workflow.add_edge("retrieval_merging", "verifier")
    workflow.add_edge("verifier", END)

    return workflow.compile(), model_config


def build_ner_llm_graph(args: Namespace):

    # set the prompts and model config
    prompts = load_from_yaml(args.prompt_path)[args.prompt_version]
    model_config = {
        "model_name": args.model_name,
        "model_kwargs": args.model_kwargs,
        "configurable": {
            "prompts": prompts,
            "model_name": args.model_name,
            "model_kwargs": args.model_kwargs,
            "additional_info": "",
            "full_features": "",
            "query": args.query if hasattr(args, "query") else None,
            "retrieval": args.retrieval if hasattr(args, "retrieval") else None,
            "rerank": args.rerank if hasattr(args, "rerank") else None,
        },
    }

    workflow = StateGraph(CPTProcedureGraphState)
    workflow.add_node("extractor", extraction_node)
    workflow.add_node("llm_retriever", llm_retrieval_node)
    workflow.add_node("retrieval_merging", retrieval_merging_node)
    workflow.add_node("verifier", verification_node)
    workflow.add_node("knowledge_enhancer", knowledge_enhancement_node)

    workflow.add_edge(START, "extractor")
    workflow.add_edge(START, "knowledge_enhancer")
    workflow.add_edge("extractor", "llm_retriever")
    workflow.add_edge("knowledge_enhancer", "llm_retriever")
    workflow.add_edge("llm_retriever", "retrieval_merging")
    workflow.add_edge("retrieval_merging", "verifier")
    workflow.add_edge("verifier", END)

    return workflow.compile(), model_config


def build_ner_rag_graph(args: Namespace):

    # set the prompts and model config
    prompts = load_from_yaml(args.prompt_path)[args.prompt_version]
    model_config = {
        "model_name": args.model_name,
        "model_kwargs": args.model_kwargs,
        "configurable": {
            "prompts": prompts,
            "model_name": args.model_name,
            "model_kwargs": args.model_kwargs,
            "additional_info": "",
            "full_features": "",
            "query": args.query if hasattr(args, "query") else None,
            "retrieval": args.retrieval if hasattr(args, "retrieval") else None,
            "rerank": args.rerank if hasattr(args, "rerank") else None,
        },
    }

    workflow = StateGraph(CPTProcedureGraphState)
    workflow.add_node("extractor", extraction_node)
    workflow.add_node("vdb_retriever", vdb_retrieval_node)
    workflow.add_node("retrieval_merging", retrieval_merging_node)
    workflow.add_node("verifier", verification_node)
    workflow.add_node("knowledge_enhancer", knowledge_enhancement_node)

    workflow.add_edge(START, "extractor")
    workflow.add_edge(START, "knowledge_enhancer")
    workflow.add_edge("extractor", "vdb_retriever")
    workflow.add_edge("knowledge_enhancer", "vdb_retriever")
    workflow.add_edge("vdb_retriever", "retrieval_merging")
    workflow.add_edge("retrieval_merging", "verifier")
    workflow.add_edge("verifier", END)

    return workflow.compile(), model_config
