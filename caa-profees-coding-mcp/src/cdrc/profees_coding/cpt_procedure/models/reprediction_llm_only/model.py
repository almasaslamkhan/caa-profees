import json
import sqlite3
from argparse import Namespace
from typing import Any, Dict, List

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from cdrc.common.data.schema import Procedure
from cdrc.common.langchain_utils import PydanticOutputParserWithRetries
from cdrc.common.llm_utils import get_llm, track_tokens
from cdrc.common.utils import load_from_yaml
from cdrc.profees_coding.cpt_procedure.models.ner_rag.model import CPTProcedureGraphState, format_procedure, verification_post_processing
from cdrc.profees_coding.cpt_procedure.models.reprediction_llm_only.data_classes import CONFIG, PROMPTS
from cdrc.profees_coding.em.models.checkpoint import SqliteCheckpointer
from cdrc.profees_coding.em.models.utils import create_markdown_diff

CODE_DESCRIPTIONS = json.load(open("cdrc/profees_coding/cpt_procedure/rag/vdb/cpt_long_descriptions.json"))
CODE_LOOKUP = {i["code"]: i["description"] for i in CODE_DESCRIPTIONS}


# ------------------------ defining the graph state ---------------------------
class CPTProcedureRepredictionGraphState(CPTProcedureGraphState):
    previous_procedure: List[Procedure] = Field(None, description="List of previously predicted CPT procedures with code and description")
    reprediction_results: List[Dict[str, Any]] = Field(
        None, description="List of reprediction results including code, description, status, and justifications"
    )
    note_diff: str = Field(None, description="Markdown formatted diff between the current and previous note")
    procedure: List[Procedure] = Field(None, description="List of final CPT procedures with code and description")


class RepredictionResult(BaseModel):
    code: str = Field(..., description="The CPT procedure code that can be verified")
    descriptions: str = Field(..., description="The description of CPT procedure code that can be verified")
    status: str = Field(..., description="one of `kept`, `removed`, or `added")
    justifications: str = Field(..., description="explanation of why it was kept/removed/added based on change tracking")


class RepredictionResultList(BaseModel):
    results: List[RepredictionResult] = Field(..., description="List of reprediction results")


def initialization_node(state: CPTProcedureRepredictionGraphState, config: RunnableConfig) -> dict:
    previous_procedure = [{"code": p.code, "description": p.description, "justifications": p.justification} for p in state.get("procedure", "")]

    return {"previous_procedure": previous_procedure}


@track_tokens("model_usage")
def reprediction_node(state: CPTProcedureRepredictionGraphState, config: RunnableConfig) -> dict:
    model_name = config["configurable"][CONFIG.MODEL_NAME]
    model_kwargs = config["configurable"][CONFIG.MODEL_KWARGS]
    model = get_llm()[model_name](**model_kwargs)

    output_parser = PydanticOutputParserWithRetries(pydantic_object=RepredictionResultList, correction_model=model, num_retries=3)

    prompts = config["configurable"][CONFIG.PROMPTS][PROMPTS.REPREDICTION]
    system_prompt = SystemMessagePromptTemplate.from_template(prompts["system"], template_format="jinja2")
    user_prompt = HumanMessagePromptTemplate.from_template(prompts["user"], template_format="jinja2")

    note_diff, add, remove = create_markdown_diff(state.get("text", ""), state.get("soap_text_previous", ""))

    messages = [
        system_prompt.format(),
        user_prompt.format(
            note_diff=note_diff,
            draft_procedure=state.get("previous_procedure", ""),
            format_instructions=output_parser.get_format_instructions(),
        ),
    ]

    chain = model | output_parser
    response = chain.invoke(messages)
    reprediction_results = [result.model_dump() for result in response.results]

    logger.debug(f"Re-prediction results: {reprediction_results}")
    verified_procedures = [procedure for procedure in reprediction_results if procedure["status"] in ["kept", "added"]]
    verified_procedures = list(map(format_procedure, verified_procedures))

    # deduplicate verified procedures
    unique_verified_procedures, seen_code_set = verification_post_processing(verified_procedures)

    return {"note_diff": note_diff, "reprediction_results": reprediction_results, "procedure": unique_verified_procedures}


def build_reprediction_llm_only_graph(args: Namespace):

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
        },
    }

    workflow = StateGraph(CPTProcedureRepredictionGraphState)
    workflow.add_node("initialization", initialization_node)
    workflow.add_node("reprediction", reprediction_node)

    workflow.add_edge(START, "initialization")
    workflow.add_edge("initialization", "reprediction")
    workflow.add_edge("reprediction", END)

    conn = sqlite3.connect(args.checkpoint_path, check_same_thread=False)
    checkpointer = SqliteCheckpointer(conn, serde=JsonPlusSerializer(pickle_fallback=True))

    return workflow.compile(checkpointer=checkpointer), model_config
