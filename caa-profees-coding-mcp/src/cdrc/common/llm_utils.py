import json
import os
import re
import threading
import time
from collections.abc import Mapping
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial, wraps
from importlib import resources
from os.path import isfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import openai
import requests
from dotenv import dotenv_values, load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI as DefaultChatOpenAI
from loguru import logger

from cdrc.common.langchain_utils import ChatOpenAIO1, ChatOpenAIWithRetries, get_openai_callback_with_time
from cdrc.common.utils import StringEnum


class LLM(StringEnum):
    # Local models
    GPT_4o = "openai/gpt-4o"
    GPT_4o_2024_05_13_ORACLE = "oracle-gpt-4o-2024-05-13"
    QWEN2_72B_INSTRUCT = "Qwen/Qwen2-72B-Instruct"
    META_LLAMA_3v1_70B_INSTRUCT = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    META_LLAMA_3v1_405B_INSTRUCT = "meta-llama/Meta-Llama-3.1-405B-Instruct"
    CAA_GPT_41_2025_04_14_ORACLE_DEV = "caa-gpt-4.1-2025-04-14-dev-aifactory"
    ORACLE_o1_PREVIEW_2024_09_12_DEV = "oracle-o1-preview-2024-09-12-dev"
    ORACLE_o1_MINI_2024_09_12_DEV = "oracle-o1-mini-2024-09-12-dev"
    ORACLE_o1_2024_12_17_DEV = "oracle-o1-2024-12-17-dev"
    ORACLE_o3_MINI_2025_01_31_DEV = "oracle-o3-mini-2025-01-31-dev"
    ORACLE_o4_2025_04_16_DEV = "oracle-o4-2025-04-16-dev"
    ORACLE_o4_MINI_2025_04_16_DEV = "oracle-o4-mini-2025-04-16-dev"
    GPT_5_2025_08_07_ORACLE_DEV = "caa-gpt-5-2025-08-07-dev-aifactory"
    GPT_51_2025_11_13_ORACLE_DEV = "caa-gpt-5.1-2025-11-13-dev-aifactory"
    GPT_52_2025_12_11_ORACLE_DEV = "caa-gpt-5.2-2025-12-11-dev-aifactory"
    GPT_54_2026_03_05_ORACLE_DEV = "caa-gpt-5.4-2026-03-05-dev-aifactory"
    GPT_51_2025_11_13_ORACLE_DEV_TOOLS = "caa-gpt-5.1-2025-11-13-dev-aifactory-tools"
    GPT_52_2025_12_11_ORACLE_DEV_TOOLS = "caa-gpt-5.2-2025-12-11-dev-aifactory-tools"
    # SDS models
    GPT_4o_2024_05_13_ORACLE_SDS = "oracle-gpt-4o-2024-05-13-sds"
    GPT_4o_2024_08_06_ORACLE_SDS = "oracle-gpt-4o-2024-08-06-dev-sds"
    GPT_4o_2024_11_20_ORACLE_DEV = "oracle-gpt-4o-2024-11-20-dev"
    GPT_4o_mini_2024_07_18_DEV_SDS = "oracle-gpt-4o-mini-2024-07-18-dev-sds"
    ORACLE_o1_PREVIEW_2024_09_12_DEV_SDS = "oracle-o1-preview-2024-09-12-dev-sds"
    ORACLE_o1_MINI_2024_09_12_DEV_SDS = "oracle-o1-mini-2024-09-12-dev-sds"
    GPT_41_2025_04_14_ORACLE_DEV_SDS = "caa-gpt-4.1-2025-04-14-dev-sds"
    GPT_41_2025_04_14_ORACLE_GLOBAL = "oracle-gpt-4.1-2025-04-14-global"
    GPT_41_MINI_2025_04_14_ORACLE_DEV = "oracle-gpt-4.1-mini-2025-04-14-dev"
    GPT_41_NANO_2025_04_14_ORACLE_DEV = "oracle-gpt-4.1-nano-2025-04-14-dev"
    GPT_5_2025_08_07_ORACLE_DEV_SDS = "caa-gpt-5-2025-08-07-dev-sds"
    GPT_5_MINI_2025_08_07_ORACLE_DEV_SDS = "caa-gpt-5-mini-2025-08-07-dev-sds"
    GPT_51_2025_11_13_ORACLE_DEV_SDS = "caa-gpt-5.1-2025-11-13-dev-sds"
    GPT_52_2025_12_11_ORACLE_DEV_SDS = "caa-gpt-5.2-2025-12-11-dev-sds"
    GPT_54_2026_03_05_ORACLE_DEV_SDS = "caa-gpt-5.4-2026-03-05-dev-sds"
    GPT_54_2026_03_17_mini_ORACLE_DEV_SDS = "caa-gpt-5.4-mini-2026-03-17-dev-sds"
    GPT_54_2026_03_17_nano_ORACLE_DEV_SDS = "caa-gpt-5.4-nano-2026-03-17-dev-sds"
    GPT_55_2026_04_23_ORACLE_DEV_SDS = "caa-gpt-5.5-2026-04-23-dev-sds"
    GPT_51_2025_11_13_ORACLE_DEV_SDS_TOOLS = "caa-gpt-5.1-2025-11-13-dev-sds-tools"
    GPT_52_2025_12_11_ORACLE_DEV_SDS_TOOLS = "caa-gpt-5.2-2025-12-11-dev-sds-tools"
    # Production models
    PROFEES_GPT_4o_2024_05_13_DEV = "profees-gpt-4o-2024-05-13-dev"
    PROFEES_GPT_4o_2024_05_13_PROD = "profees-gpt-4o-2024-05-13-prod"
    CAA_GPT_41_2025_04_14_DEV = "caa-gpt-4.1-2025-04-14-dev"
    CAA_GPT_41_2025_04_14_PROD = "caa-gpt-4.1-2025-04-14-prod"
    CAA_GPT_5_2025_08_07_DEV = "caa-gpt-5-2025-08-07-dev"
    CAA_GPT_5_2025_08_07_PROD = "caa-gpt-5-2025-08-07-prod"
    CAA_GPT_51_2025_11_13_DEV = "caa-gpt-5.1-2025-11-13-dev"
    CAA_GPT_51_2025_11_13_PROD = "caa-gpt-5.1-2025-11-13-prod"
    CAA_GPT_52_2025_12_11_DEV = "caa-gpt-5.2-2025-12-11-dev"
    CAA_GPT_52_2025_12_11_PROD = "caa-gpt-5.2-2025-12-11-prod"
    CAA_GPT_54_2026_03_05_DEV = "caa-gpt-5.4-2026-03-05-dev"
    CAA_GPT_54_2026_03_05_PROD = "caa-gpt-5.4-2026-03-05-prod"
    CAA_GPT_54_mini_2026_03_17_DEV = "caa-gpt-5.4-mini-2026-03-17-dev"
    CAA_GPT_54_mini_2026_03_17_PROD = "caa-gpt-5.4-mini-2026-03-17-prod"


ModelCatalogue = dict[LLM, Callable[..., BaseLanguageModel]]


def is_env_file_empty():
    env_path = Path(".env")

    # Check if file exists and is empty
    if env_path.exists():
        return env_path.stat().st_size == 0

    return True  # Consider non-existent file as empty


def check_and_reload_env():
    # # Get all variables defined in .env file
    # env_vars = dotenv_values(".env")

    # Get .env file path (assuming it's in the same directory)
    env_path = Path(".env")

    try:
        # Get the last modification time of the file
        last_modified = datetime.fromtimestamp(env_path.stat().st_mtime)

        # Calculate the time threshold (24 hours ago)
        threshold = datetime.now() - timedelta(hours=24)

        # Check if file was modified more than 24 hours ago to reset them
        if last_modified < threshold:
            # Clear ODA_BEARER_TOKEN from environment if it exists
            if "ODA_BEARER_TOKEN" in os.environ:
                os.environ.pop("ODA_BEARER_TOKEN")

            # Read all lines from .env file
            with open(".env", "r") as f:
                lines = f.readlines()

            # Write back all lines except ODA_BEARER_TOKEN
            with open(".env", "w") as f:
                for line in lines:
                    if not line.startswith("ODA_BEARER_TOKEN="):
                        f.write(line)
        else:
            # load env variables
            load_dotenv(Path(".env"), override=True)
    except FileNotFoundError:
        logger.warning("File .env does not exist; continuing with process environment variables.")
    except OSError:
        logger.warning("Unable to access .env; continuing with process environment variables.", exc_info=True)


def set_env_variable(key, value):
    # save the bearer token only if there is non
    if is_env_file_empty():
        # Update the .env file
        with open(".env", "a") as f:
            f.write(f"\n{key}={value}")

    check_and_reload_env()

    # Update current session
    os.environ[key] = value


def get_bearer_token(proxy: str) -> str:
    # Load environment variables from .env file
    check_and_reload_env()

    if not (os.getenv("ODA_USERNAME") and os.getenv("ODA_PASSWORD")) and not os.getenv("ODA_BEARER_TOKEN"):
        raise ValueError(
            "Either ODA username and password or bearer token are required to query the model."
            "Please either set ODA_USERNAME and ODA_PASSWORD, or alternatively set ODA_BEARER_TOKEN environment variables"
        )

    if os.environ.get("ODA_BEARER_TOKEN"):
        return os.environ.get("ODA_BEARER_TOKEN")

    if proxy is None:
        proxies = {"http": "", "https": ""}
    else:
        proxies = {"http": proxy, "https": proxy}

    if not os.getenv("ODA_USERNAME") or not os.getenv("ODA_PASSWORD"):
        raise ValueError(
            "ODA username and password are required to get the bearer token for using the model,"
            "please set ODA_USERNAME and ODA_PASSWORD environment variables"
        )

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": "Basic TExNUmVzb3VyY2VTZXJ2ZXIxX0FQUElEOjU4MTQ1ZjIzLWRhMGEtNDhiNC04MjExLWU4NWMwNzdjYjAzMQ==",
    }

    data = {
        "scope": "http://LLMResourceServer1.digitalassistant.us-phoenix-1.oci.oc-test.com/management-api/v1",
        "grant_type": "password",
        "username": os.getenv("ODA_USERNAME"),
        "password": os.getenv("ODA_PASSWORD"),
    }

    response = requests.post(
        "https://idcs-30d0b9fa41c34d5abb4733cead145c80.identity.pint.oc9qadev.com:443/oauth2/v1/token",
        headers=headers,
        data=data,
        proxies=proxies,
    )

    try:
        token = response.json()["access_token"]
        set_env_variable("ODA_BEARER_TOKEN", token)
        return token
    except Exception:
        logger.warning("Could not get bearer token.")
        raise


def get_azure_openai_key(region: Literal["eastus", "eastus2", "northcentralus"]):
    check_and_reload_env()
    return os.getenv(f"AZURE_OPENAI_{region.upper()}_API_KEY")


def read_environment_file(filename):
    result = None
    if isfile(filename):
        with open(filename, "r") as env_file:
            result = env_file.read().strip()
    return result


def get_llm(proxy: str = None, enable_rate_limit_retries: bool = True) -> ModelCatalogue:
    """Get a dictionary of LLM implementations.

    Args:
        proxy: Optional proxy server to use for requests
        enable_rate_limit_retries: If True, enables infinite rate limit retries for ChatOpenAI

    Returns:
        Dictionary mapping LLM types to their implementations
    """
    # Override ChatOpenAI class if rate limit retries are enabled
    if enable_rate_limit_retries:
        ChatOpenAI = ChatOpenAIWithRetries
    else:
        ChatOpenAI = DefaultChatOpenAI

    return {
        LLM.GPT_4o: partial(
            ChatOpenAI,
            model_name="gpt-4o-2024-05-13",
            openai_proxy=proxy,
        ),
        LLM.GPT_4o_2024_05_13_ORACLE: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000",
                azure_ad_token_provider=partial(get_bearer_token, proxy=proxy) or "EMPTY",
                azure_deployment="oracle-gpt-4o-2024-05-13",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_5_2025_08_07_ORACLE_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000",
                azure_ad_token_provider=partial(get_bearer_token, proxy=proxy) or "EMPTY",
                azure_deployment="caa-gpt-5-2025-08-07-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_51_2025_11_13_ORACLE_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000",
                azure_ad_token_provider=partial(get_bearer_token, proxy=proxy) or "EMPTY",
                azure_deployment="caa-gpt-5.1-2025-11-13-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_51_2025_11_13_ORACLE_DEV_TOOLS: partial(
            AzureChatOpenAI,
            azure_deployment="caa-gpt-5.1-2025-11-13-dev",
            api_version="2025-11-13",
            azure_endpoint="http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000",
            api_key=get_azure_openai_key(region="new") or "EMPTY",
            tools=[{"type": "web_search"}],
            default_headers={"X-model": "caa-gpt-5.1-2025-11-13-dev"},
        ),
        LLM.GPT_52_2025_12_11_ORACLE_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000",
                azure_ad_token_provider=partial(get_bearer_token, proxy=proxy) or "EMPTY",
                azure_deployment="gpt-5.2-2025-12-11",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_54_2026_03_05_ORACLE_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000",
                azure_ad_token_provider=partial(get_bearer_token, proxy=proxy) or "EMPTY",
                azure_deployment="gpt-5.4-2026-03-05",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_52_2025_12_11_ORACLE_DEV_TOOLS: partial(
            AzureChatOpenAI,
            azure_deployment="gpt-5.2-2025-12-11",
            api_version="2025-11-13",
            azure_endpoint="http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000",
            api_key=get_azure_openai_key(region="new") or "EMPTY",
            tools=[{"type": "web_search"}],
            default_headers={"X-model": "gpt-5.2-2025-12-11"},
        ),
        LLM.GPT_4o_2024_05_13_ORACLE_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://127.0.0.1:8421",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="oracle-gpt-4o-2024-05-13",
            ).chat.completions,
        ),
        LLM.GPT_4o_2024_08_06_ORACLE_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://127.0.0.1:8421",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="oracle-gpt-4o-2024-08-06-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_4o_mini_2024_07_18_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://127.0.0.1:8421",
                api_key=get_azure_openai_key(region="northcentralus") or "EMPTY",
                azure_deployment="oracle-gpt-4o-mini-2024-07-18-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.ORACLE_o1_PREVIEW_2024_09_12_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000",
                azure_ad_token_provider=partial(get_bearer_token, proxy=proxy) or "EMPTY",
                azure_deployment="oracle-o1-preview-2024-09-12-dev",
            ).chat.completions,
            openai_proxy=proxy,
            temperature=1.0,
        ),
        LLM.CAA_GPT_41_2025_04_14_ORACLE_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000",
                azure_ad_token_provider=partial(get_bearer_token, proxy=proxy) or "EMPTY",
                azure_deployment="caa-gpt-4.1-2025-04-14-dev",
            ).chat.completions,
            openai_proxy=proxy,
            temperature=1.0,
        ),
        LLM.ORACLE_o1_PREVIEW_2024_09_12_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://127.0.0.1:8421",
                api_key=get_azure_openai_key(region="eastus2") or "EMPTY",
                azure_deployment="oracle-o1-preview-2024-09-12-dev",
            ).chat.completions,
            openai_proxy=proxy,
            temperature=1.0,
        ),
        LLM.ORACLE_o1_MINI_2024_09_12_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000",
                azure_ad_token_provider=partial(get_bearer_token, proxy=proxy) or "EMPTY",
                azure_deployment="oracle-o1-mini-2024-09-12-dev",
            ).chat.completions,
            openai_proxy=proxy,
            temperature=1.0,
        ),
        LLM.ORACLE_o1_MINI_2024_09_12_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://127.0.0.1:8421",
                api_key=get_azure_openai_key(region="eastus2") or "EMPTY",
                azure_deployment="oracle-o1-mini-2024-09-12-dev",
            ).chat.completions,
            openai_proxy=proxy,
            temperature=1.0,
        ),
        LLM.QWEN2_72B_INSTRUCT: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            openai_api_base="http://127.0.0.1:18889/v1",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            openai_proxy=proxy,
        ),
        LLM.META_LLAMA_3v1_70B_INSTRUCT: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            openai_api_base="http://144.25.31.124:8492/v1",
            model="meta-llama/Meta-Llama-3.1-405B-Instruct",
            openai_proxy=proxy,
        ),
        LLM.META_LLAMA_3v1_405B_INSTRUCT: partial(
            ChatOpenAI,
            model="meta-llama/Meta-Llama-3.1-405B-Instruct",
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:19997/v1",
            openai_proxy=proxy,
        ),
        LLM.GPT_4o_2024_11_20_ORACLE_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://127.0.0.1:8421",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="oracle-gpt-4o-2024-11-20-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_41_2025_04_14_ORACLE_GLOBAL: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://127.0.0.1:8421",
                api_key=get_azure_openai_key(region="eastus2") or "EMPTY",
                azure_deployment="oracle-gpt-4.1-2025-04-14-global",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_41_2025_04_14_ORACLE_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://10.30.9.117:32501/",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="caa-gpt-4.1-2025-04-14-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_5_2025_08_07_ORACLE_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-10-21",
                azure_endpoint="http://10.30.9.117:32627",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="caa-gpt-5-2025-08-07-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_5_MINI_2025_08_07_ORACLE_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://10.30.9.117:32627",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="caa-gpt-5-mini-2025-08-07-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_51_2025_11_13_ORACLE_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://10.30.9.117:32501",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="caa-gpt-5.1-2025-11-13-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_51_2025_11_13_ORACLE_DEV_SDS_TOOLS: partial(
            AzureChatOpenAI,
            azure_deployment="caa-gpt-5.1-2025-11-13-dev",
            api_version="2025-11-13",
            azure_endpoint="http://10.30.9.117:32501",
            api_key=get_azure_openai_key(region="new") or "EMPTY",
            tools=[{"type": "web_search"}],
            default_headers={"X-model": "caa-gpt-5.1-2025-11-13-dev"},
        ),
        LLM.GPT_52_2025_12_11_ORACLE_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://10.30.9.117:32501",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="caa-gpt-5.2-2025-12-11-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_52_2025_12_11_ORACLE_DEV_SDS_TOOLS: partial(
            AzureChatOpenAI,
            azure_deployment="caa-gpt-5.2-2025-12-11-dev",
            api_version="2025-11-13",
            azure_endpoint="http://10.30.9.117:32501",
            api_key=get_azure_openai_key(region="new") or "EMPTY",
            tools=[{"type": "web_search"}],
            default_headers={"X-model": "caa-gpt-5.2-2025-12-11-dev"},
        ),
        LLM.GPT_54_2026_03_05_ORACLE_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://10.30.9.117:32501",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="caa-gpt-5.4-2026-03-05-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_54_2026_03_17_mini_ORACLE_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://10.30.9.117:32501",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="caa-gpt-5.4-mini-2026-03-17-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_54_2026_03_17_nano_ORACLE_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://10.30.9.117:32501",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="caa-gpt-5.4-nano-2026-03-17-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_55_2026_04_23_ORACLE_DEV_SDS: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://10.30.9.117:32501",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="caa-gpt-5.5-2026-04-23-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_41_MINI_2025_04_14_ORACLE_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://10.30.9.117:32501/",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="oracle-gpt-4.1-mini-2025-04-14-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.GPT_41_NANO_2025_04_14_ORACLE_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2025-11-13",
                azure_endpoint="http://10.30.9.117:32501/",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="oracle-gpt-4.1-nano-2025-04-14-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.ORACLE_o1_2024_12_17_DEV: partial(
            ChatOpenAIO1,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint="http://10.30.9.117:32501/",
                api_key=get_azure_openai_key(region="new") or "EMPTY",
                azure_deployment="oracle-o1-2024-12-17-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.ORACLE_o3_MINI_2025_01_31_DEV: partial(
            ChatOpenAIO1,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint="http://10.30.9.117:32501/",
                api_key=get_azure_openai_key(region="eastus2") or "EMPTY",
                azure_deployment="oracle-o3-mini-2025-01-31-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.ORACLE_o4_MINI_2025_04_16_DEV: partial(
            ChatOpenAIO1,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint="http://10.30.9.117:32501/",
                api_key=get_azure_openai_key(region="eastus2") or "EMPTY",
                azure_deployment="oracle-o4-mini-2025-04-16-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.ORACLE_o4_2025_04_16_DEV: partial(
            ChatOpenAIO1,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint="http://10.30.9.117:32501/",
                api_key=get_azure_openai_key(region="eastus2") or "EMPTY",
                azure_deployment="oracle-o4-2025-04-16-dev",
            ).chat.completions,
            openai_proxy=proxy,
        ),
        LLM.PROFEES_GPT_4o_2024_05_13_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="profees-gpt-4o-2024-05-13-dev",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.PROFEES_GPT_4o_2024_05_13_PROD: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2023-05-15",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="profees-gpt-4o-2024-05-13-prod",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_41_2025_04_14_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-4.1-2025-04-14-dev",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_41_2025_04_14_PROD: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2023-05-15",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-4.1-2025-04-14-prod",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_5_2025_08_07_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-5-2025-08-07-dev",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_5_2025_08_07_PROD: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2023-05-15",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-5-2025-08-07-prod",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_51_2025_11_13_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-5.1-2025-11-13-dev",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_51_2025_11_13_PROD: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-5.1-2025-11-13-prod",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_52_2025_12_11_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-5.2-2025-12-11-dev",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_52_2025_12_11_PROD: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-5.2-2025-12-11-prod",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_54_2026_03_05_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-5.4-2026-03-05-dev",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_54_2026_03_05_PROD: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-5.4-2026-03-05-prod",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_54_mini_2026_03_17_DEV: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-5.4-mini-2026-03-17-dev",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
        LLM.CAA_GPT_54_mini_2026_03_17_PROD: partial(
            ChatOpenAI,
            openai_api_key="EMPTY",
            client=openai.AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
                api_key="EMPTY",
                azure_deployment="caa-gpt-5.4-mini-2026-03-17-prod",
                default_headers={"Service-Name": "profees", "service-name": "profees"},
            ).chat.completions,
            openai_proxy=proxy,
            timeout=90,
        ),
    }


# Load model pricing data
with resources.files("cdrc.common").joinpath("model_prices.json").open("r") as f:
    _MODEL_DATA: Dict[str, Dict[str, Any]] = json.load(f)

# Build a price table and normalization rules dynamically
PRICES: Dict[str, Dict[str, float]] = {}
_NORMALIZE_RULES: List[Tuple[re.Pattern, str]] = []

for model_key, data in _MODEL_DATA.items():
    price_info = data.get("price", {})
    PRICES[model_key] = price_info

    pattern = data.get("match_pattern")
    if pattern:
        try:
            rx = re.compile(pattern)
            _NORMALIZE_RULES.append((rx, model_key))
        except re.error as e:
            raise ValueError(f"Invalid regex in match_pattern for {model_key}: {e}")


def normalize_model_for_pricing(model: Optional[str]) -> Optional[str]:
    """
    Return canonical model key if pattern matched, otherwise None.
    """
    if not model:
        return None
    s = model.strip()

    for rx, key in _NORMALIZE_RULES:
        if rx.search(s):
            return key

    # simple substring fallback (still no pricing fallback)
    s_lower = s.lower()
    for key in PRICES:
        if key in s_lower:
            return key

    return None


def get_price_table(model: Optional[str]) -> Optional[Dict[str, float]]:
    """
    Return the pricing table for the model if known, else None.
    """
    key = normalize_model_for_pricing(model)
    if key and key in PRICES:
        return PRICES[key]
    return None


def cost_from_usage(model: Optional[str], usage: Dict[str, int]) -> float:
    """
    Compute USD cost given token usage.
    Returns 0.0 if the model has no defined pricing.
    """
    p = get_price_table(model)
    if not p:
        return 0.0  # Unknown model → cost = 0

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return 0

    pt = to_int(usage.get("prompt_tokens", 0))
    ct = to_int(usage.get("completion_tokens", 0))
    cpt = to_int(usage.get("prompt_tokens_cached"))
    rzt = to_int(usage.get("reasoning_tokens", 0))

    input_cost = pt * p.get("input", 0.0)
    cached_cost = cpt * (p.get("input_cached_tokens", p.get("input_cache_read", 0.0)))
    output_cost = ct * p.get("output", 0.0)
    reasoning_cost = rzt * (p.get("output_reasoning_tokens", p.get("output_reasoning", 0.0)))
    return input_cost + cached_cost + output_cost + reasoning_cost


def attach_usage_to_tracker(usage: Dict[str, int]):
    """
    Attach detailed usage info (with cached/reasoning) to any active TokenTracker.
    """
    for inst in list(TokenTracker._instances.values()):
        if hasattr(inst, "_local"):
            inst._local.extra_usage = usage


@dataclass
class TokenStats:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_cached: int = 0
    reasoning_tokens: int = 0
    total_cost: float = 0
    successful_requests: int = 0
    requests_time_in_seconds: List[float] = field(default_factory=list)
    rate_limit_delays: List[float] = field(default_factory=list)
    init_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update_from_callback(self, cb, model: Optional[str] = None, extra_usage: Optional[Dict[str, int]] = None):
        """
        Merge LangChain callback counters and compute cost if LangChain didn't.
        """
        with self._lock:
            # aggregate tokens from LC callback
            self.total_tokens += getattr(cb, "total_tokens", 0)
            self.prompt_tokens += getattr(cb, "prompt_tokens", 0)
            self.completion_tokens += getattr(cb, "completion_tokens", 0)
            self.prompt_tokens_cached += getattr(cb, "prompt_tokens_cached", 0)
            self.reasoning_tokens += getattr(cb, "reasoning_tokens", 0)

            # Prefer LC-computed cost if non-zero
            cb_cost = float(getattr(cb, "total_cost", 0.0) or 0.0)

            # If callback didn't price (unknown model), compute locally.
            if cb_cost == 0.0:
                # Build a usage view combining callback + extra usage (cached/reasoning)
                usage = {
                    "prompt_tokens": getattr(cb, "prompt_tokens", 0),
                    "completion_tokens": getattr(cb, "completion_tokens", 0),
                    "prompt_tokens_cached": getattr(cb, "prompt_tokens_cached", 0),
                    "reasoning_tokens": getattr(cb, "reasoning_tokens", 0),
                }
                if extra_usage:
                    # take the max so we don't undercount if extra_usage captured more detail
                    for k in ("prompt_tokens", "completion_tokens", "prompt_tokens_cached", "reasoning_tokens"):
                        usage[k] = max(int(usage.get(k, 0)), int(extra_usage.get(k, 0)))
                cb_cost = cost_from_usage(model, usage)

            self.total_cost += cb_cost
            self.successful_requests += getattr(cb, "successful_requests", 0)

            if hasattr(cb, "requests_time_in_seconds") and getattr(cb, "requests_time_in_seconds"):
                self.requests_time_in_seconds.extend(cb.requests_time_in_seconds)
            if hasattr(cb, "rate_limit_delays") and getattr(cb, "rate_limit_delays"):
                self.rate_limit_delays.extend(cb.rate_limit_delays)

    def __str__(self):
        return (
            f"Tokens - Total: {self.total_tokens}, "
            f"Prompt: {self.prompt_tokens}, "
            f"Completion: {self.completion_tokens}, "
            f"Prompt Cached: {self.prompt_tokens_cached}, "
            f"Reasoning: {self.reasoning_tokens}\n"
            f"Cost: ${self.total_cost:.4f}, "
            f"Successful Requests: {self.successful_requests}\n"
            f"Request Time: {self.requests_time_in_seconds}\n"
            f"Rate Limit Delays: {self.rate_limit_delays}"
        )

    # Add methods to support serialization
    def to_dict(self):
        """Convert TokenStats to a serializable dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_tokens_cached": self.prompt_tokens_cached,
            "reasoning_tokens": self.reasoning_tokens,
            "total_cost": self.total_cost,
            "successful_requests": self.successful_requests,
            "requests_time_in_seconds": self.requests_time_in_seconds,
            "rate_limit_delays": self.rate_limit_delays,
            "init_time": self.init_time,
        }

    @classmethod
    def from_dict(cls, data):
        """Create TokenStats from a dictionary."""
        stats = cls()
        stats.total_tokens = data["total_tokens"]
        stats.prompt_tokens = data["prompt_tokens"]
        stats.completion_tokens = data["completion_tokens"]
        stats.prompt_tokens_cached = data["prompt_tokens_cached"]
        stats.reasoning_tokens = data["reasoning_tokens"]
        stats.total_cost = data["total_cost"]
        stats.successful_requests = data["successful_requests"]
        stats.requests_time_in_seconds = data["requests_time_in_seconds"]
        stats.rate_limit_delays = data["rate_limit_delays"]
        stats.init_time = data["init_time"]
        return stats


class TokenTracker(ContextDecorator):
    _instances: dict = {}
    _lock: threading.Lock = threading.Lock()  # Class-level lock for instance management

    def __new__(cls, category: str):
        with cls._lock:  # Thread-safe instance creation
            if category not in cls._instances:
                instance = super().__new__(cls)
                instance.stats = TokenStats()
                instance.category = category
                instance._local = threading.local()  # thread-local storage
                cls._instances[category] = instance
            return cls._instances[category]

    def __enter__(self):
        self._local.cb_manager = get_openai_callback_with_time()
        self._local.callback = self._local.cb_manager.__enter__()
        return self

    def __exit__(self, *exc):
        self._local.cb_manager.__exit__(*exc)
        model = getattr(self._local, "llm_model", None)
        extra_usage = getattr(self._local, "extra_usage", None)  # <= cached & reasoning info
        self.stats.update_from_callback(self._local.callback, model=model, extra_usage=extra_usage)
        # clear it so the next block starts fresh
        if hasattr(self._local, "extra_usage"):
            delattr(self._local, "extra_usage")

    @classmethod
    def get_stats(cls, category: str) -> TokenStats:
        with cls._lock:
            if category not in cls._instances:
                # Create a new TokenStats directly without going through TokenTracker constructor
                stats = TokenStats()
                # Create and store the instance manually
                instance = super(TokenTracker, cls).__new__(cls)
                instance.stats = stats
                instance.category = category
                instance._local = threading.local()
                cls._instances[category] = instance
                return stats
            return cls._instances[category].stats

    @classmethod
    def remove_instance(cls, category: str):
        """Removes the TokenTracker instance associated with the given category."""
        with cls._lock:
            if category in cls._instances:
                del cls._instances[category]

    @classmethod
    def get_all_stats(cls) -> dict:
        """Get all token stats as a serializable dictionary."""
        with cls._lock:
            return {category: stats.stats.to_dict() for category, stats in cls._instances.items()}

    @classmethod
    def restore_stats(cls, stats_dict: dict):
        """Restore token stats from a dictionary. Careful with transient instances."""
        with cls._lock:
            logger.warning("Warning: Restoring stats might recreate tracker instances for potentially transient categories.")
            for category, stats_data in stats_dict.items():
                if category not in cls._instances:
                    instance = super(TokenTracker, cls).__new__(cls)
                    instance.stats = TokenStats.from_dict(stats_data)
                    instance.category = category
                    instance._local = threading.local()
                    cls._instances[category] = instance
                else:
                    # Update existing stats
                    existing_stats = cls._instances[category].stats
                    new_stats = TokenStats.from_dict(stats_data)
                    existing_stats.total_tokens = new_stats.total_tokens
                    existing_stats.prompt_tokens = new_stats.prompt_tokens
                    existing_stats.completion_tokens = new_stats.completion_tokens
                    existing_stats.prompt_tokens_cached = new_stats.prompt_tokens_cached
                    existing_stats.reasoning_tokens = new_stats.reasoning_tokens
                    existing_stats.total_cost = new_stats.total_cost
                    existing_stats.successful_requests = new_stats.successful_requests
                    existing_stats.requests_time_in_seconds = new_stats.requests_time_in_seconds
                    existing_stats.rate_limit_delays = new_stats.rate_limit_delays
                    existing_stats.init_time = new_stats.init_time


def track_tokens(default_category: str):
    """
    Decorator to track token usage. Uses a unique 'token_tracker_id' from
    RunnableConfig metadata if available, otherwise falls back to default_category.
    and also stores 'model_name' so cost can be computed if LC callback has no price.
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            config: Optional[RunnableConfig] = None
            if len(args) > 1 and isinstance(args[1], Mapping):
                config = args[1]
            elif "config" in kwargs and isinstance(kwargs["config"], Mapping):
                config = kwargs["config"]

            category_to_use = default_category
            llm_model = None

            if config:
                md = config.get("metadata", {}) or {}
                tracker_id = md.get("token_tracker_id")
                if tracker_id:
                    category_to_use = str(tracker_id)
                llm_model = md.get("model_name")

            tracker = TokenTracker(category_to_use)
            # ensure a thread-local exists and store model for __exit__
            tracker._local = getattr(tracker, "_local", threading.local())
            tracker._local.llm_model = llm_model

            with tracker:
                return func(*args, **kwargs)

        return wrapper

    return decorator
