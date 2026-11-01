import os
import warnings
from functools import partial
from typing import Callable, List

import requests
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from cdrc.common.langchain_utils import AzureOpenAIEmbeddingsWithRateLimitRetries
from cdrc.common.llm_utils import check_and_reload_env, get_azure_openai_key, get_bearer_token

os.environ["NO_PROXY"] = "10.30.9.117"
check_and_reload_env()

ModelCatalogue = dict[str, Callable[..., Embeddings]]

tiktoken_cache_dir = "cdrc/profees_coding/em/data/tiktoken_cache"
encoding_name = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"  # corresponding to cl100k_base
if not os.path.exists(os.path.join(tiktoken_cache_dir, encoding_name)):
    warnings.warn(
        f"tiktoken encoding is not found in {tiktoken_cache_dir}. The embedding models will not work if you are on SDS. "
        "Please follow `https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer` "
        "to download the file and add it to the cache directory."
    )
else:
    os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir


def get_embedding_models(proxy: str = None) -> ModelCatalogue:
    return {
        # "text-embedding-3-large": partial(
        #     OpenAIEmbeddings,
        #     model="text-embedding-3-large",
        #     openai_proxy=proxy,
        # ),
        "oracle-text-embedding-3-large-1-dev-local": partial(
            AzureOpenAIEmbeddingsWithRateLimitRetries,
            openai_proxy=proxy,
            api_version="2024-02-01",
            azure_endpoint="http://aifactory-healthai.digitalassistant.oci.oraclecloud.com:3000",
            azure_ad_token_provider=partial(get_bearer_token, proxy=proxy) or "EMPTY",
            deployment="oracle-text-embedding-3-large-1-dev",
            tiktoken_model_name="text-embedding-3-large",
        ),
        "oracle-text-embedding-3-large-1-dev": partial(
            AzureOpenAIEmbeddingsWithRateLimitRetries,
            openai_proxy=proxy,
            api_version="2023-05-15",
            azure_endpoint="http://10.30.9.117:32627",
            api_key=get_azure_openai_key(region="eastus") or "EMPTY",
            deployment="oracle-text-embedding-3-large-1-dev",
            tiktoken_model_name="text-embedding-3-large",
        ),
        "caa-text-embedding-3-large-dev": partial(
            AzureOpenAIEmbeddingsWithRateLimitRetries,
            openai_proxy=proxy,
            api_version="2024-02-01",
            azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
            api_key="EMPTY",
            default_headers={"Service-Name": "profees", "service-name": "profees"},
            deployment="caa-text-embedding-3-large-dev",
            tiktoken_model_name="text-embedding-3-large",
        ),
        "caa-text-embedding-3-large-prod": partial(
            AzureOpenAIEmbeddingsWithRateLimitRetries,
            openai_proxy=proxy,
            api_version="2025-11-13",
            azure_endpoint="http://oda-genai-proxy-service-java.max:8421",
            api_key="EMPTY",
            default_headers={"Service-Name": "profees", "service-name": "profees"},
            deployment="caa-text-embedding-3-large-prod",
            tiktoken_model_name="text-embedding-3-large",
        ),
    }
