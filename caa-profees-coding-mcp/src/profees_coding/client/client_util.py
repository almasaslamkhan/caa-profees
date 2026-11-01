import logging
import os
import urllib
import uuid

import httpx
from adkruntime.agent_environment import AgentEnvironment
from adkruntime.messaging import MessagingContext
from oci_signing.service_accounts.service_accounts import ServiceAccountRequestDetails

from profees_coding.client.constants import EHRC_FRONTEND, MR_SERVICE_ACCOUNT, NMQAR_SERVICE_URL
from profees_coding.common import logging_context

_logger: logging.Logger = logging_context.get_logger(__name__)


def get_auth_env(agent_env: AgentEnvironment, context):
    return agent_env.get_authenticated_environment(
        context,
        service_account_request_details=ServiceAccountRequestDetails(
            name=MR_SERVICE_ACCOUNT,
            serviceResourceId="defaultId",
            serviceResourceName="Medical Coding Service",
            scopes="internal",
        ),
    )


def get_httpx_client(
    api_path: str,
    agent_env: AgentEnvironment,
    context: MessagingContext,
    base_url: str | None = None,
):
    effective_base_url = base_url or get_service_url(agent_env, context)
    service_url = f"{effective_base_url.rstrip('/')}/{api_path.strip('/')}"
    bearer_token = get_bearer_token()
    if bearer_token:
        return httpx.Client(
            base_url=service_url,
            headers={
                "Accept-Encoding": "br",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {bearer_token}",
            },
        )

    env = get_auth_env(agent_env, context)
    transport_base_url = _origin_url(service_url)
    return httpx.Client(
        base_url=service_url,
        transport=env.create_httpx_transport(base_url=transport_base_url),
        headers={"Accept-Encoding": "br"},
    )


def get_bearer_token() -> str | None:
    return os.getenv("HIM_CODING_SERVICE_BEARER_TOKEN") or os.getenv("EHR_BEARER_TOKEN")


def get_service_url(agent_env: AgentEnvironment, context: MessagingContext):
    configured_service_url = os.getenv("HIM_CODING_SERVICE_BASE_URL")
    if configured_service_url:
        return configured_service_url.rstrip("/")

    if get_bearer_token():
        frontend_runtime_url = urllib.parse.urljoin(f"{NMQAR_SERVICE_URL.rstrip('/')}/", EHRC_FRONTEND.strip("/"))
        return frontend_runtime_url

    instance_service_url = get_instance_ehr_clinical_service_url(agent_env, context)
    if instance_service_url:
        return instance_service_url.rstrip("/")

    frontend_runtime_url = urllib.parse.urljoin(f"{NMQAR_SERVICE_URL.rstrip('/')}/", EHRC_FRONTEND.strip("/"))
    return frontend_runtime_url


def _origin_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def get_instance_ehr_clinical_service_url(agent_env: AgentEnvironment, context: MessagingContext):
    instance_env = agent_env.get_instance_environment(context).get_raw_instance_metadata()
    try:
        ehr_clinical_web_url = instance_env["ehrClinical"]["webUrl"]
        return ehr_clinical_web_url
    except Exception as e:
        _logger.error(f"Failed to fetch service_url with error as : {e}")
        return ""


def get_opc_id(req_id: str):
    if req_id:
        return req_id.split("/")[0]

    return uuid.uuid4().hex.upper()
