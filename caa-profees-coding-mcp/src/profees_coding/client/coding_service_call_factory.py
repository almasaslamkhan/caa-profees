import json
import logging
from http import HTTPStatus

from adkruntime.agent_environment import AgentEnvironment

from profees_coding.common import logging_context
from profees_coding.common.debug_capture import capture_payload

from .client_util import get_httpx_client, get_service_url
from .constants import CODING_SERVICE_PATH
from .him_coding_service_api_client.api.him_coding import save_billing_codes
from .him_coding_service_api_client.client import Client
from .him_coding_service_api_client.models import BillingCodeSummary

_logger: logging.Logger = logging_context.get_logger(__name__)


class CodingClient:
    """
    Client wrapper for coding_service_client.Client
    """

    def __init__(self, agent_env: AgentEnvironment, context):
        effective_base_url = get_service_url(agent_env, context)
        self._save_billing_codes_url = f"{effective_base_url.rstrip('/')}/{CODING_SERVICE_PATH.strip('/')}/billingCodes"
        self._client = Client(effective_base_url)
        self._client.set_httpx_client(get_httpx_client(CODING_SERVICE_PATH, agent_env, context, base_url=effective_base_url))

    def save_billing_code(self, request_body: BillingCodeSummary):
        try:
            capture_payload("savebillingcodes-payload", request_body.to_dict())
            save_billing_response = save_billing_codes.sync_detailed(client=self._client, body=request_body)
            response_text = save_billing_response.content.decode("utf-8", errors="replace")
            try:
                response_body = json.loads(response_text)
            except json.JSONDecodeError:
                response_body = response_text
            capture_payload(
                "savebillingcodes-response",
                {
                    "status_code": int(save_billing_response.status_code),
                    "endpoint": self._save_billing_codes_url,
                    "body": response_body,
                },
            )
            if save_billing_response.status_code != HTTPStatus.OK:
                raise RuntimeError(f"Failed to save billing codes: HTTP {save_billing_response.status_code}")
            return save_billing_response.parsed
        except Exception as e:
            _logger.exception(f"Failed to save_billing_code: {e}")
            raise
