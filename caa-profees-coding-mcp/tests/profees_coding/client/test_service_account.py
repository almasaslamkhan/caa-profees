from unittest.mock import Mock

from profees_coding.client.client_util import get_auth_env
from profees_coding.mcp.tools import recode


def test_adk_operations_request_clinical_rev_cycle_identity():
    expected = "system:serviceaccount:clinical-rev-cycle:default"
    assert recode.PROFEES_SERVICE_ACCOUNT["name"] == expected
    assert recode.PROFEES_SERVICE_ACCOUNT["serviceResourceId"] == "defaultId"
    assert recode.PROFEES_SERVICE_ACCOUNT["serviceResourceName"] == "Medical coding agent"
    assert recode.PROFEES_SERVICE_ACCOUNT["scopes"] == "internal"
    agent = Mock()
    context = object()
    assert get_auth_env(agent, context) is agent.get_authenticated_environment.return_value
    args, kwargs = agent.get_authenticated_environment.call_args
    assert args == (context,)
    assert kwargs["service_account_request_details"]["name"] == expected
    assert kwargs["service_account_request_details"]["scopes"] == "internal"
