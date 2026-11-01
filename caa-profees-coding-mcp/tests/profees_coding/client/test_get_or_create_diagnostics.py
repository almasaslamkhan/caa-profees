import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from oci.exceptions import ServiceError
from oci_signing.service_accounts import service_accounts


@pytest.mark.parametrize("name", ["medical-coding-agent", "clinical-rev-cycle"])
def test_get_or_create_logs_only_safe_comparison_fields(monkeypatch, caplog, name):
    request = {
        "name": f"system:serviceaccount:{name}:default",
        "serviceResourceId": "defaultId",
        "serviceResourceName": "Medical coding agent",
        "scopes": "internal",
        "clientSecret": "synthetic-request-secret",
    }
    response = {"id": "synthetic-id", "clientSecret": "synthetic-response-secret"}
    client = Mock()
    client.call_api.return_value = SimpleNamespace(data=json.dumps(response))
    monkeypatch.setattr(service_accounts.oci_signing.oci_signing_utils, "create_oci_client", Mock(return_value=client))
    caplog.set_level(logging.INFO, logger=service_accounts.__name__)
    signer = object()
    result = asyncio.run(
        service_accounts.get_or_create_service_account(
            "https://user:synthetic-url-secret@auth.example.test?token=synthetic-query-secret", request, signer
        )
    )
    assert result == response
    assert "[getOrCreate_request]" in caplog.text
    assert request["name"] in caplog.text
    assert "defaultId" in caplog.text
    assert "Medical coding agent" in caplog.text
    assert "internal" in caplog.text
    assert "signer_type=builtins.object" in caplog.text
    assert "endpoint=https://auth.example.test/20240418/serviceAccountAppsServices/actions/getOrCreate" in caplog.text
    for secret in ("synthetic-request-secret", "synthetic-response-secret", "synthetic-url-secret", "synthetic-query-secret"):
        assert secret not in caplog.text
    assert client.call_api.call_args.kwargs["body"] is request


def test_request_is_logged_even_when_api_rejects_account(monkeypatch, caplog):
    client = Mock()
    client.call_api.side_effect = ServiceError(400, "InvalidParameter", {}, "Synthetic rejection")
    monkeypatch.setattr(service_accounts.oci_signing.oci_signing_utils, "create_oci_client", Mock(return_value=client))
    caplog.set_level(logging.INFO, logger=service_accounts.__name__)
    with pytest.raises(RuntimeError, match="Failed to get or create service account"):
        asyncio.run(service_accounts.get_or_create_service_account("https://auth.example.test", {"name": "clinical-rev-cycle"}))
    assert "[getOrCreate_request]" in caplog.text
    assert "signer_type=None" in caplog.text
