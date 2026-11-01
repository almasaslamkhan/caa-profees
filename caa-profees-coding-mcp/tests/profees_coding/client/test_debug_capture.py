import json
import stat
from types import SimpleNamespace

import pytest
import httpx

from profees_coding.common.debug_capture import capture_payload
from profees_coding.client import coding_service_call_factory as factory


def test_capture_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("PROFEES_DEBUG_CAPTURE_DIR", raising=False)
    assert capture_payload("model-response", {"em": []}) is None
    assert list(tmp_path.iterdir()) == []


def test_capture_preserves_alternatives_and_private_permissions(tmp_path, monkeypatch):
    root = tmp_path / "private"
    monkeypatch.setenv("PROFEES_DEBUG_CAPTURE_DIR", str(root))
    payload = {"ambulatory_em_order_catalog_candidates": [{"code": "99213", "order_catalog_candidates": [{"CATALOG_CD": "123"}]}]}
    path = capture_payload("model-response", payload)
    assert json.loads(path.read_text()) == payload
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.name == "model-response.json"
    replacement = {"em": [{"code": "99214"}]}
    assert capture_payload("model-response", replacement) == path
    assert json.loads(path.read_text()) == replacement
    assert list(root.iterdir()) == [path]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("failure", ["relative", "public", "serialization"])
def test_capture_failures_do_not_interrupt_caller(tmp_path, monkeypatch, failure):
    root = tmp_path / "capture"
    payload = {"em": []}
    if failure == "relative":
        directory = "relative-path"
    else:
        directory = str(root)
        if failure == "public":
            root.mkdir()
            root.chmod(0o755)
        else:
            payload = {"unsupported": object()}
    monkeypatch.setenv("PROFEES_DEBUG_CAPTURE_DIR", directory)
    assert capture_payload("model-response", payload) is None
    assert not root.exists() or list(root.iterdir()) == []


def test_billing_capture_occurs_before_failed_send(tmp_path, monkeypatch):
    root = tmp_path / "private"
    monkeypatch.setenv("PROFEES_DEBUG_CAPTURE_DIR", str(root))
    monkeypatch.setattr(factory, "get_service_url", lambda *_: "https://coding.test")
    monkeypatch.setattr(factory, "get_httpx_client", lambda *_args, **_kwargs: object())
    payload = {"em": [{"code": "99213", "orderCatalogCandidates": [{"CATALOG_CD": "123"}]}]}

    def fail_send(**_kwargs):
        files = list(root.glob("savebillingcodes-payload.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text()) == payload
        raise RuntimeError("synthetic TLS failure")

    monkeypatch.setattr(factory.save_billing_codes, "sync_detailed", fail_send)
    client = factory.CodingClient(object(), object())
    with pytest.raises(RuntimeError, match="synthetic TLS failure"):
        client.save_billing_code(SimpleNamespace(to_dict=lambda: payload))


def test_billing_400_response_is_captured_before_raising(tmp_path, monkeypatch):
    root = tmp_path / "private"
    monkeypatch.setenv("PROFEES_DEBUG_CAPTURE_DIR", str(root))
    body = {"code": "InvalidParameter", "message": "Synthetic validation failure", "details": {"field": "example"}}
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(400, json=body)), base_url="https://coding.test") as http_client:
        monkeypatch.setattr(factory, "get_service_url", lambda *_: "https://coding.test")
        monkeypatch.setattr(factory, "get_httpx_client", lambda *_args, **_kwargs: http_client)
        with pytest.raises(RuntimeError, match="HTTP 400"):
            factory.CodingClient(object(), object()).save_billing_code(SimpleNamespace(to_dict=lambda: {"em": []}))
    captured = json.loads((root / "savebillingcodes-response.json").read_text())
    assert captured["status_code"] == 400
    assert captured["body"] == body
    assert captured["endpoint"].endswith("/billingCodes")
