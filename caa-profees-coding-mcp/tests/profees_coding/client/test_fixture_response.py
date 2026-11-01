import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from profees_coding.mcp.tools import recode


def test_fixture_is_exact_test_sample_and_skips_model(monkeypatch, tmp_path):
    monkeypatch.setenv("PROFEES_RECODE_USE_SAMPLE_RESPONSE", "true")
    monkeypatch.setenv("PROFEES_DEBUG_CAPTURE_DIR", str(tmp_path / "capture"))
    model = SimpleNamespace(predict=Mock(side_effect=AssertionError("Live model must not be called")))
    service = recode.ProFeesCPTCodingService(si_helper=SimpleNamespace(), authenticated_environment=object(), coding_model=model)
    response, _, _ = asyncio.run(
        service.process_encounter(
            patient_id="pa-sample",
            encounter_id="en-sample",
            document_ref_id="df-sample",
            doc_content="Synthetic note",
            coding_request_context={
                "patient_info": {"type": "New", "birthdate": "1970-01-01", "gender": "female"},
                "encounter_period": {"start": "2026-10-07T09:00:00Z"},
                "orders": [],
            },
        )
    )
    expected = json.loads((Path(__file__).parent / "fixtures/profees_semantic_search_sample.json").read_text())
    assert response == expected
    model.predict.assert_not_called()
    files = list((tmp_path / "capture").glob("model-response.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text()) == expected
    assert recode._get_or_create_coding_model() is not None
