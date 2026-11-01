import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from cdrc.profees_coding.deploy.api.model.app_gen2 import ProfeesModel
from cdrc.profees_coding.deploy.api.utils import DeliveryVehicle, InputData

# Ensure this script imports `cdrc` from the current checkout, not another local clone.
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Suppress noisy metrics logging when local config is unavailable.
logging.getLogger("aoa_agent.profees_coding_agent.agent_obs.metrics").setLevel(logging.CRITICAL)

model = ProfeesModel(delivery_vehicle=DeliveryVehicle.CAAD, mode="LOCAL", log_level="info")
# model = ProfeesModel(
#     delivery_vehicle=DeliveryVehicle.SCRATCHPAD, mode="LOCAL", log_level="info"
# )

TEST_CASE_DIR = Path("cdrc/profees_coding/deploy/api/test_cases")
PAYLOAD_PATHS = [
    TEST_CASE_DIR / "payload_ambulatory_0.json",
    # TEST_CASE_DIR / "payload_ambulatory_1.json",
    # TEST_CASE_DIR / "payload_ambulatory_2.json",
    # TEST_CASE_DIR / "payload_ambulatory_3.json",
    # TEST_CASE_DIR / "payload_ambulatory_4.json",
]


async def run_prediction(payload_path: Path, idx: int = 1) -> float:
    print(f"====== Start task {idx}")
    # payload_path = Path("cdrc/profees_coding/deploy/api/model/payload_no_em.json")
    # payload_path = Path("cdrc/profees_coding/deploy/api/model/payload_ed.json")
    # payload_path = Path("cdrc/profees_coding/deploy/api/model/payload_inpatient.json")
    # payload_path = Path("cdrc/profees_coding/deploy/api/model/payload_recode.json")
    payload = json.loads(payload_path.read_text())
    inp = InputData(**payload)

    # Start time
    t0 = time.perf_counter()

    # Run prediction
    resp, _ = await model.predict(inp)

    # End time
    elapsed = time.perf_counter() - t0

    response_path = payload_path.with_name(payload_path.name.replace("payload_", "response_"))
    response_path.write_text(json.dumps(resp.model_dump(), indent=2))

    print(f"====== End task {idx} | elapsed: {elapsed:.3f} sec")
    print(f"====== Wrote response: {response_path}")
    print(json.dumps(resp.model_dump(), indent=2))

    return elapsed


async def test_concurrent_call():
    return await asyncio.gather(*(run_prediction(payload_path, idx) for idx, payload_path in enumerate(PAYLOAD_PATHS, start=1)))
    # return await asyncio.gather(run_prediction(1), run_prediction(2), run_prediction(3), run_prediction(4), run_prediction(5))


async def test_once():
    for idx, payload_path in enumerate(PAYLOAD_PATHS, start=1):
        await run_prediction(payload_path, idx)
    print("test_once() Done")


asyncio.run(test_once())
