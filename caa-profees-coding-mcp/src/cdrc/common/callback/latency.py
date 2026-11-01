import time
from typing import Any, Dict

from langchain_core.callbacks.base import BaseCallbackHandler


class ChainLatencyCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.latencies = []
        self.run_records = {}

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any):
        """Triggered when a model starts processing a batch."""
        if not kwargs["parent_run_id"]:
            start_time = time.perf_counter()  # Record start time
            self.run_records[kwargs["run_id"]] = {"start_time": start_time}

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any):
        """Triggered when a model has finished processing a batch."""
        if not kwargs["parent_run_id"]:
            end_time = time.perf_counter()  # Record end time
            latency = end_time - self.run_records[kwargs["run_id"]]["start_time"]  # Calculate latency
            self.run_records[kwargs["run_id"]]["end_time"] = end_time
            self.run_records[kwargs["run_id"]]["latency"] = latency
            self.latencies.append(latency)
