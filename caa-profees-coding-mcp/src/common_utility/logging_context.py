import logging

# from pythonjsonlogger import jsonlogger
from contextvars import ContextVar

from cda_caim_sdk.common.logging import traceutils
from cda_utils.logging_utils import ConcurrentLogger

ctx_event_tracking_id = ContextVar("event_tracking_id", default="N/A")
ctx_opc_request_id = ContextVar("opc_request_id", default="N/A")
OPC_REQUEST_ID = "opc-request-id"


def set_tracking_id(tracking_id: str):
    ctx_event_tracking_id.set(tracking_id)


def unset_tracking_id():
    ctx_event_tracking_id.set("N/A")


def set_opc_request_id(opc_request_id: str):
    ctx_opc_request_id.set(opc_request_id)
    traceutils.set_opc_request_id(opc_request_id)


def unset_opc_request_id():
    ctx_opc_request_id.set("N/A")
    traceutils.set_opc_request_id(None)


def get_opc_request_id() -> str:
    return ctx_opc_request_id.get()


def set_context(tracking_id: str, opc_request_id: str):
    set_tracking_id(tracking_id)
    set_opc_request_id(opc_request_id)


def unset_context():
    unset_tracking_id()
    unset_opc_request_id()


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.event_tracking_id = ctx_event_tracking_id.get()
        record.opc_request_id = ctx_opc_request_id.get()
        record.msg = f"[tracking-id: {record.event_tracking_id}] {record.msg}"
        if record.opc_request_id != "N/A":
            record.__dict__[OPC_REQUEST_ID] = record.opc_request_id
        return True


def list_all_log_handlers():
    """
    Lists all loggers and their attached handlers in the current application.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    loggers = {name: logging.getLogger(name) for name in logging.root.manager.loggerDict}
    # Add the root logger to the dictionary if it's not already there (it should be as '')
    if "" not in loggers:
        loggers["root"] = root_logger
    else:
        loggers[""] = root_logger  # ensure root is linked to the '' key

    print(f"--- Found {len(loggers)} Loggers and their Handlers ---")

    for name, logger in loggers.items():
        if logger.handlers:
            print(f"\nLogger: '{name}' (Level: {logging.getLevelName(logger.level)})")
            for i, handler in enumerate(logger.handlers):
                print(f"  Handler {i + 1}: {handler}")
                print(f"    Type: {type(handler).__name__}")
                print(f"    Level: {logging.getLevelName(handler.level)}")
                # Additional handler info for specific types
                if isinstance(handler, logging.FileHandler):
                    print(f"    File path: {handler.baseFilename}")
                elif isinstance(handler, logging.StreamHandler):
                    print(f"    Stream: {handler.stream.name}")
        else:
            print(f"\nLogger: '{name}' has no direct handlers.")


def get_logger(name: str = "medical_coding") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addFilter(ContextFilter())

    return logger
