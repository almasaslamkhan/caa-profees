import json
import logging
import re
from enum import Enum

from profees_coding.common import logging_context

_logger = logging_context.get_logger(__name__)


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def redact_sensitive_info(log):
    """
    Masks 'Justification' and certain 'Response' fields in a log.

    Args:
    log (str): The log output to be masked.

    Returns:
    str: The masked log output.
    """
    # Define patterns to match and mask
    patterns = [
        (r"(.*soap_text)(.*)", r"\1 : XXXXXXXX"),
        (r"(.*Justification)(.*)", r"\1 : XXXXXXXX"),
        (r"(PC Critic Response)(.*)", r"\1 : XXXXXXXX"),
        (r"(RC Critic Response)(.*)", r"\1 : XXXXXXXX"),
        (r"(DC Critic Response)(.*)", r"\1 : XXXXXXXX"),
        (r"(.*ambient_note)(.*)", r"\1 : XXXXXXXX"),
        (r"(.*patient_birthdate)(.*)", r"\1 : XXXXXXXX"),
    ]

    # Apply each pattern to the log
    for pattern, replacement in patterns:
        log = re.sub(pattern, replacement, log, flags=re.IGNORECASE)

    return log


def redact_sensitive_info_json(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if k.lower() in [
                "justification",
                "pc critic response",
                "rc critic response",
                "dc critic response",
                "soap_text",
                "patient_birthdate",
                "ambient_note",
                "soap_text_previous",
            ]:
                data[k] = "XXXXXXX"
            else:
                data[k] = redact_sensitive_info_json(v)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = redact_sensitive_info_json(item)
    return data


def custom_logger_wrapper(func):
    def wrapper(*args, **kwargs):
        log_level = kwargs.get("log_level")
        message = kwargs.get("message")
        if log_level is None or message is None:
            log_level = args[0] if len(args) > 0 else LogLevel.INFO
            message = " ".join(map(str, args[1:])) if len(args) > 1 else ""

        try:
            if isinstance(message, str):
                try:
                    message = json.dumps(redact_sensitive_info_json(message))
                except (json.JSONDecodeError, TypeError):
                    message = redact_sensitive_info(message)
            elif isinstance(message, (dict, list)):
                message = json.dumps(redact_sensitive_info_json(message))
            else:
                to_dict = getattr(message, "to_dict", None)
                if callable(to_dict):
                    message = json.dumps(redact_sensitive_info_json(to_dict()))
                else:
                    message = redact_sensitive_info(str(message))
        except Exception:
            message = redact_sensitive_info(str(message))

        kwargs["log_level"] = log_level
        kwargs["message"] = message
        return func(**kwargs)

    return wrapper


@custom_logger_wrapper
def log(*, log_level, message):
    if log_level == LogLevel.DEBUG:
        _logger.debug(message)
    elif log_level == LogLevel.INFO:
        _logger.info(message)
    elif log_level == LogLevel.WARNING:
        _logger.warning(message)
    elif log_level == LogLevel.ERROR:
        _logger.error(message)
    elif log_level == LogLevel.CRITICAL:
        _logger.critical(message)
