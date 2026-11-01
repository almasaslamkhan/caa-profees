import json
import logging
import sys
from pathlib import Path

import loguru
from gunicorn import glogging

from cdrc.config import settings

# TODO: Islam - figure out how to get the OPC_REQUEST_ID, ENCOUNTER_ID


def serialize(record):
    simplified = {
        "level": record["level"].name,
        "opc-request-id": record["extra"]["OPC_REQUEST_ID"],
        "encounterId": record["extra"]["ENCOUNTER_ID"],
        "ts": int(round(record["time"].timestamp() * 1000)),
        "thread": record["thread"].name,
        "module": record["module"],
        "pathname": record["name"],
        "msg": record["message"],
    }
    return json.dumps(simplified)


def sink(message):
    serialized = serialize(message.record)
    print(serialized)


def formatter(record):
    record["extra"]["serialized"] = serialize(record)
    return "{extra[serialized]}\n"


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class StubbedGunicornLogger(glogging.Logger):
    """Class to override gunicorn logs handler"""

    def setup(self, cfg):
        handler = logging.NullHandler()
        self.access_logger = logging.getLogger("gunicorn.access")
        self.error_logger = logging.getLogger("gunicorn.error")

        self.error_logger.addHandler(handler)
        self.access_logger.addHandler(handler)

        self.error_logger.setLevel(int(settings.get("logging.gunicorn_log_level")))
        self.access_logger.setLevel(int(settings.get("logging.gunicorn_log_level")))


class Logger:
    @classmethod
    def get_logger(cls):
        return loguru.logger

    @classmethod
    def configure_logger(cls):
        """Configures the logger"""
        intercept_handler = InterceptHandler()
        logging.root.setLevel(int(settings.get("logging.log_level")))

        seen = set()
        for name in [
            *logging.root.manager.loggerDict.keys(),
            "gunicorn",
            "gunicorn.access",
            "gunicorn.error",
            "uvicorn",
            "uvicorn.access",
            "uvicorn.error",
            "fastapi",
        ]:
            if name not in seen:
                seen.add(name.split(".")[0])
                logging.getLogger(name).handlers = [intercept_handler]

        Path(settings.get("logging.log_dir")).mkdir(parents=True, exist_ok=True)
        logger.configure(
            handlers=[
                dict(
                    sink=settings.get("logging.zip.file"),
                    serialize=False,
                    enqueue=True,
                    compression=settings.get("logging.zip.compression"),
                    level=int(settings.get("logging.app_log_level")),
                    rotation=settings.get("logging.zip.rotation"),
                    retention=settings.get("logging.zip.retention"),
                    format=settings.get("logging.format"),
                )
            ],
            extra={"OPC_REQUEST_ID": "", "ENCOUNTER_ID": ""},
        )

        logger.add(
            f"{settings.get('logging.log_dir')}/application_structured.log",
            enqueue=True,
            compression=settings.get("logging.zip.compression"),
            serialize=settings.get("logging.json_logs"),
            level=int(settings.get("logging.app_log_level")),
            rotation=settings.get("logging.zip.rotation"),
            retention=settings.get("logging.zip.retention"),
            format=formatter,
        )

        logger.add(
            sys.stdout,
            format=formatter,
            level=int(settings.get("logging.app_log_level")),
        )


logger = Logger.get_logger()
