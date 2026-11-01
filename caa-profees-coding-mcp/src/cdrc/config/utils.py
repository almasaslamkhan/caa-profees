import os
from functools import lru_cache
from os.path import isfile

from loguru import logger  # we don't use the logger from the common module to avoid circular imports
from pyhocon import ConfigFactory


def read_environment_file(filename):
    result = None
    if isfile(filename):
        with open(filename, "r") as env_file:
            result = env_file.read().strip()
    return result


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache()
def get_config():
    """Reads from the configuration files based on MODE_PROFEES environment variable"""
    VALID_MODES = ("SDS", "PROD", "DEV", "LOCAL")
    mode = os.getenv("MODE_PROFEES", "SDS" if os.path.exists("/sds") else "LOCAL").upper()

    if mode not in VALID_MODES:
        raise ValueError(f"MODE_PROFEES must be one of {VALID_MODES}, got {mode}")

    config_file = f"{CONFIG_DIR}/{mode.lower()}.conf"

    if not isfile(config_file):
        raise Exception(f"Config file {config_file} does not exist or is not a file")

    logger.info(f"Using config file {config_file}")
    return ConfigFactory.parse_file(config_file)


settings = get_config()
