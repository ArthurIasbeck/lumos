from loguru import logger
from resources.constants import constants


def get(constant_name):
    if constant_name not in constants.keys():
        raise RuntimeError(f"A constante '{constant_name}' não foi definida")

    return constants[constant_name]


def init():
    logger.add(get("LOG_PATH"), format="{time} | {level} | {message}", level="INFO")
