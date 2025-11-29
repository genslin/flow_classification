import logging
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_models"

# Logging functions
def get_model_logger(model_name):
    logger = logging.getLogger(f"model_logger.{model_name}")
    logger.setLevel("DEBUG")
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        file_handler = get_file_handler(model_name)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        console_handler.setLevel("INFO")
        file_handler.setLevel("DEBUG")
    return logger


def get_file_handler(model_name):
    subdirectory = SAVED_MODEL_DIR / model_name / "logs" 
    subdirectory.mkdir(parents=True, exist_ok=True)
    log_path =  subdirectory / f"{model_name}.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    return file_handler