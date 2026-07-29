"""
logger.py — Centralized logging configuration for VISWIR

This module provides a unified logging system using **Loguru** for flexible
logging and **Rich** for enhanced terminal output. It allows logging both
to the console and to rotating log files.

Features
--------
- Colored and formatted console output (Rich).
- Log file recording with rotation and compression (Loguru).
- Detailed stack traces with customizable format.
- Centralized configuration loaded from `logging_config.yaml`.

Usage
-----
>>> from logger import logger
>>> logger.info("Processing image {}", path_to_image)

Notes
-----
- Default configuration is loaded from `config/logging_config.yaml`.
- If the configuration file is missing, fallback defaults are applied.
"""


# import os
# import sys
# import json
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.traceback import install

from common.config_loader import load_config, ConfigError

# Initialisation Rich
console = Console()
install(show_locals=True, console=console)

# Charger la config logging
try:
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "logging_config.yaml"
    logging_cfg = load_config(config_path, defaults={
        "level": "INFO",
        "log_to_file": True,
        "log_file": "viswir.log",
        "log_to_console": True,
        "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                  "<level>{level: <8}</level> | "
                  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                  "<level>{message}</level>"
    })
except ConfigError:
    logging_cfg = {
        "level": "INFO",
        "log_to_file": True,
        "log_file": "viswir.log",
        "log_to_console": True,
        "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                  "<level>{level: <8}</level> | "
                  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                  "<level>{message}</level>"
    }

# Nettoyer les handlers par défaut
logger.remove()

# 1. Console
if logging_cfg.get("log_to_console", True):
    logger.add(lambda msg: console.print(msg, end=""),
               format=logging_cfg["format"],
               level=logging_cfg["level"])

# 2. Fichier
if logging_cfg.get("log_to_file", True):
    # Dossier logs dans src/
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / logging_cfg.get("log_file", "viswir.log")

    logger.add(log_file,
               level="DEBUG",
               rotation="10 MB",
               retention="10 days",
               compression="zip",
               enqueue=True,
               encoding="utf-8",
               backtrace=True,
               diagnose=True)

# Exemple d’utilisation si exécuté seul
if __name__ == "__main__":
    logger.debug("Mode debuggage activé avec succès.")
    logger.info("Logger initialisé avec succès.")
    logger.warning("Ceci est un avertissement de test.")
    try:
        raise RuntimeError("Erreur de démonstration.")
    except Exception:
        logger.exception("Une exception a été capturée.")