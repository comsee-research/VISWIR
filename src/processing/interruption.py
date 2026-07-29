"""
Interruption handling and signal management for long-running batch processes.
"""

import signal
import sys
import json
from pathlib import Path

from common.logger import logger

last_params = {}  # Dictionnaire mis à jour dynamiquement depuis le script principal

def save_last_params():
    """
    Save the last combination of parameters before program termination.

    This function writes the current content of the global `last_params`
    dictionary into a JSON file located at `config/last_params.json`.

    Notes
    -----
    - The file is overwritten each time this function is called.
    - Called automatically when a keyboard interruption (Ctrl+C) is detected.
    """
    json_path = Path("config") / "last_params.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(last_params, f, indent=4)
    logger.warning(f"\n💾 Derniers paramètres enregistrés dans {json_path}")

def signal_handler(sig, frame):
    """
    Handle SIGINT (Ctrl+C) signals by saving the last parameters and exiting.

    Parameters
    ----------
    sig : int
        Signal number (e.g., `signal.SIGINT`).
    frame : frame object
        Current stack frame (unused).

    Notes
    -----
    - Logs a warning message before saving.
    - Calls `save_last_params()` to persist the last parameters.
    - Exits the program with status code 0.
    """
    logger.warning("\n⚠️ Interruption détectée ! Sauvegarde des derniers paramètres...")
    save_last_params()
    sys.exit(0)

# Activer la gestion du signal
signal.signal(signal.SIGINT, signal_handler)
