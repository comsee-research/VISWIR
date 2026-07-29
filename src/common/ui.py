"""
Terminal UI formatting and license display utilities using Rich.
"""

# =============================================================================
# FILENAME:       ui.py
# DESCRIPTION:    Ce fichier s'occupe juste de faire un joli affichae dans le terminal.
#  
# REPOSITORY:     https://github.com/comsee-research/VISWIR.git
#
# AUTHOR:         [Riffard Alexandre]
# EMAIL:          [alexandre.riffard@uca.fr]
# CREATED:        [30-04-2025]
# LAST UPDATED:   [07-05-2025]
# VERSION:        1.0
#
# LICENSE:        GNU LESSER GENERAL PUBLIC LICENSE (voir LICENSE dans le dépôt)
#
# USAGE:          - Appeler ce fichier dans "batch_processing.py"
#
# DEPENDENCIES:   - rich
#
# NOTES:
#   - ...
#
# CHANGELOG:
#   - [09-05-2025]: Création initiale du fichier.
#   - [..-..-....]: ...
#
# =============================================================================

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.live import Live
from rich.markdown import Markdown
from time import sleep

def print_viswir_header():
    """
    Display the VISWIR header panel in the terminal.

    This function uses the Rich library to render a styled panel
    announcing the initialization of the VISWIR fusion pipeline.
    It also calls :func:`print_license_info` to display license details.

    Notes
    -----
    - The panel includes a title, subtitle, and a styled message.
    - Originally designed to include a spinner animation (currently disabled).
    - Intended for user-facing terminal output when launching VISWIR.
    """

    console = Console()
    
    # title_text = Text("VIS–SWIR Fusion Tool", style="bold white on blue", justify="center")
    # subtitle_text = Text("\n\nHaute performance · Fusion multi-spectrale · Version stable\n\n", style="italic cyan", justify="center")

    # spinner = Spinner("bouncingBar", text="Chargement du module VISWIR...", style="cyan")

    panel = Panel(
        Text.from_markup("\n[bold magenta]⚡ Initialisation du pipeline de fusion...\n", justify="center"),
        title="[bold yellow]VISWIR Fusion",
        subtitle="by Institut Pascal",
        padding=(1, 4),
        border_style="bright_magenta"
    )

    # console.print(title_text)
    console.print(panel)
    # console.print(subtitle_text)

    print_license_info()

    # with Live(spinner, refresh_per_second=12, console=console):
    #     sleep(1.5)  # Animation pendant un petit moment
    

def print_license_info():
    """
    Display license information for VISWIR in the terminal.

    This function renders the LGPL license notice using Rich's Markdown
    and Panel components, providing a styled and readable output.

    Notes
    -----
    - The license is **LGPL (Lesser General Public License)**.
    - Users are free to use, modify, and distribute the software,
      provided derivative works remain under the same license.
    - The license text is available at:
      https://www.gnu.org/licenses/lgpl-3.0.html
    """
    
    console = Console()
    lgpl_md = Markdown("""
<!--### 📝 Licence-->

Ce logiciel est distribué sous licence **[LGPL (Lesser General Public License)](https://www.gnu.org/licenses/lgpl-3.0.html#license-text)**.

Vous êtes libre de l'utiliser, modifier et distribuer ce logiciel, à condition de conserver la même licence pour les dérivés et de permettre le lien dynamique avec d'autres bibliothèques.

Pour plus d'information, voir [gnu.org/licenses/lgpl-3.0.html](https://www.gnu.org/licenses/lgpl-3.0.html)
""")
    panel = Panel.fit(
        lgpl_md,
        title="[bold]📝 Licence",
        border_style="green",
        padding=(1, 2)
    )
    console.print(panel)