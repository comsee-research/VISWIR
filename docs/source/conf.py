import os
import sys
from pathlib import Path

# Path to the project root (VISWIR_vQuasar/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

# Add src/ to sys.path
sys.path.insert(0, str(SRC_DIR))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VISWIR: Visible and SWIR Weighted Image Reconstruction'
copyright = '2025, Riffard Alexandre - Institut Pascal'
author = 'Riffard Alexandre - Institut Pascal'
release = '0.2 (vQuasar)'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # pour docstrings style Google/NumPy
    "sphinx.ext.viewcode",  # ajoute des liens vers le code source
    "sphinx_nefertiti",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ['_templates']
exclude_patterns = []

# -- MyST configuration ------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",   # support ::: fenced blocks
    "linkify",       # auto-detect bare links
]

# Prevent MyST from treating relative Markdown links as cross-references
myst_linkify_fuzzy_links = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_nefertiti'
html_static_path = ['_static']
html_theme_options = {
    "project_short": "VISWIR",
    "style": "indigo",
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
    # "logo": "_static/banner.svg",
    "logo_alt": "VISWIR: Visible and SWIR Weighted Image Reconstruction",
    "repository_url": "https://github.com/comsee-research/VISWIR",
    "repository_name": "VISWIR",
    "header_links": [
        {"text": "Docs", "link": "index"},
        {"text": "GitHub", "link": "https://github.com/comsee-research/VISWIR"},
    ],
    "footer_links": [
        {"text": "Contact", "link": "mailto:alexandre@uca.fr"},
        {"text": "License", "link": "license"},
    ],
    "show_colorset_choices": True,
}

# html_logo = "_static/banner.svg"
# html_favicon = "_static/favicon.ico"
