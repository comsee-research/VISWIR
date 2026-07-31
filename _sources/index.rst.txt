.. VISWIR documentation master file, created by
   sphinx-quickstart on Tue Nov  4 10:40:11 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. VISWIR documentation
.. ====================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

Welcome to VISWIR's documentation!
==================================

.. include:: ../../README.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage/quickstart
   usage/configs
   usage/optuna

.. toctree::
   :maxdepth: 2
   :caption: Miscellaneous

   usage/data
   usage/tools

.. toctree::
   :maxdepth: 2
   :caption: Advanced Usage

   help
   usage/hpc_slurm

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

.. toctree::
   :maxdepth: 2
   :caption: Project Info

   license
   authors