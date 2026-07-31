common package
==============

The ``common`` package provides **shared utilities and data structures** 
used across the VISWIR project. It includes:

- Configuration loading and validation
- Core datatypes for fusion tasks and results
- Logging utilities
- Database access for results
- User interface helpers

This package is designed to centralize functionality that is reused 
by multiple modules (fusion, optimization, processing, realtime).

Submodules
----------

common.config\_loader module
----------------------------
Utility functions for reading and validating configuration files.

.. automodule:: common.config_loader
   :members:
   :undoc-members:
   :show-inheritance:

common.datatypes module
-----------------------
Core dataclasses and type definitions for fusion tasks and results.

.. automodule:: common.datatypes
   :members:
   :undoc-members:
   :show-inheritance:

common.logger module
--------------------
Lightweight logging utilities for console and file output.

.. automodule:: common.logger
   :members:
   :undoc-members:
   :show-inheritance:

common.results\_db module
-------------------------
Database models and helpers for storing fusion results.

.. automodule:: common.results_db
   :members:
   :undoc-members:
   :show-inheritance:

common.ui module
----------------
Basic user interface helpers.

.. automodule:: common.ui
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------
The top-level ``common`` module re-exports selected utilities 
from its submodules for convenience.

.. automodule:: common
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
