processing package
==================

The ``processing`` package provides the **execution layer** of the VISWIR
pipeline. It manages batch runs, task orchestration, SQL logging, and
interruption handling. This package is responsible for coordinating the
fusion tasks and ensuring results are stored and retrievable.

Submodules
----------

processing.batch\_runner module
-------------------------------
Main batch runner for executing multiple fusion tasks in sequence.

.. automodule:: processing.batch_runner
   :members:
   :undoc-members:
   :show-inheritance:

processing.interruption module
------------------------------
Utilities for handling interruptions and safely stopping long-running jobs.

.. automodule:: processing.interruption
   :members:
   :undoc-members:
   :show-inheritance:

processing.sql\_runner module
-----------------------------
SQL interface for logging and retrieving results from the database.

.. automodule:: processing.sql_runner
   :members:
   :undoc-members:
   :show-inheritance:

processing.task\_manager module
-------------------------------
Task manager for coordinating individual fusion tasks and their dependencies.

.. automodule:: processing.task_manager
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------
The top-level ``processing`` module re-exports selected functions and classes
from its submodules for convenience.

.. automodule:: processing
   :members:
   :undoc-members:
   :show-inheritance:
