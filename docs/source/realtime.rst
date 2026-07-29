realtime package
================

The ``realtime`` package provides modules for **fast, low-latency execution**
of the VISWIR pipeline. It is designed for scenarios where image fusion and
detection must be performed in near real-time, such as embedded systems or
live video processing.

.. warning::
   The **fast mode** does not compute quality metrics.  
   It is mainly intended for quickly testing parameters 
   and obtaining a visual result without the overhead 
   of full evaluation.

Submodules
----------

realtime.fast\_config module
----------------------------
Configuration utilities optimized for fast execution.

.. automodule:: realtime.fast_config
   :members:
   :undoc-members:
   :show-inheritance:

realtime.fast\_detection module
-------------------------------
Lightweight detection routines adapted for speed in test contexts.

.. automodule:: realtime.fast_detection
   :members:
   :undoc-members:
   :show-inheritance:

realtime.fast\_fusion\_runner module
------------------------------------
Runner for executing the fusion pipeline, coordinating
fast configuration and detection.

.. automodule:: realtime.fast_fusion_runner
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------
The top-level ``realtime`` module re-exports selected functions and classes
from its submodules for convenience.

.. automodule:: realtime
   :members:
   :undoc-members:
   :show-inheritance:
