fusion package
==============

The ``fusion`` package implements the **core algorithms and utilities** 
for VISWIR image fusion. It provides modules for:

- Performing the actual fusion of visible and SWIR images
- Running object detection on fused images
- Computing quality metrics
- Utility functions for preprocessing and support tasks

Subpackages
-----------

.. toctree::
   :maxdepth: 2

   fusion.NIQE

Submodules
----------

fusion.detection\_module module
-------------------------------
YOLO-based detection module for evaluating fused images.

.. automodule:: fusion.detection_module
   :members:
   :undoc-members:
   :show-inheritance:

fusion.functions module
-----------------------
Helper functions used across the fusion pipeline.

.. automodule:: fusion.functions
   :members:
   :undoc-members:
   :show-inheritance:

fusion.fusion module
--------------------
Core fusion algorithms combining visible and SWIR images.

.. automodule:: fusion.fusion
   :members:
   :undoc-members:
   :show-inheritance:

fusion.metrics module
---------------------
Computation of image quality and detection metrics for fused and reference images.

.. automodule:: fusion.metrics
   :members:
   :undoc-members:
   :show-inheritance:

fusion.utils module
-------------------
Utility functions for file handling, preprocessing, and pipeline support.

.. automodule:: fusion.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------
The top-level ``fusion`` module re-exports selected functions and classes 
from its submodules for convenience.

.. automodule:: fusion
   :members:
   :undoc-members:
   :show-inheritance:
