Configuration Files
===================

VISWIR vQuasar relies on a set of configuration files to control input/output
paths, fusion parameters, and optional detection settings. This section
explains the purpose of each file and provides examples.

.. contents::
   :local:
   :depth: 1

Base Configuration
------------------

- ``base_config.yaml`` : defines input/output folders, reference image, mode, etc.
- ``parameters.json`` : defines fusion parameters.
- ``yolo_config.json`` : YOLO detection parameters used if  ``run_detection: true`` in ``base_config.yaml`` (optional, defaults are used if missing).

.. note::
   The base configuration is required for most runs.  
   It specifies where data is located and how the fusion pipeline should behave.

Fast Pipeline Configuration
---------------------------

- ``fast_config.yaml`` : minimal config for the fast pipeline.
- ``yolo_config.json`` : YOLO detection parameters (optional, defaults are used if missing).

.. important::
   The **fast mode** skips metric computation.  
   It is intended for rapid prototyping and quick previews, not for final evaluation.

Examples
--------

Example of a minimal base configuration:

.. literalinclude:: ../../../config/config_viswir.yaml
   :language: yaml
   :caption: Example base_config.yaml
   :lines: 1-13

Example of fusion parameters (JSON):

.. code-block:: json

   {
      "mode_fixe": true,
      "facteur_swir": 0.89,
      "beta": 1.07,
      "level": 5,
      "apply_gamma": true,
      "gamma_value": 2.82
   }

Example of YOLO detection config:

.. code-block:: json

   {
      "model_path": "yolov8x.pt",
      "confidence_threshold": 0.25,
      "iou_threshold": 0.3,
      "device": "cpu",
      "save_detection_results": false,
      "allowed_classes": [
         "truck",
         "person",
         "bus",
         "motorcycle",
         "bicycle",
         "car"
      ]
   }
