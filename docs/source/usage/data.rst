Datasets
========

VISWIR vQuasar relies on paired datasets of **visible spectrum** and **SWIR**
images. This section explains the expected folder organization and provides
guidelines for preparing your data.

.. contents::
   :local:
   :depth: 1

Folder Structure
----------------

The dataset should be organized as follows:

- ``data/visible/`` : contains visible spectrum images (e.g. RGB).
- ``data/swir/`` : contains SWIR images (paired with visible).
- ``data/ground_truth/`` : optional ground truth annotations (for detection tasks).

Example layout:

.. code-block:: text

   data/
   ├── visible/
   │   ├── img_001.png
   │   ├── img_002.png
   │   └── ...
   ├── swir/
   │   ├── img_001.png
   │   ├── img_002.png
   │   └── ...
   └── ground_truth/
       ├── img_001.xml
       ├── img_002.xml
       └── ...

Notes
-----

- **Pairing**: Visible and SWIR images must be paired and sorted consistently
  (e.g. ``img_001.png`` in both folders corresponds to the same scene).
- **Ground truth**: Annotations are optional but required if detection is enabled.
- **Formats**: Images should be in standard formats (PNG, JPEG, TIFF, ...). Ground truth
  must be in PASCAL VOC XML, default format used with Yolo and Roboflow.

.. important::
   There must also be as many RGB images as SWIR images.
   For ground truth, there are two possibilities: either the number corresponds to the number
   of RGB-SWIR pairs, or there is a single ground truth, in which case the code considers the scene
   to be fixed and uses a single ground truth for the entire execution of the code.

.. warning::
   If files are not paired correctly, the fusion pipeline will fail or produce
   meaningless results. Always verify that filenames match across ``visible/``
   and ``swir/``.

Tips
----

.. tip::
   - Keep datasets small when testing new parameters (sql mode).  
   - Use consistent naming conventions (e.g. ``img_###.png``).  
   - Store large datasets on HPC storage and mount them into the container
     using ``-B /path/to/data:/VISWIR/data``.
