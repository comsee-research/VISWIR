# ==============================================================================
# Implementation of "V-SWIR-IF: Visible and Short-Wave Infrared Image Fusion"
#
# Reference:
# H. Fang, G. Su, G. Xu, and C. Cheng, "V-SWIR-IF: Visible and Short-Wave 
# Infrared Image Fusion," 2023 4th International Symposium on Computer 
# Engineering and Intelligent Communications (ISCEIC), Suzhou, China, 2023, 
# pp. 275-280, doi: 10.1109/ISCEIC59030.2023.10271197.
#
# Re-implemented by: Alexandre Riffard
# Note: This code provides an exact reproduction of the proposed method 
# (Mattes Mutual Information registration and DWT db1 fusion).
# ==============================================================================

import cv2
import numpy as np
import SimpleITK as sitk
import pywt
import os

# ------------------------------------------------------------
# 1. Lecture des images
# ------------------------------------------------------------
def read_vis_swir(vis_path, swir_path):
    vis = cv2.imread(vis_path)              # RGB uint8
    swir = cv2.imread(swir_path, 0)         # SWIR grayscale
    return vis, swir

# ------------------------------------------------------------
# 2. Conversion RGB -> Gray
# ------------------------------------------------------------
def rgb_to_gray(vis):
    return cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

# ------------------------------------------------------------
# 3. Registration Mattes MI + 1+1 Evolutionary
# ------------------------------------------------------------
def register_swir_to_vis(vis_gray, swir_gray):
    fixed = sitk.GetImageFromArray(vis_gray.astype(np.float32))
    moving = sitk.GetImageFromArray(swir_gray.astype(np.float32))

    initial_tx = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    R = sitk.ImageRegistrationMethod()

    # Mattes MI
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingStrategy(R.RANDOM)

    total_pix = fixed.GetWidth() * fixed.GetHeight()
    pct = min(1.0, 20000 / max(1, total_pix))
    R.SetMetricSamplingPercentage(pct)

    R.SetInterpolator(sitk.sitkLinear)

    # Multi-resolution pyramid
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # 1+1 Evolutionary optimizer
    R.SetOptimizerAsOnePlusOneEvolutionary(
        numberOfIterations=300,
        initialRadius=5e-5,
        epsilon=1.5e-6,
        growthFactor=1.05,
        shrinkFactor=0.95
    )

    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(initial_tx, inPlace=False)

    final_tx = R.Execute(fixed, moving)

    swir_reg_itk = sitk.Resample(
        moving,
        fixed,
        final_tx,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32
    )

    swir_reg = sitk.GetArrayFromImage(swir_reg_itk)
    return swir_reg

# ------------------------------------------------------------
# 4. Fusion DWT
# ------------------------------------------------------------
def fuse_channel(vis_ch, swir_ch):
    cA_v, (cH_v, cV_v, cD_v) = pywt.dwt2(vis_ch, 'db1')
    cA_s, (cH_s, cV_s, cD_s) = pywt.dwt2(swir_ch, 'db1')

    # Low-frequency : average
    cA = 0.5 * (cA_v + cA_s)

    # High-frequency : absmax
    def absmax(a, b):
        return np.where(np.abs(a) >= np.abs(b), a, b)

    cH = absmax(cH_v, cH_s)
    cV = absmax(cV_v, cV_s)
    cD = absmax(cD_v, cD_s)

    fused = pywt.idwt2((cA, (cH, cV, cD)), 'db1')
    return fused

# ------------------------------------------------------------
# 5. Reconstruction RGB
# ------------------------------------------------------------
def fuse_rgb(vis, swir_reg):
    fused_R = fuse_channel(vis[:,:,0].astype(np.float32), swir_reg)
    fused_G = fuse_channel(vis[:,:,1].astype(np.float32), swir_reg)
    fused_B = fuse_channel(vis[:,:,2].astype(np.float32), swir_reg)

    fused_rgb = np.stack([fused_R, fused_G, fused_B], axis=-1)
    return fused_rgb

# ------------------------------------------------------------
# 6. Pipeline complet
# ------------------------------------------------------------
def v_swir_if_paper(vis_path, swir_path):
    vis, swir = read_vis_swir(vis_path, swir_path)
    vis_gray = rgb_to_gray(vis)
    swir_reg = register_swir_to_vis(vis_gray, swir)
    fused = fuse_rgb(vis, swir_reg)
    return fused

# ------------------------------------------------------------
# 7. Batch processing
# ------------------------------------------------------------
def process_folder_paper(visible_dir, swir_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(visible_dir):
        vis_path = os.path.join(visible_dir, filename)
        swir_path = os.path.join(swir_dir, filename)

        if not os.path.exists(swir_path):
            print(f"SWIR manquant : {filename}")
            continue

        fused = v_swir_if_paper(vis_path, swir_path)

        cv2.imwrite(os.path.join(output_dir, filename), fused)


    print("Terminé (version papier exacte).")

# ------------------------------------------------------------
# 8. Lanceur 
# ------------------------------------------------------------
if __name__ == "__main__":
    
    visible_dirs = [
        "./data/dataset_1/rgb",
        "./data/dataset_2/rgb"
    ]

    swir_dirs = [
        "./data/dataset_1/swir",
        "./data/dataset_2/swir"
    ]

    output_dirs = [
        "./output/method_name/dataset_1",
        "./output/method_name/dataset_2"
    ]

    for visible_dir, swir_dir, output_dir in zip(visible_dirs, swir_dirs, output_dirs):
        process_folder_paper(visible_dir, swir_dir, output_dir)
