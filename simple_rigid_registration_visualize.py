#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402


FIXED_PATH = Path(
    "/Users/rifqiab2708/Documents/img_regist/dataset/processed/patient_00001/fixed.nii.gz"
)
MOVING_PATH = Path(
    "/Users/rifqiab2708/Documents/img_regist/dataset/processed/patient_00001/moving.nii.gz"
)

TRANSFORM_OUT = Path(
    "/Users/rifqiab2708/Documents/img_regist/dataset/transforms/patient_00001/rigid.tfm"
)
REGISTERED_OUT = Path(
    "/Users/rifqiab2708/Documents/img_regist/dataset/results/patient_00001/moving_registered_to_fixed.nii.gz"
)
VIZ_OUT = Path(
    "/Users/rifqiab2708/Documents/img_regist/dataset/results/patient_00001/rigid_registration_viz.png"
)
METRICS_OUT = Path(
    "/Users/rifqiab2708/Documents/img_regist/dataset/results/patient_00001/rigid_registration_metrics.json"
)


def print_geometry(name: str, image: sitk.Image) -> None:
    print(f"{name}:")
    print(f"  size      : {image.GetSize()}")
    print(f"  spacing   : {image.GetSpacing()}")
    print(f"  origin    : {image.GetOrigin()}")
    print(f"  direction : {image.GetDirection()}")


def robust_normalize(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def extract_center_slices(volume_zyx: np.ndarray) -> Dict[str, np.ndarray]:
    zc = volume_zyx.shape[0] // 2
    yc = volume_zyx.shape[1] // 2
    xc = volume_zyx.shape[2] // 2

    return {
        "axial": volume_zyx[zc, :, :],
        "coronal": volume_zyx[:, yc, :],
        "sagittal": volume_zyx[:, :, xc],
    }


def display_oriented(slice_2d: np.ndarray) -> np.ndarray:
    # Rotate for consistent viewing orientation.
    return np.rot90(slice_2d)


def make_overlay_and_diff(
    fixed_slice: np.ndarray, moving_slice: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    fixed_n = robust_normalize(fixed_slice)
    moving_n = robust_normalize(moving_slice)

    rgb = np.zeros((*fixed_n.shape, 3), dtype=np.float32)
    rgb[..., 0] = moving_n  # R
    rgb[..., 1] = fixed_n  # G
    rgb[..., 2] = moving_n  # B

    diff = np.abs(fixed_n - moving_n)
    return rgb, diff


def rigid_register_3d(fixed: sitk.Image, moving: sitk.Image) -> Tuple[sitk.Transform, float, str, str]:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    fixed_norm = sitk.Normalize(fixed)
    moving_norm = sitk.Normalize(moving)

    initial = sitk.CenteredTransformInitializer(
        fixed_norm,
        moving_norm,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)

    sampling_note = "stochastic (default wall-clock seed)"
    try:
        reg.SetMetricSamplingPercentage(0.20, 42)
        sampling_note = "deterministic seed=42"
    except TypeError:
        reg.SetMetricSamplingPercentage(0.20)

    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)

    final_transform = reg.Execute(fixed_norm, moving_norm)
    final_metric = float(reg.GetMetricValue())
    stop_condition = reg.GetOptimizerStopConditionDescription()

    return final_transform, final_metric, stop_condition, sampling_note


def transform_summary(transform: sitk.Transform) -> Dict:
    summary: Dict = {
        "name": transform.GetName(),
        "dimension": transform.GetDimension(),
    }

    if hasattr(transform, "GetParameters"):
        params = list(transform.GetParameters())
        summary["parameters"] = params
        if len(params) >= 6:
            summary["rotation_xyz_radians"] = params[:3]
            summary["translation_xyz_mm"] = params[3:6]
    if hasattr(transform, "GetFixedParameters"):
        summary["fixed_parameters"] = list(transform.GetFixedParameters())
    return summary


def build_visualization(fixed: sitk.Image, moving: sitk.Image, registered: sitk.Image, out_png: Path) -> None:
    fixed_arr = sitk.GetArrayFromImage(fixed)
    moving_arr = sitk.GetArrayFromImage(moving)
    reg_arr = sitk.GetArrayFromImage(registered)

    fixed_slices = extract_center_slices(fixed_arr)
    moving_slices = extract_center_slices(moving_arr)
    reg_slices = extract_center_slices(reg_arr)

    panel_titles = [
        "Fixed (original)",
        "Moving (original)",
        "Registered overlap + diff",
    ]
    plane_order = ["axial", "coronal", "sagittal"]

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(1, 3, wspace=0.12)

    for col in range(3):
        sub = gs[0, col].subgridspec(3, 1, hspace=0.03)
        for row, plane in enumerate(plane_order):
            ax = fig.add_subplot(sub[row, 0])
            ax.set_axis_off()

            if row == 0:
                ax.set_title(panel_titles[col], fontsize=11, pad=6)
            ax.text(
                0.01,
                0.02,
                plane,
                color="white",
                fontsize=8,
                transform=ax.transAxes,
                bbox={"facecolor": "black", "alpha": 0.4, "pad": 2},
            )

            if col == 0:
                sl = display_oriented(robust_normalize(fixed_slices[plane]))
                ax.imshow(sl, cmap="gray", vmin=0.0, vmax=1.0)
            elif col == 1:
                sl = display_oriented(robust_normalize(moving_slices[plane]))
                ax.imshow(sl, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                overlay, diff = make_overlay_and_diff(fixed_slices[plane], reg_slices[plane])
                overlay = display_oriented(overlay)
                diff = display_oriented(diff)
                ax.imshow(overlay, vmin=0.0, vmax=1.0)
                ax.imshow(diff, cmap="inferno", alpha=0.35, vmin=0.0, vmax=1.0)

    fig.suptitle("Rigid Registration: moving -> fixed", fontsize=13)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def ensure_output_dirs() -> None:
    TRANSFORM_OUT.parent.mkdir(parents=True, exist_ok=True)
    REGISTERED_OUT.parent.mkdir(parents=True, exist_ok=True)
    VIZ_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_output_dirs()

    fixed = sitk.ReadImage(str(FIXED_PATH), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(MOVING_PATH), sitk.sitkFloat32)

    print("=== Input Geometry ===")
    print_geometry("fixed", fixed)
    print_geometry("moving", moving)

    final_transform, final_metric, stop_condition, sampling_note = rigid_register_3d(fixed, moving)

    registered = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )

    sitk.WriteTransform(final_transform, str(TRANSFORM_OUT))
    sitk.WriteImage(registered, str(REGISTERED_OUT))
    build_visualization(fixed, moving, registered, VIZ_OUT)

    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"fixed": str(FIXED_PATH), "moving": str(MOVING_PATH)},
        "registration_parameters": {
            "transform_type": "Euler3DTransform (rigid)",
            "metric": "MattesMutualInformation",
            "histogram_bins": 50,
            "sampling_strategy": "RANDOM",
            "sampling_percentage": 0.20,
            "sampling_note": sampling_note,
            "optimizer": "GradientDescent",
            "learning_rate": 1.0,
            "number_of_iterations": 200,
            "convergence_minimum_value": 1e-6,
            "convergence_window_size": 10,
            "shrink_factors": [4, 2, 1],
            "smoothing_sigmas_physical": [2, 1, 0],
        },
        "image_geometry": {
            "fixed": {
                "size": list(fixed.GetSize()),
                "spacing": list(fixed.GetSpacing()),
                "origin": list(fixed.GetOrigin()),
                "direction": list(fixed.GetDirection()),
            },
            "moving": {
                "size": list(moving.GetSize()),
                "spacing": list(moving.GetSpacing()),
                "origin": list(moving.GetOrigin()),
                "direction": list(moving.GetDirection()),
            },
            "registered": {
                "size": list(registered.GetSize()),
                "spacing": list(registered.GetSpacing()),
                "origin": list(registered.GetOrigin()),
                "direction": list(registered.GetDirection()),
            },
        },
        "final_metric_value": final_metric,
        "optimizer_stop_condition": stop_condition,
        "transform": transform_summary(final_transform),
        "outputs": {
            "transform": str(TRANSFORM_OUT),
            "registered_image": str(REGISTERED_OUT),
            "visualization": str(VIZ_OUT),
        },
    }

    with METRICS_OUT.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Registration Result ===")
    print(f"optimizer stop condition: {stop_condition}")
    print(f"final metric: {final_metric:.8f}")
    print("\n=== Output Paths ===")
    print(f"transform: {TRANSFORM_OUT}")
    print(f"registered image: {REGISTERED_OUT}")
    print(f"visualization: {VIZ_OUT}")
    print(f"metrics: {METRICS_OUT}")


if __name__ == "__main__":
    main()
