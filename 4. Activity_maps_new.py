"""
Activity-map visualization for Suite2p response maps.

This script loads per-trial response images computed in the previous step,
keeps only positive CS responses, normalizes them, and overlays the activity
on top of the anatomy template. The outputs include a plane-by-trial summary
figure and per-plane multi-page TIFF stacks for downstream ROI inspection.
"""

import pickle
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import tifffile
from tqdm import tqdm

import my_functions_imaging_new as fi
import plotting_style_new as plotting_style
from my_paths_new import fish_name, path_home


RELOAD_MODULES = True
USE_PLOTLY_DARK = True

PANDAS_OPTIONS = {
    "mode.copy_on_write": True,
    "compute.use_numba": True,
    "compute.use_numexpr": True,
    "compute.use_bottleneck": True,
}


def configure_environment(use_plotly_dark: bool) -> None:
    # Keep plotting styles and pandas settings consistent with the pipeline.
    for option, value in PANDAS_OPTIONS.items():
        pd.set_option(option, value)
    if use_plotly_dark:
        pio.templates.default = "plotly_dark"
    plotting_style.set_plot_style(use_constrained_layout=False)


if RELOAD_MODULES:
    reload(fi)
configure_environment(USE_PLOTLY_DARK)


fish_id = "_".join(fish_name.split("_")[:2])

results_root = Path(r"F:\Results (paper)") / path_home.stem
results_figs_path_save = results_root / "Neurons" / fish_name
results_figs_path_save.mkdir(parents=True, exist_ok=True)

whole_data_path_save = Path(r"H:\2-P imaging") / path_home.stem / fish_name
path_pkl_responses = whole_data_path_save / f"{fish_id}_3. Responses_Suite2p.pkl"
if not path_pkl_responses.exists():
    raise FileNotFoundError(f"Responses pickle not found: {path_pkl_responses}")


with open(path_pkl_responses, "rb") as file:
    all_data = pickle.load(file)

print("Analyzing fish: ", fish_name)

if not getattr(all_data, "planes", None):
    raise ValueError("No planes found in the response data.")

num_planes = len(all_data.planes)
num_trials_per_plane = len(all_data.planes[0].trials)
if num_trials_per_plane == 0:
    raise ValueError("No trials found in the response data.")

frame_shape = all_data.planes[0].trials[0].images.shape[1:]
image_height = frame_shape[0]
image_width = frame_shape[1]


fig, axs = plt.subplots(num_planes, num_trials_per_plane, squeeze=False, facecolor="w")

# Create activity overlays for each plane and trial, then save per-plane stacks.
tiff_output_dir = Path.home() / "Desktop"
tiff_output_dir.mkdir(parents=True, exist_ok=True)

for plane_i, plane in tqdm(enumerate(all_data.planes), total=num_planes):
    plane_frames = []
    for trial_i, trial in enumerate(plane.trials):
        if not hasattr(trial, "cs_us_vs_pre"):
            raise AttributeError(f"Trial missing cs_us_vs_pre (plane {plane_i}, trial {trial_i}).")
        if not hasattr(trial, "anatomy"):
            raise AttributeError(f"Trial missing anatomy (plane {plane_i}, trial {trial_i}).")

        # Keep only positive CS responses and normalize for visualization.
        response = np.where(trial.cs_us_vs_pre > 0, trial.cs_us_vs_pre, 0)
        response_frame = fi.normalize_image(response, quantiles=(0, 1))

        # Overlay the response on anatomy using a fixed threshold for display.
        overlay = fi.add_colors_to_world_improved_2(
            trial.anatomy * 2.5,
            response_frame,
            colormap="inferno",
            activity_threshold=0.4,
            alpha=1,
        )

        axs[plane_i, trial_i].imshow(overlay, interpolation="none")
        axs[plane_i, trial_i].axis("off")
        plane_frames.append(overlay)

    tiff_path = tiff_output_dir / f"{fish_id}_plane_{plane_i}_positive_responses.tiff"
    tifffile.imwrite(tiff_path, plane_frames, photometric="rgb")


# Build a compact summary figure covering all planes and trials.
fig.subplots_adjust(hspace=0, wspace=0.05)
fig.set_size_inches(num_trials_per_plane * image_width / 100, num_planes * image_height / 100)
fig.suptitle(f"{fish_name} Positive responses to CS", fontsize=10, y=0.95)
fig.savefig(
    results_figs_path_save / "6.1. Positive responses to CS_Suite2p.png",
    dpi=600,
    facecolor="white",
    bbox_inches="tight",
)
