"""
Pixel-level analysis on Suite2p-registered imaging data.

This script loads motion-corrected trial data, segments each trial into
pre-CS, CS-US, and post-US windows, then computes normalized response maps.
The results are smoothed and saved for activity-map visualization.
"""

# Imports
import pickle
from importlib import reload
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
import xarray as xr
from natsort import natsorted
from scipy.ndimage import gaussian_filter
from suite2p.io.binary import BinaryFile

import my_functions_imaging_new as fi
import plotting_style_new as plotting_style
from my_paths_new import fish_name, path_home


# Configuration
RELOAD_MODULES = True
USE_PLOTLY_DARK = True

PANDAS_OPTIONS = {
    "mode.copy_on_write": True,
    "compute.use_numba": True,
    "compute.use_numexpr": True,
    "compute.use_bottleneck": True,
}


def configure_environment(use_plotly_dark: bool) -> None:
    # Match plotting and pandas behavior with the rest of the pipeline.
    for option, value in PANDAS_OPTIONS.items():
        pd.set_option(option, value)
    if use_plotly_dark:
        pio.templates.default = "plotly_dark"
    plotting_style.set_plot_style(use_constrained_layout=False)


if RELOAD_MODULES:
    reload(fi)

configure_environment(USE_PLOTLY_DARK)

# Paths and dataset selection
fish_ID = "_".join(fish_name.split("_")[:2])

whole_data_path_save = Path(r'H:\2-P imaging') / path_home.stem / fish_name
whole_data_path_save.mkdir(parents=True, exist_ok=True)


path_pkl_after_motion_correction = whole_data_path_save / (fish_ID + '_2. After motion correction_Suite2p.pkl')
path_pkl_responses = whole_data_path_save / (fish_ID + '_3. Responses_Suite2p.pkl')


# Small offset to stabilize division when baseline intensities are low.
softthresh = 100


# Use naming conventions to infer CS->US interval when the protocol lacks explicit US timing.
if 'delay' in fish_name:
    interval_between_cs_onset_us_onset = 9
elif 'trace' in fish_name:
    interval_between_cs_onset_us_onset = 13
elif 'control' in fish_name:
    interval_between_cs_onset_us_onset = 9
else:
    interval_between_cs_onset_us_onset = None


# Load motion-corrected trials saved by the previous pipeline step.
if not path_pkl_after_motion_correction.exists():
    raise FileNotFoundError(f"Missing input pickle: {path_pkl_after_motion_correction}")
with open(path_pkl_after_motion_correction, 'rb') as file:
    all_data = pickle.load(file)


print('Analyzing fish: ', fish_name)


# Registered binaries produced by Suite2p.
reg_dir = whole_data_path_save / "registeredBin"
if not reg_dir.is_dir():
    raise FileNotFoundError(f"Suite2p registeredBin not found: {reg_dir}")

# Use the first trial to infer frame dimensions.
Ly = int(all_data.planes[0].trials[0].images.shape[1])
Lx = int(all_data.planes[0].trials[0].images.shape[2])


reg_files = natsorted([f.name for f in reg_dir.glob("*_reg.bin")])
if not reg_files:
    raise FileNotFoundError(f"No registered binaries found in {reg_dir}")

# Process each registered trial: load registered frames and segment in time.
for fname in reg_files:
    plane_number = int(fname.split('_')[1])
    trial_number = int(fname.split('_')[3])
    path_in = reg_dir / fname

    trial = all_data.planes[plane_number].trials[trial_number]
    frames_per_plane = int(trial.images.shape[0])
    f_raw = BinaryFile(Ly=Ly, Lx=Lx, filename=str(path_in), n_frames=frames_per_plane)
    images = [f_raw[j] for j in range(frames_per_plane)]

    # Replace the raw images with the registered frames from Suite2p.
    if images and images[0].shape == tuple(trial.images.shape[1:]) and len(images) == trial.images.shape[0]:
        trial.images.values = np.asarray(images)
    else:
        trial.images = xr.DataArray(
            np.asarray(images),
            dims=trial.images.dims,
            coords={
                trial.images.dims[0]: trial.images[trial.images.dims[0]].values[:len(images)],
                'y': trial.images['y'],
                'x': trial.images['x'],
            },
            name='images',
        )

    # Extract stimulus timing from the trial protocol table.
    protocol = trial.protocol

    cs_beg_candidates = protocol.loc[protocol['CS beg'] != 0, 'Time (ms)'].values
    cs_end_candidates = protocol.loc[protocol['CS end'] != 0, 'Time (ms)'].values
    if cs_beg_candidates.size == 0 or cs_end_candidates.size == 0:
        raise ValueError(f"Missing CS timing in protocol for plane {plane_number}, trial {trial_number}")
    cs_beg_time = cs_beg_candidates[0]

    us_beg_series = protocol.loc[protocol['US beg'] != 0, 'Time (ms)']
    if us_beg_series.empty:
        if interval_between_cs_onset_us_onset is None:
            raise ValueError("US timing missing and no interval specified in fish name.")
        us_beg_time = cs_beg_time + interval_between_cs_onset_us_onset * 1000
    else:
        us_beg_time = us_beg_series.values[0]
    us_end_series = protocol.loc[protocol['US end'] != 0, 'Time (ms)']
    if us_end_series.empty:
        # Default to a 100 ms US when only onset timing is available.
        us_end_time = us_beg_time + 100
    else:
        us_end_time = us_end_series.values[0]

    # Ensure good-frame masking exists so averages ignore bad frames.
    trial_images = trial.images.copy().fillna(0)
    if 'mask good frames' not in trial_images.coords:
        trial_images = trial_images.assign_coords(
            {'mask good frames': ('Time (ms)', np.ones(trial_images.shape[0], dtype=bool))}
        )

    # Build time masks for baseline, CS-US, and post-US windows.
    trial_images['mask before cs'] = trial_images['Time (ms)'] < cs_beg_time
    trial_images['mask cs-us'] = (trial_images['Time (ms)'] > cs_beg_time) & (trial_images['Time (ms)'] < us_beg_time)
    trial_images['mask after us'] = trial_images['Time (ms)'] > us_end_time

    # Compute mean images for each window, restricted to good frames.
    trial.pre_cs_mean = trial_images.sel({'Time (ms)': (trial_images['mask before cs']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')
    trial.cs_us_mean = trial_images.sel({'Time (ms)': (trial_images['mask cs-us']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')
    trial.post_us_mean = trial_images.sel({'Time (ms)': (trial_images['mask after us']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')

    # Persist changes back into the data structure.
    all_data.planes[plane_number].trials[trial_number] = trial


for plane_i, plane in enumerate(all_data.planes):
    for trial_i, trial in enumerate(plane.trials):

        # Normalize CS-US and post-US responses by pre-CS baseline, then smooth.
        pre_cs_data = trial.pre_cs_mean.values

        trial.cs_us_vs_pre = gaussian_filter(((trial.cs_us_mean.values - pre_cs_data) / (pre_cs_data + softthresh)), sigma=2)
        trial.post_us_vs_pre = gaussian_filter(((trial.post_us_mean.values - pre_cs_data) / (pre_cs_data + softthresh)), sigma=2)

        # Keep a lightly normalized anatomy image for overlay visualization.
        trial.anatomy = fi.normalize_image(all_data.planes[plane_i].trials[trial_i].template_image, (0.01,0.99)) / 10


# Save response maps for downstream activity-map visualization.
with open(path_pkl_responses, 'wb') as file:
    pickle.dump(all_data, file)


# Run the activity-map generation step on the freshly saved responses.
exec(open('4.Activity_maps.py').read())


print('END')
