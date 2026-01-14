"""
5. Cell Detection (ROI Identification)

This script performs automated cell/ROI detection on registered imaging data using Suite2p algorithms.

Intended Workflow (under development):
1. Load motion-corrected imaging data
2. Apply Suite2p cell detection algorithms:
    - Compute spatial correlation maps
    - Identify candidate ROIs based on correlation thresholds
    - Extract fluorescence traces for detected cells
3. Validate detected ROIs against quality metrics
4. Export ROI masks and time-series data

Dependencies:
- suite2p.detection: Core cell detection algorithms
- scipy.ndimage: Spatial filtering operations
- skimage.measure: Region analysis tools

Note: This script is currently under development. For ROI-based analysis,
consider using Suite2p's full pipeline or custom correlation-based methods
implemented in earlier script versions (see commented code in 3.x scripts).

Key Suite2p Detection Parameters:
- threshold: Correlation threshold for ROI identification
- max_pixels: Maximum pixels per ROI
- spatial_scale: Gaussian smoothing scale for correlation maps

Output (planned):
- ROI masks (spatial footprints)
- Fluorescence time-series per ROI
- Quality metrics (SNR, correlation scores)
"""

#* Imports

# %% 
# region Imports
import pickle
from importlib import reload
from pathlib import Path
from xml.etree.ElementPath import \
    ops  # xml helper (not used heavily here, kept for compatibility)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import tifffile
import xarray as xr
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from natsort import natsorted
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from skimage.measure import block_reduce
from suite2p import default_ops, detection, io, registration
from suite2p.registration import register
from suite2p.registration.register import compute_reference, register_frames
from suite2p.run_s2p import run_s2p
from tqdm import tqdm

#* Load custom functions and classes
import my_classes as c
# import my_experiment_specific_variables as spec_var
import my_functions_imaging as fi
import my_parameters as p
from my_general_variables import *

# Save all data in a single pickle file.
# Anatomical stack images and imaging data are median filtered.

#* Imports (end)

# endregion

# Reload user modules to pick up changes during interactive development.
reload(fi)
reload(c)
reload(p)

#* Settings
##    Settings
# region Settings

# Use Plotly dark template by default for any plotly figures
pio.templates.default = "plotly_dark"

# Pandas performance / behaviour options used across the pipeline
pd.set_option("mode.copy_on_write", True)
pd.set_option("compute.use_numba", True)
pd.set_option("compute.use_numexpr", True)
pd.set_option("compute.use_bottleneck", True)
#endregionx


#* Paths and dataset selection
# region Paths
# Base folder for current experiment (change per experiment in my_paths.py normally)
path_home = Path(r'D:\2024 09_Delay 2-P 4 planes JC neurons')

# Folder where results will be stored
path_results_save = Path(r'F:\Results (paper)') / path_home.stem

# Current fish dataset identifier (folder name)
fish_name = r'20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'

# Color/phase helper lists used for plotting labelled tiles later
color_list = ['cyan']*2 + ['red']*8 + ['yellow']*6
phase_list = ['Pre-Train']*2 + ['Train']*8 + ['Test']*6
helper = np.array([0,1, 0,1, 10,11, 20,21, 30,31, 0,1, 12,13, 22,23])+1
trial_numbers_list = np.array([helper, helper+2, helper+4, helper+6])

# If path contains 'top' change lists to accommodate a different session format
if 'top' in str(path_home):
     color_list = ['white']*2 + ['red']*2
     phase_list = ['Pre-Train']*2 + ['Train']*2
     helper = np.array([0,1, 10,11])+1
     trial_numbers_list = np.array([helper+i*2 for i in range(15)])

# Fish ID prefix used for filenames
fish_ID = '_'.join(fish_name.split('_')[:2])

# Local paths for behavior and imaging subfolders
behavior_path_home = path_home / 'Tail'
imaging_path_home = path_home / 'Neurons' / fish_name

behavior_path_save = path_results_save / 'Tail'
results_figs_path_save = path_results_save / 'Neurons' / fish_name

# Where processed imaging data is stored (H: is a network drive in this environment)
whole_data_path_save = Path(r'H:\2-P imaging') / path_home.stem / fish_name

# Pickle file containing responses (after Suite2p motion correction / processing)
path_pkl_responses = whole_data_path_save / (fish_ID + '_3. Responses_Suite2p.pkl')

# Create interval between CS and US onsets depending on experiment type
if 'delay' in fish_name:
     interval_between_cs_onset_us_onset = 9  # seconds
elif 'trace' in fish_name:
     interval_between_cs_onset_us_onset = 13  # seconds
elif 'control' in fish_name:
     interval_between_cs_onset_us_onset = 9  # seconds
else:
     interval_between_cs_onset_us_onset = None

#* Load the data before motion correction.
# region Load the data before motion correction
with open(path_pkl_responses, 'rb') as file:
     all_data = pickle.load(file)

print('Analyzing fish: ', fish_name)
# endregion

# Get number of imaging planes and trials per plane (assumes uniform counts)
num_planes = len(all_data.planes)
num_trials_per_plane = len(all_data.planes[0].trials)

# Image shape info from first trial
shape_ = all_data.planes[0].trials[0].images.shape[1:]
image_dim_1 = shape_[0]
image_dim_2 = shape_[1]

# region Maps per single trials
##* CS positive response - analysis per trial on a per-plane basis

# Use default Suite2p ops as a starting template then update values used for detection
ops = default_ops()
ops.update({
     # Registration parameters (kept as reference; many are not used if only detecting on already registered data)
     'nonrigid': True,
     'two_step_registration': True,
     'maxregshift': 0.1,
     'maxregshiftNR': 5,
     'nimg_init': 180,
     'th_badframes': 0.7,
     'batch_size': 1,
     'smooth_sigma_time': 0.5,
     'smooth_sigma_space': 0.5,
     'fs': 30,
     'max_iterations': 100,
     'align_by_chan': 1,
     'chan2_use': 1,
     'tau': 0.7,
     'force_smooth': True,
     'keep_movie_raw': False,
     'keep_movie_rig': False,
     'keep_movie_dff': False,
     'keep_movie_corr': False,

     # Cell detection parameters (tune these for your data)
     'connected': True,             # whether to keep ROIs fully connected
     'navg_frames_svd': 5000,         # max frames for SVD computation
     'nsvd_for_roi': 1000,          # number of SVD components for ROI detection
     'threshold_scaling': 10.0,       # adjust threshold (higher = fewer ROIs)
     'max_overlap': 0.75,           # maximum allowed overlap between ROIs
     'high_pass': 100,              # running mean subtraction window (set 0 to disable)
     'spatial_scale': 0,            # spatial scale for filtering (0 = no spatial filtering)
     'sparse_mode': True,           # use sparse matrix operations
     'diameter': 10,                # expected diameter of cells in pixels
     'spatial_hp_detect': 25,         # window for spatial high-pass filtering before detection
     'threshold_scaling': 1.0,        # scaling of the threshold (redundant here but kept)
     'max_iterations': 20,          # max iterations for cell detection

     # Neuropil correction parameters
     'clust_model': 'surround',
     'neuropil_radius': 30,
     'inner_neuropil_radius': 2,
     'max_neuropil_corr': 0.7,
     'min_neuropil_pixels': 100,

     # Classifier parameters (optional)
     'classifier_path': None,
     'soma_crop': True,
})

# Loop through each plane and each trial to compute voxelized signals and a simple voxel-based ΔF/F
for plane_i, plane in tqdm(enumerate(all_data.planes), desc='Processing planes'):

     # Example: process only plane index 2 in this run (comment out to process all planes)
     if plane_i != 2:
        continue

     # Get concatenated images for the plane and crop edges to avoid border artifacts
     plane_images = all_data.planes[plane_i].get_all_images()[:,100:-100, 100:]

     # Compute a plane-level correlation map to mask out low-correlation pixels (inspired by Suite2p Vmap)
     correlation_map = np.linalg.norm(
        ndimage.gaussian_filter(plane_images, sigma=p.correlation_map_sigma, axes=(1,2)),
        axis=0
     )**2 / ndimage.gaussian_filter(np.linalg.norm(plane_images, axis=0), sigma=p.correlation_map_sigma)**2

     # Small mask for replacing low-correlation pixels with a median value later
     mask = correlation_map <= 0.5  # default correlation threshold

     # Iterate trials within the plane
     for trial_i, trial in enumerate(tqdm(plane.trials, desc=f'Plane {plane_i} trials', leave=False)):

        # Copy and crop trial images to align with plane crop above
        trial_images = trial.images.copy()[:,100:-100, 100:]

        # Apply median and gaussian filtering to reduce noise before voxelization/detection
        trial_images.values = median_filter(trial_images, size=(1,3,3))   # filter across spatial dims, leave time intact
        trial_images.values = gaussian_filter(trial_images, sigma=(0,1,1)) # small gaussian blur per-frame

        # Mask out pixels with low correlation (set to zero) to reduce false positives
        correlation_thr = 0.5
        trial_images = trial_images.where(correlation_map > correlation_thr, 0)

        # Compute a median frame and replace masked low-correlation pixels with median to avoid artifacts
        median_frame = np.median(trial_images, axis=0)
        for t in range(trial_images.shape[0]):
              frame = trial_images[t,:,:].values
              frame[mask] = median_frame[mask]
              trial_images[t] = frame

        # Visual check: mean image for the trial after processing
        plt.imshow(trial_images.mean(dim='Time (ms)'), interpolation='none')

        # Bin the images into voxels using block_reduce (time axis left unchanged)
        voxel_bin_size = p.voxel_bin_size  # read configured voxel bin size from parameters module
        # block_reduce can accept a tuple block_size that matches the array shape; here first axis = time (1)
        trial_images_binned = block_reduce(trial_images.values, block_size=(1, p.voxel_bin_size, p.voxel_bin_size), func=np.mean, cval=0)

        # Store binned voxel data on the trial object for downstream steps
        trial.voxel_images = trial_images_binned

        # Create a binned correlation map to mask out voxels with low correlation
        correlation_map_binned = block_reduce(correlation_map, block_size=(p.voxel_bin_size, p.voxel_bin_size), func=np.mean, cval=0)
        voxel_mask = correlation_map_binned > correlation_thr

        # Flatten voxel data to traces (time x voxels) and keep only valid voxels
        voxel_traces = trial_images_binned.reshape(trial_images_binned.shape[0], -1)
        voxel_mask_flat = voxel_mask.flatten()
        voxel_traces = voxel_traces[:, voxel_mask_flat]

        # Extract stimulus timing from the trial protocol
        cs_beg_time = trial.protocol.loc[trial.protocol['CS beg']!=0, 'Time (ms)'].values[0]
        cs_end_time = trial.protocol.loc[trial.protocol['CS end']!=0, 'Time (ms)'].values[0]
        us_beg_time = cs_beg_time + interval_between_cs_onset_us_onset*1000  # compute US onset from CS onset + interval

        # Convert stimulus times to frame indices for time axis alignment
        time_coords = trial.images['Time (ms)'].values
        cs_start_frame = np.argmin(np.abs(time_coords - cs_beg_time))
        cs_end_frame = np.argmin(np.abs(time_coords - cs_end_time))
        us_start_frame = np.argmin(np.abs(time_coords - us_beg_time))

        # Baseline frames: before CS onset
        baseline_frames = time_coords < cs_beg_time
        baseline = np.mean(voxel_traces[baseline_frames, :], axis=0)

        # Soft threshold for denominator in ΔF/F to avoid division by zero and dampen small numerators
        softthresh = 20

        # Calculate ΔF/F per voxel over time. Keep shape (time x voxels).
        deltaF_over_F = (voxel_traces - baseline) / (baseline + softthresh)

        # Replace invalid numbers (NaN/inf) with zeros
        deltaF_over_F = np.where(np.isnan(deltaF_over_F) | np.isinf(deltaF_over_F), 0, deltaF_over_F)

        # Store voxel ΔF/F on the trial object for later analysis
        trial.voxel_deltaF_over_F = deltaF_over_F

        # Compute indices/times to highlight CS window for plotting
        cs_period_mask = (time_coords >= cs_beg_time) & (time_coords <= cs_end_time-1000)







        #!!!!!!!!!!!!!!!!!! Select voxels with highest peak during CS period for visualization
        voxel_max_values = np.max(deltaF_over_F[cs_period_mask, :], axis=0)
        top_voxel_indices = np.argsort(voxel_max_values)[-25:][::-1]  # top 25 voxels (descending)






        num_traces_to_plot = min(25, len(top_voxel_indices))

        # Plot individual voxel ΔF/F traces for the top voxels
        fig, axs = plt.subplots(num_traces_to_plot, 1, figsize=(14, num_traces_to_plot*0.6), sharex=True)
        if num_traces_to_plot == 1:
              axs = [axs]

        # X-axis time in milliseconds
        time_ms = time_coords

        for i in range(num_traces_to_plot):
              voxel_idx = top_voxel_indices[i]
              trace = deltaF_over_F[:, voxel_idx]

              # Plot trace as black line
              axs[i].plot(time_ms, trace, linewidth=0.8, color='black', alpha=0.8)

              # Highlight CS and US periods with colored spans
              axs[i].axvspan(cs_beg_time, cs_end_time, alpha=0.15, color='green', label='CS' if i == 0 else '')
              axs[i].axvspan(us_beg_time, us_beg_time + 500, alpha=0.15, color='red', label='US' if i == 0 else '')

              # Clean up axis appearance for stacked traces
              axs[i].set_yticks([])
              axs[i].spines['top'].set_visible(False)
              axs[i].spines['right'].set_visible(False)
              axs[i].spines['left'].set_visible(False)
              axs[i].spines['bottom'].set_visible(False)
              axs[i].set_xlim(time_ms[0], time_ms[-1])
              axs[i].axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

              # Add a small label to mark ranking of voxel
              axs[i].text(1.01, 0.5, f'#{i+1}', transform=axs[i].transAxes,
                          fontsize=8, va='center', color='gray')

        # Enable bottom x-axis on last subplot
        axs[-1].spines['bottom'].set_visible(True)
        axs[-1].set_xlabel('Time (ms)', fontsize=11)
        axs[-1].tick_params(axis='x', labelsize=9)

        # Legend only on first axis to avoid clutter
        if num_traces_to_plot > 0:
              axs[0].legend(loc='upper left', fontsize=8, frameon=False)

        # Title and y-label for the whole figure
        fig.suptitle(f'Top {num_traces_to_plot} Voxel ΔF/F Traces - Plane {plane_i}, Trial {trial_i}',
                        fontsize=12, fontweight='bold', y=0.995)
        fig.text(0.02, 0.5, 'ΔF/F (a.u.)', va='center', rotation='vertical', fontsize=11)

        plt.tight_layout(rect=[0.03, 0, 0.97, 0.99])
        plt.subplots_adjust(hspace=0.05)
        plt.show()

        # Store flattened voxel traces (time x selected_voxels mask) on trial object (raw binned traces)
        trial.voxel_traces = voxel_traces

        # Plot median ΔF/F across top voxels to see mean response shape
        median_deltaF = np.median(deltaF_over_F[:, top_voxel_indices[:25]], axis=1)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(time_ms, median_deltaF, linewidth=1.5, color='black')

        # Add stimulus shading again for the median trace
        ax.axvspan(cs_beg_time, cs_end_time, alpha=0.15, color='green', label='CS')
        ax.axvspan(us_beg_time, us_beg_time + 500, alpha=0.15, color='red', label='US')

        # Baseline line at zero and labels
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Median ΔF/F (a.u.)', fontsize=11)
        ax.set_title(f'Median ΔF/F Across Top 25 Voxels - Plane {plane_i}, Trial {trial_i}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(time_ms[0], time_ms[-1])
        plt.tight_layout()
        plt.show()

        plt.savefig(results_figs_path_save / f'Plane_{plane_i}_Trial_{trial_i}_Top25VoxelMedianDeltaF.png', dpi=300)

     # Note: The code above only used a subset of Suite2p functions. Suite2p detection
     # (detection.detection_wrapper or full run_s2p) is left commented in later sections
     # for cases where the full pipeline should be executed.

# Visualize the anatomical stack (summed projection)
plt.imshow(np.sum(all_data.anatomical_stack, axis=0))

# Quick introspection of an example trial to show available attributes
all_data.planes[0].trials[0].__dict__.keys()

# Build a tiled figure of positive responses (CS) for all planes and trials.
fig, axs = plt.subplots(num_planes, num_trials_per_plane, squeeze=False, facecolor='w')
for plane_i, plane in enumerate(all_data.planes):
     plane_frames = []  # List to store image frames to save as multipage TIFF later
     for trial_i, trial in enumerate(plane.trials):
        # Use precomputed cs_us_vs_pre response and only keep positive values for visualization
        response = np.where(trial.cs_us_vs_pre > 0, trial.cs_us_vs_pre, 0)
        response_frame = fi.normalize_image(response, quantiles=(0, 1))

        # Create a RGB overlay by mapping response_frame onto a colormap over the anatomy
        image = fi.add_colors_to_world_improved_2(trial.anatomy*2.5, response_frame, colormap='inferno', activity_threshold=0.4, alpha=1)
        axs[plane_i, trial_i].imshow(image, interpolation='none')
        axs[plane_i, trial_i].axis('off')

        # Save frame for TIFF export (keeps the RGB frame)
        plane_frames.append(image)
        
        break
     
     # Export frames for this plane as a multipage TIFF to desktop for quick inspection
     tiff_path = Path(r"C:\Users\joaquim\Desktop") / f'{fish_ID}_plane_{plane_i}_positive_responses.tiff'
     tifffile.imwrite(tiff_path, plane_frames, photometric='rgb')
     
     break


# Adjust spacing and save the summary figure
fig.subplots_adjust(hspace=0, wspace=0)
fig.set_size_inches(len(all_data.planes[0].trials) * image_dim_2/100, len(all_data.planes) * image_dim_1/100)
fig.suptitle(f'{fish_name} Positive responses to CS', fontsize=10, y=0.95)
fig.savefig(results_figs_path_save / '6.1. Positive responses to CS_Suite2p.png', dpi=600, facecolor='white', bbox_inches='tight')





                        # # ##* CS negative response
                        # # fig, axs = plt.subplots(num_planes, num_trials_per_plane, figsize=(15 * image_dim_1/614, 7 * image_dim_2/350), squeeze=False)
                        # # for plane_i, plane in enumerate(all_data.planes):
                        # # 	for trial_i, trial in enumerate(plane.trials):
                        # # 		response = np.where(trial.cs_us_vs_pre < 0, trial.cs_us_vs_pre, 0)
                        # # 		response_frame = fi.normalize_image(-response, quantiles=(0, 1))
                        # # 		# response_thr = np.quantile(response_frame, 0.99)
                        # # 		axs[plane_i,trial_i].imshow(fi.add_colors_to_world_improved_2(trial.anatomy * 2.5, response_frame, colormap='inferno', activity_threshold=0.5, alpha=1), interpolation='none')
                        # # 		axs[plane_i,trial_i].axis('off')
                        # # fig.subplots_adjust(hspace=0.01, wspace=0)
                        # # fig.tight_layout()
                        # # fig.suptitle(f'{fish_name}/nNegative responses to CS', fontsize=10, y=0.95)
                        # # fig.savefig(results_figs_path_save / '7.1. Negative responses to CS.png', dpi=600, facecolor='white', bbox_inches='tight')

                        # # ##* US positive response
                        # fig, axs = plt.subplots(num_planes, num_trials_per_plane, figsize=(15 * image_dim_1/614, 7 * image_dim_2/350), squeeze=False)
                        # for plane_i, plane in enumerate(all_data.planes):
                        # 	for trial_i, trial in enumerate(plane.trials):
                        # 		response = np.where(trial.post_us_vs_pre > 0, trial.post_us_vs_pre, 0)
                        # 		response_frame = fi.normalize_image(response, quantiles=(0, 1))
                        # 		axs[plane_i,trial_i].imshow(fi.add_colors_to_world_improved_2(trial.anatomy * 2.5, response_frame, colormap='inferno', activity_threshold=0.3, alpha=1), interpolation='none')
                        # 		axs[plane_i,trial_i].axis('off')
                        # fig.subplots_adjust(hspace=0.01, wspace=0)
                        # fig.tight_layout()
                        # fig.suptitle(f'{fish_name}/nPositive responses to US', fontsize=10, y=0.95)
                        # fig.savefig(results_figs_path_save / '8. Positive responses to US.png', dpi=600, facecolor='white', bbox_inches='tight')

                        # # ##* US negative response
                        # # fig, axs = plt.subplots(num_planes, num_trials_per_plane, figsize=(15 * image_dim_1/614, 7 * image_dim_2/350), squeeze=False)
                        # # for plane_i, plane in enumerate(all_data.planes):
                        # # 	for trial_i, trial in enumerate(plane.trials):
                        # # 		response = np.where(trial.post_us_vs_pre < 0, trial.cs_us_vs_pre, 0)
                        # # 		response_frame = fi.normalize_image(-response, quantiles=(0, 1))
                        # # 		axs[plane_i,trial_i].imshow(fi.add_colors_to_world_improved_2(trial.anatomy * 2.5, response_frame, colormap='inferno', activity_threshold=0, alpha=1), interpolation='none')
                        # # 		axs[plane_i,trial_i].axis('off')
                        # # fig.subplots_adjust(hspace=0.01, wspace=0)
                        # # fig.tight_layout()
                        # # fig.suptitle(f'{fish_name}/nNegative responses to US', fontsize=10, y=0.95)
                        # # fig.savefig(results_figs_path_save / '9. Negative responses to US.png', dpi=600, facecolor='white', bbox_inches='tight')
                        # # #endregion



                        # #region Maps per pair of trials of the same plane
                        # #* Create anatomical scaffold and calculate response for every pair of trials of a plane
                        # for plane_i, plane in tqdm(enumerate(all_data.planes)):

                        # 	plane_anatomies = np.stack([all_data.planes[plane_i].trials[i].template_image for i in range(len(all_data.planes[plane_i].trials))])
                        # 	#* Anatomy of a pair of trials of the same plane

                        # 	plane_anatomies = [np.mean(np.stack(plane_anatomies[start_index : start_index + p.step], axis=0), axis=0) for start_index in range(0, num_trials_per_plane, p.step)]

                        # 	plane_anatomies = [fi.normalize_image(anatomy, (0.01,0.99)) / 5 for anatomy in plane_anatomies]


                        # 	plane_cs_us_vs_pre = np.stack([all_data.planes[plane_i].trials[i].cs_us_vs_pre for i in range(len(all_data.planes[plane_i].trials))])

                        # 	plane_cs_us_vs_pre = [np.mean(np.stack(plane_cs_us_vs_pre[start_index : start_index + p.step], axis=0), axis=0) for start_index in range(0, num_trials_per_plane, p.step)]


                        # 	plane.anatomies = plane_anatomies
                        # 	plane.cs_us_vs_pre = plane_cs_us_vs_pre

                        # 	all_data.planes[plane_i] = plane

                        # ##* CS positive response
                        # num_pairs_trials = len(all_data.planes[0].trials) // 2

                        # # phase_list = ['Pre-Train'] + ['Train']
                        # # color_list = ['white', 'red']

                        # phase_list = phase_list[::2]
                        # color_list = color_list[::2]


                        # # fig, axs = plt.subplots(num_planes, num_pairs_trials, figsize=(9 * image_dim_1 / 614, 7 * image_dim_2 / 350), squeeze=False)
                        # for plane_i, plane in enumerate(all_data.planes):
                        # 	plane_frames = []  # List to store frames for the current plane

                        # 	for trial_pair_i in range(num_pairs_trials):
                        # 		anatomy = plane.anatomies[trial_pair_i]
                        # 		cs_us_vs_pre = plane.cs_us_vs_pre[trial_pair_i]

                        # 		response = np.where(cs_us_vs_pre > 0, cs_us_vs_pre, 0)
                        # 		response_frame = fi.normalize_image(response, quantiles=(0, 1))
                        # 		image = fi.add_colors_to_world_improved_2(anatomy, response_frame, colormap='inferno', activity_threshold=0.3, alpha=1)

                        # 		# Rotate the image by 90 degrees to the left
                        # 		image = np.rot90(image, k=1)

                        # 		# Add plane and trial pair number as text in the top-right corner
                        # 		plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
                        # 		plt.imshow(image, interpolation='none')
                        # 		plt.axis('off')

                        # 		plt.text(10, 10, f'Plane {plane_i+1}', color='darkgray', fontsize=18, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

                        # 		trials_numbers = trial_numbers_list[plane_i,trial_pair_i*2:trial_pair_i*2+2]

                        # 		plt.text(image.shape[1] - 10, 10, f'{phase_list[trial_pair_i]} ({trials_numbers[0]},{trials_numbers[1]})', color=color_list[trial_pair_i], fontsize=18, ha='right', va='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

                        # 		plt.tight_layout(pad=0)

                        # 		# Render the image with the label
                        # 		plt_canvas = plt.gca().figure.canvas
                        # 		plt_canvas.draw()

                        # 		plt_canvas = FigureCanvas(plt.gcf())
                        # 		plt_canvas.draw()
                        # 		labeled_image = np.frombuffer(plt_canvas.buffer_rgba(), dtype=np.uint8).reshape(plt_canvas.get_width_height()[::-1] + (4,))
                        # 		plt.close()

                        # 		plane_frames.append(labeled_image)  # Add labeled frame to the list

                        # 	# Save the frames as a multipage TIFF for the current plane
                        # 	tiff_path = Path(r"C:\Users\joaquim\Desktop") / f'{fish_name}_plane_{plane_i}_positive_responses_pairs.tiff'
                        # 	tifffile.imwrite(tiff_path, plane_frames, photometric='rgb')









                        # # Build one 5D stack for all planes and save a single multipage TIFF
                        # # Axes order: P (planes), C (channels: 0=response, 1=anatomy), T (trials), Y, X
                        # response_pos_planes = []
                        # response_neg_planes = []
                        # anatomy_planes = []

                        # for plane in all_data.planes:
                        # 	response_pos_frames = []
                        # 	response_neg_frames = []
                        # 	anatomy_frames = []

                        # 	for trial in [plane.trials[-1]]:
                        # 		resp_pos = np.where(trial.cs_us_vs_pre > 0, trial.cs_us_vs_pre, 0)
                        # 		resp_pos_norm = fi.normalize_image(resp_pos, quantiles=(0, 1))

                        # 		resp_neg = np.where(trial.cs_us_vs_pre < 0, trial.cs_us_vs_pre, 0)
                        # 		resp_neg_norm = fi.normalize_image(-resp_neg, quantiles=(0, 1))

                        # 		# Rotate by 90 degrees left
                        # 		response_pos_frames.append(np.rot90(resp_pos_norm, k=1))
                        # 		response_neg_frames.append(np.rot90(resp_neg_norm, k=1))
                        # 		anatomy_frames.append(np.rot90(trial.anatomy, k=1))

                        # 	response_pos_planes.append(np.asarray(response_pos_frames, dtype=np.float32))   # (T, Y, X)
                        # 	response_neg_planes.append(np.asarray(response_neg_frames, dtype=np.float32))   # (T, Y, X)
                        # 	anatomy_planes.append(np.asarray(anatomy_frames, dtype=np.float32))     # (T, Y, X)

                        # # responses = np.stack(response_planes, axis=0).astype(np.float32)   # (P, T, Y, X)
                        # # anatomies = np.stack(anatomy_planes, axis=0).astype(np.float32)    # (P, T, Y, X)

                        # # combined_all = np.stack([responses, anatomies], axis=1)          # (P, C=2, T, Y, X)

                        # tiff_path = Path(r"C:\Users\joaquim\Desktop") / f"{fish_name}_all_planes_anatomy.tiff"
                        # tifffile.imwrite(tiff_path, anatomy_planes)

                        # tiff_path = Path(r"C:\Users\joaquim\Desktop") / f"{fish_name}_all_planes_positiveResponses.tiff"
                        # tifffile.imwrite(tiff_path, response_pos_planes)

                        # tiff_path = Path(r"C:\Users\joaquim\Desktop") / f"{fish_name}_all_planes_negativeResponses.tiff"
                        # tifffile.imwrite(tiff_path, response_neg_planes)




                        # # 	break

                        # # fig.subplots_adjust(hspace=0, wspace=0)
                        # # fig.tight_layout()
                        # # fig.suptitle(f'{fish_name} Positive responses to CS_pairs of trials', fontsize=10, y=1)

                        # # fig.savefig(results_figs_path_save / '6.2. Positive responses to CS.png', dpi=600, facecolor='white', bbox_inches='tight')


                        # # #* CS negative response
                        # # fig, axs = plt.subplots(num_planes, num_pairs_trials, figsize=(9 * image_dim_1/614, 7 * image_dim_2/350), squeeze=False)
                        # # for plane_i, plane in enumerate(all_data.planes):

                        # # 	for trial_pair_i in range(num_pairs_trials):

                        # # 		anatomy = plane.anatomies[trial_pair_i]
                        # # 		cs_us_vs_pre = plane.cs_us_vs_pre[trial_pair_i]

                        # # 		response = np.where(cs_us_vs_pre < 0, cs_us_vs_pre, 0)
                        # # 		response_frame = fi.normalize_image(-response, quantiles=(0, 1))
                        # # 		axs[plane_i,trial_pair_i].imshow(fi.add_colors_to_world_improved_2(anatomy, response_frame, colormap='inferno', activity_threshold=0.5, alpha=1), interpolation='none')
                        # # 		axs[plane_i,trial_pair_i].axis('off')
                        # # fig.subplots_adjust(hspace=0, wspace=0)
                        # # fig.tight_layout()
                        # # fig.suptitle(f'{fish_name}\nNegative responses to CS_pairs of trials', fontsize=10, y=1)

                        # # fig.savefig(results_figs_path_save / '7.2. Negative responses to CS.png', dpi=600, facecolor='white', bbox_inches='tight')


                        # # exec(open('5.Save_data_as_HDF5.py').read())


                        # print('END')



                        # #endregion




                        # # #region Create anatomical scaffold for every plane
                        # # plane_anatomy = [[] for _ in range(num_planes)]


                        # # fig, axs = plt.subplots(num_planes, len(all_data.planes[0].trials)//2, figsize=(10, 50), squeeze=False)

                        # # for plane_i, plane in tqdm(enumerate(all_data.planes)):

                        # # 	plane_anatomies = np.stack([all_data.planes[plane_i].trials[i].template_image for i in range(len(all_data.planes[plane_i].trials))])

                        # # 	plane_anatomies = [np.mean(plane_anatomies[start_index : start_index + p.step], axis=0) for start_index in range(0, num_trials_per_plane, p.step)]
                        # # 	# plt.figure('Processed Anatomy Channel')
                        # # 	plt.imshow(plane_anatomies[0], interpolation='none')
                        # # 	plt.title(f'Plane {plane_i} Processed Anatomy Channel')
                        # # 	plt.colorbar(shrink=0.5)
                        # # 	plt.show()

                        # # 	# break

                        # # 	# for trial_i in tqdm(range(len(all_data.planes[0].trials))):



                        # # 	# 	break
                        # # 	# break

                        # # 		# trial_images = all_data.planes[plane_i].trials[trial_i].images.copy()
                        # # 		# trial_images.values = fi.align_frames(trial_images.to_numpy(),
                        # # 		# 								np.stack([trial_images['shift correction in X'].to_numpy(),
                        # # 		# 								trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))
                        # # 		# trial_images = trial_images.sel({'Time (ms)':trial_images['mask good frames']})

                        # # 		# # plane_anatomy[plane_i].append(trial_images.values)



                        # # 	plane_anatomies = [
                        # # 		fi.calculate_anatomy(
                        # # 			np.concatenate(plane_anatomies[start_index : start_index + p.step], axis=0).astype(np.float64),
                        # # 			border_size
                        # # 		)
                        # # 		for start_index in range(0, num_trials_per_plane, p.step)
                        # # 	]

                        # # 	all_data.planes[plane_i].anatomy_channel = plane_anatomies





                        # # 	for i in range(len(plane_anatomies)):
                              
                        # # 		im = axs[plane_i,i].imshow(plane_anatomies[i], interpolation='none')
                        # # 		# fig.colorbar(im, ax=axs[plane_i,], shrink=0.5)

                        # # 		axs[plane_i,i].set_xticks([])
                        # # 		axs[plane_i,i].set_yticks([])
                        # # 		axs[plane_i,i].set_xticklabels([])
                        # # 		axs[plane_i,i].set_yticklabels([])

                        # # 		axs[plane_i,i].set_title(f"Plane {plane_i}")

                        # # 	# break


                        # # fig.set_size_inches(3, 28)
                        # # fig.subplots_adjust(hspace=0, wspace=0)
                        # # fig.savefig(results_figs_path_save / '7. Planes anatomy.png', dpi=600, facecolor='white', bbox_inches='tight', facecolor='white')



                        # # # #* Using a general template per plane

                        # # # for plane_i, plane in enumerate(all_data.planes):

                        # # # 	plane_baseline_color = plane.template

                        # # # 	for trial_i, trial in enumerate(plane.trials):

                        # # # 		color_frame_original = np.clip(trial.cs_us_vs_pre, 0, 1)

                        # # # 		trial_red_channel = plane_baseline_color * (1 - color_frame_original)
                        # # # 		trial_green_channel = trial_red_channel + color_frame_original * color_frame_original
                        # # # 		trial_blue_channel = trial_red_channel

                        # # # 		trial_cs_response = np.stack([trial_red_channel,trial_green_channel,trial_blue_channel])


                        # # # 		trial_cs_response = np.moveaxis(trial_cs_response, 0, -1)

                        # # # 		plt.imshow(Image.fromarray((trial_cs_response*256).astype(np.uint8)))
                        # # # 		plt.show()

                              
                        # # # 		# break
                        # # # 	break

                        # # # plane_numbers = np.zeros((15,4))

                        # # # all_data = c.Data(all_data.planes, anatomical_stack_images)

                        # # #endregion







                        # # #* Save the images as a multipage tiff
                        # # #region Save the images as a multipage tiff
                        # # tifffile = []
                        # # for trial_i, trial in enumerate(all_data.planes[2].trials):
                        # # 	tifffile.append(Image.fromarray(trial.cs_positive_response))

                        # # # Save tifffile as a multipage tiff
                        # # tiff_path = imaging_path_ / 'cs_positive_response_multipage new 1.tiff'
                        # # tifffile[0].save(tiff_path, save_all=True, append_images=tifffile[1:])



                        # # # Load the TIFF file
                        # # tiff_path = imaging_path_ / 'cs_positive_response_multipage new 1.tiff'


                        # # # Label a specific frame in the TIFF stack
                        # # frame_index = 0  # Change this to the index of the frame you want to label
                        # # labeled_frame = tifffile[1]

                        # # # Display the labeled frame
                        # # plt.imshow(labeled_frame, cmap='gray')
                        # # plt.title(f'Labeled Frame {1}')
                        # # plt.colorbar()
                        # # plt.show()



















                        # # # region Voxel analysis
                        # # #* Voxel analysis

                        # # #* Bin the 2D images.

                        # # fig, axs = plt.subplots(num_planes, len(all_data.planes[0].trials), figsize=(15 * image_dim_1/614, 7 * image_dim_2/350), squeeze=False)

                        # # for plane_i, plane in enumerate(all_data.planes):
                        # # 	for trial_i, trial in enumerate(plane.trials):


                        # # 		protocol = trial.protocol

                        # # 		cs_beg_time = protocol.loc[protocol['CS beg']!=0, 'Time (ms)'].values[0]
                        # # 		cs_end_time = protocol.loc[protocol['CS end']!=0, 'Time (ms)'].values[0]
                                    
                        # # 		if (us_beg_time := protocol.loc[protocol['US beg']!=0, 'Time (ms)']).empty:
                        # # 			us_beg_time = cs_beg_time + interval_between_cs_onset_us_onset*1000
                        # # 			us_end_time = us_beg_time + 100
                        # # 		else:
                        # # 			us_beg_time = us_beg_time.values[0]
                        # # 			us_end_time = protocol.loc[protocol['US end']!=0, 'Time (ms)'].values[0]


                        # # 		trial_images = trial.images.copy().fillna(0)

                        # # 		trial_images.values = fi.align_frames(trial_images.to_numpy(), np.stack([trial_images['shift correction in X'].to_numpy(), trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))


                        # # 		trial_images = block_reduce(trial_images, block_size=(1, p.voxel_bin_size, p.voxel_bin_size), func=np.mean, cval=0)


                        # # 		trial_images['mask before cs'] = trial_images['Time (ms)'] < cs_beg_time
                        # # 		trial_images['mask cs-us'] = (trial_images['Time (ms)'] > cs_beg_time) & (trial_images['Time (ms)'] < us_beg_time)
                        # # 		trial_images['mask after us'] = trial_images['Time (ms)'] > us_beg_time


                        # # 		plt.imshow(np.mean(trial_images, axis=0), interpolation='none')

                        # # 		break
                        # # 	break








                        # # 	deltaF = []
                        # # 	deltaF_ratio = []

                        # # 	for i in range(len(cs_indices)):

                        # # 		baseline = np.nanmean(trial_images_binned[[cs_indices[i, 0] - 20, cs_indices[i, 0]]], axis=0)
                              
                        # # 		during_cs = np.nanmean(trial_images_binned[[cs_indices[i, 0], cs_indices[i, 1]]], axis=0)

                        # # 		deltaF_ratio.append((during_cs - baseline) / baseline)

                        # # 		if i == 0:
                                
                        # # 			deltaF.append((trial_images_binned[ : plane_trials_number_images[0]] - baseline) / baseline)

                        # # 		elif i < len(cs_indices)-1:

                        # # 			deltaF.append((trial_images_binned[np.cumsum(plane_trials_number_images)[i-1] : np.cumsum(plane_trials_number_images)[i]] - baseline) / baseline)

                        # # 		else:
                        # # 			deltaF.append((trial_images_binned[np.cumsum(plane_trials_number_images)[i-1] : ] - baseline) / baseline)

                        # # 	deltaF = np.concatenate(deltaF)

                        # # 	deltaF = np.where(np.isnan(deltaF), 0, deltaF)

                        # # 	deltaF_ratio = np.array(deltaF_ratio)


                        # # 	for i in range(len(cs_indices)):
                        # # 		plt.imshow(deltaF_ratio[i], interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
                        # # 		plt.colorbar(shrink=0.5)
                        # # 		plt.title('DeltaF_SR')
                        # # 		plt.show()


                        # # 	A = np.mean(np.array([deltaF_ratio[0], deltaF_ratio[1]]), axis=0)
                        # # 	B = np.mean(np.array([deltaF_ratio[2], deltaF_ratio[3]]), axis=0)



                        # # 	plt.imshow(A, interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
                        # # 	plt.colorbar(shrink=0.5)
                        # # 	plt.title('DeltaF_SR A')
                        # # 	plt.show()

                        # # 	plt.imshow(B, interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
                        # # 	plt.colorbar(shrink=0.5)
                        # # 	plt.title('DeltaF_SR B')
                        # # 	plt.show()

                        # # 	plt.imshow(B/A, interpolation='none', vmin=-100, vmax=100, cmap='RdBu_r')
                        # # 	plt.colorbar(shrink=0.5)
                        # # 	plt.title('DeltaF_SR B / DeltaF_SR A')
                        # # 	plt.savefig(imaging_path_ /  fish_name / (fish_name + '_deltaF_SR_voxels_plane ' + str(plane_i) + '.tif'))
                        # # 	plt.show()

                        # # 	deltaF_ratio = np.concatenate(deltaF_ratio)

                        # # #!!!
                        # # 	# deltaF_ = np.empty(tuple([plane_trials_all_images.shape[0]] + list(deltaF.shape[1:]))) * np.nan
                        # # 	# deltaF_[~plane_trials_mask_good_frames, :, :] = deltaF

                        # # 	# deltaF = deltaF_.copy()

                        # # 	# del deltaF_

                        # # 	for i in range(len(cs_indices)):
                        # # 		deltaF[cs_indices[i,0]:cs_indices[i,1],:20,-20:] = -100

                        # # 	plt.imshow(np.nanmean(deltaF, axis=0))
                        # # 	plt.colorbar(shrink=0.5)

                        # # 	#* Save rois_zscore_over_time as a TIFF file.
                        # # 	tifffile.imwrite(imaging_path_ /  fish_name / (fish_name + '_deltaF_voxels_plane ' + str(plane_i) + '.tif'), deltaF.astype('float32'))

                        # # 	# endregion





                        # # # #* Plot the position in the anatomical stack.
                        # # # # region Position in the anatomical stack
                        # # # try:
                        # # # 	A = []
                        # # # 	B = []

                        # # # 	C = []
                        # # # 	D = []

                        # # # 	for i in range(num_planes):

                        # # # 		for j in range(2):

                        # # # 			A.append(all_data.planes[i].trials[j].position_anatomical_stack)

                        # # # 			C.append(all_data.planes[i].trials[j].template_image)

                        # # # 		for l in range(2,4):

                        # # # 			B.append(all_data.planes[i].trials[l].position_anatomical_stack)

                        # # # 			D.append(all_data.planes[i].trials[l].template_image)


                        # # # 	A = np.array(A)
                        # # # 	B = np.array(B)

                        # # # 	C = np.array(C)
                        # # # 	D = np.array(D)


                        # # # 	sns.set_style('whitegrid')


                        # # # 	path_ = path_home / fish_name


                        # # # 	plt.xlabel('Trial before or after initial train')
                        # # # 	plt.ylabel('Plane number in anatomical stack')
                        # # # 	plt.plot(A, 'blue')
                        # # # 	plt.plot(B, 'red')
                        # # # 	plt.legend(['Before initial train', 'After initial train'])
                        # # # 	plt.savefig(path_ / ('Where in the anatomical stack' + '.png'), dpi=300, bbox_inches='tight')


                        # # # 	plt.xlabel('Trial before or after initial train')
                        # # # 	plt.ylabel('Difference between planes imaged\n before and after initial train (μm)')
                        # # # 	plt.plot(A-B, 'k')
                        # # # 	plt.ylim(-10, 10)
                        # # # 	plt.savefig(path_ / ('Difference when revisiting planes' + '.png'), dpi=300, bbox_inches='tight')


                        # # # 	sns.set_style('white')
                          
                        # # # except:
                        # # # 	pass
















                        # # # 	# except:
                        # # # 	# 	continue

                        # # # # compression_level = 4
                        # # # # compression_library = 'zlib'

                        # # # # with pd.HDFStore(path_pkl, complevel=compression_level, complib=compression_library) as store:
                          
                        # # # # 	store.append(fish_name, planes_list, data_columns=[cs, us], expectedrows=len(fish.raw_data), append=False)

                        # # # # 	store.get_storer(fish.dataset_key()).attrs['metadata'] = fish.metadata._asdict()





                        # # # #* For correlation map.

                        # # # ##* Preparing the data for the correlation map.

                        # # # # for plane_i, plane in enumerate(Data.planes):
                        # # # # 	for trial_i, trial in enumerate(plane):

                        # # # # A = [Data.planes[plane_i].trials[trial_i].images.values for trial_i, trial in enumerate(plane.trials) for plane_i, plane in enumerate(Data.planes)]

                        # # # # B = np.sum([np.sum(x, axis=0) for x in A], axis=0)

                        # # # # plt.imshow(B)


                        # # # eye_mask = np.ones(all_data.planes[0].trials[0].images.shape[1:], dtype='bool')

                        # # # #!
                        # # # eye_mask[350:, 350:450] = False
                        # # # eye_mask[:50, 350:450] = False
                        # # # plt.imshow(eye_mask)

                        # # # # # A = ndimage.uniform_filter(trial_images, size=(30, 30), axes=(1,2))
                        # # # # ndimage.gaussian_filter(trial_images, sigma=gaussian_filter_sigma, axes=(1,2))

                        # # # # plt.imshow(np.mean(A, axis=0))
                        # # # # plt.colorbar()











                        # # # for plane_i, plane in enumerate(all_data.planes):

                        # # # 	# if plane_i not in [0,1,3,6,8,9,10,13]:
                        # # # 	# 	continue
                        # # # 	# break
                        # # # #!
                        # # # 	# plane.trials = plane.trials



                        # # # 	#!!!!!!!!!!!!!!!!!!!!!!!!! DO ALL OF THIS FOR SINGLE TRIAL AND THEN CONCATENATE TO GET PLANE DATA




                        # # # 	#* To get a correlation map for the whole plane data, we need to concatenate all the images of the trials.
                        # # # 	# plane_trials_all_images = np.concatenate([t.images.values for t in plane.trials])
                        # # # 	plane_trials_all_images = plane.get_all_images()

                        # # # 	plt.title('All images from plane')
                        # # # 	plt.imshow(np.mean(plane_trials_all_images, axis=0))
                        # # # 	plt.colorbar
                        # # # 	plt.show()


                        # # # 	#* Get the number of images per trial.
                        # # # 	plane_trials_number_images = np.array([t.images.shape[0] for t in plane.trials])


                        # # # 	#* Get the indices of the cs in the images of the trials.
                        # # # 	cs_indices = np.array([trial.get_stim_index(cs) for trial in plane.trials])

                        # # # 	cs_indices[1:,0] += np.cumsum(plane_trials_number_images[:-1])
                        # # # 	cs_indices[1:,1] += np.cumsum(plane_trials_number_images[:-1])









                        # # # 	#* Discard good frames due to motion, gating of the PMT or plane change.
                        # # # 	plane_trials_mask_good_frames = np.concatenate([t.mask_good_frames for t in plane.trials])
                        # # # 	plane_bad_frames_index = np.where(plane_trials_mask_good_frames)[0]
                        # # # 	trial_images = plane_trials_all_images[~plane_trials_mask_good_frames].copy()

                        # # # 	plt.title('All good images from plane')
                        # # # 	plt.imshow(np.mean(trial_images, axis=0))
                        # # # 	plt.colorbar
                        # # # 	plt.show()










                        # # # 	#* Filter in space.
                        # # # 	trial_images_filtered = ndimage.gaussian_filter(trial_images, sigma=gaussian_filter_sigma, axes=(1,2))

                        # # # 	plt.title('All good images from plane filtered')
                        # # # 	plt.imshow(np.mean(trial_images_filtered, axis=0))
                        # # # 	plt.colorbar

                        # # # #!!!!!!!!!!!!!!!!! move it further down
                        # # # 	#* Calcultate the correlation map.
                        # # # 	# Inspired in Suit2p. There, the function that computes the correlation map is celldetect2.getVmap.
                        # # # 	correlation_map = np.linalg.norm(ndimage.gaussian_filter(trial_images, sigma=correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(trial_images, axis=0), sigma=correlation_map_sigma)**2

                        # # # 	plt.figure('Correlation map')
                        # # # 	plt.imshow(correlation_map)
                        # # # 	plt.colorbar(shrink=0.5)
                        # # # 	plt.show()






                        # # # 	#* Subtract the background.
                        # # # 	# Pixel values equal to 0 are ignored to discard the artificial edges of the images that were introduced during the motion correction.
                        # # # 	images_mean = np.nanmean(np.where(trial_images == 0, np.nan, trial_images), axis=(1,2))

                        # # # 	images_mean = np.nanmean(trial_images, axis=(1,2))
                        # # # 	for image_i in range(trial_images.shape[0]):
                        # # # 		trial_images[image_i] -= images_mean[image_i]

                        # # # 	del images_mean

                        # # # 	#* Mask the background.
                        # # # 	plane_images_mask_fish = np.where(np.median(trial_images, axis=0) <= 0, 0, 1).astype(dtype='bool')

                        # # # 	plane_images_mask_fish_without_eyes = plane_images_mask_fish & eye_mask

                        # # # 	#* Set to 0 the pixels that are not part of the fish in the images. Also, mask the eyes.
                        # # # 	trial_images = np.where(plane_images_mask_fish_without_eyes, trial_images, 0)

                        # # # 	plt.title('All good images from plane masked background')
                        # # # 	plt.imshow(np.mean(trial_images, axis=0))
                        # # # 	plt.colorbar(shrink=0.5)
                        # # # 	plt.show()









                        # # # 	# region ROI analysis for the whole plane

                        # # # 	#* Set to 0 the pixels that are not part of the fish in the correlation map.
                        # # # 	correlation_map = np.where(plane_images_mask_fish_without_eyes, correlation_map, 0)

                        # # # 	plt.title('Correlation map masked background')
                        # # # 	plt.imshow(np.where(plane_images_mask_fish_without_eyes, correlation_map, 0))
                        # # # 	plt.colorbar(shrink=0.5)
                        # # # 	plt.show()





                        # # # 	#* ROIs for the all the trials of the same plane.
                        # # # 	#TODO need to rewrite all this part, using Mike's and Ruben's code
                        # # # 	all_traces, all_rois, used_pixels, correlation_map_ = f.get_ROIs(Nrois=100, correlation_map=correlation_map, images=trial_images_filtered, threshold=0.3, max_pixels=60)

                        # # # 	plt.imshow(zscore(all_traces, 1), aspect="auto", cmap="RdBu_r")
                        # # # 	plt.savefig(imaging_path_ / fish_name / (fish_name + 'zscore ' + str(plane_i) + '.tif'))
                        # # # 	plt.show()
                        # # # 	plt.imshow(all_rois)
                        # # # 	plt.colorbar()
                        # # # 	plt.show()
                        # # # 	plt.imshow(correlation_map_)
                        # # # 	plt.show()
                        # # # 	plt.imshow(np.sum(plane_trials_all_images, axis=0))
                        # # # 	plt.show()
                        # # # 	plt.imshow(correlation_map)
                        # # # 	plt.show()



                        # # # 		#* Create array to then make movie.
                        # # # 		all_rois = all_rois.astype('int')

                        # # # 		rois_zscore_over_time = np.zeros_like(plane_trials_all_images)


                        # # # 		#* Consider the periods of good frames in the array with the Z score of the ROI traces.
                        # # # 		all_traces_z_score = zscore(all_traces, 1)

                        # # # 		all_traces_z_score_ = np.empty((all_traces.shape[0], len(plane_trials_all_images))) * np.nan
                        # # # 		all_traces_z_score_[:, ~plane_trials_mask_good_frames] = all_traces_z_score

                        # # # 		plt.imshow(all_traces_z_score_, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)

                        # # # 		all_traces_z_score = all_traces_z_score_
                        # # # 		del all_traces_z_score_

                        # # # 		rois_mask = np.zeros(rois_zscore_over_time.shape, dtype='bool')

                        # # # 		#* Get mask of the ROIs.
                        # # # 		for roi_i in range(1, all_rois.max()):
                        # # # 			# break
                        # # # 			rois_mask[roi_i] = all_rois == roi_i

                        # # # 			# roi_mask = rois_mask[roi_i]
                        # # # 			# [np.newaxis, :, :]

                        # # # 			for t in range(rois_zscore_over_time.shape[0]):
                        # # # 				# break
                        # # # 				rois_zscore_over_time[t,:,:] += np.where(rois_mask[roi_i], all_traces_z_score[roi_i, t], 0)

                        # # # 				rois_zscore_over_time[t,:,:]

                        # # # 		for i in range(len(cs_indices)):
                        # # # 			rois_zscore_over_time[cs_indices[i,0]:cs_indices[i,1],:50,-50:] = -100

                        # # # 		# plt.imshow(np.mean(plane_trials_all_images, axis=0))
                        # # # 		plt.imshow(np.nansum(rois_zscore_over_time, axis=0), aspect="auto", cmap="RdBu_r")
                        # # # 		plt.colorbar()

                        # # # 		#* Save rois_zscore_over_time as a TIFF file.
                        # # # 		tifffile.imwrite(imaging_path_ / fish_name / (fish_name + 'rois_zscore_over_time ' + str(plane_i) + '.tif'), rois_zscore_over_time.astype('float32'))

                        # # # 		# endregion






                        # # # 	# region ROI analysis for each trial

                        # # # 	for trial in plane.trials:

                        # # # 		#!
                        # # # 		# break
                        # # # 		trial = plane.trials[3]
                        # # # 	# break
                        # # # 		# trial.images = trial.images


                        # # # 		#* Discard good frames due to motion, gating of the PMT or trial change.
                        # # # 		trial_good_images = trial.images.values[~trial.mask_good_frames]
                        # # # 		trial_bad_frames_index = np.where(trial.mask_good_frames)[0]

                        # # # 		plt.title('All images from trial')
                        # # # 		plt.imshow(np.mean(trial_good_images, axis=0))
                        # # # 		plt.colorbar(shrink=0.5)
                        # # # 		plt.show()

                        # # # 		#* Subtract the background.
                        # # # 		# Pixel values equal to 0 are ignored to discard the artificial edges of the images that were introduced during the motion correction.
                        # # # 		images_mean = np.nanmean(np.where(trial_good_images == 0, np.nan, trial_good_images), axis=(1,2))

                        # # # 		images_mean = np.nanmean(trial_good_images, axis=(1,2))
                        # # # 		for image_i in range(trial_good_images.shape[0]):
                        # # # 			trial_good_images[image_i] -= images_mean[image_i]

                        # # # 		del images_mean

                        # # # 		#* Mask the background.
                        # # # 		trial.images_mask_fish = np.where(np.median(trial_good_images, axis=0) <= 0, 0, 1).astype(dtype='bool')

                        # # # 		trial.images_mask_fish_without_eyes = trial.images_mask_fish & eye_mask

                        # # # 		#* Set to 0 the pixels that are not part of the fish in the images. Also, mask the eyes.
                        # # # 		trial_good_images = np.where(trial.images_mask_fish_without_eyes, trial_good_images, 0)

                        # # # 		plt.title('All good images from trial masked background')
                        # # # 		plt.imshow(np.mean(trial_good_images, axis=0))
                        # # # 		plt.colorbar(shrink=0.5)
                        # # # 		plt.show()



                        # # # #!!!!!!!!!!! Voxel analysis per trial
                        # # # 		#* Bin the 2D images.
                        # # # 		trial_good_images_binned = block_reduce(trial_good_images, block_size=(1, voxel_bin_size, voxel_bin_size), func=np.mean, cval=0)

                        # # # 		plt.imshow(np.mean(trial_good_images_binned, axis=0), interpolation='none')


                        # # # 		trial_images_binned = np.empty(tuple([trial.images.shape[0]] + list(trial_good_images_binned.shape[1:]))) * np.nan
                        # # # 		trial_images_binned[~trial.mask_good_frames, :, :] = trial_good_images_binned

                        # # # 		trial_good_images_binned = trial_images_binned.copy()

                        # # # 		del trial_images_binned

                        # # # 		plt.title('All good images from trial binned')
                        # # # 		plt.imshow(np.mean(trial_good_images_binned, axis=0))
                        # # # 		plt.colorbar(shrink=0.5)
                        # # # 		plt.show()

                          

















                        # # # 		#* Filter in space.
                        # # # 		trial_images_filtered = ndimage.gaussian_filter(trial_images, sigma=gaussian_filter_sigma, axes=(1,2))
                        # # # 		trial_images_images_filtered = trial_images_filtered[~trial.mask_good_frames].copy()




                        # # # 		#* Correlation map
                        # # # 		# In Suit2p, the function that computes the correlation map is celldetect2.getVmap.
                        # # # 		correlation_map = np.linalg.norm(ndimage.gaussian_filter(trial_images_images, sigma=correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(trial_images_images, axis=0), sigma=correlation_map_sigma)**2

                        # # # 		plt.figure('Correlation map')
                        # # # 		plt.imshow(correlation_map)
                        # # # 		plt.colorbar(shrink=0.5)
                        # # # 		plt.show()

                        # # # 		#* Subtract the background.
                        # # # 	#! Here I should take the average ignoring the sharp edges of the images.
                        # # # 		images_mean = np.mean(trial_images_images, axis=(1,2))

                        # # # 		for image_i in range(trial_images_images.shape[0]):
                        # # # 			trial_images_images[image_i] -= images_mean[image_i]

                        # # # 		del images_mean

                        # # # 		#* Mask the background.
                        # # # 		trial_images_mask_fish = np.where(np.median(trial_images_images, axis=0) <= 0, 0, 1).astype(dtype='bool')
                              
                        # # # 		#* Mask the background and the eyes.
                        # # # 		trial_images_mask_fish_without_eyes = trial_images_mask_fish & eye_mask


                        # # # 		#* Set to 0 the pixels that are not part of the fish in the images.
                        # # # 		trial_images_images = np.where(trial_images_mask_fish_without_eyes, trial_images_images, 0)

                        # # # 		plt.title('All good images from plane masked background')
                        # # # 		plt.imshow(np.mean(trial_images_images, axis=0))
                        # # # 		plt.colorbar(shrink=0.5)
                        # # # 		plt.show()

                        # # # 		#* Set to 0 the pixels that are not part of the fish in the correlation map.
                        # # # 		correlation_map = np.where(trial_images_mask_fish_without_eyes, correlation_map, 0)

                        # # # 		plt.title('Correlation map masked background')
                        # # # 		plt.imshow(np.where(trial_images_mask_fish_without_eyes, correlation_map, 0))
                        # # # 		plt.colorbar(shrink=0.5)
                        # # # 		plt.show()





                        # # # 		#* ROIs

                        # # # 		all_traces, all_rois, used_pixels, correlation_map_ = get_ROIs(Nrois=100, correlation_map=correlation_map, images=trial_images_images_filtered, threshold=0.3, max_pixels=60)

                        # # # 		images_times = trial_images.time.values


                        # # # 		trial_time_ref = images_times[0]

                        # # # 		trial_protocol = trial.protocol

                        # # # 		cs_times = trial_protocol[trial_protocol[cs]!=0]
                        # # # 		cs_times = cs_times.iloc[[0,-1]] if cs_times.shape[0] > 1 else cs_times

                        # # # 		us_times = trial_protocol[trial_protocol[us]!=0]
                        # # # 		us_times = us_times.iloc[[0,-1]] if us_times.shape[0] > 1 else us_times


                        # # # 		images_times = images_times - trial_time_ref
                        # # # 		cs_times = cs_times['Time (ms)'].values - trial_time_ref
                        # # # 		us_times = us_times['Time (ms)'].values - trial_time_ref


                        # # # 		number_traces = 50

                        # # # 		fig, axs = plt.subplots(number_traces, 1, sharex=True)
                        # # # 		# figsize=(10, 8)

                        # # # 		for i in range(number_traces):

                        # # # #!
                        # # # 			axs[i].plot(images_times[:110], all_traces[i+50][:110])

                        # # # 			if cs_times.shape[0] > 0:
                        # # # 				axs[i].axvline(x=cs_times[0], color='g', linestyle='-')
                        # # # 				axs[i].axvline(x=cs_times[1], color='g', linestyle='--')
                                
                        # # # 			if us_times.shape[0] > 0:
                        # # # 				axs[i].axvline(x=us_times[0], color='r', linestyle='-')
                        # # # 				axs[i].axvline(x=us_times[1], color='r', linestyle='--')

                        # # # 		fig.show()


                        # # # 		plt.imshow(zscore(all_traces, 1), aspect="auto", vmin=-3, vmax=3, cmap="RdBu_r")

                        # # # 		plt.show()
                        # # # 		plt.imshow(all_rois)
                        # # # 		plt.show()
                        # # # 		plt.imshow(correlation_map_)
                        # # # 		plt.show()
                        # # # 		plt.imshow(np.sum(imag, axis=0))
                        # # # 		plt.show()
                        # # # 		plt.imshow(original_correlation_map)
                        # # # 		plt.show()
                        # # # # fig,(ax1,ax2,ax3,ax4)= plt.subplots(1,4)
                        # # # # ax1 = plt.subplot(121)
                        # # # # img=ax1.imshow(zscore(all_traces, 1), aspect="auto", vmin=-3, vmax=3, cmap="RdBu_r")
                        # # # # ax1.set_ylabel("trace ROI number")
                        # # # # ax1.set_xlabel("frame number")
                        # # # # fig.colorbar(img,ax=ax1)
                        # # # # ax2 = plt.subplot(322)
                        # # # # ax2.imshow(all_rois)
                        # # # # ax3 = plt.subplot(324)
                        # # # # ax3.imshow(correlation_map_)
                        # # # # ax4 = plt.subplot(326)
                        # # # # ax4.imshow(original_correlation_map)
                        # # # # # plt.show()
                        # # # # fig.tight_layout()


                        # # # a = Data.planes[0].trials[0].images.values

                        # # # plt.imshow(np.mean(a, axis=0))


                        # # # b = Data.planes[0].trials[3].images.values

                        # # # plt.imshow(np.mean(b, axis=0))


                        # # # plt.imshow(np.mean(a, axis=0) - np.mean(b, axis=0), cmap='viridis')
                        # # # plt.colorbar(shrink=0.5)


                        # # # __dict__.keys()