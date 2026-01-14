"""
3.1 Pixel-Level Imaging Data Analysis

This script performs temporal segmentation and response analysis on motion-corrected 2-photon calcium imaging data.

Workflow:
1. Loads motion-corrected data from pickle file (output of script 2.Motion_correction.py)
2. Segments each trial into three temporal periods based on experimental protocol:
   - Pre-CS: frames before conditioned stimulus (CS) onset
   - CS-US: frames between CS onset and unconditioned stimulus (US) onset
   - Post-US: frames after US onset
3. Applies frame alignment using previously computed shift corrections
4. Computes mean fluorescence for each temporal period (excluding bad frames)
5. Calculates normalized activity responses:
   - CS-US vs Pre-CS: (cs_us_mean - pre_cs_mean) / (pre_cs_mean + softthresh)
   - Post-US vs Pre-CS: (post_us_mean - pre_cs_mean) / (pre_cs_mean + softthresh)
6. Applies gaussian smoothing (sigma=2) to response maps to reduce noise
7. Generates normalized anatomical reference images from template images
8. Saves processed results to pickle file for downstream activity map generation (script 4.Activity_maps.py)

Protocol-Specific Parameters:
- Delay conditioning: CS-US interval = 9s
- Trace conditioning: CS-US interval = 13s  
- Control: CS-US interval = 9s

Key Parameters:
- softthresh: Baseline offset (100) to prevent division by near-zero values
- gaussian_filter sigma: 2 pixels for spatial smoothing of response maps
- border_size: 2 * voxel_bin_size for anatomy calculation

Output Structure:
Each trial object contains:
- pre_cs_mean, cs_us_mean, post_us_mean: temporal period averages (xarray)
- cs_us_vs_pre, post_us_vs_pre: normalized response maps (numpy arrays)
- anatomy: normalized template image for overlay visualization

Note: This is the custom motion correction version. For Suite2p-registered data,
use 3.1.Analysis_of_imaging_data_pixels_Suite2p.py instead.
"""

#* Imports

# %%
# region Imports

##   
# region Imports



#* Imports

##   
# region Imports
import os
import pickle
from collections import defaultdict
# import pickle
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import tifffile
import xarray as xr
from natsort import natsorted
from scipy.ndimage import gaussian_filter
from suite2p import default_ops, io, registration
from suite2p.io.binary import BinaryFile
from suite2p.registration import register
from suite2p.registration.register import compute_reference, register_frames
from suite2p.run_s2p import run_s2p
from tqdm import tqdm

#* Load custom functions and classes
#* Load custom functions and classes
import my_classes as c
# import my_experiment_specific_variables as spec_var
import my_functions_imaging as fi
import my_parameters as p
from my_general_variables import *
from my_paths import fish_name, path_home

# Save all data in a single pickle file.
# Anatomical stack images and imaging data are median filtered.


#* Imports


# endregion

reload(fi)
reload(c)
reload(p)

#* Settings
##    Settings
# region Settings

# %matplotlib ipympl

pio.templates.default = "plotly_dark"

pd.set_option("mode.copy_on_write", True)
pd.set_option("compute.use_numba", True)
pd.set_option("compute.use_numexpr", True)
pd.set_option("compute.use_bottleneck", True)
#endregion



#* Paths
##   
# region Paths
path_home =Path(r'D:\2024 10_Delay 2-P 15 planes ca8 neurons')
# Path(r'D:\2024 03_Delay 2-P 15 planes top part')
# Path(r'D:\2024 09_Delay 2-P 4 planes JC neurons')
# Path(r'D:\2024 10_Delay 2-P 15 planes bottom part')
# Path(r'D:\2024 10_Delay 2-P single plane')
# Path(r'D:\2024 09_Delay 2-P zoom in multiplane imaging')

path_results_save = Path(r'F:\Results (paper)') / path_home.stem

# fish_list = [f for f in (path_home / 'Imaging').iterdir() if fi.is_dir()]
# fish_names_list = [fi.stem for f in fish_list]

fish_name = r'20241015_03_delay_2p-9_mitfaminusminus,ca8e1bgcamp6s_6dpf'

# '20240415_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf'

# '20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241022_01_delay_2p-10_mitfaminusminus,ca8e1bgcamp6s_5dpf'
# '20241015_01_delay_2p-7_mitfaminusminus,ca8e1bgcamp6s_6dpf'
# '20241008_03_delay_2p-6_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241007_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'

# '20241013_01_control_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240930_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241014_01_trace_2p-4_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241014_03_trace_2p-6_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241010_01_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241008_02_delay_2p-5_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241008_01_delay_2p-4_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241007_03_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20241015_03_delay_2p-9_mitfaminusminus,ca8e1bgcamp6s_6dpf'
# '20241017_01_delay_2p-4_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241016_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240417_01_delay_2p-4_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
# '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'
# '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf'




fish_ID = '_'.join(fish_name.split('_')[:2])


behavior_path_home = path_home / 'Tail'
imaging_path_home = path_home / 'Neurons' / fish_name

behavior_path_save = path_results_save / 'Tail'
results_figs_path_save = path_results_save / 'Neurons' / fish_name

whole_data_path_save = Path(r'H:\2-P imaging') / path_home.stem / fish_name


path_pkl_after_motion_correction = whole_data_path_save / (fish_ID + '_2. After motion correction.pkl')

path_pkl_responses = whole_data_path_save / (fish_ID + '_3. Responses.pkl')

			# # Create the directory if it does not exist
			# (results_path / 'ROI traces').mkdir(parents=True, exist_ok=True)
			# (results_path / 'Videos with deltaF over F').mkdir(parents=True, exist_ok=True)
			# (results_path / 'Anatomy with ROIs').mkdir(parents=True, exist_ok=True)
			# os.makedirs(whole_data_path_save, exist_ok=True)




border_size = 2*p.voxel_bin_size
correlation_thr = 0.3
median_thr = 5

softthresh=100

step = 2 # Process trials in pairs


if 'delay' in fish_name:
	interval_between_cs_onset_us_onset = 9  # s
elif 'trace' in fish_name:
	interval_between_cs_onset_us_onset = 13  # s
elif 'control' in fish_name:
	interval_between_cs_onset_us_onset = 9  # s
else:
	interval_between_cs_onset_us_onset = None





#* Load the data before motion correction.
# region Load the data before motion correction
with open(path_pkl_after_motion_correction, 'rb') as file:
	all_data = pickle.load(file)



print('Analyzing fish: ', fish_name)


# endregion


shape_ = all_data.planes[0].trials[0].images.shape[1:]

# x_black_box_beg, x_black_box_end, y_black_box_beg, y_black_box_end = all_data.black_box

# x_black_box_beg = shape_[0] - 20
# x_black_box_end = shape_[0] - 5
# y_black_box_beg = shape_[1] - 20
# y_black_box_end = shape_[1] - 5






#* Split data into 3 periods: before cs, from cs onset to us onset and after us onset.
for plane_i, plane in tqdm(enumerate(all_data.planes)):
	for trial_i, trial in enumerate(plane.trials):

		# if trial_i != 2:
	# 	# 	continue
	# 	break
	# break

		protocol = trial.protocol

		cs_beg_time = protocol.loc[protocol['CS beg']!=0, 'Time (ms)'].values[0]
		cs_end_time = protocol.loc[protocol['CS end']!=0, 'Time (ms)'].values[0]
				
		if (us_beg_time := protocol.loc[protocol['US beg']!=0, 'Time (ms)']).empty:
			us_beg_time = cs_beg_time + interval_between_cs_onset_us_onset*1000
			us_end_time = us_beg_time + 100
		else:
			us_beg_time = us_beg_time.values[0]
			us_end_time = protocol.loc[protocol['US end']!=0, 'Time (ms)'].values[0]


		trial_images = trial.images.copy().fillna(0)


		trial_images['mask before cs'] = trial_images['Time (ms)'] < cs_beg_time
		trial_images['mask cs-us'] = (trial_images['Time (ms)'] > cs_beg_time) & (trial_images['Time (ms)'] < us_beg_time)
		trial_images['mask after us'] = trial_images['Time (ms)'] > us_beg_time



		#* Align the frames
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! correct for single plane
		trial_images.values = fi.align_frames(trial_images.to_numpy(), np.stack([trial_images['shift correction in X'].to_numpy(), trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))


		trial.pre_cs_mean = trial_images.sel({'Time (ms)':(trial_images['mask before cs']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')
		trial.cs_us_mean = trial_images.sel({'Time (ms)':(trial_images['mask cs-us']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')
		trial.post_us_mean = trial_images.sel({'Time (ms)':(trial_images['mask after us']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')


		all_data.planes[plane_i].trials[trial_i] = trial




for plane_i, plane in enumerate(all_data.planes):
	for trial_i, trial in enumerate(plane.trials):

		pre_cs_data = trial.pre_cs_mean.values

		trial.cs_us_vs_pre = gaussian_filter(((trial.cs_us_mean.values - pre_cs_data) / (pre_cs_data + softthresh)), sigma=2)
		trial.post_us_vs_pre = gaussian_filter((trial.post_us_mean.values - pre_cs_data / (pre_cs_data + softthresh)), sigma=2)

		# anatomy = fi.normalize_image(fi.calculate_anatomy(trial_images.values.astype(np.float64), border_size), (0.01,0.99))

		# anatomy = anatomies[trial_i//2]
		# trial.anatomy = fi.scale_slide(all_data.planes[plane_i].trials[trial_i].template_image, p.min_intensity_threshold) / 5
		trial.anatomy = fi.normalize_image(all_data.planes[plane_i].trials[trial_i].template_image, (0.01,0.99)) / 10

		#* Give some colors to the world.
		# color_frame_original = fi.scale_slide(trial.cs_us_vs_pre, p.min_intensity_threshold)
		# color_frame_original.max()

		# plt.imshow(fi.add_colors_to_world(anatomy/5, color_frame_original), interpolation='none')
		# plt.imshow(fi.add_colors_to_world_improved(anatomy, color_frame_original, activity_scaling=2.0, anatomy_brightness=0.5), interpolation='none')


with open(path_pkl_responses, 'wb') as file:
	pickle.dump(all_data, file)


exec(open('4.Activity_maps.py').read())


print('END')

