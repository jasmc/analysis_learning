"""
Legacy Analysis Script (Deprecated)

This script contains the previous version of imaging data analysis code.
It has been superseded by scripts 3.1 and 3.2 but is kept for reference.

Original Functionality:
- Combined pixel-level and voxel-level analysis
- Single-pass processing of all planes and trials
- Different response calculation methods
- Alternative visualization approaches

Replaced By:
- 3.1.Analysis_of_imaging_data_pixels.py (pixel-level analysis)
- 3.2.Analysis_of_imaging_data_voxels.py (voxel-level analysis)
- 3.1.Analysis_of_imaging_data_pixels_Suite2p.py (Suite2p version)

Why Deprecated:
- Code organization: separated pixel vs voxel analysis
- Performance: optimized individual scripts
- Maintainability: clearer separation of concerns
- Suite2p integration: added dedicated version

Keep For:
- Reference for algorithm development
- Comparison with current implementation
- Recovery of deprecated features if needed
- Historical documentation

Note: This script is no longer maintained or used in the production pipeline.
Refer to scripts 3.1.x and 3.2.x for current analysis methods.
"""

#* Imports

# %%
# region Imports

##   
# region Imports
import os
import pickle
from importlib import reload
from pathlib import Path

import cv2
# import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import scipy.ndimage as ndimage
import seaborn as sns
import tifffile
import tifffile as tiff
import xarray as xr
from PIL import Image
from scipy import signal
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
from tqdm import tqdm

#* Load custom functions and classes
import my_classes as c
# import my_experiment_specific_variables as spec_var
import my_functions_imaging as fi
import my_parameters as p
from my_general_variables import *

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
path_home = Path(r'D:\2024 03_Delay 2-P 15 planes top part')
# Path(r'E:\2024 09_Delay 2-P 4 planes JC neurons')
# Path(r'E:\2024 10_Delay 2-P single plane')
# Path(r'E:\2024 10_Delay 2-P 15 planes ca8 neurons')
# Path(r'E:\2024 09_Delay 2-P zoom in multiplane imaging')

path_results_save = Path(r'F:\Results (paper)') / path_home.stem

# fish_list = [f for f in (path_home / 'Imaging').iterdir() if f.is_dir()]
# fish_names_list = [f.stem for f in fish_list]

fish_name = r'20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf'


# '20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'

# '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'


# '20241013_01_control_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241013_02_control_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241009_03_delay_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'

# '20241024_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
# '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240415_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'

# 20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf

# '20241015_03_delay_2p-9_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf'

# '20240926_03_trace_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20240927_02_control_2p-5_mitfaMinusMinus,elavl3BGCaMP6f_6dpf'
# '20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'

		# # fish_list = [f for f in (path_home / 'Imaging').iterdir() if f.is_dir()]
		# # fish_names_list = [f.stem for f in fish_list]

		# fish_name = r'20241013_01_control_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
		# # '20241015_03_delay_2p-9_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf'
		# # '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
		# # '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
		# # '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
		# # !'20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
		# #! 20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf
		# #! 20240415_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf
		# #! 20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf

		# # '20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
		# # r'20240926_03_trace_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
		# # '20241015_03_delay_2p-9_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf'
		# # '20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'
		# # '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'



		# imaging_path = path_home / 'Imaging'


		# # for fish_i, fish_name in enumerate(fish_names_list):

		# # 	try:

		# imaging_path_ = imaging_path / fish_name / 'Imaging'


		# path_pkl_after_motion_correction = path_home / fish_name / (fish_name + '_after motion correction' + '.pkl')


		# # h5_path = path_home / fish_name / (fish_name + '_before_motion_correction.h5')

		# # h5_path = imaging_path_ / (fish_name + '_before_motion_correction.h5')

		# # h5_path = r"E:\2024 03_Delay 2-P 15 planes top part\Imaging\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf\Imaging\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf_before_motion_correction.h5"



fish_ID = '_'.join(fish_name.split('_')[:2])


behavior_path_home = path_home / 'Tail'
imaging_path_home = path_home / 'Neurons' / fish_name

behavior_path_save = path_results_save / 'Tail'
results_figs_path_save = path_results_save / 'Neurons' / fish_name

whole_data_path_save = Path(r'H:\2-P imaging') / path_home.stem / fish_name


path_pkl_before_motion_correction = whole_data_path_save / (fish_ID + '_before motion correction' + '.pkl')
path_pkl_after_motion_correction = whole_data_path_save / (fish_ID + '_after motion correction' + '.pkl')



			# # Create the directory if it does not exist
			# (results_path / 'ROI traces').mkdir(parents=True, exist_ok=True)
			# (results_path / 'Videos with deltaF over F').mkdir(parents=True, exist_ok=True)
			# (results_path / 'Anatomy with ROIs').mkdir(parents=True, exist_ok=True)
			# os.makedirs(whole_data_path_save, exist_ok=True)




border_size = 2*p.voxel_bin_size
correlation_thr = 0.3
median_thr = 5

softthresh=50


if 'delay' in fish_name:
	interval_between_CS_onset_US_onset = 9  # s
elif 'trace' in fish_name:
	interval_between_CS_onset_US_onset = 13  # s
elif 'control' in fish_name:
	interval_between_CS_onset_US_onset = 9  # s





#* Load the data before motion correction.
# region Load the data before motion correction
with open(path_pkl_after_motion_correction, 'rb') as file:
	all_data = pickle.load(file)



print('Analyzing fish: ', fish_name)


# endregion

# all_data.__dict__.keys()

# all_data.planes[3].trials[3].shift_correction



shape_ = all_data.planes[0].trials[0].images.shape[1:]

# x_black_box_beg, x_black_box_end, y_black_box_beg, y_black_box_end = all_data.black_box

x_black_box_beg = shape_[0] - 20
x_black_box_end = shape_[0] - 5
y_black_box_beg = shape_[1] - 20
y_black_box_end = shape_[1] - 5
			# if ('ca8' in str(path_home)) | ('4' in str(path_home)):
			# 	x_black_box_beg = shape_[0] - 20
			# 	x_black_box_end = shape_[0] - 5
			# 	y_black_box_beg = shape_[1] - 20
			# 	y_black_box_end = shape_[1] - 5


			# elif 'single' in str(path_home):
					
					
			# 	x_black_box_beg = shape_[0] - 10
			# 	x_black_box_end = shape_[0] - 5
			# 	y_black_box_beg = shape_[1] - 10
			# 	y_black_box_end = shape_[1] - 5

			# else:
			# 	x_black_box_beg = 330
			# 	x_black_box_end = 345
			# 	y_black_box_beg = 594
			# 	y_black_box_end = 609









plt.figure()
plt.imshow(all_data.planes[0].trials[0].images[100][x_black_box_beg:x_black_box_end, y_black_box_beg:y_black_box_end], vmin=0, vmax=None)
plt.colorbar(shrink=0.5)



plt.imshow(all_data.planes[0].trials[0].images.mean('Time (ms)'), vmin=10, vmax=100)
plt.colorbar(shrink=0.5)


len(all_data.planes)



#* Subtract the background (calculated from the black box) and clip the values to 0.
for plane_i, plane in enumerate(all_data.planes):
	for trial_i, trial in enumerate(plane.trials):
		
		all_data.planes[plane_i].trials[trial_i].images = (trial.images - trial.images[:, x_black_box_beg:x_black_box_end, y_black_box_beg:y_black_box_end].mean(dim=('x','y'))).clip(0,None)


#!!!!!!!!!!!!!!!!!!! IF IN CERTAIN PERIODS OF THE EXPERIMENT
# TRAINING



# ALL_DATA = deepcopy(all_data)
# all_data = deepcopy(ALL_DATA)



# region Voxel analysis
#* Voxel analysis

#* Bin the 2D images.
#! 15-plane experiment
# for plane_i, plane in enumerate(all_data.planes):
# 	for trial_i, trial in enumerate(reversed(plane.trials)):

# 		trial_i = len(plane.trials) - trial_i - 1
# 	# 	break
# 	# break


# #!!!!!!!!!!!!!!!!!!!!
# 		# trial_i=3
# 		# plane_i=12
# #!!!!!!!!!!!!!!!!!!!!


# 		protocol = all_data.planes[plane_i].trials[trial_i].protocol

# 		cs_beg_time = protocol.loc[protocol['CS beg']!=0, 'Time (ms)'].values[0]
# 		cs_end_time = protocol.loc[protocol['CS end']!=0, 'Time (ms)'].values[0]
				
# 		if (us_beg_time := protocol.loc[protocol['US beg']!=0, 'Time (ms)']).empty:
# 			us_beg_time = cs_beg_time + interval_between_CS_onset_US_onset*1000
# 			us_end_time = us_beg_time + 100
# 		else:
# 			us_beg_time = us_beg_time.values[0]
# 			us_end_time = protocol.loc[protocol['US end']!=0, 'Time (ms)'].values[0]
# 		# us_beg_time = protocol.loc[protocol['US beg']!=0, 'Time (ms)'].values[0]

# 		trial_images = all_data.planes[plane_i].trials[trial_i].images
# 		trial_images.values = fi.align_frames(trial_images.to_numpy(), np.stack([trial_images['shift correction in X'].to_numpy(), trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))


# 		trial_images_good = trial_images.sel({'Time (ms)':trial_images['mask good frames']}).copy()


# 		#* Calcultate the correlation map.
# 		# Inspired in Suit2p. There, the function that computes the correlation map is celldetect2.getVmap.
# 		correlation_map = np.linalg.norm(ndimage.gaussian_filter(trial_images_good, sigma=p.correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(trial_images_good, axis=0), sigma=p.
# 		correlation_map_sigma)**2

# 		# Set the correlation around the image to 0
# 		correlation_map[:border_size, :] = 0
# 		correlation_map[-border_size:, :] = 0
# 		correlation_map[:, :border_size] = 0
# 		correlation_map[:, -border_size:] = 0

# 		# plt.figure('Correlation map')
# 		# plt.imshow(correlation_map, interpolation='none', vmin=0.3)
# 		# plt.colorbar(shrink=0.5)
# 		# plt.show()

# 		##* Set to 0 pixels where the correlation is below a threshold.
# 		trial_images_good = trial_images_good.where(correlation_map > correlation_thr, 0)


# 		# Take the median of trial_images_good over time
# 		median_image = np.median(trial_images_good, axis=0)

# 		##* Set to 0 pixels where the median is below a threshold.
# 		trial_images_good = trial_images_good.where(median_image >= median_thr, 0)

# 		# plt.imshow(trial_images_good.mean(dim='Time (ms)'))
		

# 		trial_images_good_binned = block_reduce(trial_images_good, block_size=(1, p.voxel_bin_size, p.voxel_bin_size), func=np.mean, cval=0)

# 		trial_images_good_binned = xr.DataArray(trial_images_good_binned, coords={'index': ('Time (ms)', trial_images_good.index.values), 'Time (ms)': trial_images_good.coords['Time (ms)'].values, 'x': np.arange(trial_images_good_binned.shape[1]), 'y': np.arange(trial_images_good_binned.shape[2]) }, name='trial_images_good_binned',  dims=['Time (ms)', 'x', 'y'])

# 		# plt.imshow(trial_images_good_binned.mean(dim='Time (ms)'), interpolation='none')
# 		# plt.colorbar(shrink=0.5)

# 		baseline_images = trial_images_good_binned.loc[trial_images_good_binned['Time (ms)'] < cs_beg_time]
# 		# CS_US_images = trial_images_good_binned.loc[(trial_images_good_binned['Time (ms)'] > cs_beg_time) & (trial_images_good_binned['Time (ms)'] < us_beg_time)]


# 		cs_beg_position = np.where(trial_images_good_binned['Time (ms)'].values >= cs_beg_time)[0][0]
# 		cs_end_position = np.where(trial_images_good_binned['Time (ms)'].values <= cs_end_time)[0][-1]
# 		us_beg_position = np.where(trial_images_good_binned['Time (ms)'].values >= us_beg_time)[0][0]
# 		us_end_position = np.where(trial_images_good_binned['Time (ms)'].values <= us_end_time)[0][-1]


# 		# cs_beg_time = trial_images_good_binned.sel({'Time (ms)' : cs_beg}, method='bfill')['Time (ms)'].values
# 		# cs_end_time = trial_images_good_binned.sel({'Time (ms)' : cs_end}, method='bfill')['Time (ms)'].values
# 		# us_beg_time = trial_images_good_binned.sel({'Time (ms)' : us_beg}, method='bfill')['Time (ms)'].values
# 		# us_end_time = trial_images_good_binned.sel({'Time (ms)' : us_end}, method='bfill')['Time (ms)'].values

# 		cs_beg_index = trial_images_good_binned.sel({'Time (ms)' : cs_beg_time}, method='bfill')['Time (ms)'].index.values
# 		cs_end_index = trial_images_good_binned.sel({'Time (ms)' : cs_end_time}, method='bfill')['Time (ms)'].index.values
# 		us_beg_index = trial_images_good_binned.sel({'Time (ms)' : us_beg_time}, method='bfill')['Time (ms)'].index.values
# 		us_end_index = trial_images_good_binned.sel({'Time (ms)' : us_end_time}, method='bfill')['Time (ms)'].index.values

# 		##* Calculate deltaF/F for each pixel
# 		# baseline = trial_images_good_binned.sel({'Time (ms)': slice(0, 'CS beg')}).mean(dim='Time (ms)')

# 		baseline = baseline_images.mean(dim='Time (ms)')
# 		# plt.imshow(baseline, interpolation='none', cmap='viridis')
# 		deltaF = (trial_images_good_binned - baseline) / (baseline + softthresh)
# 		# deltaF = (trial_images_good_binned - baseline) / baseline

# 				# # Get a list of traces for each pixel
# 				# traces = []
# 				# for x in range(deltaF.shape[1]):
# 				# 	for y in range(deltaF.shape[2]):
# 				# 		traces.append(deltaF[:, x, y])


# 		# deltaF = (np.clip(deltaF, 1, 3.5) - 1 )/ 2.5

# 		#* Give some colors to the world.

# 		##* Background color

# 		anatomy = trial_images_good_binned.mean(dim='Time (ms)')
# 		anatomy /= np.median(anatomy *30)
# 		anatomy = np.clip(anatomy,0,1).fillna(0)

# 		deltaF_images = []

# 		for frame_i, frame in enumerate(deltaF):

# 			# anatomy = trial_images_good_binned[frame_i]
# 			# anatomy /= np.median(anatomy *30)
# 			# anatomy = np.clip(anatomy,0,1)
			
# 			color_frame_original = (np.clip(frame, 0.5, 3) - 0.5) / 3
		
# 			frame = add_colors_to_world(anatomy, color_frame_original)

# 			deltaF_images.append(frame)

# 			if cs_beg_position <= frame_i <= cs_end_position:
# 				frame = cv2.rectangle(frame, (frame.shape[1] - 10, 0), (frame.shape[1], 10), (255, 255, 0), -1)

# 			# break

		

# 		#* Save deltaF as a multipage TIFF
# 		deltaF_images = [Image.fromarray(frame) for frame in deltaF_images]
# 		tiff_path = results_path / 'Videos with deltaF over F' / f"{fish_name}_deltaF_voxels_plane_{plane_i, trial_i}.tiff"
# 		deltaF_images[0].save(tiff_path, save_all=True, append_images=deltaF_images[1:])


# 		# # Save deltaF as a multipage TIFF
# 		# deltaF_images = [Image.fromarray((frame.values * 255).astype(np.uint8)) for frame in deltaF]
# 		# tiff_path = results_path / 'Videos with deltaF over F' / f"{fish_name}_deltaF_voxels_plane_{plane_i, trial_i}.tiff"
# 		# deltaF_images[0].save(tiff_path, save_all=True, append_images=deltaF_images[1:])


# 		if trial_i == 3:
# 			ROI_list = []
# 			# Create a matplotlib figure
# 			fig, axs = plt.subplots(1, 4, figsize=(30, 6), squeeze=False, sharey=True)

# 		if trial_i in [2,3]:

# 			# Get a list of traces for each pixel along with their coordinates
# 			traces = []
# 			coords = []
# 			for x in range(deltaF.shape[1]):
# 				for y in range(deltaF.shape[2]):
# 					traces.append(deltaF[:, x, y])
# 					coords.append((x, y))

# 			# Add traces for each pixel
# 			for trace, (x, y) in zip(traces, coords):
# 				if max(trace[cs_beg_position:us_beg_position].values) > 3:
# 					axs[0,trial_i].plot(trace['Time (ms)'], trace.values, color='k', linewidth=1, label=f'Pixel ({x}, {y})', alpha=0.7)
# 					ROI_list.append((x,y))

# 			# # Add vertical lines for CS and US events
# 			# axs[0,trial_i].axvline(x=cs_beg_time, color='green', linestyle='--', label='CS Begin')
# 			# axs[0,trial_i].axvline(x=cs_end_time, color='green', linestyle='--', label='CS End')
# 			# axs[0,trial_i].axvline(x=us_beg_time, color='red', linestyle='--', label='US Begin')
# 			# axs[0,trial_i].axvline(x=us_end_time, color='red', linestyle='--', label='US End')

# 			# # Update layout
# 			# axs[0,trial_i].set_title('DeltaF/F Traces for Each Pixel plane ' + str(plane_i) + ' trial ' + str(trial_i))
# 			# axs[0,trial_i].set_xlabel('Time (ms)')
# 			# axs[0,trial_i].set_ylabel('DeltaF/F')
# 			# axs[0,trial_i].legend(loc='upper left', bbox_to_anchor=(1, 1))



# 			anatomy_binarized = trial_images_good_binned.mean(dim='Time (ms)')
# 			anatomy_binarized = np.where(anatomy_binarized > 0, 0.3,0)

# 			anatomy_binarized_rgb = np.stack([anatomy_binarized, anatomy_binarized, anatomy_binarized], axis=-1) 

# 			# mark in the anatomy_binarized image the coords of the pixels in ROI_list
# 			for x,y in ROI_list:
# 				anatomy_binarized_rgb[x,y] = [1,0,0]

# 			anatomy_binarized_rgb = Image.fromarray((anatomy_binarized_rgb * 255).astype(np.uint8))

# 			# Save the anatomy_binarized_rgb image as a PNG file
# 			anatomy_binarized_rgb = anatomy_binarized_rgb.resize((anatomy_binarized_rgb.width * 5, anatomy_binarized_rgb.height * 5))
# 			anatomy_binarized_rgb.save(results_path / 'Anatomy with ROIs' / (f'{fish_name}_CS ROIs during train_plane ' + str(plane_i) + ' trial ' + str(trial_i) + '.png'), dpi=(600, 600))



# 		if trial_i in [0,1]:

# 			# Plot the traces of the pixels with the coordinates in ROI_list.
# 			# Create a matplotlib figure
# 			# fig, axs[0,trial_i] = plt.subplots(figsize=(15, 6))

# 			ROI_list_before_training = []

# 			for x,y in ROI_list:
# 				trace = deltaF[:, x, y]
# 				# Add traces for each pixel
# 				axs[0,trial_i].plot(trace['Time (ms)'], trace.values, color='k', linewidth=1, label=f'Pixel ({x}, {y})', alpha=0.7)

# 				if max(trace[cs_beg_position:us_beg_position].values) > 3:
# 					ROI_list_before_training.append((x,y))

# 			anatomy_binarized = trial_images_good_binned.mean(dim='Time (ms)')
# 			anatomy_binarized = np.where(anatomy_binarized > 0, 0.3,0)

# 			anatomy_binarized_rgb = np.stack([anatomy_binarized, anatomy_binarized, anatomy_binarized], axis=-1)

# 			# mark in the anatomy_binarized image the coords of the pixels in ROI_list
# 			for x,y in ROI_list_before_training:
# 				anatomy_binarized_rgb[x,y] = [0,1,0]

# 			anatomy_binarized_rgb = Image.fromarray((anatomy_binarized_rgb * 255).astype(np.uint8))
			
# 			# Save the anatomy_binarized_rgb image as a PNG file
# 			anatomy_binarized_rgb = anatomy_binarized_rgb.resize((anatomy_binarized_rgb.width * 5, anatomy_binarized_rgb.height * 5))
# 			anatomy_binarized_rgb.save(results_path / 'Anatomy with ROIs' / (f'{fish_name}_CS ROIs before train from CS ROIs during train_plane ' + str(plane_i) + ' trial ' + str(trial_i) + '.png'), dpi=(600, 600))



# 		# Add vertical lines for CS and US events
# 		axs[0,trial_i].axvline(x=cs_beg_time, color='green', linestyle='--', label='CS Begin')
# 		axs[0,trial_i].axvline(x=cs_end_time, color='green', linestyle='--', label='CS End')
# 		axs[0,trial_i].axvline(x=us_beg_time, color='red', linestyle='--', label='US Begin')
# 		axs[0,trial_i].axvline(x=us_end_time, color='red', linestyle='--', label='US End')

# 		# Update layout
# 		axs[0,trial_i].set_title('DeltaF/F Traces for Each Pixel plane ' + str(plane_i) + ' trial ' + str(trial_i))
# 		axs[0,trial_i].set_xlabel('Time (ms)')
# 		axs[0,trial_i].set_ylabel('DeltaF/F')
# 		axs[0,trial_i].legend(loc='upper left', bbox_to_anchor=(1, 1))

# 		if trial_i == 0:
# 			# Save the figure as a png with tight_layout
# 			fig.tight_layout()
# 			fig.savefig(results_path / 'ROI traces' / f"{fish_name}_traces deltaF over F_plane {plane_i}.png")

# 		# break
# 	# break








#! 4-plane experiment

# number_imaged_planes = 4
# number_reps_plane_consective = 2
# # relevant_cs = [range(5,15),
# # 			  range(15,25), range(25,55), range(55,45), range(45,55),
# # 			  range(55,65), range(65,75), range(75,85)]
# relevant_cs = [range(5,13),
# 				range(15,23), range(25,33), range(35,43), range(45,53),
# 				range(55,63), range(67,75), range(77,85)]
# index_list = [np.concatenate([[i+number_reps_plane_consective*x*number_imaged_planes, i+number_reps_plane_consective*x*number_imaged_planes+1] for x in range(len(relevant_cs))]) for i in range(0, number_reps_plane_consective * number_imaged_planes, number_reps_plane_consective)]


# trials_list = [11, 12] + list(range(0,10)) + list(range(13,16))

# plane_images.shape
plane_anatomy = [[] for _ in range(len(all_data.planes))]

for plane_i, plane in tqdm(enumerate(all_data.planes)):

	for trial_i in tqdm(range(len(all_data.planes[0].trials))):

		trial_images = all_data.planes[plane_i].trials[trial_i].images
		trial_images.values = fi.align_frames(trial_images.to_numpy(), np.stack([trial_images['shift correction in X'].to_numpy(), trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))
		trial_images = trial_images.sel({'Time (ms)':trial_images['mask good frames']})

		plane_anatomy[plane_i].append(trial_images.values)
		# plane_images[plane_i] += trial_images.mean(dim='Time (ms)')
		# if trial_i == 1:
		# 	break
		# break
	plane_anatomy[plane_i] = np.concatenate(plane_anatomy[plane_i], axis=0)
	# break
	# plane_images[plane_i] /= len(trials_list)

	#* Calcultate the correlation map.
	# Inspired in Suit2p. There, the function that computes the correlation map is celldetect2.getVmap.
	correlation_map = np.linalg.norm(ndimage.gaussian_filter(plane_anatomy[plane_i], sigma=p.correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(plane_anatomy[plane_i], axis=0), sigma=p.correlation_map_sigma)**2

	# Set the correlation around the image to 0
	correlation_map[:border_size, :] = 0
	correlation_map[-border_size:, :] = 0
	correlation_map[:, :border_size] = 0
	correlation_map[:, -border_size:] = 0

	plt.figure('Correlation map')
	plt.imshow(correlation_map, interpolation='none', vmin=0.3)
	plt.colorbar(shrink=0.5)
	plt.show()
	# break

	##* Set to 0 pixels where the correlation is below a threshold.
	plane_anatomy[plane_i] = np.mean(plane_anatomy[plane_i], axis=0)
	plane_anatomy[plane_i] = np.where(correlation_map > correlation_thr, plane_anatomy[plane_i], 0)

# plt.imshow(np.mean(plane_images[0],axis=0), interpolation='none')
# plt.colorbar(shrink=0.5)


for plane_i, plane in tqdm(enumerate(all_data.planes)):

	# if plane_i not in [0,1]:
	# 	continue

	plane_anatomy_mask = np.where(plane_anatomy[plane_i] > 0, 0.5, 0).copy()

	for trial_i in tqdm(trials_list):

	# 	break
	# break

#!!!!!!!!!!!!!!!!!!!!
		# trial_i=3
		# plane_i=12
#!!!!!!!!!!!!!!!!!!!!

		protocol = all_data.planes[plane_i].trials[trial_i].protocol

		cs_beg_time = protocol.loc[protocol['CS beg']!=0, 'Time (ms)'].values[0]
		cs_end_time = protocol.loc[protocol['CS end']!=0, 'Time (ms)'].values[0]
				
		if (us_beg_time := protocol.loc[protocol['US beg']!=0, 'Time (ms)']).empty:
			us_beg_time = cs_beg_time + interval_between_CS_onset_US_onset*1000
			us_end_time = us_beg_time + 100
		else:
			us_beg_time = us_beg_time.values[0]
			us_end_time = protocol.loc[protocol['US end']!=0, 'Time (ms)'].values[0]
		# us_beg_time = protocol.loc[protocol['US beg']!=0, 'Time (ms)'].values[0]

		trial_images = all_data.planes[plane_i].trials[trial_i].images
		trial_images.values = fi.align_frames(trial_images.to_numpy(), np.stack([trial_images['shift correction in X'].to_numpy(), trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))


		trial_images_good = trial_images.sel({'Time (ms)':trial_images['mask good frames']}).copy()


		# #* Calcultate the correlation map.
		# # Inspired in Suit2p. There, the function that computes the correlation map is celldetect2.getVmap.
		# correlation_map = np.linalg.norm(ndimage.gaussian_filter(trial_images_good, sigma=p.correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(trial_images_good, axis=0), sigma=p.
		# correlation_map_sigma)**2

		# # Set the correlation around the image to 0
		# correlation_map[:border_size, :] = 0
		# correlation_map[-border_size:, :] = 0
		# correlation_map[:, :border_size] = 0
		# correlation_map[:, -border_size:] = 0

		# # plt.figure('Correlation map')
		# # plt.imshow(correlation_map, interpolation='none', vmin=0.3)
		# # plt.colorbar(shrink=0.5)
		# # plt.show()

		# ##* Set to 0 pixels where the correlation is below a threshold.
		# trial_images_good = trial_images_good.where(correlation_map > correlation_thr, 0)


		# # Take the median of trial_images_good over time
		# median_image = np.median(trial_images_good, axis=0)

		# ##* Set to 0 pixels where the median is below a threshold.
		# trial_images_good = trial_images_good.where(median_image >= median_thr, 0)

		# # plt.imshow(trial_images_good.mean(dim='Time (ms)'))
		

		##* Set to 0 pixels where the anatomy mask is below a threshold.
		trial_images_good = trial_images_good.where(plane_anatomy_mask > 0, 0)
		# plt.imshow(np.mean(trial_images_good, axis=0))


		trial_images_good_binned = block_reduce(trial_images_good, block_size=(1, p.voxel_bin_size, p.voxel_bin_size), func=np.mean, cval=0)

		trial_images_good_binned = xr.DataArray(trial_images_good_binned, coords={'index': ('Time (ms)', trial_images_good.index.values), 'Time (ms)': trial_images_good.coords['Time (ms)'].values, 'x': np.arange(trial_images_good_binned.shape[1]), 'y': np.arange(trial_images_good_binned.shape[2]) }, name='trial_images_good_binned',  dims=['Time (ms)', 'x', 'y'])

		# plt.imshow(trial_images_good_binned.mean(dim='Time (ms)'), interpolation='none')
		# plt.colorbar(shrink=0.5)

		baseline_images = trial_images_good_binned.loc[trial_images_good_binned['Time (ms)'] < cs_beg_time]
		# CS_US_images = trial_images_good_binned.loc[(trial_images_good_binned['Time (ms)'] > cs_beg_time) & (trial_images_good_binned['Time (ms)'] < us_beg_time)]


		cs_beg_position = np.where(trial_images_good_binned['Time (ms)'].values >= cs_beg_time)[0][0]
		cs_end_position = np.where(trial_images_good_binned['Time (ms)'].values <= cs_end_time)[0][-1]
		us_beg_position = np.where(trial_images_good_binned['Time (ms)'].values >= us_beg_time)[0][0]
		us_end_position = np.where(trial_images_good_binned['Time (ms)'].values <= us_end_time)[0][-1]


		# cs_beg_time = trial_images_good_binned.sel({'Time (ms)' : cs_beg}, method='bfill')['Time (ms)'].values
		# cs_end_time = trial_images_good_binned.sel({'Time (ms)' : cs_end}, method='bfill')['Time (ms)'].values
		# us_beg_time = trial_images_good_binned.sel({'Time (ms)' : us_beg}, method='bfill')['Time (ms)'].values
		# us_end_time = trial_images_good_binned.sel({'Time (ms)' : us_end}, method='bfill')['Time (ms)'].values

		cs_beg_index = trial_images_good_binned.sel({'Time (ms)' : cs_beg_time}, method='bfill')['Time (ms)'].index.values
		cs_end_index = trial_images_good_binned.sel({'Time (ms)' : cs_end_time}, method='bfill')['Time (ms)'].index.values
		us_beg_index = trial_images_good_binned.sel({'Time (ms)' : us_beg_time}, method='bfill')['Time (ms)'].index.values
		us_end_index = trial_images_good_binned.sel({'Time (ms)' : us_end_time}, method='bfill')['Time (ms)'].index.values

		##* Calculate deltaF/F for each pixel
		# baseline = trial_images_good_binned.sel({'Time (ms)': slice(0, 'CS beg')}).mean(dim='Time (ms)')

		baseline = baseline_images.mean(dim='Time (ms)')
		# plt.imshow(baseline, interpolation='none', cmap='viridis')
		deltaF = (trial_images_good_binned - baseline) / (baseline + softthresh)
		# deltaF = (trial_images_good_binned - baseline) / baseline

				# # Get a list of traces for each pixel
				# traces = []
				# for x in range(deltaF.shape[1]):
				# 	for y in range(deltaF.shape[2]):
				# 		traces.append(deltaF[:, x, y])


		# deltaF = (np.clip(deltaF, 1, 3.5) - 1 )/ 2.5

		#* Give some colors to the world.

		##* Background color

		anatomy = trial_images_good_binned.mean(dim='Time (ms)')
		anatomy /= np.median(anatomy *30)
		anatomy = np.clip(anatomy,0,1).fillna(0)

		deltaF_images = []

		for frame_i, frame in enumerate(deltaF):

			# anatomy = trial_images_good_binned[frame_i]
			# anatomy /= np.median(anatomy *30)
			# anatomy = np.clip(anatomy,0,1)
			
			color_frame_original = (np.clip(frame, 0.5, 3) - 0.5) / 3
		
			frame = add_colors_to_world(anatomy, color_frame_original)

			deltaF_images.append(frame)

			if cs_beg_position <= frame_i <= cs_end_position:
				frame = cv2.rectangle(frame, (frame.shape[1] - 10, 0), (frame.shape[1], 10), (255, 255, 0), -1)

# 			# break

		

# 		#* Save deltaF as a multipage TIFF
# 		deltaF_images = [Image.fromarray(frame) for frame in deltaF_images]
# 		tiff_path = results_path / 'Videos with deltaF over F' / f"{fish_name}_deltaF_voxels_plane_{plane_i, trial_i}.tiff"
# 		deltaF_images[0].save(tiff_path, save_all=True, append_images=deltaF_images[1:])


# 		# # Save deltaF as a multipage TIFF
# 		# deltaF_images = [Image.fromarray((frame.values * 255).astype(np.uint8)) for frame in deltaF]
# 		# tiff_path = results_path / 'Videos with deltaF over F' / f"{fish_name}_deltaF_voxels_plane_{plane_i, trial_i}.tiff"
# 		# deltaF_images[0].save(tiff_path, save_all=True, append_images=deltaF_images[1:])


# 		if trial_i == 3:
# 			ROI_list = []
# 			# Create a matplotlib figure
# 			fig, axs = plt.subplots(1, 4, figsize=(30, 6), squeeze=False, sharey=True)

# 		if trial_i in [2,3]:

# 			# Get a list of traces for each pixel along with their coordinates
# 			traces = []
# 			coords = []
# 			for x in range(deltaF.shape[1]):
# 				for y in range(deltaF.shape[2]):
# 					traces.append(deltaF[:, x, y])
# 					coords.append((x, y))

# 			# Add traces for each pixel
# 			for trace, (x, y) in zip(traces, coords):
# 				if max(trace[cs_beg_position:us_beg_position].values) > 3:
# 					axs[0,trial_i].plot(trace['Time (ms)'], trace.values, color='k', linewidth=1, label=f'Pixel ({x}, {y})', alpha=0.7)
# 					ROI_list.append((x,y))

# 			# # Add vertical lines for CS and US events
# 			# axs[0,trial_i].axvline(x=cs_beg_time, color='green', linestyle='--', label='CS Begin')
# 			# axs[0,trial_i].axvline(x=cs_end_time, color='green', linestyle='--', label='CS End')
# 			# axs[0,trial_i].axvline(x=us_beg_time, color='red', linestyle='--', label='US Begin')
# 			# axs[0,trial_i].axvline(x=us_end_time, color='red', linestyle='--', label='US End')

# 			# # Update layout
# 			# axs[0,trial_i].set_title('DeltaF/F Traces for Each Pixel plane ' + str(plane_i) + ' trial ' + str(trial_i))
# 			# axs[0,trial_i].set_xlabel('Time (ms)')
# 			# axs[0,trial_i].set_ylabel('DeltaF/F')
# 			# axs[0,trial_i].legend(loc='upper left', bbox_to_anchor=(1, 1))



# 			anatomy_binarized = trial_images_good_binned.mean(dim='Time (ms)')
# 			anatomy_binarized = np.where(anatomy_binarized > 0, 0.3,0)

# 			anatomy_binarized_rgb = np.stack([anatomy_binarized, anatomy_binarized, anatomy_binarized], axis=-1) 

# 			# mark in the anatomy_binarized image the coords of the pixels in ROI_list
# 			for x,y in ROI_list:
# 				anatomy_binarized_rgb[x,y] = [1,0,0]

# 			anatomy_binarized_rgb = Image.fromarray((anatomy_binarized_rgb * 255).astype(np.uint8))

# 			# Save the anatomy_binarized_rgb image as a PNG file
# 			anatomy_binarized_rgb = anatomy_binarized_rgb.resize((anatomy_binarized_rgb.width * 5, anatomy_binarized_rgb.height * 5))
# 			anatomy_binarized_rgb.save(results_path / 'Anatomy with ROIs' / (f'{fish_name}_CS ROIs during train_plane ' + str(plane_i) + ' trial ' + str(trial_i) + '.png'), dpi=(600, 600))



# 		if trial_i in [0,1]:

# 			# Plot the traces of the pixels with the coordinates in ROI_list.
# 			# Create a matplotlib figure
# 			# fig, axs[0,trial_i] = plt.subplots(figsize=(15, 6))

# 			ROI_list_before_training = []

# 			for x,y in ROI_list:
# 				trace = deltaF[:, x, y]
# 				# Add traces for each pixel
# 				axs[0,trial_i].plot(trace['Time (ms)'], trace.values, color='k', linewidth=1, label=f'Pixel ({x}, {y})', alpha=0.7)

# 				if max(trace[cs_beg_position:us_beg_position].values) > 3:
# 					ROI_list_before_training.append((x,y))

# 			anatomy_binarized = trial_images_good_binned.mean(dim='Time (ms)')
# 			anatomy_binarized = np.where(anatomy_binarized > 0, 0.3,0)

# 			anatomy_binarized_rgb = np.stack([anatomy_binarized, anatomy_binarized, anatomy_binarized], axis=-1)

# 			# mark in the anatomy_binarized image the coords of the pixels in ROI_list
# 			for x,y in ROI_list_before_training:
# 				anatomy_binarized_rgb[x,y] = [0,1,0]

# 			anatomy_binarized_rgb = Image.fromarray((anatomy_binarized_rgb * 255).astype(np.uint8))
			
# 			# Save the anatomy_binarized_rgb image as a PNG file
# 			anatomy_binarized_rgb = anatomy_binarized_rgb.resize((anatomy_binarized_rgb.width * 5, anatomy_binarized_rgb.height * 5))
# 			anatomy_binarized_rgb.save(results_path / 'Anatomy with ROIs' / (f'{fish_name}_CS ROIs before train from CS ROIs during train_plane ' + str(plane_i) + ' trial ' + str(trial_i) + '.png'), dpi=(600, 600))



# 		# Add vertical lines for CS and US events
# 		axs[0,trial_i].axvline(x=cs_beg_time, color='green', linestyle='--', label='CS Begin')
# 		axs[0,trial_i].axvline(x=cs_end_time, color='green', linestyle='--', label='CS End')
# 		axs[0,trial_i].axvline(x=us_beg_time, color='red', linestyle='--', label='US Begin')
# 		axs[0,trial_i].axvline(x=us_end_time, color='red', linestyle='--', label='US End')

# 		# Update layout
# 		axs[0,trial_i].set_title('DeltaF/F Traces for Each Pixel plane ' + str(plane_i) + ' trial ' + str(trial_i))
# 		axs[0,trial_i].set_xlabel('Time (ms)')
# 		axs[0,trial_i].set_ylabel('DeltaF/F')
# 		axs[0,trial_i].legend(loc='upper left', bbox_to_anchor=(1, 1))

# 		if trial_i == 0:
# 			# Save the figure as a png with tight_layout
# 			fig.tight_layout()
# 			fig.savefig(results_path / 'ROI traces' / f"{fish_name}_traces deltaF over F_plane {plane_i}.png")

# 		# break
# 	# break










trial_images_good = trial_images.sel({'Time (ms)':trial_images['mask good frames']}).copy()

A = trial_images_good.copy()
A.values = np.array([trial_images_good.median(dim='Time (ms)').values]*trial_images_good.shape[0])
# A = trial_images_good.median(dim='Time (ms)')


B = np.where(A < 0,0,trial_images_good)
A.shape
B.shape
trial_images_good.shape
# B = trial_images_good.where(A > 0,0,1)

plt.imshow(A.mean(dim='Time (ms)'), interpolation='none')

plt.imshow(B.mean(axis=0), vmax=10)
plt.colorbar(shrink=0.5)




trial_images_good_fish = np.where(np.median(trial_images_good, axis=0) <= 0, 0, 1).astype(dtype='bool')
trial_images_good_fish = xr.DataArray(
	np.where(np.median(trial_images_good, axis=0) <= 0, 0, 1).astype(dtype='bool'),
	dims=['x', 'y'],
	coords={'x': trial_images_good.coords['x'], 'y': trial_images_good.coords['y']}
)


trial_images_good.sel({'Time (ms)':trial_images_good_fish})

trial_template_aligned = f.get_template_image(trial_images_aligned.sel(time=trial_images['mask good frames']))








# Plot all traces in a line plot with grey color
fig, ax = plt.subplots(figsize=(50, 6))
for trace in traces:
	ax.plot(trace['Time (ms)'], trace.values, color='grey', alpha=0.5)

ax.plot(trace['Time (ms)'], np.zeros(len(trace)), color='white', alpha=0)

ax.set_xlabel('Time (ms)')
ax.set_ylabel('DeltaF/F')
ax.set_title('DeltaF/F Traces for Each Pixel')


# ax.axvline(x=cs_beg_index, color='green', linestyle='--', label='CS Begin')
# ax.axvline(x=cs_end_index, color='green', linestyle='--', label='CS End')
# ax.axvline(x=us_beg_index, color='red', linestyle='--', label='US Begin')
# ax.axvline(x=us_end_index, color='red', linestyle='--', label='US End')

ax.axvline(x=cs_beg_time, color='green', linestyle='--', label='CS Begin')
ax.axvline(x=cs_end_time, color='green', linestyle='--', label='CS End')
ax.axvline(x=us_beg_time, color='red', linestyle='--', label='US Begin')
ax.axvline(x=us_end_time, color='red', linestyle='--', label='US End')

ax.legend()





# Get a list of traces for each pixel along with their coordinates
traces = []
coords = []
for x in range(deltaF.shape[1]):
	for y in range(deltaF.shape[2]):
		traces.append(deltaF[:, x, y])
		coords.append((x, y))


ROI_list = []

# Create a matplotlib figure
fig, ax = plt.subplots(figsize=(15, 6))
# Add traces for each pixel
for trace, (x, y) in zip(traces, coords):
	if max(trace[cs_beg_position:cs_end_position].values) > 3:
		ax.plot(trace['Time (ms)'], trace.values, color='grey', linewidth=1, label=f'Pixel ({x}, {y})')
		ROI_list.append((x,y))

# Add vertical lines for CS and US events
ax.axvline(x=cs_beg_time, color='green', linestyle='--', label='CS Begin')
ax.axvline(x=cs_end_time, color='green', linestyle='--', label='CS End')
ax.axvline(x=us_beg_time, color='red', linestyle='--', label='US Begin')
ax.axvline(x=us_end_time, color='red', linestyle='--', label='US End')

# Update layout
ax.set_title('DeltaF/F Traces for Each Pixel')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('DeltaF/F')
ax.legend()
plt.show()






# Get a list of traces for each pixel along with their coordinates
traces = []
coords = []
for x in range(deltaF.shape[1]):
	for y in range(deltaF.shape[2]):
		traces.append(deltaF[:, x, y])
		coords.append((x, y))

# Create a plotly figure
fig = go.Figure()

# Add traces for each pixel
for trace, (x, y) in zip(traces, coords):
	if max(trace[cs_beg_position:cs_end_position].values) > 3:
		fig.add_trace(go.Scatter(x=trace['Time (ms)'], y=trace.values, mode='lines', line=dict(color='grey', width=1), name=f'Pixel ({x}, {y})'))

# Add vertical lines for CS and US events
fig.add_vline(x=cs_beg_time, line=dict(color='green', dash='dash'), name='CS Begin')
fig.add_vline(x=cs_end_time, line=dict(color='green', dash='dash'), name='CS End')
fig.add_vline(x=us_beg_time, line=dict(color='red', dash='dash'), name='US Begin')
fig.add_vline(x=us_end_time, line=dict(color='red', dash='dash'), name='US End')

# Update layout
fig.update_layout(
	title='DeltaF/F Traces for Each Pixel',
	xaxis_title='Time (ms)',
	yaxis_title='DeltaF/F',
	showlegend=False,
	width=1500,
	height=600
)

fig.update_layout(showlegend=True)

fig.show()


plt.imshow(trial_images_good_binned.mean(dim='Time (ms)'), interpolation='none')

plt.imshow((CS_US_images.mean(dim='Time (ms)') - baseline)/(baseline + softthresh), interpolation='none')

plt.imshow(CS_US_images.mean(dim='Time (ms)'), interpolation='none')






#* Save rois_zscore_over_time as a TIFF file.
tifffile.imwrite(imaging_path_ /  fish_name / (fish_name + '_deltaF_voxels_plane ' + str(plane_i) + '.tif'), deltaF.astype('float32'))

# endregion






























#* Split data into 3 periods: before CS, from CS onset to US onset and after US onset.
for plane_i, plane in enumerate(all_data.planes):

	# if plane_i < 3:
	# 	continue

	plane_template = np.zeros(plane.trials[0].images.shape[1:])


	# fig, axs = plt.subplots()


	for trial_i, trial in enumerate(plane.trials):

	# 	break
	# break

		cs_beg_time = trial.protocol.loc[trial.protocol['CS beg'] != 0, 'Time (ms)'].to_numpy()[0]

		if (us_beg_time := trial.protocol.loc[trial.protocol['US beg'] != 0, 'Time (ms)']).empty:
			us_beg_time = cs_beg_time + interval_between_CS_onset_US_onset*1000

		else:
	#! issue when multiple US in trial
			us_beg_time = us_beg_time.to_numpy()[0]



		trial_images = trial.images.copy()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		trial_images = trial_images.fillna(0)
# trial.__dict__.keys()
# trial.mask_good_frames
	#!remove
		# trial_images = trial_images.assign_coords({'mask good frames' : ('Time (ms)', trial.mask_good_frames), 'shift correction in X' : ('Time (ms)', trial.shift_correction[:,0].astype('float32')), 'shift correction in Y' : ('Time (ms)', trial.shift_correction[:,1].astype('float32'))})
	#!

		trial_images['mask before CS'] = trial_images['Time (ms)'] < cs_beg_time
		trial_images['mask CS-US'] = (trial_images['Time (ms)'] > cs_beg_time) & (trial_images['Time (ms)'] < us_beg_time)
		trial_images['mask after US'] = trial_images['Time (ms)'] > us_beg_time



		#* Align the frames
		# trial_images_aligned = trial_images.copy()
		# trial_images_aligned.values = f.align_frames(trial_images.to_numpy(), np.stack([trial_images['shift correction in X'].to_numpy(), trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))

		# print(plane_i, trial_i)
		# print(1)

		# plt.imshow(trial.template_image)
		# plt.show()

		# trial_template_aligned = f.get_template_image(trial_images_aligned.sel(time=trial_images['mask good frames']))

		# plt.imshow(trial_template_aligned)

		# trial_images_aligned = trial_images.copy()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! correct for single plane
		trial_images.values = fi.align_frames(trial_images.to_numpy(), np.stack([trial_images['shift correction in X'].to_numpy(), trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))

		trial.images = trial_images



		trial.template_image = fi.get_template_image(trial_images.sel({'Time (ms)':trial_images['mask good frames']}))

		# print(2)

		# plt.imshow(trial.template_image)
		# plt.show()


		# break
	# break



		trial.pre_CS_mean = trial_images.sel({'Time (ms)':(trial_images['mask before CS']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')
		trial.CS_US_mean = trial_images.sel({'Time (ms)':(trial_images['mask CS-US']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')
		trial.post_US_mean = trial_images.sel({'Time (ms)':(trial_images['mask after US']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')

		pre_CS_data = trial.pre_CS_mean.values

		trial.CS_US_vs_pre = gaussian_filter(((trial.CS_US_mean.values - pre_CS_data) / (pre_CS_data + softthresh)), sigma=2)
		trial.post_US_vs_pre = gaussian_filter((trial.post_US_mean.values - pre_CS_data / (pre_CS_data + softthresh)), sigma=2)



		#* Give some colors to the world.

		##* Background color
		anatomy = trial.template_image.copy()
#!!!!!!!!!!!   USE QUANTILESSSSS
		anatomy /= np.median(anatomy *30)
		anatomy = np.clip(anatomy,0,1)
		
		trial.anatomy_channel = anatomy

		plt.imshow(anatomy, vmax=1)
		# plt.colorbar()
		# plt.imshow(trial.template_image)


		###* CS responses

		####* CS positive response
		color_frame_original = np.clip(trial.CS_US_vs_pre, 0, 1)
		trial.CS_positive_response = fi.add_colors_to_world(anatomy, color_frame_original)


		###* CS negative response
		# color_frame_original = np.clip(-trial.CS_US_vs_pre*5, 0, 1)
		# trial.CS_negative_response = fi.add_colors_to_world(anatomy, color_frame_original)


		# ###* US positive response
		# color_frame_original = np.clip(trial.post_US_vs_pre, 0, 1)
		# trial.CS_negative_response = fi.add_colors_to_world(anatomy, color_frame_original)


		# ###* US negative response
		# color_frame_original = np.clip(-trial.post_US_vs_pre*5, 0, 1)
		# trial.CS_negative_response = fi.add_colors_to_world(anatomy, color_frame_original)



		trial.images = trial_images
		all_data.planes[plane_i].trials[trial_i] = trial

# all_data.planes[plane_i].trials[trial_i].images['shift_correction in X'].values

# __dict__.keys()




		# plane_template = plane_template + trial.pre_CS_mean.to_numpy()
		# plane_template += trial.template_image

	# plane_template = ndimage.median_filter(plane_template, size=p.median_filter_kernel, axes=(0,1))

	# plane_template = trial.template_image.copy()
	# plane_template /= np.median(plane_template   *4)
	# plane_template = np.clip(plane_template,0,1)

	# plt.imshow(plane_template)
	# plt.colorbar()
	# plt.show()
	# break

	# plt.imshow(trial.pre_CS_mean)

	# plane.template = plane_template

	# all_data.planes[plane_i].template = plane_template



	# break






# #* Using a general template per plane

# for plane_i, plane in enumerate(all_data.planes):

# 	plane_baseline_color = plane.template

# 	for trial_i, trial in enumerate(plane.trials):

# 		color_frame_original = np.clip(trial.CS_US_vs_pre, 0, 1)

# 		trial_red_channel = plane_baseline_color * (1 - color_frame_original)
# 		trial_green_channel = trial_red_channel + color_frame_original * color_frame_original
# 		trial_blue_channel = trial_red_channel

# 		trial_CS_response = np.stack([trial_red_channel,trial_green_channel,trial_blue_channel])


# 		trial_CS_response = np.moveaxis(trial_CS_response, 0, -1)

# 		plt.imshow(Image.fromarray((trial_CS_response*256).astype(np.uint8)))
# 		plt.show()

		
# 		# break
# 	break

# plane_numbers = np.zeros((15,4))

fig, axs = plt.subplots(len(all_data.planes), len(plane.trials), figsize=(10, 50), squeeze=False)

for plane_i in range(len(all_data.planes)):
	for trial_i in range(len(plane.trials)):
		
		
		# plane_numbers[plane_i,trial_i] = all_data.planes[plane_i].trials[trial_i].position_anatomical_stack


		# position = plane_position_stack[plane_i]
		# all_data.planes[plane_i].template_image_position_anatomical_stack
		# print(position)
		
		axs[plane_i,trial_i].imshow(all_data.planes[plane_i].trials[trial_i].CS_positive_response)
		axs[plane_i,trial_i].set_xticks([])
		axs[plane_i,trial_i].set_yticks([])



# plane_numbers = plane_numbers.astype('int')

# plane_position_stack = np.argsort(plane_numbers[:,0])
	# break

fig.set_size_inches(30, 70)
fig.subplots_adjust(hspace=0.05, wspace=0.02)

fig.savefig(imaging_path_ / 'CS positive response.png', dpi=400)



# all_data = c.Data(all_data.planes, anatomical_stack_images)

path_pkl_analysis_1 = path_home / fish_name / (fish_name + '_analysis 1' + '.pkl')

with open(path_pkl_analysis_1, 'wb') as file:
	pickle.dump(all_data, file)


print('END')



fig, axs = plt.subplots(len(all_data.planes), len(plane.trials), figsize=(10, 50), squeeze=False)

for plane_i in range(len(all_data.planes)):
	for trial_i in range(len(plane.trials)):
		
		
		# plane_numbers[plane_i,trial_i] = all_data.planes[plane_i].trials[trial_i].position_anatomical_stack


		# position = plane_position_stack[plane_i]
		# all_data.planes[plane_i].template_image_position_anatomical_stack
		# print(position)
		
		axs[plane_i,trial_i].imshow(all_data.planes[plane_i].trials[trial_i].CS_positive_response)
		axs[plane_i,trial_i].set_xticks([])
		axs[plane_i,trial_i].set_yticks([])



# plane_numbers = plane_numbers.astype('int')

# plane_position_stack = np.argsort(plane_numbers[:,0])
	# break

fig.set_size_inches(50, 30)
fig.subplots_adjust(hspace=0.05, wspace=0.02)

fig.tight_layout()

fig.savefig(imaging_path_ / 'CS positive response 2.png', dpi=400)



tifffile = []

for trial_i, trial in enumerate(all_data.planes[2].trials):

	tifffile.append(Image.fromarray(trial.CS_positive_response))





# Save tifffile as a multipage tiff
tiff_path = imaging_path_ / 'CS_positive_response_multipage new 1.tiff'
tifffile[0].save(tiff_path, save_all=True, append_images=tifffile[1:])



# Load the TIFF file
tiff_path = imaging_path_ / 'CS_positive_response_multipage new 1.tiff'


# Label a specific frame in the TIFF stack
frame_index = 0  # Change this to the index of the frame you want to label
labeled_frame = tifffile[1]

# Display the labeled frame
plt.imshow(labeled_frame, cmap='gray')
plt.title(f'Labeled Frame {1}')
plt.colorbar()
plt.show()







# all_data.planes[plane_i].trials[0].__dict__.keys()





