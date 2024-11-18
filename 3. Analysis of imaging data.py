#* Imports

# %%
# region Imports

import os
import pickle
from copy import deepcopy
from importlib import reload
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import scipy.ndimage as ndimage
import seaborn as sns
import tifffile as tiff
import xarray as xr
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

#* Load custom functions and classes
import my_classes as c
import my_functions_imaging as fi
import my_parameters as p
from my_general_variables import *

reload(fi)
reload(c)
reload(p)
# endregion

#* Settings
# %% Settings
# region Settings
pio.templates.default = "plotly_dark"
pd.set_option("mode.copy_on_write", True)
# pd.set_option("compute.use_numba", True)
pd.set_option("compute.use_numexpr", True)
pd.set_option("compute.use_bottleneck", True)
#endregion



#* Load CSV data
# %% Load CSV data
# region Load CSV data

csv_path = r"C:\Users\joaqc\Desktop\Book1.csv"
csv_data = pd.read_csv(csv_path, sep=';')

csv_data = csv_data.iloc[:,1:]

for i in range(4):
	plt.plot(csv_data.index, csv_data.iloc[:,i])
plt.xlim(-1, len(csv_data.index))
plt.ylabel('deltaF/F')
plt.xlabel('Trial')
plt.savefig(r"C:\Users\joaqc\Desktop\deltaFF.svg", format="svg")

# Display the first few rows of the CSV data
print(csv_data.head())

# endregion

#* Paths
# %%
# region Paths
path_home = Path(r'E:\2024 09_Delay 2-P 4 planes JC neurons')
# Path(r'E:\2024 10_Delay 2-P single plane')
# Path(r'E:\2024 03_Delay 2-P 15 planes top part')
# Path(r'E:\2024 10_Delay 2-P 15 planes ca8 neurons')
# Path(r'E:\2024 09_Delay 2-P zoom in multiplane imaging')

# fish_list = [f for f in (path_home / 'Imaging').iterdir() if f.is_dir()]
# fish_names_list = [f.stem for f in fish_list]

fish_name = r'20241013_01_control_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241015_03_delay_2p-9_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf'
# '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
# '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# !'20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
#! 20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf
#! 20240415_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf
#! 20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf

# '20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# r'20240926_03_trace_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241015_03_delay_2p-9_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf'
# '20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'
# '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'



imaging_path = path_home / 'Imaging'


# for fish_i, fish_name in enumerate(fish_names_list):

# 	try:

imaging_path_ = imaging_path / fish_name / 'Imaging'


path_pkl_after_motion_correction = path_home / fish_name / (fish_name + '_after motion correction' + '.pkl')


# h5_path = path_home / fish_name / (fish_name + '_before_motion_correction.h5')

# h5_path = imaging_path_ / (fish_name + '_before_motion_correction.h5')

# h5_path = r"E:\2024 03_Delay 2-P 15 planes top part\Imaging\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf\Imaging\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf_before_motion_correction.h5"

#endregion



if 'delay' in fish_name:
	interval_between_CS_onset_US_onset = 9  # s
elif 'trace' in fish_name:
	interval_between_CS_onset_US_onset = 13  # s
elif 'control' in fish_name:
	interval_between_CS_onset_US_onset = 9  # s



#* Load the data before motion correction.
# %%
# region Load the data before motion correction
with open(path_pkl_after_motion_correction, 'rb') as file:
	all_data = pickle.load(file)

# endregion

# all_data.__dict__.keys()

# all_data.planes[3].trials[3].shift_correction



softthresh=50


shape_ = all_data.planes[0].trials[0].images.shape[1:]

# x_black_box_beg, x_black_box_end, y_black_box_beg, y_black_box_end = all_data.black_box


if ('ca8' in str(path_home)) | ('4' in str(path_home)):
	x_black_box_beg = shape_[0] - 20
	x_black_box_end = shape_[0] - 5
	y_black_box_beg = shape_[1] - 20
	y_black_box_end = shape_[1] - 5


elif 'single' in str(path_home):
		
		
	x_black_box_beg = shape_[0] - 10
	x_black_box_end = shape_[0] - 5
	y_black_box_beg = shape_[1] - 10
	y_black_box_end = shape_[1] - 5

else:
	x_black_box_beg = 330
	x_black_box_end = 345
	y_black_box_beg = 594
	y_black_box_end = 609






def add_colors_to_world(anatomy, color_frame_original):
	trial_red_channel = anatomy * (1 - color_frame_original)
	trial_green_channel = trial_red_channel + color_frame_original*color_frame_original
	trial_blue_channel = trial_red_channel
	
	image = (np.stack([trial_red_channel,trial_green_channel,trial_blue_channel], axis=-1)*255).astype(np.uint8)
	
	plt.imshow(image)
	plt.show()

	return image












plt.figure()
plt.imshow(all_data.planes[0].trials[0].images[100][x_black_box_beg:x_black_box_end, y_black_box_beg:y_black_box_end], vmin=0, vmax=None)
plt.colorbar(shrink=0.5)

plt.imshow(all_data.planes[0].trials[0].images.mean('Time (ms)'), vmin=10, vmax=100)
plt.colorbar(shrink=0.5)





H







#* Subtract the background (calculated from the black box) and clip the values to 0.
for plane_i, plane in enumerate(all_data.planes):
	for trial_i, trial in enumerate(plane.trials):
		
		all_data.planes[plane_i].trials[trial_i].images = (trial.images - trial.images[:, x_black_box_beg:x_black_box_end, y_black_box_beg:y_black_box_end].mean(dim=('x','y'))).clip(0,None)


#!!!!!!!!!!!!!!!!!!! IF IN CERTAIN PERIODS OF THE EXPERIMENT
# TRAINING



# ALL_DATA = deepcopy(all_data)
# all_data = deepcopy(ALL_DATA)


#* Split data into 3 periods: before CS, from CS onset to US onset and after US onset.
for plane_i, plane in enumerate(all_data.planes):

	# if plane_i < 3:
	# 	continue

	plane_template = np.zeros(plane.trials[0].images.shape[1:])


	# fig, axs = plt.subplots()


	for trial_i, trial in enumerate(plane.trials):

	# 	break
	# break

		cs_beg = trial.protocol.loc[trial.protocol['CS beg'] != 0, 'Time (ms)'].to_numpy()[0]

		if (us_beg := trial.protocol.loc[trial.protocol['US beg'] != 0, 'Time (ms)']).empty:
			us_beg = cs_beg + interval_between_CS_onset_US_onset*1000

		else:
	#! issue when multiple US in trial
			us_beg = us_beg.to_numpy()[0]



		trial_images = trial.images.copy()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		trial_images = trial_images.fillna(0)
# trial.__dict__.keys()
# trial.mask_good_frames
	#!remove
		# trial_images = trial_images.assign_coords({'mask good frames' : ('Time (ms)', trial.mask_good_frames), 'shift correction in X' : ('Time (ms)', trial.shift_correction[:,0].astype('float32')), 'shift correction in Y' : ('Time (ms)', trial.shift_correction[:,1].astype('float32'))})
	#!

		trial_images['mask before CS'] = trial_images['Time (ms)'] < cs_beg
		trial_images['mask CS-US'] = (trial_images['Time (ms)'] > cs_beg) & (trial_images['Time (ms)'] < us_beg)
		trial_images['mask after US'] = trial_images['Time (ms)'] > us_beg



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
		trial.CS_positive_response = add_colors_to_world(anatomy, color_frame_original)


		###* CS negative response
		# color_frame_original = np.clip(-trial.CS_US_vs_pre*5, 0, 1)
		# trial.CS_negative_response = add_colors_to_world(anatomy, color_frame_original)


		# ###* US positive response
		# color_frame_original = np.clip(trial.post_US_vs_pre, 0, 1)
		# trial.CS_negative_response = add_colors_to_world(anatomy, color_frame_original)


		# ###* US negative response
		# color_frame_original = np.clip(-trial.post_US_vs_pre*5, 0, 1)
		# trial.CS_negative_response = add_colors_to_world(anatomy, color_frame_original)



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







# #!!!!!!!!!!!!!!!!!!!!!!



# #* Plot the position in the anatomical stack.
# # region Position in the anatomical stack
# try:
# 	A = []
# 	B = []

# 	C = []
# 	D = []

# 	for i in range(len(all_data.planes)):

# 		for j in range(2):

# 			A.append(all_data.planes[i].trials[j].position_anatomical_stack)

# 			C.append(all_data.planes[i].trials[j].template_image)

# 		for l in range(2,4):

# 			B.append(all_data.planes[i].trials[l].position_anatomical_stack)

# 			D.append(all_data.planes[i].trials[l].template_image)


# 	A = np.array(A)
# 	B = np.array(B)

# 	C = np.array(C)
# 	D = np.array(D)


# 	sns.set_style('whitegrid')


# 	path_ = path_home / fish_name


# 	plt.xlabel('Trial before or after initial train')
# 	plt.ylabel('Plane number in anatomical stack')
# 	plt.plot(A, 'blue')
# 	plt.plot(B, 'red')
# 	plt.legend(['Before initial train', 'After initial train'])
# 	plt.savefig(path_ / ('Where in the anatomical stack' + '.png'), dpi=300, bbox_inches='tight')


# 	plt.xlabel('Trial before or after initial train')
# 	plt.ylabel('Difference between planes imaged\n before and after initial train (μm)')
# 	plt.plot(A-B, 'k')
# 	plt.ylim(-10, 10)
# 	plt.savefig(path_ / ('Difference when revisiting planes' + '.png'), dpi=300, bbox_inches='tight')


# 	sns.set_style('white')
	
# except:
# 	pass
















# 	# except:
# 	# 	continue

# # compression_level = 4
# # compression_library = 'zlib'

# # with pd.HDFStore(path_pkl, complevel=compression_level, complib=compression_library) as store:
	
# # 	store.append(fish_name, planes_list, data_columns=[cs, us], expectedrows=len(fish.raw_data), append=False)

# # 	store.get_storer(fish.dataset_key()).attrs['metadata'] = fish.metadata._asdict()





# #* For correlation map.

# ##* Preparing the data for the correlation map.

# # for plane_i, plane in enumerate(Data.planes):
# # 	for trial_i, trial in enumerate(plane):

# # A = [Data.planes[plane_i].trials[trial_i].images.values for trial_i, trial in enumerate(plane.trials) for plane_i, plane in enumerate(Data.planes)]

# # B = np.sum([np.sum(x, axis=0) for x in A], axis=0)

# # plt.imshow(B)


# eye_mask = np.ones(all_data.planes[0].trials[0].images.shape[1:], dtype='bool')

# #!
# eye_mask[350:, 350:450] = False
# eye_mask[:50, 350:450] = False
# plt.imshow(eye_mask)

# # # A = ndimage.uniform_filter(plane_trials_good_images, size=(30, 30), axes=(1,2))
# # ndimage.gaussian_filter(plane_trials_good_images, sigma=gaussian_filter_sigma, axes=(1,2))

# # plt.imshow(np.mean(A, axis=0))
# # plt.colorbar()











# for plane_i, plane in enumerate(all_data.planes):

# 	# if plane_i not in [0,1,3,6,8,9,10,13]:
# 	# 	continue
# 	# break
# #!
# 	# plane.trials = plane.trials



# 	#!!!!!!!!!!!!!!!!!!!!!!!!! DO ALL OF THIS FOR SINGLE TRIAL AND THEN CONCATENATE TO GET PLANE DATA




# 	#* To get a correlation map for the whole plane data, we need to concatenate all the images of the trials.
# 	# plane_trials_all_images = np.concatenate([t.images.values for t in plane.trials])
# 	plane_trials_all_images = plane.get_all_images()

# 	plt.title('All images from plane')
# 	plt.imshow(np.mean(plane_trials_all_images, axis=0))
# 	plt.colorbar
# 	plt.show()


# 	#* Get the number of images per trial.
# 	plane_trials_number_images = np.array([t.images.shape[0] for t in plane.trials])


# 	#* Get the indices of the CS in the images of the trials.
# 	cs_indices = np.array([trial.get_stim_index(cs) for trial in plane.trials])

# 	cs_indices[1:,0] += np.cumsum(plane_trials_number_images[:-1])
# 	cs_indices[1:,1] += np.cumsum(plane_trials_number_images[:-1])









# 	#* Discard good frames due to motion, gating of the PMT or plane change.
# 	plane_trials_mask_good_frames = np.concatenate([t.mask_good_frames for t in plane.trials])
# 	plane_bad_frames_index = np.where(plane_trials_mask_good_frames)[0]
# 	plane_trials_good_images = plane_trials_all_images[~plane_trials_mask_good_frames].copy()

# 	plt.title('All good images from plane')
# 	plt.imshow(np.mean(plane_trials_good_images, axis=0))
# 	plt.colorbar
# 	plt.show()









# #! not doing anything here
# 	#* Filter in space.
# 	plane_trials_good_images_filtered = ndimage.gaussian_filter(plane_trials_good_images, sigma=gaussian_filter_sigma, axes=(1,2))

# 	plt.title('All good images from plane filtered')
# 	plt.imshow(np.mean(plane_trials_good_images_filtered, axis=0))
# 	plt.colorbar





# 	# break




# #!!!!!!!!!!!!!!!!! move it further down
# 	#* Calcultate the correlation map.
# 	# Inspired in Suit2p. There, the function that computes the correlation map is celldetect2.getVmap.
# 	correlation_map = np.linalg.norm(ndimage.gaussian_filter(plane_trials_good_images, sigma=correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(plane_trials_good_images, axis=0), sigma=correlation_map_sigma)**2

# 	plt.figure('Correlation map')
# 	plt.imshow(correlation_map, interpolation='none')
# 	plt.colorbar(shrink=0.5)
# 	plt.show()






# 	#* Subtract the background.
# 	# Pixel values equal to 0 are ignored to discard the artificial edges of the images that were introduced during the motion correction.
# 	images_mean = np.nanmean(np.where(plane_trials_good_images == 0, np.nan, plane_trials_good_images), axis=(1,2))

# 	images_mean = np.nanmean(plane_trials_good_images, axis=(1,2))
# 	for image_i in range(plane_trials_good_images.shape[0]):
# 		plane_trials_good_images[image_i] -= images_mean[image_i]

# 	del images_mean

# 	#* Mask the background.
# 	plane_images_mask_fish = np.where(np.median(plane_trials_good_images, axis=0) <= 0, 0, 1).astype(dtype='bool')

# 	plane_images_mask_fish_without_eyes = plane_images_mask_fish & eye_mask

# 	#* Set to 0 the pixels that are not part of the fish in the images. Also, mask the eyes.
# 	plane_trials_good_images = np.where(plane_images_mask_fish_without_eyes, plane_trials_good_images, 0)

# 	plt.title('All good images from plane masked background')
# 	plt.imshow(np.mean(plane_trials_good_images, axis=0))
# 	plt.colorbar(shrink=0.5)
# 	plt.show()




# 	# region Voxel analysis
# 	#* Voxel analysis

# 	#* Bin the 2D images.
	

# 	plane_trials_good_images_binned = block_reduce(plane_trials_good_images, block_size=(1, voxel_bin_size, voxel_bin_size), func=np.mean, cval=0)

# 	plt.imshow(np.mean(plane_trials_good_images_binned, axis=0), interpolation='none')


# 	plane_trials_good_images_binned_ = np.empty(tuple([plane_trials_all_images.shape[0]] + list(plane_trials_good_images_binned.shape[1:]))) * np.nan
# 	plane_trials_good_images_binned_[~plane_trials_mask_good_frames, :, :] = plane_trials_good_images_binned

# 	plane_trials_good_images_binned = plane_trials_good_images_binned_.copy()

# 	del plane_trials_good_images_binned_

# 	plt.title('All good images from plane binned')
# 	plt.imshow(np.mean(plane_trials_good_images_binned, axis=0))
# 	plt.colorbar(shrink=0.5)
# 	plt.show()



# 	deltaF = []
# 	deltaF_ratio = []

# 	for i in range(len(cs_indices)):

# 		baseline = np.nanmean(plane_trials_good_images_binned[[cs_indices[i, 0] - 20, cs_indices[i, 0]]], axis=0)
		
# 		during_cs = np.nanmean(plane_trials_good_images_binned[[cs_indices[i, 0], cs_indices[i, 1]]], axis=0)

# 		deltaF_ratio.append((during_cs - baseline) / baseline)

# 		if i == 0:
			
# 			deltaF.append((plane_trials_good_images_binned[ : plane_trials_number_images[0]] - baseline) / baseline)

# 		elif i < len(cs_indices)-1:

# 			deltaF.append((plane_trials_good_images_binned[np.cumsum(plane_trials_number_images)[i-1] : np.cumsum(plane_trials_number_images)[i]] - baseline) / baseline)

# 		else:
# 			deltaF.append((plane_trials_good_images_binned[np.cumsum(plane_trials_number_images)[i-1] : ] - baseline) / baseline)

# 	deltaF = np.concatenate(deltaF)

# 	deltaF = np.where(np.isnan(deltaF), 0, deltaF)

# 	deltaF_ratio = np.array(deltaF_ratio)


# 	for i in range(len(cs_indices)):
# 		plt.imshow(deltaF_ratio[i], interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
# 		plt.colorbar(shrink=0.5)
# 		plt.title('DeltaF_SR')
# 		plt.show()


# 	A = np.mean(np.array([deltaF_ratio[0], deltaF_ratio[1]]), axis=0)
# 	B = np.mean(np.array([deltaF_ratio[2], deltaF_ratio[3]]), axis=0)



# 	plt.imshow(A, interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
# 	plt.colorbar(shrink=0.5)
# 	plt.title('DeltaF_SR A')
# 	plt.show()

# 	plt.imshow(B, interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
# 	plt.colorbar(shrink=0.5)
# 	plt.title('DeltaF_SR B')
# 	plt.show()

# 	plt.imshow(B/A, interpolation='none', vmin=-100, vmax=100, cmap='RdBu_r')
# 	plt.colorbar(shrink=0.5)
# 	plt.title('DeltaF_SR B / DeltaF_SR A')
# 	plt.savefig(imaging_path_ /  fish_name / (fish_name + '_deltaF_SR_voxels_plane ' + str(plane_i) + '.tif'))
# 	plt.show()

# 	deltaF_ratio = np.concatenate(deltaF_ratio)

# #!!!
# 	# deltaF_ = np.empty(tuple([plane_trials_all_images.shape[0]] + list(deltaF.shape[1:]))) * np.nan
# 	# deltaF_[~plane_trials_mask_good_frames, :, :] = deltaF

# 	# deltaF = deltaF_.copy()

# 	# del deltaF_

# 	for i in range(len(cs_indices)):
# 		deltaF[cs_indices[i,0]:cs_indices[i,1],:20,-20:] = -100

# 	plt.imshow(np.nanmean(deltaF, axis=0))
# 	plt.colorbar(shrink=0.5)

# 	#* Save rois_zscore_over_time as a TIFF file.
# 	tifffile.imwrite(imaging_path_ /  fish_name / (fish_name + '_deltaF_voxels_plane ' + str(plane_i) + '.tif'), deltaF.astype('float32'))

# 	# endregion







# 	# region ROI analysis for the whole plane

# 	#* Set to 0 the pixels that are not part of the fish in the correlation map.
# 	correlation_map = np.where(plane_images_mask_fish_without_eyes, correlation_map, 0)

# 	plt.title('Correlation map masked background')
# 	plt.imshow(np.where(plane_images_mask_fish_without_eyes, correlation_map, 0))
# 	plt.colorbar(shrink=0.5)
# 	plt.show()





# 	#* ROIs for the all the trials of the same plane.
# 	#TODO need to rewrite all this part, using Mike's and Ruben's code
# 	all_traces, all_rois, used_pixels, correlation_map_ = f.get_ROIs(Nrois=100, correlation_map=correlation_map, images=plane_trials_good_images_filtered, threshold=0.3, max_pixels=60)

# 	plt.imshow(zscore(all_traces, 1), aspect="auto", cmap="RdBu_r")
# 	plt.savefig(imaging_path_ / fish_name / (fish_name + 'zscore ' + str(plane_i) + '.tif'))
# 	plt.show()
# 	plt.imshow(all_rois)
# 	plt.colorbar()
# 	plt.show()
# 	plt.imshow(correlation_map_)
# 	plt.show()
# 	plt.imshow(np.sum(plane_trials_all_images, axis=0))
# 	plt.show()
# 	plt.imshow(correlation_map)
# 	plt.show()



# 	#* Create array to then make movie.
# 	all_rois = all_rois.astype('int')

# 	rois_zscore_over_time = np.zeros_like(plane_trials_all_images)


# 	#* Consider the periods of good frames in the array with the Z score of the ROI traces.
# 	all_traces_z_score = zscore(all_traces, 1)

# 	all_traces_z_score_ = np.empty((all_traces.shape[0], len(plane_trials_all_images))) * np.nan
# 	all_traces_z_score_[:, ~plane_trials_mask_good_frames] = all_traces_z_score

# 	plt.imshow(all_traces_z_score_, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)

# 	all_traces_z_score = all_traces_z_score_
# 	del all_traces_z_score_

# 	rois_mask = np.zeros(rois_zscore_over_time.shape, dtype='bool')

# 	#* Get mask of the ROIs.
# 	for roi_i in range(1, all_rois.max()):
# 		# break
# 		rois_mask[roi_i] = all_rois == roi_i

# 		# roi_mask = rois_mask[roi_i]
# 		# [np.newaxis, :, :]

# 		for t in range(rois_zscore_over_time.shape[0]):
# 			# break
# 			rois_zscore_over_time[t,:,:] += np.where(rois_mask[roi_i], all_traces_z_score[roi_i, t], 0)

# 			rois_zscore_over_time[t,:,:]

# 	for i in range(len(cs_indices)):
# 		rois_zscore_over_time[cs_indices[i,0]:cs_indices[i,1],:50,-50:] = -100

# 	# plt.imshow(np.mean(plane_trials_all_images, axis=0))
# 	plt.imshow(np.nansum(rois_zscore_over_time, axis=0), aspect="auto", cmap="RdBu_r", interpolation='none')
# 	plt.colorbar()

# 	#* Save rois_zscore_over_time as a TIFF file.
# 	tifffile.imwrite(imaging_path_ / fish_name / (fish_name + 'rois_zscore_over_time ' + str(plane_i) + '.tif'), rois_zscore_over_time.astype('float32'))

# 	# endregion




# for plane_i, plane in enumerate(all_data.planes):

# 	deltaF = [_ for _ in range(len(plane.trials))]
# 	deltaF_ratio = deltaF.copy()


# 	# region ROI analysis for each trial
# 	for trial_i, trial in enumerate(plane.trials):
		
# 		# trial_i = trial.trial_number
		
# 		# trial = plane.trials[1]

# 		# break
# 	# break


# 		#* Get the indices of the CS in the images of the trials.
# 		cs_indices = trial.get_stim_index(cs)



# 		#* Discard good frames due to motion, gating of the PMT or trial change.
# 		trial_good_images = trial.images.values[~trial.mask_good_frames]
# 		trial_bad_frames_index = np.where(trial.mask_good_frames)[0]

# 		# plt.title('All images from trial')
# 		# plt.imshow(np.mean(trial_good_images, axis=0))
# 		# plt.colorbar(shrink=0.5)
# 		# plt.show()





# 		#* Filter in space.
# 		trial_good_images_filtered = ndimage.gaussian_filter(trial_good_images, sigma=gaussian_filter_sigma, axes=(1,2))

# 		# plt.title('All good images from trial filtered')
# 		# plt.imshow(np.mean(trial_good_images_filtered, axis=0))
# 		# plt.colorbar(shrink=0.5)
# 		# plt.show()




# #! calculate the correlation map for each trial
# 		#* Correlation map
# 		# In Suit2p, the function that computes the correlation map is celldetect2.getVmap.
# 		correlation_map = np.linalg.norm(ndimage.gaussian_filter(trial_good_images, sigma=correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(trial_good_images, axis=0), sigma=correlation_map_sigma)**2

# 		# plt.figure('Correlation map')
# 		# plt.imshow(correlation_map)
# 		# plt.colorbar(shrink=0.5)
# 		# plt.show()




# 		#* Subtract the background.
# 		# Pixel values equal to 0 are ignored to discard the artificial edges of the images that were introduced during the motion correction.
# 		images_mean = np.nanmean(np.where(trial_good_images == 0, np.nan, trial_good_images), axis=(1,2))

# 		images_mean = np.nanmean(trial_good_images, axis=(1,2))
# 		for image_i in range(trial_good_images.shape[0]):
# 			trial_good_images[image_i] -= images_mean[image_i]

# 		del images_mean

# 		#* Mask the background.
# 		trial.images_mask_fish = np.where(np.median(trial_good_images, axis=0) <= 0, 0, 1).astype(dtype='bool')

# 		trial.images_mask_fish_without_eyes = trial.images_mask_fish & eye_mask

# 		#* Set to 0 the pixels that are not part of the fish in the images. Also, mask the eyes.
# 		trial_good_images = np.where(trial.images_mask_fish_without_eyes, trial_good_images, 0)

# 		# plt.title('All good images from trial masked background')
# 		# plt.imshow(np.mean(trial_good_images, axis=0))
# 		# plt.colorbar(shrink=0.5)
# 		# plt.show()

# plane = all_data.planes[7]

# trial = plane.trials[3]

# trial.images.to_numpy()
# tifffile.imwrite(imaging_path_ / fish_name / (fish_name + '_trial_images.tif'), trial.images.to_numpy().astype('float32'))


# 		# region Voxel analysis
# 		#* Voxel analysis

# 		#* Bin the 2D images.
		
# 		trial_good_images_binned = block_reduce(trial_good_images, block_size=(1, voxel_bin_size, voxel_bin_size), func=np.mean, cval=0)

# 		# plt.imshow(np.mean(trial_good_images_binned, axis=0), interpolation='none')


# 		trial_images_binned = np.empty(tuple([trial.images.shape[0]] + list(trial_good_images_binned.shape[1:]))) * np.nan
# 		trial_images_binned[~trial.mask_good_frames, :, :] = trial_good_images_binned
		
# 		# del trial_good_images_binned

# 		# plt.title('All good images from trial binned')
# 		# plt.imshow(np.mean(trial_images_binned, axis=0))
# 		# plt.colorbar(shrink=0.5)
# 		# plt.show()







# 		baseline = np.nanmean(trial_images_binned[:cs_indices[0], :, :], axis=0)
		
# 		during_cs = np.nanmean(trial_images_binned[cs_indices[0] : cs_indices[1], :, :], axis=0)

# 		deltaF[trial_i] = (trial_images_binned - baseline) / baseline



# 		deltaF_ratio[trial_i] = (during_cs - baseline) / baseline
# 		deltaF_ratio[trial_i] = np.where(np.isnan(deltaF_ratio[trial_i]), 0, deltaF_ratio[trial_i])

# 		# plt.imshow(deltaF_ratio[trial_i], interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
# 		# plt.colorbar(shrink=0.5)
# 		# plt.title('DeltaF / F')
# 		# plt.show()



# 		#!
# 		# deltaF[trial_i][cs_indices[0]:cs_indices[1],:20,-20:] = -100
# 		# deltaF[trial_i] = np.where(np.isnan(deltaF[trial_i]), 0, deltaF[trial_i])

# 		# plt.imshow(np.nanmean(deltaF[trial_i], axis=0))
# 		# plt.colorbar(shrink=0.5)
# 		# plt.show()

# 		#* Save rois_zscore_over_time as a TIFF file.
# 		# tifffile.imwrite(imaging_path_ /  fish_name / (fish_name + '_deltaF_voxels_trial ' + str(trial_i) + '.tif'), deltaF[trial_i].astype('float32'))

# 		# endregion



# 	fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

# 	for i in range(len(deltaF)):

# 		a = 0 if (i==0 or i==1) else 1
# 		b = 0 if (i==0 or i==2) else 1

# 		C = np.reshape(deltaF[i], (deltaF[i].shape[0], deltaF[i].shape[1]*deltaF[i].shape[2]))

# 		for j in range(C.shape[1]):
			
# 			axs[a,b].plot(C[:,j], color='gray', alpha=0.5)

# 		axs[a,b].set_title('Trial ' + str(i+1))

# 		axs[a,b].axvline(x=cs_indices[0], color='red', linestyle='--')
# 		axs[a,b].axvline(x=cs_indices[1], color='blue', linestyle='--')

# 		axs[a,b].set_ylim((-5, 30))

# 	fig.suptitle('DeltaF/F plane ' + str(plane_i), fontsize=20)

# 	fig.savefig(imaging_path_ / fish_name / (fish_name + '_deltaF_voxels_plane ' + str(plane_i) + '.png'), dpi=100, bbox_inches='tight')



# A = np.mean(np.array([deltaF[0][:-1,:,:], deltaF[1][:,:,:]]), axis=0)
# # for trial_i in range(len(deltaF)):
# # 	deltaF[trial_i] = np.where(np.isnan(deltaF[trial_i]), 0, deltaF[trial_i])
# A = np.where(np.isnan(A), 0, A)

# B = np.mean(np.array([deltaF[2], deltaF[3]]), axis=0)


# np.nanmedian(A, axis=0).shape

# cs_indices

# plt.imshow(np.nanmedian(A, axis=0), interpolation='none')






# A = np.mean(np.array([deltaF_ratio[trial_i][0], deltaF_ratio[trial_i][1]]), axis=0)
# B = np.mean(np.array([deltaF_ratio[trial_i][2], deltaF_ratio[trial_i][3]]), axis=0)



# plt.imshow(A, interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
# plt.colorbar(shrink=0.5)
# plt.title('DeltaF_SR A')
# plt.show()

# plt.imshow(B, interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
# plt.colorbar(shrink=0.5)
# plt.title('DeltaF_SR B')
# plt.show()

# plt.imshow(B/A, interpolation='none', vmin=-100, vmax=100, cmap='RdBu_r')
# plt.colorbar(shrink=0.5)
# plt.title('DeltaF_SR B / DeltaF_SR A')
# plt.savefig(imaging_path_ /  fish_name / (fish_name + '_deltaF_SR_voxels_trial ' + str(trial_i) + '.tif'))
# plt.show()

# deltaF_ratio[trial_i] = np.concatenate(deltaF_ratio)

# !!!
# deltaF_ = np.empty(tuple([trial.images.shape[0]] + list(deltaF.shape[1:]))) * np.nan
# deltaF_[~trial.mask_good_frames, :, :] = deltaF

# deltaF = deltaF_.copy()

# del deltaF_

		







# 		# region ROI analysis for each trial

# 		#* Set to 0 the pixels that are not part of the fish in the correlation map.
# 		correlation_map = np.where(trial.images_mask_fish_without_eyes , correlation_map, 0)

# 		plt.title('Correlation map masked background')
# 		plt.imshow(np.where(trial.images_mask_fish_without_eyes , correlation_map, 0))
# 		plt.colorbar(shrink=0.5)
# 		plt.show()



# 		#* ROIs for the all the trials of the same plane.
# 		#TODO need to rewrite all this part, using Mike's and Ruben's code
# 		all_traces, all_rois, used_pixels, correlation_map_ = get_ROIs(Nrois=100, correlation_map=correlation_map, images=trial_good_images_filtered, threshold=0.3, max_pixels=60)

# 		plt.imshow(zscore(all_traces, 1), aspect="auto", cmap="RdBu_r")
# 		plt.savefig(imaging_path_ / fish_name / (fish_name + 'zscore ' + str(plane_i) + '.tif'))
# 		plt.show()
# 		plt.imshow(all_rois)
# 		plt.colorbar()
# 		plt.show()
# 		plt.imshow(correlation_map_)
# 		plt.show()
# 		plt.imshow(np.sum(plane_trials_all_images, axis=0))
# 		plt.show()
# 		plt.imshow(correlation_map)
# 		plt.show()


# 	# if plane_i not in [0,1,3,6,8,9,10,13]:
# 	# 	continue
# 	# break
# #!
# 	# plane.trials = plane.trials



# 	#!!!!!!!!!!!!!!!!!!!!!!!!! DO ALL OF THIS FOR SINGLE TRIAL AND THEN CONCATENATE TO GET PLANE DATA




# 	#* To get a correlation map for the whole plane data, we need to concatenate all the images of the trials.
# 	# plane_trials_all_images = np.concatenate([t.images.values for t in plane.trials])
# 	plane_trials_all_images = plane.get_all_images()

# 	plt.title('All images from plane')
# 	plt.imshow(np.mean(plane_trials_all_images, axis=0))
# 	plt.colorbar
# 	plt.show()


# 	#* Get the number of images per trial.
# 	plane_trials_number_images = np.array([t.images.shape[0] for t in plane.trials])


# 	#* Get the indices of the CS in the images of the trials.
# 	cs_indices = np.array([trial.get_stim_index(cs) for trial in plane.trials])

# 	cs_indices[1:,0] += np.cumsum(plane_trials_number_images[:-1])
# 	cs_indices[1:,1] += np.cumsum(plane_trials_number_images[:-1])









# 	#* Discard good frames due to motion, gating of the PMT or plane change.
# 	plane_trials_mask_good_frames = np.concatenate([t.mask_good_frames for t in plane.trials])
# 	plane_bad_frames_index = np.where(plane_trials_mask_good_frames)[0]
# 	plane_trials_good_images = plane_trials_all_images[~plane_trials_mask_good_frames].copy()

# 	plt.title('All good images from plane')
# 	plt.imshow(np.mean(plane_trials_good_images, axis=0))
# 	plt.colorbar
# 	plt.show()










# 	#* Filter in space.
# 	plane_trials_good_images_filtered = ndimage.gaussian_filter(plane_trials_good_images, sigma=gaussian_filter_sigma, axes=(1,2))

# 	plt.title('All good images from plane filtered')
# 	plt.imshow(np.mean(plane_trials_good_images_filtered, axis=0))
# 	plt.colorbar

# #!!!!!!!!!!!!!!!!! move it further down
# 	#* Calcultate the correlation map.
# 	# Inspired in Suit2p. There, the function that computes the correlation map is celldetect2.getVmap.
# 	correlation_map = np.linalg.norm(ndimage.gaussian_filter(plane_trials_good_images, sigma=correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(plane_trials_good_images, axis=0), sigma=correlation_map_sigma)**2

# 	plt.figure('Correlation map')
# 	plt.imshow(correlation_map)
# 	plt.colorbar(shrink=0.5)
# 	plt.show()






# 	#* Subtract the background.
# 	# Pixel values equal to 0 are ignored to discard the artificial edges of the images that were introduced during the motion correction.
# 	images_mean = np.nanmean(np.where(plane_trials_good_images == 0, np.nan, plane_trials_good_images), axis=(1,2))

# 	images_mean = np.nanmean(plane_trials_good_images, axis=(1,2))
# 	for image_i in range(plane_trials_good_images.shape[0]):
# 		plane_trials_good_images[image_i] -= images_mean[image_i]

# 	del images_mean

# 	#* Mask the background.
# 	plane_images_mask_fish = np.where(np.median(plane_trials_good_images, axis=0) <= 0, 0, 1).astype(dtype='bool')

# 	plane_images_mask_fish_without_eyes = plane_images_mask_fish & eye_mask

# 	#* Set to 0 the pixels that are not part of the fish in the images. Also, mask the eyes.
# 	plane_trials_good_images = np.where(plane_images_mask_fish_without_eyes, plane_trials_good_images, 0)

# 	plt.title('All good images from plane masked background')
# 	plt.imshow(np.mean(plane_trials_good_images, axis=0))
# 	plt.colorbar(shrink=0.5)
# 	plt.show()









# 	# region ROI analysis for the whole plane

# 	#* Set to 0 the pixels that are not part of the fish in the correlation map.
# 	correlation_map = np.where(plane_images_mask_fish_without_eyes, correlation_map, 0)

# 	plt.title('Correlation map masked background')
# 	plt.imshow(np.where(plane_images_mask_fish_without_eyes, correlation_map, 0))
# 	plt.colorbar(shrink=0.5)
# 	plt.show()





# 	#* ROIs for the all the trials of the same plane.
# 	#TODO need to rewrite all this part, using Mike's and Ruben's code
# 	all_traces, all_rois, used_pixels, correlation_map_ = f.get_ROIs(Nrois=100, correlation_map=correlation_map, images=plane_trials_good_images_filtered, threshold=0.3, max_pixels=60)

# 	plt.imshow(zscore(all_traces, 1), aspect="auto", cmap="RdBu_r")
# 	plt.savefig(imaging_path_ / fish_name / (fish_name + 'zscore ' + str(plane_i) + '.tif'))
# 	plt.show()
# 	plt.imshow(all_rois)
# 	plt.colorbar()
# 	plt.show()
# 	plt.imshow(correlation_map_)
# 	plt.show()
# 	plt.imshow(np.sum(plane_trials_all_images, axis=0))
# 	plt.show()
# 	plt.imshow(correlation_map)
# 	plt.show()



# 	#* Create array to then make movie.
# 	all_rois = all_rois.astype('int')

# 	rois_zscore_over_time = np.zeros_like(plane_trials_all_images)


# 	#* Consider the periods of good frames in the array with the Z score of the ROI traces.
# 	all_traces_z_score = zscore(all_traces, 1)

# 	all_traces_z_score_ = np.empty((all_traces.shape[0], len(plane_trials_all_images))) * np.nan
# 	all_traces_z_score_[:, ~plane_trials_mask_good_frames] = all_traces_z_score

# 	plt.imshow(all_traces_z_score_, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)

# 	all_traces_z_score = all_traces_z_score_
# 	del all_traces_z_score_

# 	rois_mask = np.zeros(rois_zscore_over_time.shape, dtype='bool')

# 	#* Get mask of the ROIs.
# 	for roi_i in range(1, all_rois.max()):
# 		# break
# 		rois_mask[roi_i] = all_rois == roi_i

# 		# roi_mask = rois_mask[roi_i]
# 		# [np.newaxis, :, :]

# 		for t in range(rois_zscore_over_time.shape[0]):
# 			# break
# 			rois_zscore_over_time[t,:,:] += np.where(rois_mask[roi_i], all_traces_z_score[roi_i, t], 0)

# 			rois_zscore_over_time[t,:,:]

# 	for i in range(len(cs_indices)):
# 		rois_zscore_over_time[cs_indices[i,0]:cs_indices[i,1],:50,-50:] = -100

# 	# plt.imshow(np.mean(plane_trials_all_images, axis=0))
# 	plt.imshow(np.nansum(rois_zscore_over_time, axis=0), aspect="auto", cmap="RdBu_r", interpolation='none')
# 	plt.colorbar()

# 	#* Save rois_zscore_over_time as a TIFF file.
# 	tifffile.imwrite(imaging_path_ / fish_name / (fish_name + 'rois_zscore_over_time ' + str(plane_i) + '.tif'), rois_zscore_over_time.astype('float32'))

# 	# endregion






# 	# region ROI analysis for each trial

# 	for trial in plane.trials:

# 		#!
# 		# break
# 		trial = plane.trials[3]
# 	# break
# 		# trial.images = trial.images


# 		#* Discard good frames due to motion, gating of the PMT or trial change.
# 		trial_good_images = trial.images.values[~trial.mask_good_frames]
# 		trial_bad_frames_index = np.where(trial.mask_good_frames)[0]

# 		plt.title('All images from trial')
# 		plt.imshow(np.mean(trial_good_images, axis=0))
# 		plt.colorbar(shrink=0.5)
# 		plt.show()

# 		#* Subtract the background.
# 		# Pixel values equal to 0 are ignored to discard the artificial edges of the images that were introduced during the motion correction.
# 		images_mean = np.nanmean(np.where(trial_good_images == 0, np.nan, trial_good_images), axis=(1,2))

# 		images_mean = np.nanmean(trial_good_images, axis=(1,2))
# 		for image_i in range(trial_good_images.shape[0]):
# 			trial_good_images[image_i] -= images_mean[image_i]

# 		del images_mean

# 		#* Mask the background.
# 		trial.images_mask_fish = np.where(np.median(trial_good_images, axis=0) <= 0, 0, 1).astype(dtype='bool')

# 		trial.images_mask_fish_without_eyes = trial.images_mask_fish & eye_mask

# 		#* Set to 0 the pixels that are not part of the fish in the images. Also, mask the eyes.
# 		trial_good_images = np.where(trial.images_mask_fish_without_eyes, trial_good_images, 0)

# 		plt.title('All good images from trial masked background')
# 		plt.imshow(np.mean(trial_good_images, axis=0))
# 		plt.colorbar(shrink=0.5)
# 		plt.show()



# #!!!!!!!!!!! Voxel analysis per trial
# 		#* Bin the 2D images.
# 		trial_good_images_binned = block_reduce(trial_good_images, block_size=(1, voxel_bin_size, voxel_bin_size), func=np.mean, cval=0)

# 		plt.imshow(np.mean(trial_good_images_binned, axis=0), interpolation='none')


# 		trial_images_binned = np.empty(tuple([trial.images.shape[0]] + list(trial_good_images_binned.shape[1:]))) * np.nan
# 		trial_images_binned[~trial.mask_good_frames, :, :] = trial_good_images_binned

# 		trial_good_images_binned = trial_images_binned.copy()

# 		del trial_images_binned

# 		plt.title('All good images from trial binned')
# 		plt.imshow(np.mean(trial_good_images_binned, axis=0))
# 		plt.colorbar(shrink=0.5)
# 		plt.show()

	

















# 		#* Filter in space.
# 		trial_images_filtered = ndimage.gaussian_filter(trial_images, sigma=gaussian_filter_sigma, axes=(1,2))
# 		trial_images_good_images_filtered = trial_images_filtered[~trial.mask_good_frames].copy()




# 		#* Correlation map
# 		# In Suit2p, the function that computes the correlation map is celldetect2.getVmap.
# 		correlation_map = np.linalg.norm(ndimage.gaussian_filter(trial_images_good_images, sigma=correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(trial_images_good_images, axis=0), sigma=correlation_map_sigma)**2

# 		plt.figure('Correlation map')
# 		plt.imshow(correlation_map)
# 		plt.colorbar(shrink=0.5)
# 		plt.show()

# 		#* Subtract the background.
# 	#! Here I should take the average ignoring the sharp edges of the images.
# 		images_mean = np.mean(trial_images_good_images, axis=(1,2))

# 		for image_i in range(trial_images_good_images.shape[0]):
# 			trial_images_good_images[image_i] -= images_mean[image_i]

# 		del images_mean

# 		#* Mask the background.
# 		trial_images_mask_fish = np.where(np.median(trial_images_good_images, axis=0) <= 0, 0, 1).astype(dtype='bool')
		
# 		#* Mask the background and the eyes.
# 		trial_images_mask_fish_without_eyes = trial_images_mask_fish & eye_mask


# 		#* Set to 0 the pixels that are not part of the fish in the images.
# 		trial_images_good_images = np.where(trial_images_mask_fish_without_eyes, trial_images_good_images, 0)

# 		plt.title('All good images from plane masked background')
# 		plt.imshow(np.mean(trial_images_good_images, axis=0))
# 		plt.colorbar(shrink=0.5)
# 		plt.show()

# 		#* Set to 0 the pixels that are not part of the fish in the correlation map.
# 		correlation_map = np.where(trial_images_mask_fish_without_eyes, correlation_map, 0)

# 		plt.title('Correlation map masked background')
# 		plt.imshow(np.where(trial_images_mask_fish_without_eyes, correlation_map, 0))
# 		plt.colorbar(shrink=0.5)
# 		plt.show()





# 		#* ROIs

# 		all_traces, all_rois, used_pixels, correlation_map_ = get_ROIs(Nrois=100, correlation_map=correlation_map, images=trial_images_good_images_filtered, threshold=0.3, max_pixels=60)

# 		images_times = trial_images.time.values


# 		trial_time_ref = images_times[0]

# 		trial_protocol = trial.protocol

# 		cs_times = trial_protocol[trial_protocol[cs]!=0]
# 		cs_times = cs_times.iloc[[0,-1]] if cs_times.shape[0] > 1 else cs_times

# 		us_times = trial_protocol[trial_protocol[us]!=0]
# 		us_times = us_times.iloc[[0,-1]] if us_times.shape[0] > 1 else us_times


# 		images_times = images_times - trial_time_ref
# 		cs_times = cs_times['Time (ms)'].values - trial_time_ref
# 		us_times = us_times['Time (ms)'].values - trial_time_ref


# 		number_traces = 50

# 		fig, axs = plt.subplots(number_traces, 1, sharex=True)
# 		# figsize=(10, 8)

# 		for i in range(number_traces):

# #!
# 			axs[i].plot(images_times[:110], all_traces[i+50][:110])

# 			if cs_times.shape[0] > 0:
# 				axs[i].axvline(x=cs_times[0], color='g', linestyle='-')
# 				axs[i].axvline(x=cs_times[1], color='g', linestyle='--')
			
# 			if us_times.shape[0] > 0:
# 				axs[i].axvline(x=us_times[0], color='r', linestyle='-')
# 				axs[i].axvline(x=us_times[1], color='r', linestyle='--')

# 		fig.show()


# 		plt.imshow(zscore(all_traces, 1), aspect="auto", vmin=-3, vmax=3, cmap="RdBu_r")

# 		plt.show()
# 		plt.imshow(all_rois)
# 		plt.show()
# 		plt.imshow(correlation_map_)
# 		plt.show()
# 		plt.imshow(np.sum(trial_images_good_images, axis=0))
# 		plt.show()
# 		plt.imshow(correlation_map)
# 		plt.show()






# 		#* Create array to then make movie.
# 		all_rois = all_rois.astype('int')

# 		rois_zscore_over_time = np.zeros_like(trial_images)


# 		#* Consider the periods of good frames in the array with the Z score of the ROI traces.
# 		all_traces_z_score = zscore(all_traces, 1)

# 		all_traces_z_score_ = np.empty((all_traces.shape[0], len(trial_images))) * np.nan
# 		all_traces_z_score_[:, ~trial.mask_good_frames] = all_traces_z_score

# 		plt.imshow(all_traces_z_score_, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)

# 		all_traces_z_score = all_traces_z_score_
# 		del all_traces_z_score_

# 		rois_mask = np.zeros(rois_zscore_over_time.shape, dtype='bool')

# 		#* Get mask of the ROIs.
# 		for roi_i in range(1, all_rois.max()):
# 			# break
# 			rois_mask[roi_i] = all_rois == roi_i

# 			# roi_mask = rois_mask[roi_i]
# 			# [np.newaxis, :, :]

# 			for t in range(rois_zscore_over_time.shape[0]):
# 				# break
# 				rois_zscore_over_time[t,:,:] += np.where(rois_mask[roi_i], all_traces_z_score[roi_i, t], 0)

# 				rois_zscore_over_time[t,:,:]

# 		for i in range(len(cs_indices)):
# 			rois_zscore_over_time[cs_indices[i,0]:cs_indices[i,1],:50,-50:] = -100

# 		# plt.imshow(np.mean(plane_trials_all_images, axis=0))
# 		plt.imshow(np.nansum(rois_zscore_over_time, axis=0), aspect="auto", cmap="RdBu_r")
# 		plt.colorbar()

# 		#* Save rois_zscore_over_time as a TIFF file.
# 		tifffile.imwrite(imaging_path_ / fish_name / (fish_name + 'trial_rois_zscore_over_time' + '.tif'), rois_zscore_over_time.astype('float32'))

# 		# endregion



















# #* Discard good frames due to motion, gating of the PMT or plane change.
# trial_images_good_images = trial_images[~trial.mask_good_frames].copy()

# #* Filter in space.
# trial_images_good_images_filtered = ndimage.gaussian_filter(trial_images_good_images, sigma=gaussian_filter_sigma, axes=(1,2))


# # #* Subtract the background.
# # images_mean = np.mean(trial_images_good_images, axis=(1,2))

# # for image_i in range(trial_images_good_images.shape[0]):
# # 	trial_images_good_images[image_i] -= images_mean[image_i]
	
# # # # signal.detrend(trial_images_good_images, axis=0, type='constant', bp=0, overwrite_data=True)

# # # # for image_i in range(trial_images_good_images.shape[0]):
# # # # 	trial_images_good_images[image_i] -= np.mean(trial_images_good_images[image_i,-50:-5,-50:-5])

# # # images_mean = np.mean(trial_images_good_images, axis=(1,2))
# # # images_filtered_mean = np.mean(trial_images_good_images_filtered, axis=(1,2))

# # # for image_i in range(trial_images_good_images.shape[0]):
# # # 	trial_images_good_images[image_i] -= images_mean[image_i]
# # # 	trial_images_good_images_filtered[image_i] -= images_filtered_mean[image_i]

# # # #* Clip values below the mean background.
# # # np.clip(trial_images_good_images, 0, None, out=trial_images_good_images)

# # #* Mask the background.
# # trial_images_mask_fish = np.where(np.median(trial_images_good_images, axis=0) < 0, 0, 1).astype(dtype='bool')
# # # plt.imshow(np.sum(np.where(trial_images_good_images<0,0,trial_images_good_images), axis=0))
# # # plt.colorbar(shrink=0.5)
# # # plt.show()

# # plt.imshow(np.where(trial_images_mask_fish, np.sum(trial_images_good_images, axis=0), 0))
# # plt.colorbar(shrink=0.5)
# # plt.show()
# # plt.imshow(np.sum(trial_images_good_images_filtered, axis=0))
# # plt.colorbar(shrink=0.5)
# # plt.show()




# #* Actual correlation map.

# # imag = trial_images_good_images[:,100:300, 150:250].copy()
# imag = trial_images_good_images[:,:, :].copy()

# correlation_map=np.zeros(imag.shape[1:])

# for i in tqdm(range(imag.shape[1])):
# 	if i>0 and i<(imag.shape[1]-1):
# 		for j in range(imag.shape[2]):

# 			if j>0 and j<(imag.shape[2]-1):

# 				this_pixel=np.squeeze(imag[:,i,j])
# 				surr_pixels=np.squeeze(np.sum(np.sum(np.squeeze(imag[:,i-1:i+2,j-1:j+2]),2),1))-this_pixel
# 				C, _ = pearsonr(this_pixel, surr_pixels)
# 				correlation_map[i,j]=C



# import numpy as np
# from scipy.stats import pearsonr
# from tqdm import tqdm

# # Assuming 'imag' is your 3D image data with shape (time, height, width)
# correlation_map = np.zeros(imag.shape[1:])

# # Precompute the sum of the surrounding pixels
# surr_sum = np.zeros_like(imag)
# for t in range(imag.shape[0]):
#     surr_sum[t,:,:] = np.pad(imag[t, :, :], ((1, 1), (1, 1)), 'constant', constant_values=0)[1:-1, 1:-1]


# import numpy as np
# from scipy.signal import convolve2d

# # Assuming 'image' is your 2D image data
# kernel = np.ones((3, 3))
# kernel[1, 1] = 0  # Exclude the center pixel if you don't want to include it in the sum

# # Convolve the image with the kernel
# surr_sum = np.zeros_like(imag)

# for t in range(imag.shape[0]):
# 	surr_sum[] = convolve2d(imag[t, :, :], kernel, mode='same')




# surr_sum.shape





# # Calculate the correlation using vectorized operations
# for i in tqdm(range(1, imag.shape[1] - 1)):
#     for j in range(1, imag.shape[2] - 1):
#         this_pixel = imag[:, i, j]
#         surr_pixels = np.sum(surr_sum[:, i-1:i+2, j-1:j+2], axis=(1,2)) - this_pixel
#         C, _ = pearsonr(this_pixel, surr_pixels)
#         correlation_map[i, j] = C







# original_correlation_map=np.copy(correlation_map)

# MAP = original_correlation_map.copy()

# plt.imshow(MAP, vmin=0.3)
# plt.colorbar(shrink=0.5)



# correlation_map = np.copy(original_correlation_map)
# correlation_map = ndimage.gaussian_filter(correlation_map, sigma=3)

# plt.imshow(correlation_map)
# plt.colorbar(shrink=0.5)





# #* Correlation map

# # Assuming trial_images_good_images, trial_images_good_images_filtered, and gausswidth are defined

# # In Suit2p, the function that computes the correlation map is celldetect2.getVmap.

# # trial_images_good_images_mean = np.mean(trial_images_good_images**2, axis=0)
# # trial_images_good_images_filtered_mean = np.mean(trial_images_good_images_filtered**2, axis=0)

# trial_images_good_images_mean = np.linalg.norm(trial_images_good_images, axis=0)
# trial_images_good_images_filtered_mean = np.linalg.norm(trial_images_good_images_filtered, axis=0)

# trial_images_good_images_mean = ndimage.gaussian_filter(trial_images_good_images_mean, sigma=gaussian_filter_sigma)

# trial_images_good_images_mean = trial_images_good_images_mean**2
# trial_images_good_images_filtered_mean = trial_images_good_images_filtered_mean**2

# correlation_map = trial_images_good_images_filtered_mean / trial_images_good_images_mean
# # correlation_map = trial_images_good_images_filtered_mean / trial_images_good_images_mean
# # correlation_map = 1 - trial_images_good_images_filtered_mean / trial_images_good_images_mean
# # correlation_map -= np.nanmean(correlation_map[-50:-20,-50:-20])
# # np.clip(correlation_map, 0, None, out=correlation_map)

# plt.imshow(correlation_map)
# plt.colorbar(shrink=0.5)

# plt.imshow(np.mean(trial_images_good_images, axis=0))
# plt.colorbar(shrink=0.5)

# plt.imshow(np.mean(trial_images_good_images_filtered, axis=0))
# plt.colorbar(shrink=0.5)

# plt.imshow(correlation_map, vmin=0.4, vmax=0.8)
# plt.colorbar(shrink=0.5)



# #* Subtract the background.
# images_mean = np.mean(trial_images_good_images, axis=(1,2))

# for image_i in range(trial_images_good_images.shape[0]):
# 	trial_images_good_images[image_i] -= images_mean[image_i]

# #* Mask the background.
# plane_images_mask_fish = np.where(np.median(trial_images_good_images, axis=0) < 0, 0, 1).astype(dtype='bool')
# # plt.imshow(np.sum(np.where(trial_images_good_images<0,0,trial_images_good_images), axis=0))
# # plt.colorbar(shrink=0.5)
# # plt.show()

# plt.imshow(np.where(plane_images_mask_fish, np.sum(trial_images_good_images, axis=0), 0))
# plt.colorbar(shrink=0.5)
# plt.show()
# plt.imshow(np.sum(trial_images_good_images_filtered, axis=0))
# plt.colorbar(shrink=0.5)
# plt.show()


# #* Set to 0 the pixels that are not part of the fish in the correlation map.
# correlation_map = np.where(plane_images_mask_fish, correlation_map, 0)

# plt.imshow(np.where(plane_images_mask_fish,correlation_map,0))
# # , vmin=0.4, vmax=0.8
# plt.colorbar(shrink=0.5)




# #! Careful with border. think now it is fine.





# # for i in tqdm(range(1, imag.shape[1]-1)):
# # 	for j in range(1, imag.shape[2]-1):
# # 		this_pixel = np.squeeze(imag[:, i, j])
# # 		surr_pixels = np.squeeze(np.sum(np.sum(np.squeeze(imag[:, i-1:i+2, j-1:j+2]), 2), 1)) - this_pixel
# # 		C, _ = pearsonr(this_pixel, surr_pixels)
# # 		correlation_map[i, j] = C

# # original_correlation_map = np.copy(correlation_map)



# 				# imag_background = trial_images_good_images[:,-50:-5,-50:-5].copy()

# 				# correlation_map_background = np.empty(imag_background.shape[1:])*np.nan

# 				# for i in tqdm(range(imag_background.shape[1])):
# 				# 	if i>0 and i<(imag_background.shape[1]-1):
# 				# 		for j in range(imag_background.shape[2]):

# 				# 			if j>0 and j<(imag_background.shape[2]-1):

# 				# 				this_pixel=np.squeeze(imag_background[:,i,j])
# 				# 				surr_pixels=np.squeeze(np.sum(np.sum(np.squeeze(imag_background[:,i-1:i+2,j-1:j+2]),2),1))-this_pixel
# 				# 				C, _ = pearsonr(this_pixel, surr_pixels)
# 				# 				correlation_map_background[i,j]=C

# 				# original_correlation_map_background=np.copy(correlation_map_background)


# 				# original_correlation_map -= np.nanmean(original_correlation_map_background)

# # A=original_correlation_map.copy()

# plt.imshow(np.sum(imag, axis=0))
# plt.show()
# plt.imshow(original_correlation_map)
# plt.colorbar(shrink=0.5)
# plt.show()



# # plt.figure()
# # plt.subplot(1,2,1)
# # plt.imshow(template_image)
# # plt.subplot(1,2,2)
# # plt.imshow(correlation_map_background)
# # plt.show()


# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(np.mean(imag, axis=0))
# plt.colorbar(shrink=0.5)
# plt.subplot(1,2,2)
# # plt.imshow(np.where((original_correlation_map>0.75) & (original_correlation_map<0.9), original_correlation_map, 0))
# # plt.imshow(np.where((original_correlation_map>0.9), original_correlation_map, 0))
# plt.imshow(np.sum(imag, axis=0))

# plt.imshow(original_correlation_map)
# plt.colorbar(shrink=0.5)
# plt.show()



# # plt.figure()
# # plt.subplot(1,2,1)
# # plt.imshow(np.mean(imag_background, axis=0))
# # plt.colorbar(shrink=0.5)
# # plt.subplot(1,2,2)
# # # plt.imshow(np.where(correlation_map_background>0.3, correlation_map_background, np.nan))
# # plt.imshow(correlation_map_background)
# # plt.colorbar(shrink=0.5)
# # plt.show()


# imag = trial_images_good_images_filtered



# #* ROIs

# def next_roi(Vcorrelation_map, Vframes, corr_thresh, Vsize):
	
#     this_max=np.max(Vcorrelation_map)
#     #print(this_max)
#     result = np.where(Vcorrelation_map== this_max)
#     coords=list(zip(result[0], result[1]))
#     I=coords[0][0]
#     J=coords[0][1]
#     this_roi_trace=np.squeeze(Vframes[:,I,J])
#     this_roi=np.zeros(Vcorrelation_map.shape)
#     this_roi[I,J]=1;
#     this_correlation_map=np.copy(Vcorrelation_map)
#     this_correlation_map[I,J]=0;

#     added=1
#     while (np.sum(np.sum(this_roi,1),0)<Vsize and added==1):
#         added=0
#         dilated=morphology.binary_dilation(this_roi, np.ones((3,3))).astype(np.uint8)
#         new_pixels=dilated-this_roi
#         result = np.where(new_pixels == 1)
#         coords=list(zip(result[0], result[1]))
#         coords2=np.asarray(coords, dtype=np.int32)
#         for a in range(coords2.shape[0]):
#             I=coords2[a][0]
#             J=coords2[a][1]
#             if not(this_correlation_map[I,J]==0):
#                 Y=np.squeeze(Vframes[:,I,J])
#                 C, _ = pearsonr(this_roi_trace, Y)
#                 if C>corr_thresh:
#                     this_roi[I,J]=1
#                     this_correlation_map[I,J]=0
#                     this_roi_trace=this_roi_trace+Y
#                     added=1

#     return this_roi, this_roi_trace, np.sum(np.sum(this_roi,1),0), this_correlation_map


# correlation_map_ = correlation_map.copy()

# original_correlation_map = correlation_map_

# aligned_frames = imag.copy()

# Nrois=100
# all_traces=np.zeros((Nrois,aligned_frames.shape[0]))
# all_rois=np.zeros(original_correlation_map.shape)
# used_pixels=np.zeros(original_correlation_map.shape)
# original_correlation_map[:5,:]=0
# original_correlation_map[:,:5]=0
# original_correlation_map[-5:,:]=0
# original_correlation_map[:,-5:]=0

# correlation_map_=np.copy(original_correlation_map)
# # correlation_map_ = np.where((original_correlation_map<0.9), original_correlation_map, 0)

# for i in tqdm(range(Nrois)):
#     this_roi3,this_roi_trace,N,this_correlation_map=next_roi(correlation_map_, aligned_frames, 0.4, 100)
#     all_traces[i,:]=this_roi_trace
#     all_rois=all_rois+(i+1)*this_roi3
#     used_pixels=used_pixels+this_roi3
#     correlation_map_[all_rois>0]=0

# plt.imshow(zscore(all_traces, 1), aspect="auto", vmin=-3, vmax=3, cmap="RdBu_r")
# plt.show()
# plt.imshow(all_rois)
# plt.show()
# plt.imshow(correlation_map_)
# plt.show()
# plt.imshow(np.sum(imag, axis=0))
# plt.show()
# plt.imshow(original_correlation_map)
# plt.show()
# # fig,(ax1,ax2,ax3,ax4)= plt.subplots(1,4)
# # ax1 = plt.subplot(121)
# # img=ax1.imshow(zscore(all_traces, 1), aspect="auto", vmin=-3, vmax=3, cmap="RdBu_r")
# # ax1.set_ylabel("trace ROI number")
# # ax1.set_xlabel("frame number")
# # fig.colorbar(img,ax=ax1)
# # ax2 = plt.subplot(322)
# # ax2.imshow(all_rois)
# # ax3 = plt.subplot(324)
# # ax3.imshow(correlation_map_)
# # ax4 = plt.subplot(326)
# # ax4.imshow(original_correlation_map)
# # # plt.show()
# # fig.tight_layout()


# a = Data.planes[0].trials[0].images.values

# plt.imshow(np.mean(a, axis=0))


# b = Data.planes[0].trials[3].images.values

# plt.imshow(np.mean(b, axis=0))


# plt.imshow(np.mean(a, axis=0) - np.mean(b, axis=0), cmap='viridis')
# plt.colorbar(shrink=0.5)


# __dict__.keys()