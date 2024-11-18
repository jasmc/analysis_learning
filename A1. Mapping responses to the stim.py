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
import xarray as xr
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

#* Load custom functions and classes
import my_classes as c
import my_functions_imaging as fi
import my_parameters as p
from my_general_variables import *
from PIL import ImageSequence
from PIL import ImageDraw, ImageFont

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






range_color_thr = 30











#* Paths
# %%
# region Paths
path_home = Path(r'E:\2024 09_Delay 2-P 4 planes JC neurons')
# Path(r'E:\2024 03_Delay 2-P 15 planes top part')
# Path(r'E:\2024 10_Delay 2-P single plane')
# Path(r'E:\2024 10_Delay 2-P 15 planes ca8 neurons')
# Path(r'E:\2024 09_Delay 2-P zoom in multiplane imaging')

# fish_list = [f for f in (path_home / 'Imaging').iterdir() if f.is_dir()]
# fish_names_list = [f.stem for f in fish_list]

fish_name = r'20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'

# '20241013_01_control_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'

# '20241015_03_delay_2p-9_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf'
# '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
# '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'

# '20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241024_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241015_03_delay_2p-9_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf'
# '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240926_03_trace_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20240926_03_trace_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'


imaging_path = path_home / 'Imaging'


# for fish_i, fish_name in enumerate(fish_names_list):

# 	try:

imaging_path_ = imaging_path / fish_name / 'Imaging'


path_pkl_analysis_1 = path_home / fish_name / (fish_name + '_analysis 1' + '.pkl')


# h5_path = path_home / fish_name / (fish_name + '_before_motion_correction.h5')

# h5_path = imaging_path_ / (fish_name + '_before_motion_correction.h5')

# h5_path = r"E:\2024 03_Delay 2-P 15 planes top part\Imaging\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf\Imaging\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf_before_motion_correction.h5"

#endregion



if 'delay' in fish_name:
	interval_between_CS_onset_US_onset = 9  # s
if 'trace' in fish_name:
	interval_between_CS_onset_US_onset = 13  # s


#* Load the data before motion correction.
# %%
# region Load the data before motion correction
with open(path_pkl_analysis_1, 'rb') as file:
	all_data = pickle.load(file)

# endregion

# all_data.__dict__.keys()

# all_data.planes[3].trials[3].shift_correction



# softthresh=50

plt.imshow((all_data.planes[0].trials[3].images.mean('Time (ms)')))



shape_ = all_data.planes[0].trials[0].images.shape[1:]


# plt.imshow((all_data.planes[0].trials[3].images[0,:,:].values))
# A=all_data.planes[0].trials[3].images[0,:,:].values
# A[A>0]
# x_black_box_beg = 330
# x_black_box_end = 345
# y_black_box_beg = 594
# y_black_box_end = 609

if 'ca8' in str(path_home):
		
	x_black_box_beg = shape_[0] - 20
	x_black_box_end = shape_[0] - 5
	y_black_box_beg = shape_[1] - 20
	y_black_box_end = shape_[1] - 5


else:
	x_black_box_beg = 330
	x_black_box_end = 345
	y_black_box_beg = 594
	y_black_box_end = 609







def add_colors_to_world(anatomy, color_frame_original):
	trial_red_channel = anatomy * (1 - color_frame_original)
	trial_green_channel = trial_red_channel + color_frame_original * color_frame_original
	trial_blue_channel = trial_red_channel
	
	image = (np.stack([trial_red_channel,trial_green_channel,trial_blue_channel], axis=-1)*255).astype(np.uint8)
	
	plt.imshow(image)
	plt.show()

	return image








# #* Subtract the background (calculated from the black box) and clip the values to 0.
# for plane_i, plane in enumerate(all_data.planes):
# 	for trial_i, trial in enumerate(plane.trials):
		
# 		all_data.planes[plane_i].trials[trial_i].images = (trial.images - trial.images[:, x_black_box_beg:x_black_box_end, y_black_box_beg:y_black_box_end].mean(dim=('x','y'))).clip(0,None)




ALL_DATA = deepcopy(all_data)
# all_data = deepcopy(ALL_DATA)


#* Split data into 3 periods: before CS, from CS onset to US onset and after US onset.
for plane_i, plane in enumerate(all_data.planes):

	plane_template = np.zeros(plane.trials[0].images.shape[1:])


	# fig, axs = plt.subplots()


	for trial_i, trial in enumerate(plane.trials):




		trial_images = trial.images.copy()


		# trial_images['mask after US']

	# 	break
	# break


		#* Give some colors to the world.

		##* Background color
		anatomy = trial.template_image.copy()
		#!!!!!!!!!!!
		anatomy /= np.median(anatomy*range_color_thr)
		anatomy = np.clip(anatomy,0,1)
		
		trial.anatomy_channel = anatomy

		# plt.imshow(anatomy)
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


		# break
	# break





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



plane_numbers = np.zeros((len(all_data.planes),len(all_data.planes[0].trials)))



for plane_i in range(len(all_data.planes)):
	for trial_i in range(len(plane.trials)):
	
		plane_numbers[plane_i,trial_i] = all_data.planes[plane_i].trials[trial_i].position_anatomical_stack


plane_numbers = plane_numbers.astype('int')

plane_position_stack = np.argsort(plane_numbers[:,0])




fig, axs = plt.subplots(len(all_data.planes), len(plane.trials), figsize=(10, 50), squeeze=False)

for plane_i in range(len(all_data.planes)):
	
	# position = plane_position_stack[plane_i]
	position=plane_i

	
	for trial_i in range(len(plane.trials)):
		
		# all_data.planes[plane_i].template_image_position_anatomical_stack
		# print(position)
		
		axs[position,trial_i].imshow(all_data.planes[plane_i].trials[trial_i].CS_positive_response)
		axs[position,trial_i].set_xticks([])
		axs[position,trial_i].set_yticks([])



	# break

fig.set_size_inches(30, 70)
fig.subplots_adjust(hspace=0.05, wspace=0.02)


fig.tight_layout()


fig.savefig(imaging_path_ / 'CS positive response new.png', dpi=600)


# Rotate the frames in each trial by 90 degrees
for plane in all_data.planes:
	for trial in plane.trials:
		trial.CS_positive_response = np.rot90(trial.CS_positive_response)



All_data = deepcopy(all_data)

all_data = deepcopy(All_data)


tifffile = []

for trial_i, trial in enumerate(all_data.planes[3].trials):

	tifffile.append(Image.fromarray(trial.CS_positive_response))




All_data = deepcopy(tifffile)

# tifffile = deepcopy(All_data)
#* Save all_data.planes[0].trials in a tiff




plt.imshow(tifffile[-3])


# Define a function to add text to an image
def add_text_to_image(image, text, position, font=None, font_size=40, color=None, bg_color=(255,255,255)):
	draw = ImageDraw.Draw(image)
	try:
		if font is None:
			font = ImageFont.load_default()
		else:
			font = ImageFont.truetype(font, font_size)
	except IOError:
		font = ImageFont.load_default()
	# text_size = draw.textsize(text, font=font)
	text_position = (position[0], position[1])
	draw.rectangle([text_position, (text_position[0] , text_position[1])], fill=bg_color)
	draw.text(text_position, text, font=font, fill=color)
	return image



text = ['Pre-train (trial 7)', 'Pre-train (trial 8)',
		'Train (trial 7)', 'Train (trial 8)', 'Train (trial 17)', 'Train (trial 18)', 'Train (trial 27)', 'Train (trial 28)', 'Train (trial 37)', 'Train (trial 38)',
		'Test (trial 7)', 'Test (trial 8)', 'Test (trial 19)', 'Test (trial 20)', 'Test (trial 29)', 'Test (trial 30)']

color = ['white', 'white',
		 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red',
		 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow']

position = (10, 10)

# Add text to each frame
for i, img in enumerate(tifffile):

	img_with_text = add_text_to_image(img, text[i], position, font="arial.ttf", font_size=40, color=color[i])
	tifffile[i] = img_with_text

plt.imshow(tifffile[-9])




# Rotate the images in the tiff by 90 degrees
rotated_tifffile = [img.rotate(90) for img in tifffile]
plt.imshow(rotated_tifffile[0])

# Read the rotated tiff file

rotated_images = []
# with Image.open(rotated_tiff_path) as img:
	for frame in ImageSequence.Iterator(img):
		rotated_images.append(frame.copy())

# Display the first image to verify
plt.imshow(rotated_images[0])
plt.show()



# Save the rotated tifffile as a multipage tiff
rotated_tiff_path = imaging_path_ / 'CS_positive_response_multipage_rotated.tiff'
rotated_tifffile[0].save(rotated_tiff_path, save_all=True, append_images=rotated_tifffile[1:])




# Save tifffile as a multipage tiff
tiff_path = imaging_path_ / 'CS_positive_response_multipage talk new.tiff'
tifffile[0].save(tiff_path, save_all=True, append_images=tifffile[1:])