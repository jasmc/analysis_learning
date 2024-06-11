
#* Imports
# %%
# region Imports
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import scipy.ndimage as ndimage
import tifffile
import xarray as xr
from scipy import interpolate, signal
from scipy.ndimage import shift
from scipy.stats import pearsonr, zscore
from skimage import morphology
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
from skimage.measure import block_reduce

import my_functions as f
import my_classes as c
from my_general_variables import *
# endregion

#* Settings
# %% Settings
# region Settings

# %matplotlib ipympl

pio.templates.default = "plotly_dark"

pd.set_option("mode.copy_on_write", True)
pd.set_option("compute.use_numba", True)
pd.set_option("compute.use_numexpr", True)
pd.set_option("compute.use_bottleneck", True)
#endregion

#* Parameters
# %%
#region Parameters
light_percentage_increase_thr = 5
average_light_derivative_thr = 10
top_bottom_frame_slice = 50  # number pixels
front_back_frame_slice = 50  # number pixels


# step_size = 0.001  # mm
# number_repetitions_the_plane = 20
images_bin_size = 30
number_repetitions_the_plane_consecutively_stable = 40
# step_between_repetitions_of_the_plane = 1

median_filter_kernel = 3
gaussian_filter_sigma = 1

# kernel_size = 3
# ddepth = cv2.CV_16S

total_motion_thr = 0.5


#! debug
nrows = None
# 100000000

number_rows_read = None


galvo_value_height_threshold = 0.5
galvo_value_distance_threshold = 100
galvo_value_width_threshold = 20


xy_movement_allowed = 0.15  # fraction of the real image


number_imaged_planes = 15
number_reps_plane_consective = 2
relevant_cs = np.concatenate([range(5,35), range(45,75)])

motion_thr_from_trial_average = 5

correlation_map_sigma = 2
voxel_bin_size = 5


#! this is overwriting the one in my_general_variables.py
# cols_to_use_orig = ['FrameID'] + ['x15'] + ['y15'] + ['angle15']
# data_cols = ['X 14'] + ['Y 14'] + ['Angle (deg) 14']
# angle_name = 'Angle (deg) 14'
# angle_cols = [angle_name]

time_experiment_f = frame_id
#endregion

#* Paths
# %%
# region Paths
path_home = Path(r'E:\2024 03_Delay 2-P multiple planes')

# fish_names = [folder.stem for folder in path_home.iterdir() if folder.is_dir()]
# fish_names.remove('Behavior')

# for fish_name in fish_names:
# fish_name = r'20240228_01_delay_2p-1_mitfaMinusMinus,elavl3H2BCaMP6s_7dpf'

fish_name = r'20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'

#! '20240304_01_delay_2p-1_mitfaMinusMinus,Gal4elavl3GCaMP6s_6dpf' is good

# behavior_path = Path(r'D:\2024 02_Delay 2p\Behavior')
# imaging_path = Path(r'D:\2024 02_Delay 2p') / fish_name / 'Imaging'

behavior_path = path_home / 'Behavior'
imaging_path = path_home / fish_name / 'Imaging'

protocol_path = behavior_path / (fish_name + '_stim control.txt')
camera_path = behavior_path / (fish_name + '_cam.txt')
tracking_path = behavior_path / (fish_name + '_mp tail tracking.txt')


galvo_path = imaging_path / 'signalsfeedback.xls'
images_path = imaging_path / (fish_name + '_green.tif')

anatomy_1_path = path_home / fish_name / 'Anatomical stack 1.tif'
anatomy_1_filtered_path = path_home / fish_name / 'Anatomical stack 1 binned and filtered.tif'
#endregion

#* Classes
# %%
# region Classes
@dataclass
class Trial:
	
	trial_number : int

	# position_anatomical_stack : int
	# reference_image : np.ndarray

	protocol : pd.DataFrame
	behavior : pd.DataFrame
	images : xr.DataArray

	def get_stim_index(self, cs_us: str):

		#* Get the indices of the CS in the images of the trials.

		# self = self.trials[trial_number]

		protocol = self.protocol

		cs_beg, cs_end = protocol.loc[protocol[cs_us] != 0, 'Time (ms)'].values[[0,-1]]

		a = self.images.time.values < cs_beg
		cs_beg_index = np.where(np.diff(a))[0][0]

		b = self.images.time.values > cs_end
		cs_end_index = np.where(np.diff(b))[0][0]

		return np.array([cs_beg_index, cs_end_index])
	

@dataclass
class Plane:

	trials : list['Trial']

	# reference_image_position_anatomical_stack : int
	# reference_image : np.ndarray

	# order_planes_sequence : int

	def get_reference_position(self):

		return round(np.median([trial.position_anatomical_stack for trial in self.trials]))
	
	def get_all_images(self):

		return np.concatenate([trial.images.values for trial in self.trials])

@dataclass
class Data:
	
	planes : list['Plane']
	anatomical_stack : np.ndarray


	def get_planes(self, plane_numbers: list[int]):
		
		return [self.planes[i] for i in plane_numbers]


	def get_trials(self, plane_numbers: list[int] | str, trial_numbers: list[int]):
		
		if plane_numbers == 'all':
			
			plane_numbers = range(len(self.planes))

		return [self.planes[i].trials[j] for i in plane_numbers for j in trial_numbers]


	# def get_images(self, plane_numbers: list[int] | str, trial_numbers: list[int]):
		
	# 	if plane_numbers is None:
			
	# 		plane_numbers = len(self.planes)

	# 	return [self.planes[i].trials[j].images for i in plane_numbers for j in trial_numbers]


# endregion





#! flag summer time
if (date := int(fish_name.split('_')[0][4:6])) >= 4 and date <= 10:
	Summer_time = True
else:
	Summer_time = False


#* Read the behavior camera data and preprocess it.
# %%
# region Behavior camera
data = read_camera(camera_path)
data[abs_time] = data[abs_time].astype('float64')

print('Behavior camera started: ', pd.to_datetime(data[abs_time].iat[0], unit='ms'))
 

#* Estimate the true framerate.
predicted_framerate, reference_frame_id = framerate_and_reference_frame(data)


data = data.drop(columns=ela_time)

#* Discard frames that will not be used (in camera and hence further down).
# The calculated interframe interval before the reference frame is variable. Discard what happens up to then (also achieved by using how='inner' in merge_camera_with_data).
data = data[data[frame_id] >= reference_frame_id]

		# #! reverse data_cols to what we want
		# #! #* Open tail tracking data.
		# data = read_tail_tracking_data(tracking_path)

		# # if (data := read_tail_tracking_data(data_path)) is None: # type: ignore
		# # 	return None

		# # plot_behavior_overview(data, fish_name, fig_behavior_name)

		# # #* Look for possible tail tracking errors.
		# # if tracking_errors(data, single_point_tracking_error_thr):
		# # 	return None

		# #! #* Add information about the time of each frame to data.
		# data = merge_camera_with_data(data, camera)
		# # data = camera

#* Fix abs_time so that the time of each frame becomes closer to the time at which the frames were acquired by the data and not when they were caught by the computer.
# The delay between acquiring and catching the frame is unknown and therefore disregarded.
data[abs_time] = np.linspace(data[abs_time].iat[0], data[abs_time].iat[0] + len(data) * (1000 / predicted_framerate), len(data))

#! Need to join with galvo before doing this.
#! #* Interpolate data to the expected framerate.
# data = interpolate_data(data, predicted_framerate)
#endregion

#* Read the stim log and merge it with the behavior camera data.
# %%
# region Stim log

#* Open the stim log.
protocol = read_protocol(protocol_path)


# protocol.iloc[:,1] - protocol.iloc[:,0]

#* Identify the stimuli, trials of the experiment.
data_cols = []
data = identify_trials(data, protocol)

# plt.plot(data[abs_time])
# data[cs].unique()
#endregion

#* Read the galvo signal and find the peaks.
# %%
#region Galvo signal
galvo = pd.read_csv(galvo_path, sep='\t', decimal=',', usecols=[0,1], names=[abs_time, 'GalvoValue'], dtype={'GalvoValue':'float64'}, parse_dates=[abs_time], date_format=r'%d/%m/%Y  %H:%M:%S,%f', skip_blank_lines=True, skipinitialspace=True, nrows=nrows).dropna(axis=0)
galvo = galvo.reset_index(drop=True)
#* Convert the time in galvo to unixtime in ms
galvo[abs_time] = galvo[abs_time].astype('int64') / 10**6

#* To align the galvo signals to the respective frames with need to consider at least from the beginning of the galvo signal.
# if (first_timepoint_galvo := galvo[abs_time].iat[0]) <= (first_timepoint_data := data[abs_time].iat[0]):

# 	galvo = galvo[galvo[abs_time] >= first_timepoint_data]
# 	first_timepoint_galvo = galvo[abs_time].iat[0]

# plt.plot(galvo[abs_time])



print('Galvo started: ', pd.to_datetime(galvo[abs_time].iat[0], unit='ms'))

if Summer_time:
	galvo[abs_time] -= 60*60*1000




#* Remove consecutive duplicates...
galvo = galvo[galvo[galvo_value].ne(galvo[galvo_value].shift())]


galvo = galvo.reset_index(drop=True)



#* Find the beginning of the frames.
beg_first_image = signal.find_peaks(galvo[galvo_value], height=0.5, prominence=0.05)[0][0]
beg_first_image_time = galvo[abs_time].iat[beg_first_image]

peaks = signal.find_peaks(galvo[galvo_value].iloc[beg_first_image+1000:], height=[0.5, 5], distance=100, prominence=[0.5, 5], width=20)[0] + beg_first_image + 1000
number_peaks = len(peaks)

galvo_sub = galvo.loc[0:10000]

fig, ax = plt.subplots()
ax.set_title('')
ax.plot(galvo_sub[abs_time].to_numpy(), galvo_sub[galvo_value].to_numpy(), 'k')
ax.plot(beg_first_image_time, 1, 'ro')
ax.plot(galvo_sub[abs_time].iloc[peaks[:5]], galvo_sub[galvo_value].iloc[peaks[:5]], 'bo')
ax.set_xlabel('Interframe interval (ms)')
ax.set_ylabel('Galvo value')
fig.show()



#* Calculate the interframe interval.
interframe_interval_array = galvo[abs_time].iloc[peaks].diff()
interframe_interval = interframe_interval_array.median()


print('The median of the interframe interval is:', interframe_interval)
print('Min and max interframe interval:', interframe_interval_array.min(), interframe_interval_array.max())


#* Discard a few images at the beginning where the imaging is not good or we are unsure of the true beginning of the images.
beg_image_to_consider_index = interframe_interval_array.index[np.where(interframe_interval_array == interframe_interval)[0][0] - 1]
beg_image_to_consider_time = galvo[abs_time].iat[beg_image_to_consider_index]

peaks = peaks[peaks >= beg_image_to_consider_index]


#* Number of images to discard at the beginning.

# need to round donw
number_images_before_first_image_to_consider = round((beg_image_to_consider_time - beg_first_image_time) / interframe_interval)
# beg_first_image_time = beg_image_to_consider_time - number_images_before_first_image_to_consider * interframe_interval




#* Get info about the images in the multipage tiff with the imaging data.
bytes_header, height, width = get_bytes_header_and_image(images_path)
bytes_header_and_image = bytes_header + height * width * 2

#* Find where we started imaging in the anatomical stack.
number_images = get_number_images(images_path, bytes_header_and_image)
# - number_images_before_first_image_to_consider


print('Number of galvo peaks identified:', number_peaks)
print('Number of images in the tiff:', number_images)


# Here I use NAN and not False to later be able to discard rows where a frame did not start and where there is no tracking data.
galvo['Frame beg'] = np.nan
galvo.loc[galvo.iloc[peaks].index[:number_images-number_images_before_first_image_to_consider], 'Frame beg'] = 1
# np.arange(1, len(peaks)+1)

galvo_sub = galvo.iloc[0:5000].copy()

#* Add the beginning of the missed images to galvo.
galvo = pd.merge_ordered(galvo, pd.DataFrame({abs_time: np.arange(beg_image_to_consider_time - number_images_before_first_image_to_consider * interframe_interval, beg_image_to_consider_time, interframe_interval), 'Frame beg': np.ones(number_images_before_first_image_to_consider)}))



fig, ax = plt.subplots()
ax.plot(galvo_sub[abs_time].to_numpy(), galvo_sub[galvo_value].to_numpy(), 'k')
ax.plot(galvo_sub[abs_time].to_numpy(), galvo_sub['Frame beg'].to_numpy(), 'ro')
ax.plot(np.arange(beg_image_to_consider_time - number_images_before_first_image_to_consider * interframe_interval, beg_image_to_consider_time, interframe_interval), np.ones(number_images_before_first_image_to_consider)*2, 'yo')
ax.plot(galvo.iloc[0:5000][abs_time].to_numpy(), galvo.iloc[0:5000]['Frame beg'].to_numpy()*3, 'bo')
ax.set_xlabel('Interframe interval (ms)')
ax.set_ylabel('Galvo value')



galvo_sub = galvo.iloc[-5000:]

galvo_sub.loc[galvo_sub['Frame beg'].notna(), 'Frame beg'] = 1

fig, ax = plt.subplots()
ax.plot(galvo_sub[abs_time].to_numpy(), galvo_sub[galvo_value].to_numpy(), 'k')
ax.plot(galvo_sub[abs_time].to_numpy(), galvo_sub['Frame beg'].to_numpy(), 'ro')
ax.set_xlabel('Interframe interval (ms)')
ax.set_ylabel('Galvo value')
fig.show()

del galvo_sub

fig, ax = plt.subplots()
ax.plot(interframe_interval_array, 'k.')
ax.set_xlabel('Interframe interval (ms)')
ax.set_ylabel('Galvo value')
fig.show()
#endregion

#* Read the behavior data.
# %%
# region Behavior data
#! reverse data_cols to what we want
data_cols = x_cols + y_cols + angle_cols
behavior = read_tail_tracking_data(tracking_path).astype('float32')

# behavior.dtypes
# if (tail := read_tail_tracking_data(data_path)) is None: # type: ignore
# 	return None

# plot_behavior_overview(tail, fish_name, fig_behavior_name)

# #* Look for possible tail tracking errors.
# if tracking_errors(tail, single_point_tracking_error_thr):
# 	return None
# endregion

#* Merge the galvo signal, stim log, behavior camera data and behavior data.
# %%
# region Merge of galvo signal, stim log and behavior camera data
#* Discard imaging before the tracking started (in some cases, the tracking might start after the imaging).
if (first_timepoint_galvo := galvo[abs_time].iat[0]) <= (first_timepoint_data := data[abs_time].iat[0]):

	#* Do not discard images because only the galvo signal txt file is not overwritten when one stops and restarts the imaging.
	# number_images_discard = len(galvo[(galvo[abs_time] < first_timepoint_data) & (galvo['Frame beg'].notna())])
	# images = images[number_images_discard:]


	galvo = galvo[galvo[abs_time] >= first_timepoint_data]
	# first_timepoint_galvo = galvo[abs_time].iat[0]

	first_timepoint = galvo[abs_time].iat[0]

	data[abs_time] -= first_timepoint
	galvo[abs_time] -= first_timepoint

	data = data[data[abs_time] >= 0]
	galvo = galvo[galvo[abs_time] <= data[abs_time].iat[-1]]

	print('Galvo signal started before the tracking.')

	plt.plot(galvo[abs_time])
	plt.plot(data[abs_time])

first_timepoint = galvo[abs_time].iat[0]

data[abs_time] -= first_timepoint
galvo[abs_time] -= first_timepoint

data = data[data[abs_time] >= 0]
galvo = galvo[galvo[abs_time] <= data[abs_time].iat[-1]]


#* Update the number of images.
# Discard all images after the tracking is over.
number_images = len(galvo.loc[galvo['Frame beg'].notna(),:])


#* Merge galvo with data.
# data = pd.merge_ordered(data, galvo.drop(columns=galvo_value), on=abs_time, how='outer')
data = pd.merge_ordered(data, galvo, on=abs_time, how='outer')

del galvo, protocol

# Cannot use pd.merge_ordered because of NANs
# data = pd.merge_ordered(data, behavior, on=frame_id, how='left')

data = pd.merge(data, behavior, on=frame_id, how='left')

del behavior

data.reset_index(drop=True, inplace=True)
data.rename(columns={abs_time : 'Time (ms)'}, inplace=True)
abs_time = 'Time (ms)'
data[abs_time] -= data[abs_time].iat[0]
# endregion

#* Read the imaging data.
# %%
# region Imaging data
#* Get the images and align them to data.
#! Do not forget to discard the first images.
#!!!!! images = np.array([get_image_from_tiff(images_path, image_i, bytes_header, height, width) for image_i in range(number_images_before_first_image_to_consider, len(peaks))])
# images_subset_mean = [np.mean(image[-30:-10][-30:-10]).astype('float32') for image in images]
images = np.array([get_image_from_tiff(images_path, image_i, bytes_header, height, width).astype('float32') for image_i in tqdm(range(number_images))])

images.shape
# endregion

#* Check whether the different pieces of data are aligned.
# %%
# region Confirmation of data alingment
images_mean = [image.mean() for image in images]

# Remove all colummns where there is no tracking data and no frame started.
# Really need to convert to dense...
data[[cs,us]] = data[[cs,us]].sparse.to_dense()
data = data.dropna(subset=[frame_id, 'Frame beg', cs, us], how='all')
# data.loc[data['Frame beg'].notna()]
# data[[frame_id, 'Frame beg']] = data[[frame_id, 'Frame beg']].fillna(0)


# data[abs_time] -= data[abs_time].iat[0]

# first_timepoint = galvo[abs_time].iat[0]

# galvo[abs_time] -= first_timepoint


# data[abs_time] -= first_timepoint
# data = data[data[abs_time] >= 0]

# galvo = galvo[galvo[abs_time] <= data[abs_time].iat[-1]]

# fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# axs[0].plot(galvo[abs_time])
# axs[0].set_title('Galvo Signal')
# axs[1].plot(data[abs_time])
# axs[1].set_title('Data')
# plt.tight_layout()
# plt.show()

# plt.plot(data[abs_time].to_numpy())
# plt.plot(galvo[abs_time].to_numpy())

# data[abs_time] -= data[abs_time].iat[0]

# del galvo
# data = pd.merge_ordered(data, galvo, on=abs_time, how='outer')

data['Image mean'] = np.nan

data.loc[data['Frame beg'].notna(), 'Image mean'] = images_mean[:len(data.loc[data['Frame beg'].notna(), 'Image mean'])]
# data.loc[data['Frame beg'].notna(), 'Image mean'] = images_subset_mean_top[:len(data[data['Frame beg'].notna()])]

data = data.reset_index(drop=True)


data[[cs, us]] = data[[cs, us]].fillna(0)

# data[cs].cat.remove_unused_categories()
# data[us].cat.remove_unused_categories()

x0 = 2500
x1 = 2500

stim_numbers = data.loc[data[us] != 0, us].unique()


fig, axs = plt.subplots(stim_numbers.size-1, 1, figsize=(10, 50), sharex=True, sharey=True)

# fig, axs = plt.subplots(3, 1, figsize=(10, 50))

# fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)

# ! SHOULD PLOT AS WELL THE BEHAVIOR DATA

for stim_number_i, stim_number in enumerate(stim_numbers[1:]):


	data_ = data.loc[data[us] == stim_number, abs_time]

	us_beg_ = data_.iat[0]

	us_end_ = data_.iat[-1]


	# print(stim_number_i, us_end_ - us_beg_)


	data_plot = data.loc[data[abs_time].between(us_beg_-x0, us_end_+x1)]


	axs[stim_number_i].plot(data_plot[abs_time].to_numpy() - us_beg_, data_plot['Image mean'].to_numpy(), 'k.')
	# axs[stim_number_i].plot(data_plot[abs_time].to_numpy() - us_beg_, data_plot['Angle (deg) 14'].diff().to_numpy(), 'k.')
	# axs[stim_number_i].plot(data_plot[abs_time].to_numpy() - us_beg_, data_plot[galvo_value].to_numpy() + 100, 'bo')
	#! axs[stim_number_i].plot(data__[abs_time].to_numpy() - us_beg_, data__[galvo_value].to_numpy(), 'm')

	axs[stim_number_i].axvline(x=us_beg_ - us_beg_, color='g', linestyle='--')
	axs[stim_number_i].axvline(x=us_end_ - us_beg_, color='r', linestyle='--')
	axs[stim_number_i].set_title(f"Stimulus Number: {stim_number}")
	# plt.plot(time, data__[galvo_value]], 'k')
	# plt.plot(time, data__['Frame beg'], 'bo')
	# plt.plot(time, data__[us], 'yo')
	# plt.plot(time, data__[us_end], 'mo')

	# data__.plot(x=abs_time, y=['Frame beg', 'Image mean', us, us_end], ls='.')

	# break
fig.show()


del data_, data_plot

# data.loc[data['Frame beg'].notna()]

data.drop(columns=['Image mean', 'GalvoValue'], inplace=True)

#endregion

#* Separate the different pieces of data in different dataframes.
# %%
# region Separation of different pieces of data
protocol = data[[abs_time, cs, us]].copy()
protocol = protocol[((protocol[cs]!=0) | (protocol[us]!=0))]

behavior = data[[abs_time] + data_cols].dropna().rename(columns={frame_id : 'Frame number (behavior)'}).copy()
# behavior_array = xr.DataArray(behavior, coords=[('time', all_data.loc[all_data['Frame number'].notna(), :].index), ('parameters', behavior.columns)], dims=['time', 'parameters'])
# behavior_array.to_dataframe()

imaging = xr.DataArray(images, coords={'index': ('time', data.loc[data['Frame beg'].notna(), :].index), 'time': data.loc[data['Frame beg'].notna(), abs_time].to_numpy(), 'x': range(images.shape[1]), 'y': range(images.shape[2])}, dims=['time', 'x', 'y'])

imaging.name = 'Imaging data'
# endregion

#* Arrange the data in planes data.
# %%
# region Planes data

cs_onset_index = np.array([protocol.loc[protocol[cs] == relevant_cs[i], :].index[0] for i in range(len(relevant_cs))])

index_list = [[i, i+1, i+2*number_imaged_planes, i+2*number_imaged_planes+1] for i in range(0, number_reps_plane_consective * number_imaged_planes, 2)]

planes_cs_onset_indices = cs_onset_index

planes_cs_onset_indices = [cs_onset_index[[j for j in i]] for i in index_list]

del cs_onset_index, index_list


all_data = []

i = 0
for plane_i, plane_cs_onset_indices in tqdm(enumerate(planes_cs_onset_indices)):

	trials_list = []
	for trial_i, trial_cs_onset_index in enumerate(plane_cs_onset_indices):

		time_start = protocol.loc[trial_cs_onset_index, abs_time] - 45000
		time_end = protocol.loc[trial_cs_onset_index, abs_time] + 15000

		# index = protocol[protocol[abs_time].between(time_start, time_end)].index

		trial_images = imaging.loc[time_start : time_end,:,:]

		# plt.imshow(np.mean(trial_images.values, axis=0))
		# plt.show()


		# images_average = ndimage.median_filter(np.mean(get_good_last_images(trial_images), axis=0), size=median_filter_kernel)
		# # images_average = ndimage.median_filter(np.mean(trial_images[i], axis=0), size=median_filter_kernel)

		# if trial_images[i] is not None:

		# 	planes_numbers[i] = find_plane_in_anatomical_stack(anatomical_stack_images, images_average.astype('float32'), None, x_dim, y_dim)[0]

		# 	#* Bin and filter 3 planes of the anatomical stack where the plane is included.
		# 	reference_images[i] = ndimage.median_filter(np.mean(anatomical_stack_images[planes_numbers[i]-1 : planes_numbers[i]+2], axis=0), size=median_filter_kernel)
		# 	# anatomical_stack_images_sub = ndimage.median_filter(anatomical_stack_images[planes_numbers[i]], size=median_filter_kernel)

		# 	# plt.imshow(reference_images[i])
		# 	# plt.show()

		# else:
		# 	print('Look here')
		# 	planes_numbers[i] = np.nan
		# 	continue


		trials_list.append(Trial(i, protocol[protocol[abs_time].between(time_start, time_end)], behavior[behavior[abs_time].between(time_start, time_end)], trial_images))

		i += 1

	all_data.append(Plane(trials_list))

	# break

del trials_list

for i in range(len(all_data)):
	plt.imshow(np.mean(all_data[i].trials[0].images.values, axis=0))
	plt.show()

#endregion

#* Read the anatomical stack.
# %%
# region Anatomical stack
anatomical_stack_images = tifffile.imread(anatomy_1_path).astype('float32')
# anatomical_stack_images = tifffile.imread(anatomy_1_filtered_path).astype('float32')

anatomical_stack_images = ndimage.median_filter(anatomical_stack_images, size=median_filter_kernel, axes=(1,2))


#ToDo this should take the pixel spacing into account!!!

_, y_dim, x_dim = np.array(anatomical_stack_images.shape)

x_dim = int(x_dim * xy_movement_allowed/2)
y_dim = int(y_dim * xy_movement_allowed/2)
# endregion

#* Save the data.
# %%
# region Save the data
path_pkl = path_home / fish_name / (fish_name + '.pkl')

all_data = Data(all_data, anatomical_stack_images)

with open(path_pkl, 'wb') as f:
	pickle.dump(all_data, f)
# endregion

#* Correct motion within and across trials.
# %%
# region Correct motion

no final, ha valores negativos de pixels

for plane_i, plane in tqdm(enumerate(all_data)):
	# break

	print('Plane: ', plane_i)

	motions = [_ for _ in range(4)]
	template_images = np.zeros((len(plane.trials), plane.trials[0].images.shape[1], plane.trials[0].images.shape[2]))
	plane_numbers = np.zeros(len(plane.trials), dtype='int32')

	#* Motion correction within trial.
	for trial_i, trial in enumerate(plane.trials):

		#* 1.1. Motion correction relative to trials average.

		##* Discard bad frames due to motion, gating of the PMT or plane change when making a template image for the trial.
		# plane.trials[trial_i].images.values, plane.trials[trial_i].template_image, plane.trials[trial_i].position_anatomical_stack 
		motions[trial_i], template_images[trial_i], plane_numbers[trial_i] = correct_motion_within_trial(trial, 5)

		plt.imshow(anatomical_stack_images[plane_numbers[trial_i]])
		plt.show()


		#* Frames to ignore due to too much motion (or gating of the PMT, which causes a huge "motion").
		trial_images = trial.images.values

		# Mask with True where the frames are bad (due to gating of the PMT or motion).
		mask_bad_frames = (~get_good_images_indices(trial_images)) | (np.where(get_total_motion(motions[trial_i]) > motion_thr_from_trial_average, True, False))

		all_data[plane_i].trials[trial_i].mask_bad_frames = mask_bad_frames


	#* Motion correction across trials of the same plane.
	for trial_i, trial in enumerate(plane.trials):
		
		if trial_i > 0:
			#* Measure motion of each frame using phase cross-correlation.
			motion = measure_motion(np.expand_dims(template_images[trial_i][5:-5, 5:-5], axis=0), template_images[0][5:-5, 5:-5], normalization=None)[0]

			motions[trial_i] += motion

		#* Measure motion of each frame using phase cross-correlation.
		total_motion = get_total_motion(motions[trial_i])
		# Use half of the frames to get the template image.
		motion_thr = np.median(total_motion)

		
		
		#* Align the frames to their average.
		aligned_frames = align_frames(trial.images.to_numpy(), motions[trial_i], total_motion, [5,10,30,35][trial_i])

		template_image = get_template_image(aligned_frames[np.where(total_motion <= motion_thr)[0]])


		#* Identify the plane number of the trial.
		plane_number, _ = find_plane_in_anatomical_stack(anatomical_stack_images, template_image.astype('float32'), None, x_dim, y_dim)


		plt.imshow(ndimage.median_filter(np.mean(aligned_frames, axis=0), size=median_filter_kernel))
		plt.colorbar(shrink=0.5)
		plt.show()

		all_data[plane_i].trials[trial_i].images.values = aligned_frames
		all_data[plane_i].trials[trial_i].template_image = template_image
		all_data[plane_i].trials[trial_i].position_anatomical_stack = plane_number

		# break
	# break
	print('Plane:', plane_i, plane_numbers)
#endregion

#* Plot the position in the anatomical stack.
# region Position in the anatomical stack

A = []
B = []

C = []
D = []

for i in range(len(all_data)):

	for j in range(2):

		A.append(all_data[i].trials[j].position_anatomical_stack)

		C.append(all_data[i].trials[j].template_image)

	for l in range(2,4):

		B.append(all_data[i].trials[l].position_anatomical_stack)

		D.append(all_data[i].trials[l].template_image)


A = np.array(A)
B = np.array(B)

C = np.array(C)
D = np.array(D)


import seaborn as sns
sns.set_style('whitegrid')


plt.xlabel('Trial before or after initial train')
plt.ylabel('Plane number in anatomical stack')
plt.plot(A, 'blue')
plt.plot(B, 'red')
plt.legend(['Before initial train', 'After initial train'])
plt.savefig(r'H:\My Drive\PhD\Lab meetings\A and B.png', dpi=300, bbox_inches='tight')


plt.xlabel('Trial before or after initial train')
plt.ylabel('Difference between planes imaged\n before and after initial train (μm)')
plt.plot(A-B, 'k')
plt.ylim(-6, 6)
plt.savefig(r'H:\My Drive\PhD\Lab meetings\difference.png', dpi=300, bbox_inches='tight')




sns.set_style('white')



fig, axs = plt.subplots(15, 2, figsize=(10, 50))

for i in range(30):
	if i<=14:
		im = axs[i,0].imshow(C[i*2], interpolation=None, cmap='RdBu_r', vmin=80, vmax=500)
		axs[i,0].axis('off')

	if i>14 and i<=29:
		axs[i-15,1].imshow(C[(i-15)*2+1], interpolation=None, cmap='RdBu_r', vmin=80, vmax=500)
		axs[i-15,1].axis('off')

fig.tight_layout()
# fig.suptitle('Templates Before Correction', fontsize=16)
fig.savefig(r'H:\My Drive\PhD\Lab meetings\templates before.png', dpi=300, bbox_inches='tight')



fig, axs = plt.subplots(15, 2, figsize=(10, 50))

for i in range(30):
	if i<=14:
		axs[i,0].imshow(D[i*2], interpolation=None, cmap='RdBu_r', vmin=80, vmax=500)
		axs[i,0].axis('off')

	if i>14 and i<=29:
		axs[i-15,1].imshow(D[(i-15)*2+1], interpolation=None, cmap='RdBu_r', vmin=80, vmax=500)
		axs[i-15,1].axis('off')

fig.tight_layout()
# fig.suptitle('Templates After Correction', fontsize=16)
fig.savefig(r'H:\My Drive\PhD\Lab meetings\templates after.png', dpi=300, bbox_inches='tight')

#  endregion



#* Load the data.
path_pkl = path_home / fish_name / (fish_name + '.pkl')
# path_pkl = r"E:\2024 03_Delay 2-P multiple planes\20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf\20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf.pkl"

with open(path_pkl, 'rb') as file:
	all_data = pickle.load(file)

all_data.planes[0].trials[0].__dict__.keys()


# compression_level = 4
# compression_library = 'zlib'

# with pd.HDFStore(path_pkl, complevel=compression_level, complib=compression_library) as store:
	
# 	store.append(fish_name, planes_list, data_columns=[cs, us], expectedrows=len(fish.raw_data), append=False)

# 	store.get_storer(fish.dataset_key()).attrs['metadata'] = fish.metadata._asdict()





#* For correlation map.

##* Preparing the data for the correlation map.

# for plane_i, plane in enumerate(Data.planes):
# 	for trial_i, trial in enumerate(plane):

# A = [Data.planes[plane_i].trials[trial_i].images.values for trial_i, trial in enumerate(plane.trials) for plane_i, plane in enumerate(Data.planes)]

# B = np.sum([np.sum(x, axis=0) for x in A], axis=0)

# plt.imshow(B)


eye_mask = np.ones(all_data.planes[0].trials[0].images.shape[1:], dtype='bool')

#!
eye_mask[350:, 350:450] = False
eye_mask[:50, 350:450] = False
plt.imshow(eye_mask)

# # A = ndimage.uniform_filter(plane_trials_good_images, size=(30, 30), axes=(1,2))
# ndimage.gaussian_filter(plane_trials_good_images, sigma=gaussian_filter_sigma, axes=(1,2))

# plt.imshow(np.mean(A, axis=0))
# plt.colorbar()


for plane_i, plane in enumerate(all_data.planes):

	# if plane_i not in [0,1,3,6,8,9,10,13]:
	# 	continue
	# break
#!
	# plane.trials = plane.trials



	#!!!!!!!!!!!!!!!!!!!!!!!!! DO ALL OF THIS FOR SINGLE TRIAL AND THEN CONCATENATE TO GET PLANE DATA




	#* To get a correlation map for the whole plane data, we need to concatenate all the images of the trials.
	# plane_trials_all_images = np.concatenate([t.images.values for t in plane.trials])
	plane_trials_all_images = plane.get_all_images()

	plt.title('All images from plane')
	plt.imshow(np.mean(plane_trials_all_images, axis=0))
	plt.colorbar
	plt.show()


	#* Get the number of images per trial.
	plane_trials_number_images = np.array([t.images.shape[0] for t in plane.trials])


	#* Get the indices of the CS in the images of the trials.
	cs_indices = np.array([trial.get_stim_index(cs) for trial in plane.trials])

	cs_indices[1:,0] += np.cumsum(plane_trials_number_images[:-1])
	cs_indices[1:,1] += np.cumsum(plane_trials_number_images[:-1])









	#* Discard bad frames due to motion, gating of the PMT or plane change.
	plane_trials_mask_bad_frames = np.concatenate([t.mask_bad_frames for t in plane.trials])
	plane_bad_frames_index = np.where(plane_trials_mask_bad_frames)[0]
	plane_trials_good_images = plane_trials_all_images[~plane_trials_mask_bad_frames].copy()

	plt.title('All good images from plane')
	plt.imshow(np.mean(plane_trials_good_images, axis=0))
	plt.colorbar
	plt.show()










	#* Filter in space.
	plane_trials_good_images_filtered = ndimage.gaussian_filter(plane_trials_good_images, sigma=gaussian_filter_sigma, axes=(1,2))

	plt.title('All good images from plane filtered')
	plt.imshow(np.mean(plane_trials_good_images_filtered, axis=0))
	plt.colorbar

#!!!!!!!!!!!!!!!!! move it further down
	#* Calcultate the correlation map.
	# Inspired in Suit2p. There, the function that computes the correlation map is celldetect2.getVmap.
	correlation_map = np.linalg.norm(ndimage.gaussian_filter(plane_trials_good_images, sigma=correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(plane_trials_good_images, axis=0), sigma=correlation_map_sigma)**2

	plt.figure('Correlation map')
	plt.imshow(correlation_map)
	plt.colorbar(shrink=0.5)
	plt.show()






	#* Subtract the background.
	# Pixel values equal to 0 are ignored to discard the artificial edges of the images that were introduced during the motion correction.
	images_mean = np.nanmean(np.where(plane_trials_good_images == 0, np.nan, plane_trials_good_images), axis=(1,2))

	images_mean = np.nanmean(plane_trials_good_images, axis=(1,2))
	for image_i in range(plane_trials_good_images.shape[0]):
		plane_trials_good_images[image_i] -= images_mean[image_i]

	del images_mean

	#* Mask the background.
	plane_images_mask_fish = np.where(np.median(plane_trials_good_images, axis=0) <= 0, 0, 1).astype(dtype='bool')

	plane_images_mask_fish_without_eyes = plane_images_mask_fish & eye_mask

	#* Set to 0 the pixels that are not part of the fish in the images. Also, mask the eyes.
	plane_trials_good_images = np.where(plane_images_mask_fish_without_eyes, plane_trials_good_images, 0)

	plt.title('All good images from plane masked background')
	plt.imshow(np.mean(plane_trials_good_images, axis=0))
	plt.colorbar(shrink=0.5)
	plt.show()




	# region Voxel analysis
	#* Voxel analysis

	#* Bin the 2D images.
	

	plane_trials_good_images_binned = block_reduce(plane_trials_good_images, block_size=(1, voxel_bin_size, voxel_bin_size), func=np.mean, cval=0)

	plt.imshow(np.mean(plane_trials_good_images_binned, axis=0), interpolation='none')


	plane_trials_good_images_binned_ = np.empty(tuple([plane_trials_all_images.shape[0]] + list(plane_trials_good_images_binned.shape[1:]))) * np.nan
	plane_trials_good_images_binned_[~plane_trials_mask_bad_frames, :, :] = plane_trials_good_images_binned

	plane_trials_good_images_binned = plane_trials_good_images_binned_.copy()

	del plane_trials_good_images_binned_

	plt.title('All good images from plane binned')
	plt.imshow(np.mean(plane_trials_good_images_binned, axis=0))
	plt.colorbar(shrink=0.5)
	plt.show()



	deltaF = []
	deltaF_SR = []

	for i in range(len(cs_indices)):

		baseline = np.nanmean(plane_trials_good_images_binned[[cs_indices[i, 0] - 20, cs_indices[i, 0]]], axis=0)
		
		during_cs = np.nanmean(plane_trials_good_images_binned[[cs_indices[i, 0], cs_indices[i, 1]]], axis=0)

		deltaF_SR.append((during_cs - baseline) / baseline)

		if i == 0:
			
			deltaF.append((plane_trials_good_images_binned[ : plane_trials_number_images[0]] - baseline) / baseline)

		elif i < len(cs_indices)-1:

			deltaF.append((plane_trials_good_images_binned[np.cumsum(plane_trials_number_images)[i-1] : np.cumsum(plane_trials_number_images)[i]] - baseline) / baseline)

		else:
			deltaF.append((plane_trials_good_images_binned[np.cumsum(plane_trials_number_images)[i-1] : ] - baseline) / baseline)

	deltaF = np.concatenate(deltaF)

	deltaF = np.where(np.isnan(deltaF), 0, deltaF)

	deltaF_SR = np.array(deltaF_SR)


	for i in range(len(cs_indices)):
		plt.imshow(deltaF_SR[i], interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
		plt.colorbar(shrink=0.5)
		plt.title('DeltaF_SR')
		plt.show()


	A = np.mean(np.array([deltaF_SR[0], deltaF_SR[1]]), axis=0)
	B = np.mean(np.array([deltaF_SR[2], deltaF_SR[3]]), axis=0)



	plt.imshow(A, interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
	plt.colorbar(shrink=0.5)
	plt.title('DeltaF_SR A')
	plt.show()

	plt.imshow(B, interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
	plt.colorbar(shrink=0.5)
	plt.title('DeltaF_SR B')
	plt.show()

	plt.imshow(B/A, interpolation='none', vmin=-100, vmax=100, cmap='RdBu_r')
	plt.colorbar(shrink=0.5)
	plt.title('DeltaF_SR B / DeltaF_SR A')
	plt.savefig(path_home /  fish_name / (fish_name + '_deltaF_SR_voxels_plane ' + str(plane_i) + '.tif'))
	plt.show()

	deltaF_SR = np.concatenate(deltaF_SR)

#!!!
	# deltaF_ = np.empty(tuple([plane_trials_all_images.shape[0]] + list(deltaF.shape[1:]))) * np.nan
	# deltaF_[~plane_trials_mask_bad_frames, :, :] = deltaF

	# deltaF = deltaF_.copy()

	# del deltaF_

	for i in range(len(cs_indices)):
		deltaF[cs_indices[i,0]:cs_indices[i,1],:20,-20:] = -100

	plt.imshow(np.nanmean(deltaF, axis=0))
	plt.colorbar(shrink=0.5)

	#* Save rois_zscore_over_time as a TIFF file.
	tifffile.imwrite(path_home /  fish_name / (fish_name + '_deltaF_voxels_plane ' + str(plane_i) + '.tif'), deltaF.astype('float32'))

	# endregion







	# region ROI analysis for the whole plane

	#* Set to 0 the pixels that are not part of the fish in the correlation map.
	correlation_map = np.where(plane_images_mask_fish_without_eyes, correlation_map, 0)

	plt.title('Correlation map masked background')
	plt.imshow(np.where(plane_images_mask_fish_without_eyes, correlation_map, 0))
	plt.colorbar(shrink=0.5)
	plt.show()





	#* ROIs for the all the trials of the same plane.
	#TODO need to rewrite all this part, using Mike's and Ruben's code
	all_traces, all_rois, used_pixels, correlation_map_ = get_ROIs(Nrois=100, correlation_map=correlation_map, images=plane_trials_good_images_filtered, threshold=0.3, max_pixels=60)

	plt.imshow(zscore(all_traces, 1), aspect="auto", cmap="RdBu_r")
	plt.savefig(path_home / fish_name / (fish_name + 'zscore ' + str(plane_i) + '.tif'))
	plt.show()
	plt.imshow(all_rois)
	plt.colorbar()
	plt.show()
	plt.imshow(correlation_map_)
	plt.show()
	plt.imshow(np.sum(plane_trials_all_images, axis=0))
	plt.show()
	plt.imshow(correlation_map)
	plt.show()



	#* Create array to then make movie.
	all_rois = all_rois.astype('int')

	rois_zscore_over_time = np.zeros_like(plane_trials_all_images)


	#* Consider the periods of bad frames in the array with the Z score of the ROI traces.
	all_traces_z_score = zscore(all_traces, 1)

	all_traces_z_score_ = np.empty((all_traces.shape[0], len(plane_trials_all_images))) * np.nan
	all_traces_z_score_[:, ~plane_trials_mask_bad_frames] = all_traces_z_score

	plt.imshow(all_traces_z_score_, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)

	all_traces_z_score = all_traces_z_score_
	del all_traces_z_score_

	rois_mask = np.zeros(rois_zscore_over_time.shape, dtype='bool')

	#* Get mask of the ROIs.
	for roi_i in range(1, all_rois.max()):
		# break
		rois_mask[roi_i] = all_rois == roi_i

		# roi_mask = rois_mask[roi_i]
		# [np.newaxis, :, :]

		for t in range(rois_zscore_over_time.shape[0]):
			# break
			rois_zscore_over_time[t,:,:] += np.where(rois_mask[roi_i], all_traces_z_score[roi_i, t], 0)

			rois_zscore_over_time[t,:,:]

	for i in range(len(cs_indices)):
		rois_zscore_over_time[cs_indices[i,0]:cs_indices[i,1],:50,-50:] = -100

	# plt.imshow(np.mean(plane_trials_all_images, axis=0))
	plt.imshow(np.nansum(rois_zscore_over_time, axis=0), aspect="auto", cmap="RdBu_r", interpolation='none')
	plt.colorbar()

	#* Save rois_zscore_over_time as a TIFF file.
	tifffile.imwrite(path_home / fish_name / (fish_name + 'rois_zscore_over_time ' + str(plane_i) + '.tif'), rois_zscore_over_time.astype('float32'))

	# endregion



	# region ROI analysis for each trial

	for trial in plane.trials:

		#!
		# break
		trial = plane.trials[3]
	# break
		# trial.images = trial.images


		#* Discard bad frames due to motion, gating of the PMT or trial change.
		trial_good_images = trial.images.values[~trial.mask_bad_frames]
		trial_bad_frames_index = np.where(trial.mask_bad_frames)[0]

		plt.title('All images from trial')
		plt.imshow(np.mean(trial_good_images, axis=0))
		plt.colorbar(shrink=0.5)
		plt.show()

		#* Subtract the background.
		# Pixel values equal to 0 are ignored to discard the artificial edges of the images that were introduced during the motion correction.
		images_mean = np.nanmean(np.where(trial_good_images == 0, np.nan, trial_good_images), axis=(1,2))

		images_mean = np.nanmean(trial_good_images, axis=(1,2))
		for image_i in range(trial_good_images.shape[0]):
			trial_good_images[image_i] -= images_mean[image_i]

		del images_mean

		#* Mask the background.
		trial.images_mask_fish = np.where(np.median(trial_good_images, axis=0) <= 0, 0, 1).astype(dtype='bool')

		trial.images_mask_fish_without_eyes = trial.images_mask_fish & eye_mask

		#* Set to 0 the pixels that are not part of the fish in the images. Also, mask the eyes.
		trial_good_images = np.where(trial.images_mask_fish_without_eyes, trial_good_images, 0)

		plt.title('All good images from trial masked background')
		plt.imshow(np.mean(trial_good_images, axis=0))
		plt.colorbar(shrink=0.5)
		plt.show()



#!!!!!!!!!!! Voxel analysis per trial
		#* Bin the 2D images.
		trial_good_images_binned = block_reduce(trial_good_images, block_size=(1, voxel_bin_size, voxel_bin_size), func=np.mean, cval=0)

		plt.imshow(np.mean(trial_good_images_binned, axis=0), interpolation='none')


		trial_good_images_binned_ = np.empty(tuple([trial.images.shape[0]] + list(trial_good_images_binned.shape[1:]))) * np.nan
		trial_good_images_binned_[~trial.mask_bad_frames, :, :] = trial_good_images_binned

		trial_good_images_binned = trial_good_images_binned_.copy()

		del trial_good_images_binned_

		plt.title('All good images from trial binned')
		plt.imshow(np.mean(trial_good_images_binned, axis=0))
		plt.colorbar(shrink=0.5)
		plt.show()

	

















		#* Filter in space.
		trial_images_filtered = ndimage.gaussian_filter(trial_images, sigma=gaussian_filter_sigma, axes=(1,2))
		trial_images_good_images_filtered = trial_images_filtered[~trial.mask_bad_frames].copy()




		#* Correlation map
		# In Suit2p, the function that computes the correlation map is celldetect2.getVmap.
		correlation_map = np.linalg.norm(ndimage.gaussian_filter(trial_images_good_images, sigma=correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(trial_images_good_images, axis=0), sigma=correlation_map_sigma)**2

		plt.figure('Correlation map')
		plt.imshow(correlation_map)
		plt.colorbar(shrink=0.5)
		plt.show()

		#* Subtract the background.
	#! Here I should take the average ignoring the sharp edges of the images.
		images_mean = np.mean(trial_images_good_images, axis=(1,2))

		for image_i in range(trial_images_good_images.shape[0]):
			trial_images_good_images[image_i] -= images_mean[image_i]

		del images_mean

		#* Mask the background.
		trial_images_mask_fish = np.where(np.median(trial_images_good_images, axis=0) <= 0, 0, 1).astype(dtype='bool')
		
		#* Mask the background and the eyes.
		trial_images_mask_fish_without_eyes = trial_images_mask_fish & eye_mask


		#* Set to 0 the pixels that are not part of the fish in the images.
		trial_images_good_images = np.where(trial_images_mask_fish_without_eyes, trial_images_good_images, 0)

		plt.title('All good images from plane masked background')
		plt.imshow(np.mean(trial_images_good_images, axis=0))
		plt.colorbar(shrink=0.5)
		plt.show()

		#* Set to 0 the pixels that are not part of the fish in the correlation map.
		correlation_map = np.where(trial_images_mask_fish_without_eyes, correlation_map, 0)

		plt.title('Correlation map masked background')
		plt.imshow(np.where(trial_images_mask_fish_without_eyes, correlation_map, 0))
		plt.colorbar(shrink=0.5)
		plt.show()





		#* ROIs

		all_traces, all_rois, used_pixels, correlation_map_ = get_ROIs(Nrois=100, correlation_map=correlation_map, images=trial_images_good_images_filtered, threshold=0.3, max_pixels=60)

		images_times = trial_images.time.values


		trial_time_ref = images_times[0]

		trial_protocol = trial.protocol

		cs_times = trial_protocol[trial_protocol[cs]!=0]
		cs_times = cs_times.iloc[[0,-1]] if cs_times.shape[0] > 1 else cs_times

		us_times = trial_protocol[trial_protocol[us]!=0]
		us_times = us_times.iloc[[0,-1]] if us_times.shape[0] > 1 else us_times


		images_times = images_times - trial_time_ref
		cs_times = cs_times['Time (ms)'].values - trial_time_ref
		us_times = us_times['Time (ms)'].values - trial_time_ref


		number_traces = 50

		fig, axs = plt.subplots(number_traces, 1, sharex=True)
		# figsize=(10, 8)

		for i in range(number_traces):

#!
			axs[i].plot(images_times[:110], all_traces[i+50][:110])

			if cs_times.shape[0] > 0:
				axs[i].axvline(x=cs_times[0], color='g', linestyle='-')
				axs[i].axvline(x=cs_times[1], color='g', linestyle='--')
			
			if us_times.shape[0] > 0:
				axs[i].axvline(x=us_times[0], color='r', linestyle='-')
				axs[i].axvline(x=us_times[1], color='r', linestyle='--')

		fig.show()


		plt.imshow(zscore(all_traces, 1), aspect="auto", vmin=-3, vmax=3, cmap="RdBu_r")

		plt.show()
		plt.imshow(all_rois)
		plt.show()
		plt.imshow(correlation_map_)
		plt.show()
		plt.imshow(np.sum(trial_images_good_images, axis=0))
		plt.show()
		plt.imshow(correlation_map)
		plt.show()






		#* Create array to then make movie.
		all_rois = all_rois.astype('int')

		rois_zscore_over_time = np.zeros_like(trial_images)


		#* Consider the periods of bad frames in the array with the Z score of the ROI traces.
		all_traces_z_score = zscore(all_traces, 1)

		all_traces_z_score_ = np.empty((all_traces.shape[0], len(trial_images))) * np.nan
		all_traces_z_score_[:, ~trial.mask_bad_frames] = all_traces_z_score

		plt.imshow(all_traces_z_score_, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)

		all_traces_z_score = all_traces_z_score_
		del all_traces_z_score_

		rois_mask = np.zeros(rois_zscore_over_time.shape, dtype='bool')

		#* Get mask of the ROIs.
		for roi_i in range(1, all_rois.max()):
			# break
			rois_mask[roi_i] = all_rois == roi_i

			# roi_mask = rois_mask[roi_i]
			# [np.newaxis, :, :]

			for t in range(rois_zscore_over_time.shape[0]):
				# break
				rois_zscore_over_time[t,:,:] += np.where(rois_mask[roi_i], all_traces_z_score[roi_i, t], 0)

				rois_zscore_over_time[t,:,:]

		for i in range(len(cs_indices)):
			rois_zscore_over_time[cs_indices[i,0]:cs_indices[i,1],:50,-50:] = -100

		# plt.imshow(np.mean(plane_trials_all_images, axis=0))
		plt.imshow(np.nansum(rois_zscore_over_time, axis=0), aspect="auto", cmap="RdBu_r")
		plt.colorbar()

		#* Save rois_zscore_over_time as a TIFF file.
		tifffile.imwrite(path_home / fish_name / (fish_name + 'trial_rois_zscore_over_time' + '.tif'), rois_zscore_over_time.astype('float32'))

		# endregion



















#* Discard bad frames due to motion, gating of the PMT or plane change.
trial_images_good_images = trial_images[~trial.mask_bad_frames].copy()

#* Filter in space.
trial_images_good_images_filtered = ndimage.gaussian_filter(trial_images_good_images, sigma=gaussian_filter_sigma, axes=(1,2))


# #* Subtract the background.
# images_mean = np.mean(trial_images_good_images, axis=(1,2))

# for image_i in range(trial_images_good_images.shape[0]):
# 	trial_images_good_images[image_i] -= images_mean[image_i]
	
# # # signal.detrend(trial_images_good_images, axis=0, type='constant', bp=0, overwrite_data=True)

# # # for image_i in range(trial_images_good_images.shape[0]):
# # # 	trial_images_good_images[image_i] -= np.mean(trial_images_good_images[image_i,-50:-5,-50:-5])

# # images_mean = np.mean(trial_images_good_images, axis=(1,2))
# # images_filtered_mean = np.mean(trial_images_good_images_filtered, axis=(1,2))

# # for image_i in range(trial_images_good_images.shape[0]):
# # 	trial_images_good_images[image_i] -= images_mean[image_i]
# # 	trial_images_good_images_filtered[image_i] -= images_filtered_mean[image_i]

# # #* Clip values below the mean background.
# # np.clip(trial_images_good_images, 0, None, out=trial_images_good_images)

# #* Mask the background.
# trial_images_mask_fish = np.where(np.median(trial_images_good_images, axis=0) < 0, 0, 1).astype(dtype='bool')
# # plt.imshow(np.sum(np.where(trial_images_good_images<0,0,trial_images_good_images), axis=0))
# # plt.colorbar(shrink=0.5)
# # plt.show()

# plt.imshow(np.where(trial_images_mask_fish, np.sum(trial_images_good_images, axis=0), 0))
# plt.colorbar(shrink=0.5)
# plt.show()
# plt.imshow(np.sum(trial_images_good_images_filtered, axis=0))
# plt.colorbar(shrink=0.5)
# plt.show()




#* Actual correlation map.

# imag = trial_images_good_images[:,100:300, 150:250].copy()
imag = trial_images_good_images[:,:, :].copy()

correlation_map=np.zeros(imag.shape[1:])

for i in tqdm(range(imag.shape[1])):
	if i>0 and i<(imag.shape[1]-1):
		for j in range(imag.shape[2]):

			if j>0 and j<(imag.shape[2]-1):

				this_pixel=np.squeeze(imag[:,i,j])
				surr_pixels=np.squeeze(np.sum(np.sum(np.squeeze(imag[:,i-1:i+2,j-1:j+2]),2),1))-this_pixel
				C, _ = pearsonr(this_pixel, surr_pixels)
				correlation_map[i,j]=C



import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

# Assuming 'imag' is your 3D image data with shape (time, height, width)
correlation_map = np.zeros(imag.shape[1:])

# Precompute the sum of the surrounding pixels
surr_sum = np.zeros_like(imag)
for t in range(imag.shape[0]):
    surr_sum[t,:,:] = np.pad(imag[t, :, :], ((1, 1), (1, 1)), 'constant', constant_values=0)[1:-1, 1:-1]


import numpy as np
from scipy.signal import convolve2d

# Assuming 'image' is your 2D image data
kernel = np.ones((3, 3))
kernel[1, 1] = 0  # Exclude the center pixel if you don't want to include it in the sum

# Convolve the image with the kernel
surr_sum = np.zeros_like(imag)

for t in range(imag.shape[0]):
	surr_sum[] = convolve2d(imag[t, :, :], kernel, mode='same')




surr_sum.shape





# Calculate the correlation using vectorized operations
for i in tqdm(range(1, imag.shape[1] - 1)):
    for j in range(1, imag.shape[2] - 1):
        this_pixel = imag[:, i, j]
        surr_pixels = np.sum(surr_sum[:, i-1:i+2, j-1:j+2], axis=(1,2)) - this_pixel
        C, _ = pearsonr(this_pixel, surr_pixels)
        correlation_map[i, j] = C







original_correlation_map=np.copy(correlation_map)

MAP = original_correlation_map.copy()

plt.imshow(MAP, vmin=0.3)
plt.colorbar(shrink=0.5)



correlation_map = np.copy(original_correlation_map)
correlation_map = ndimage.gaussian_filter(correlation_map, sigma=3)

plt.imshow(correlation_map)
plt.colorbar(shrink=0.5)





#* Correlation map

# Assuming trial_images_good_images, trial_images_good_images_filtered, and gausswidth are defined

# In Suit2p, the function that computes the correlation map is celldetect2.getVmap.

# trial_images_good_images_mean = np.mean(trial_images_good_images**2, axis=0)
# trial_images_good_images_filtered_mean = np.mean(trial_images_good_images_filtered**2, axis=0)

trial_images_good_images_mean = np.linalg.norm(trial_images_good_images, axis=0)
trial_images_good_images_filtered_mean = np.linalg.norm(trial_images_good_images_filtered, axis=0)

trial_images_good_images_mean = ndimage.gaussian_filter(trial_images_good_images_mean, sigma=gaussian_filter_sigma)

trial_images_good_images_mean = trial_images_good_images_mean**2
trial_images_good_images_filtered_mean = trial_images_good_images_filtered_mean**2

correlation_map = trial_images_good_images_filtered_mean / trial_images_good_images_mean
# correlation_map = trial_images_good_images_filtered_mean / trial_images_good_images_mean
# correlation_map = 1 - trial_images_good_images_filtered_mean / trial_images_good_images_mean
# correlation_map -= np.nanmean(correlation_map[-50:-20,-50:-20])
# np.clip(correlation_map, 0, None, out=correlation_map)

plt.imshow(correlation_map)
plt.colorbar(shrink=0.5)

plt.imshow(np.mean(trial_images_good_images, axis=0))
plt.colorbar(shrink=0.5)

plt.imshow(np.mean(trial_images_good_images_filtered, axis=0))
plt.colorbar(shrink=0.5)

plt.imshow(correlation_map, vmin=0.4, vmax=0.8)
plt.colorbar(shrink=0.5)



#* Subtract the background.
images_mean = np.mean(trial_images_good_images, axis=(1,2))

for image_i in range(trial_images_good_images.shape[0]):
	trial_images_good_images[image_i] -= images_mean[image_i]

#* Mask the background.
plane_images_mask_fish = np.where(np.median(trial_images_good_images, axis=0) < 0, 0, 1).astype(dtype='bool')
# plt.imshow(np.sum(np.where(trial_images_good_images<0,0,trial_images_good_images), axis=0))
# plt.colorbar(shrink=0.5)
# plt.show()

plt.imshow(np.where(plane_images_mask_fish, np.sum(trial_images_good_images, axis=0), 0))
plt.colorbar(shrink=0.5)
plt.show()
plt.imshow(np.sum(trial_images_good_images_filtered, axis=0))
plt.colorbar(shrink=0.5)
plt.show()


#* Set to 0 the pixels that are not part of the fish in the correlation map.
correlation_map = np.where(plane_images_mask_fish, correlation_map, 0)

plt.imshow(np.where(plane_images_mask_fish,correlation_map,0))
# , vmin=0.4, vmax=0.8
plt.colorbar(shrink=0.5)




#! Careful with border. think now it is fine.





# for i in tqdm(range(1, imag.shape[1]-1)):
# 	for j in range(1, imag.shape[2]-1):
# 		this_pixel = np.squeeze(imag[:, i, j])
# 		surr_pixels = np.squeeze(np.sum(np.sum(np.squeeze(imag[:, i-1:i+2, j-1:j+2]), 2), 1)) - this_pixel
# 		C, _ = pearsonr(this_pixel, surr_pixels)
# 		correlation_map[i, j] = C

# original_correlation_map = np.copy(correlation_map)



				# imag_background = trial_images_good_images[:,-50:-5,-50:-5].copy()

				# correlation_map_background = np.empty(imag_background.shape[1:])*np.nan

				# for i in tqdm(range(imag_background.shape[1])):
				# 	if i>0 and i<(imag_background.shape[1]-1):
				# 		for j in range(imag_background.shape[2]):

				# 			if j>0 and j<(imag_background.shape[2]-1):

				# 				this_pixel=np.squeeze(imag_background[:,i,j])
				# 				surr_pixels=np.squeeze(np.sum(np.sum(np.squeeze(imag_background[:,i-1:i+2,j-1:j+2]),2),1))-this_pixel
				# 				C, _ = pearsonr(this_pixel, surr_pixels)
				# 				correlation_map_background[i,j]=C

				# original_correlation_map_background=np.copy(correlation_map_background)


				# original_correlation_map -= np.nanmean(original_correlation_map_background)

# A=original_correlation_map.copy()

plt.imshow(np.sum(imag, axis=0))
plt.show()
plt.imshow(original_correlation_map)
plt.colorbar(shrink=0.5)
plt.show()



# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(template_image)
# plt.subplot(1,2,2)
# plt.imshow(correlation_map_background)
# plt.show()


plt.figure()
plt.subplot(1,2,1)
plt.imshow(np.mean(imag, axis=0))
plt.colorbar(shrink=0.5)
plt.subplot(1,2,2)
# plt.imshow(np.where((original_correlation_map>0.75) & (original_correlation_map<0.9), original_correlation_map, 0))
# plt.imshow(np.where((original_correlation_map>0.9), original_correlation_map, 0))
plt.imshow(np.sum(imag, axis=0))

plt.imshow(original_correlation_map)
plt.colorbar(shrink=0.5)
plt.show()



# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(np.mean(imag_background, axis=0))
# plt.colorbar(shrink=0.5)
# plt.subplot(1,2,2)
# # plt.imshow(np.where(correlation_map_background>0.3, correlation_map_background, np.nan))
# plt.imshow(correlation_map_background)
# plt.colorbar(shrink=0.5)
# plt.show()


imag = trial_images_good_images_filtered



#* ROIs

def next_roi(Vcorrelation_map, Vframes, corr_thresh, Vsize):
    
    this_max=np.max(Vcorrelation_map)
    #print(this_max)
    result = np.where(Vcorrelation_map== this_max)
    coords=list(zip(result[0], result[1]))
    I=coords[0][0]
    J=coords[0][1]
    this_roi_trace=np.squeeze(Vframes[:,I,J])
    this_roi=np.zeros(Vcorrelation_map.shape)
    this_roi[I,J]=1;
    this_correlation_map=np.copy(Vcorrelation_map)
    this_correlation_map[I,J]=0;

    added=1
    while (np.sum(np.sum(this_roi,1),0)<Vsize and added==1):
        added=0
        dilated=morphology.binary_dilation(this_roi, np.ones((3,3))).astype(np.uint8)
        new_pixels=dilated-this_roi
        result = np.where(new_pixels == 1)
        coords=list(zip(result[0], result[1]))
        coords2=np.asarray(coords, dtype=np.int32)
        for a in range(coords2.shape[0]):
            I=coords2[a][0]
            J=coords2[a][1]
            if not(this_correlation_map[I,J]==0):
                Y=np.squeeze(Vframes[:,I,J])
                C, _ = pearsonr(this_roi_trace, Y)
                if C>corr_thresh:
                    this_roi[I,J]=1
                    this_correlation_map[I,J]=0
                    this_roi_trace=this_roi_trace+Y
                    added=1

    return this_roi, this_roi_trace, np.sum(np.sum(this_roi,1),0), this_correlation_map


correlation_map_ = correlation_map.copy()

original_correlation_map = correlation_map_

aligned_frames = imag.copy()

Nrois=100
all_traces=np.zeros((Nrois,aligned_frames.shape[0]))
all_rois=np.zeros(original_correlation_map.shape)
used_pixels=np.zeros(original_correlation_map.shape)
original_correlation_map[:5,:]=0
original_correlation_map[:,:5]=0
original_correlation_map[-5:,:]=0
original_correlation_map[:,-5:]=0

correlation_map_=np.copy(original_correlation_map)
# correlation_map_ = np.where((original_correlation_map<0.9), original_correlation_map, 0)

for i in tqdm(range(Nrois)):
    this_roi3,this_roi_trace,N,this_correlation_map=next_roi(correlation_map_, aligned_frames, 0.4, 100)
    all_traces[i,:]=this_roi_trace
    all_rois=all_rois+(i+1)*this_roi3
    used_pixels=used_pixels+this_roi3
    correlation_map_[all_rois>0]=0

plt.imshow(zscore(all_traces, 1), aspect="auto", vmin=-3, vmax=3, cmap="RdBu_r")
plt.show()
plt.imshow(all_rois)
plt.show()
plt.imshow(correlation_map_)
plt.show()
plt.imshow(np.sum(imag, axis=0))
plt.show()
plt.imshow(original_correlation_map)
plt.show()
# fig,(ax1,ax2,ax3,ax4)= plt.subplots(1,4)
# ax1 = plt.subplot(121)
# img=ax1.imshow(zscore(all_traces, 1), aspect="auto", vmin=-3, vmax=3, cmap="RdBu_r")
# ax1.set_ylabel("trace ROI number")
# ax1.set_xlabel("frame number")
# fig.colorbar(img,ax=ax1)
# ax2 = plt.subplot(322)
# ax2.imshow(all_rois)
# ax3 = plt.subplot(324)
# ax3.imshow(correlation_map_)
# ax4 = plt.subplot(326)
# ax4.imshow(original_correlation_map)
# # plt.show()
# fig.tight_layout()


a = Data.planes[0].trials[0].images.values

plt.imshow(np.mean(a, axis=0))


b = Data.planes[0].trials[3].images.values

plt.imshow(np.mean(b, axis=0))


plt.imshow(np.mean(a, axis=0) - np.mean(b, axis=0), cmap='viridis')
plt.colorbar(shrink=0.5)


__dict__.keys()