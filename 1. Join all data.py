# Save all data in a single pickle file.
# Anatomical stack images and imaging data are median filtered.


#* Imports

##   
# region Imports
import os
import pickle
from importlib import reload
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import scipy.ndimage as ndimage
import tifffile
import xarray as xr
from scipy import signal
from tqdm import tqdm

#* Load custom functions and classes
import my_classes as c
import my_functions_imaging as fi
import my_parameters as p
from my_general_variables import *

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
# pd.set_option("compute.use_numba", True)
pd.set_option("compute.use_numexpr", True)
pd.set_option("compute.use_bottleneck", True)
#endregion


#* Paths
##   
# region Paths
path_home = Path(r'C:\Users\joaqc\Desktop\WIP')
# Path(r'E:\2024 03_Delay 2-P 15 planes top part')
# Path(r'E:\2024 09_Delay 2-P 4 planes JC neurons')
# Path(r'E:\2024 10_Delay 2-P single plane')
# Path(r'E:\2024 10_Delay 2-P 15 planes ca8 neurons')
# Path(r'E:\2024 09_Delay 2-P zoom in multiplane imaging')

# fish_list = [f for f in (path_home / 'Imaging').iterdir() if f.is_dir()]
# fish_names_list = [f.stem for f in fish_list]

fish_name = r'20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
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
# '20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'

behavior_path = path_home / fish_name / 'Behavior'
imaging_path = path_home / fish_name / 'Imaging'




# for fish_i, fish_name in enumerate(fish_names_list):

# 	try:

# imaging_path_ = imaging_path / 'Imaging'

if (path_pkl_before_motion_correction := imaging_path / fish_name / (fish_name + '_before motion correction' + '.pkl')).exists():

	print('Already preprocessed: ', fish_name)
	print(path_pkl_before_motion_correction)
	
	# continue

print('Analyzing fish: ', fish_name)

protocol_path = behavior_path / (fish_name + '_stim control.txt')
camera_path = behavior_path / (fish_name + '_cam.txt')
tracking_path = behavior_path / (fish_name + '_mp tail tracking.txt')

galvo_path = imaging_path / 'signalsfeedback.xls'
images_path = imaging_path / (fish_name + '_green.tif')

anatomy_1_path = imaging_path / fish_name / 'Anatomical stack 1.tif'
# anatomy_1_filtered_path = imaging_path / fish_name / 'Anatomical stack 1 binned and filtered.tif'

#endregion


#! flag summer time
if (date := int(fish_name.split('_')[0][4:6])) >= 4 and date <= 10:
	Summer_time = True
else:
	Summer_time = False




match str(path_home):

	case r'E:\2024 03_Delay 2-P 15 planes top part' | r'E:\2024 10_Delay 2-P 15 planes bottom part' | r'E:\2024 10_Delay 2-P 15 planes ca8 neurons' | r'C:\Users\joaqc\Desktop\WIP':

		number_imaged_planes = 15
		number_reps_plane_consective = 2
		relevant_cs = [range(5,35), range(45,75)]

		index_list = [np.concatenate([[i+number_reps_plane_consective*x*number_imaged_planes, i+number_reps_plane_consective*x*number_imaged_planes+1] for x in range(len(relevant_cs))]) for i in range(0, number_reps_plane_consective * number_imaged_planes, number_reps_plane_consective)]

	case r'E:\2024 10_Delay 2-P single plane':
		number_imaged_planes = 1
		number_reps_plane_consective = 80
		relevant_cs = [np.arange(5,85)]

		index_list = relevant_cs

	case r'E:\2024 09_Delay 2-P 4 planes JC neurons':

		#! one of the planes it's when the drift correction happens
		number_imaged_planes = 4
		number_reps_plane_consective = 2
		# relevant_cs = [range(5,15),
		# 			  range(15,25), range(25,55), range(55,45), range(45,55),
		# 			  range(55,65), range(65,75), range(75,85)]
		relevant_cs = [range(5,13),
					  range(15,23), range(25,33), range(35,43), range(45,53),
					  range(55,63), range(67,75), range(77,85)]
		index_list = [np.concatenate([[i+number_reps_plane_consective*x*number_imaged_planes, i+number_reps_plane_consective*x*number_imaged_planes+1] for x in range(len(relevant_cs))]) for i in range(0, number_reps_plane_consective * number_imaged_planes, number_reps_plane_consective)]
		# [np.array([0,1, 10,11, 20,21, 30,31, 40,41, 50,51, 62,63, 72,73]),
		# 	np.array([2,3, 12,13, 22,23, 32,33, 42,43, 52,53, 64,65, 74,75]),
		# 	np.array([4,5, 14,15, 24,25, 34,35, 44,45, 54,55, 66,67, 76,77]),
		# 	np.array([6,7, 16,17, 26,27, 36,37, 46,47, 56,57, 68,69, 78,79])]
		
		
relevant_cs = np.concatenate(relevant_cs)






#!!!!!!!!!!!! label Trial objects with the period of the experiment





















#* Read the behavior camera data and preprocess it.
##   
# region Behavior camera
data = fi.read_camera(camera_path)

# camera = pd.read_csv(r"C:\Users\joaqc\Desktop\WIP\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf\Behavior\20240910_02_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_6dpf_cam.txt", engine='pyarrow', sep=' ', header=0, decimal='.')

data[abs_time] = data[abs_time].astype('float64')


print('Behavior camera started: ', pd.to_datetime(data[abs_time].iat[0], unit='ms'))


#* Estimate the true framerate.
predicted_framerate, reference_frame_id = fi.framerate_and_reference_frame(data)


data = data.drop(columns=ela_time)

#* Discard frames that will not be used (in camera and hence further down).
# The calculated interframe interval before the reference frame is variable. Discard what happens up to then (also achieved by using how='inner' in merge_camera_with_data).
data = data[data['Frame number'] >= reference_frame_id]

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
##   
# region Stim log

#* Open the stim log.
protocol = fi.read_protocol(protocol_path)


# protocol.iloc[:,1] - protocol.iloc[:,0]

#* Identify the stimuli, trials of the experiment.
# data_cols = []
data = fi.identify_trials(data, protocol)

# plt.plot(data[abs_time])
# data[cs].unique()
#endregion

#* Read the galvo signal and find the peaks.
##   
#region Galvo signal
galvo = pd.read_csv(galvo_path, sep='\t', decimal=',', usecols=[0,1], names=[abs_time, 'GalvoValue'], dtype={'GalvoValue':'float64'}, parse_dates=[abs_time], date_format=r'%d/%m/%Y  %H:%M:%S,%f', skip_blank_lines=True, skipinitialspace=True, nrows=p.nrows).dropna(axis=0)
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

fig, axs = plt.subplots(nrows=4)
axs[0].set_title('')
axs[0].plot(galvo_sub[abs_time].to_numpy(), galvo_sub[galvo_value].to_numpy(), 'k')
axs[0].plot(beg_first_image_time, 1, 'ro')
axs[0].plot(galvo_sub[abs_time].iloc[peaks[:5]], galvo_sub[galvo_value].iloc[peaks[:5]], 'bo')
axs[0].set_xlabel('Interframe interval (ms)')
axs[0].set_ylabel('Galvo value')
# fig.show()



#* Calculate the interframe interval.
interframe_interval_array = galvo[abs_time].iloc[peaks].diff()
interframe_interval = interframe_interval_array.median()


print('The median of the interframe interval is:', interframe_interval)
print('Min and max interframe interval:', interframe_interval_array.min(), interframe_interval_array.max())


#* Discard a few images at the beginning where the imaging is not good or when we are unsure of the true beginning of the images.
beg_image_to_consider_index = interframe_interval_array.index[np.where(interframe_interval_array == interframe_interval)[0][0] - 1]
beg_image_to_consider_time = galvo[abs_time].iat[beg_image_to_consider_index]

peaks = peaks[peaks >= beg_image_to_consider_index]


#* Number of images to discard at the beginning.
# need to round down
number_images_before_first_image_to_consider = round((beg_image_to_consider_time - beg_first_image_time) / interframe_interval)
# beg_first_image_time = beg_image_to_consider_time - number_images_before_first_image_to_consider * interframe_interval



#* Get info about the images in the multipage tiff with the imaging data.
bytes_header, height, width = fi.get_bytes_header_and_image(images_path)
bytes_header_and_image = bytes_header + height * width * 2

#* Find where we started imaging in the anatomical stack.
number_images = fi.get_number_images(images_path, bytes_header_and_image)
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



# fig, ax = plt.subplots()
axs[1].plot(galvo_sub[abs_time].to_numpy(), galvo_sub[galvo_value].to_numpy(), 'k')
axs[1].plot(galvo_sub[abs_time].to_numpy(), galvo_sub['Frame beg'].to_numpy(), 'ro')
axs[1].plot(np.arange(beg_image_to_consider_time - number_images_before_first_image_to_consider * interframe_interval, beg_image_to_consider_time, interframe_interval), np.ones(number_images_before_first_image_to_consider)*2, 'yo')
axs[1].plot(galvo.iloc[0:5000][abs_time].to_numpy(), galvo.iloc[0:5000]['Frame beg'].to_numpy()*3, 'bo')
axs[1].set_xlabel('Interframe interval (ms)')
axs[1].set_ylabel('Galvo value')


galvo_sub = galvo.iloc[-5000:]

galvo_sub.loc[galvo_sub['Frame beg'].notna(), 'Frame beg'] = 1

# fig, ax = plt.subplots()
axs[2].plot(galvo_sub[abs_time].to_numpy(), galvo_sub[galvo_value].to_numpy(), 'k')
axs[2].plot(galvo_sub[abs_time].to_numpy(), galvo_sub['Frame beg'].to_numpy(), 'ro')
axs[2].set_xlabel('Interframe interval (ms)')
axs[2].set_ylabel('Galvo value')
# fig.show()

del galvo_sub

# fig, axs = plt.subplots()
axs[3].plot(interframe_interval_array, 'k.')
axs[3].set_xlabel('Interframe interval (ms)')
axs[3].set_ylabel('Galvo value')
# fig.show()

fig.savefig(path_home / fish_name / 'Galvo signal and frames.png')
#endregion

#* Read the behavior data.
##   
# region Behavior data
#! reverse data_cols to what we want
# data_cols = x_cols + y_cols + angle_cols
#! behavior = f.read_tail_tracking_data(tracking_path).astype('float32')

# behavior.dtypes
# if (tail := read_tail_tracking_data(data_path)) is None: # type: ignore
# 	return None

# plot_behavior_overview(tail, fish_name, fig_behavior_name)

# #* Look for possible tail tracking errors.
# if tracking_errors(tail, single_point_tracking_error_thr):
# 	return None
# endregion

#* Merge the galvo signal, stim log, behavior camera data and behavior data.
##   
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

	# plt.plot(galvo[abs_time])
	# plt.plot(data[abs_time])

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


#!!!!!!!!!!!!!!!!!!! call function that analyzes the behavior data and returns the relevant columns

behavior = pd.read_pickle(r'C:\\Users\\joaqc\\Desktop\\WIP\\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf\\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf_behavior.pkl')

behavior.rename(columns={'Original frame number' : 'Frame number'}, inplace=True)







data = pd.merge(data, behavior, on='Frame number', how='outer')





#!!!!! DATA = data.copy()
# data = DATA.copy()






min_index, max_index = data['GalvoValue'].dropna().index[[0,-1]]
min_index_, max_index_ = data['Angle of point 15 (deg)'].dropna().index[[0,-1]]

data = data[(data.index >= max(min_index, min_index_)) & (data.index <= min(max_index, max_index_))]

del behavior



data.reset_index(drop=True, inplace=True)
data.rename(columns={abs_time : 'Time (ms)'}, inplace=True)
abs_time = 'Time (ms)'
data[abs_time] -= data[abs_time].iat[0]









data['GalvoValue'].dropna()



from scipy import interpolate

data.iloc[:,0] = data.iloc[:,0] * expected_framerate/predicted_framerate

interp_function = interpolate.interp1d(data.iloc[:,0], data.loc[:,data.columns[1:]], kind='slinear', axis=0, assume_sorted=True, bounds_error=False, fill_value="extrapolate")

data_ = pd.DataFrame(np.arange(data.iat[0,0], data.iat[-1,0]), columns=['Time (frame) [{} FPS]'.format(expected_framerate)])
data_[data.columns[1:]] = interp_function(data_.iloc[:,0])



# endregion

#* Read the imaging data.
##   
# region Imaging data
#* Get the images and align them to data.
#! Do not forget to discard the first images.
#!!!!! images = np.array([get_image_from_tiff(images_path, image_i, bytes_header, height, width) for image_i in range(number_images_before_first_image_to_consider, len(peaks))])
# images_subset_mean = [np.mean(image[-30:-10][-30:-10]).astype('float32') for image in images]
images = np.array([fi.get_image_from_tiff(images_path, image_i, bytes_header, height, width).astype('float32') for image_i in tqdm(range(number_images))])


# images.shape
# endregion

#* Check whether the different pieces of data are aligned.
##   
# region Confirmation of data alingment
images_mean = [image.mean() for image in images]

# Remove all colummns where there is no tracking data and no frame started.
# Really need to convert to dense...
data[[cs,us]] = data[[cs,us]].sparse.to_dense()
data = data.dropna(subset=['Frame number', 'Frame beg', cs, us], how='all')





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

# x0 = 2500
# x1 = 2500

stim_numbers = data.loc[data[us] != 0, us].unique()

stim_numbers = stim_numbers.astype('int')


fig, axs = plt.subplots(stim_numbers.size, 1, figsize=(10, 50), sharex=True, sharey=True)

# ! SHOULD PLOT AS WELL THE BEHAVIOR data

for stim_number_i, stim_number in enumerate(stim_numbers):


	data_ = data.loc[data[us] == stim_number, abs_time]

	us_beg_ = data_.iat[0]

	us_end_ = data_.iat[-1]

	# print(stim_number_i, us_end_ - us_beg_)

	data_plot = data.loc[data[abs_time].between(us_beg_-5000, us_end_+5000)]

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
fig.savefig(path_home / fish_name / 'US and frames alignment.png')


del data_, data_plot

# data.loc[data['Frame beg'].notna()]

data.drop(columns=['Image mean', 'GalvoValue'], inplace=True, errors='ignore')
#endregion

#* Separate the different pieces of data in different dataframes.
##   
# region Separation of different pieces of data
protocol = data[[abs_time, cs, us]].copy()
protocol[['CS', 'US']] = protocol[['CS', 'US']].astype('int')
protocol = protocol[((protocol[cs]!=0) | (protocol[us]!=0))]

behavior = data[[abs_time] + ['Angle of point {} (deg)'.format(i) for i in range(15)]].dropna().rename(columns={'Frame number' : 'Frame number (behavior)'}).copy()
#! behavior = data[[abs_time] + data_cols].dropna().rename(columns={'Frame number' : 'Frame number (behavior)'}).copy()
# behavior_array = xr.DataArray(behavior, coords=[('Time (ms)', all_data.loc[all_data['Frame number'].notna(), :].index), ('parameters', behavior.columns)], dims=['Time (ms)', 'parameters'])
# behavior_array.to_dataframe()

imaging = xr.DataArray(images, coords={'index': ('Time (ms)', data.loc[data['Frame beg'].notna(), :].index), 'Time (ms)': data.loc[data['Frame beg'].notna(), abs_time].to_numpy(), 'x': range(images.shape[1]), 'y': range(images.shape[2])}, dims=['Time (ms)', 'x', 'y'])

imaging.name = 'Imaging data'
# endregion

#* Arrange the data in planes data.
##   
# region Planes data
cs_onset_index = np.array([protocol.loc[protocol[cs] == relevant_cs[i], :].index[0] for i in range(len(relevant_cs))])
len(cs_onset_index)
# print(protocol.to_string())
# index_list = [[i, i+1, i+2*number_imaged_planes, i+2*number_imaged_planes+1] for i in range(0, number_reps_plane_consective * number_imaged_planes, 2)]

# planes_cs_onset_indices = cs_onset_index

len(index_list)

try:
	planes_cs_onset_indices = [cs_onset_index[[j for j in i]] for i in index_list]
except:
	planes_cs_onset_indices = [cs_onset_index]



del cs_onset_index


all_data = []


i = 0
for plane_i, plane_cs_onset_indices in tqdm(enumerate(planes_cs_onset_indices)):

	trials_list = []
	for trial_i, trial_cs_onset_index in enumerate(plane_cs_onset_indices):

		time_start = protocol.loc[trial_cs_onset_index, abs_time] - p.time_bef_cs_onset
		time_end = protocol.loc[trial_cs_onset_index, abs_time] + p.time_aft_cs_onset


		trial_protocol = protocol[protocol[abs_time].between(time_start, time_end)]

		protocol_ = trial_protocol[['Time (ms)', 'CS']]

		protocol_ = protocol_[protocol_['CS'] > 0]

		
		cs_beg = protocol_[protocol_['CS'].ne(protocol_['CS'].shift(periods=1))]
		cs_beg.rename(columns={'CS' : 'CS beg'}, inplace=True)

		cs_end = protocol_[protocol_['CS'].ne(protocol_['CS'].shift(periods=-1))]
		cs_end.rename(columns={'CS' : 'CS end'}, inplace=True)


		protocol_ = trial_protocol[['Time (ms)', 'US']]

		protocol_ = protocol_[protocol_['US'] > 0]
		
		us_beg = protocol_[protocol_['US'].ne(protocol_['US'].shift(periods=1))]
		us_beg.rename(columns={'US' : 'US beg'}, inplace=True)

		us_end = protocol_[protocol_['US'].ne(protocol_['US'].shift(periods=-1))]
		us_end.rename(columns={'US' : 'US end'}, inplace=True)

		trial_protocol = pd.concat([cs_beg, cs_end, us_beg, us_end])

		trial_protocol = trial_protocol.sort_values(by='Time (ms)').fillna(0)
		trial_protocol[['CS beg', 'CS end', 'US beg', 'US end']] = trial_protocol[['CS beg', 'CS end', 'US beg', 'US end']].astype(pd.SparseDtype("int", 0), copy=False)


		#* Filter the images with a median filter.
		trial_images = imaging.loc[time_start : time_end,:,:]
		trial_images.values = ndimage.median_filter(trial_images, size=p.median_filter_kernel, axes=(1,2))

		trials_list.append(c.Trial(i, trial_protocol, behavior[behavior[abs_time].between(time_start, time_end)], trial_images))

		i += 1

	all_data.append(c.Plane(trials_list))

	# break

# del trials_list, protocol_, trial_protocol, trial_cs_onset_index, time_start, time_end, plane_cs_onset_indices, index_list, relevant_cs, predicted_framerate, reference_frame_id, interframe_interval_array, interframe_interval, beg_first_image, beg_first_image_time, peaks, number_peaks, beg_image_to_consider_index, beg_image_to_consider_time, number_images_before_first_image_to_consider, bytes_header, height, width, bytes_header_and_image, number_images, images_mean, stim_numbers, fig, axs, protocol, behavior, images


fig, axs = plt.subplots(len(all_data), len(all_data[0].trials), figsize=(10, 50), squeeze=False)

for i in range(len(all_data)):
	for j in range(len(all_data[0].trials)):

		axs[i,j].imshow(np.mean(all_data[i].trials[j].images, axis=0), vmin=0, vmax=500)
		axs[i,j].set_xticks([])
		axs[i,j].set_yticks([])

fig.set_size_inches(40, 50)
fig.subplots_adjust(hspace=0, wspace=0)

fig.savefig(path_home / fish_name / 'Summary of imaged planes.png')

#endregion



#* Read the anatomical stack.
##   
# region Anatomical stack
anatomical_stack_images = tifffile.imread(anatomy_1_path).astype('float32')
# anatomical_stack_images = tifffile.imread(anatomy_1_filtered_path).astype('float32')

#!!!!!!!!!!!!!!!

# MAYBE SHOULD MEDIAN FILTER EACH FRAME FROM THE ANATOMICAL STACK AND THEN USE THE MEAN
anatomical_stack_images = xr.DataArray(anatomical_stack_images, coords={'index': ('plane_number', range(anatomical_stack_images.shape[0])), 'plane_number': range(anatomical_stack_images.shape[0]), 'x': range(anatomical_stack_images.shape[2]), 'y': range(anatomical_stack_images.shape[1])}, dims=['plane_number', 'y', 'x'])

anatomical_stack_images = ndimage.median_filter(anatomical_stack_images, size=p.median_filter_kernel, axes=(1,2))

#!!!!!! maybe here subtract background and save image with projection

# endregion


#* Create an object with all the data.
all_data = c.Data(all_data, anatomical_stack_images)
# all_data.__dict__.keys()



#* Save the data.
##   
# region Save the data


path_pkl_before_motion_correction = path_home / fish_name / (fish_name + '_before motion correction' + '.pkl')
os.makedirs(os.path.dirname(path_pkl_before_motion_correction), exist_ok=True)

with open(path_pkl_before_motion_correction, 'wb') as file:
	pickle.dump(all_data, file)


print('END')




fig, axs = plt.subplots(len(all_data.planes), len(all_data.planes[0].trials), figsize=(50, 50), squeeze=False)

for i in range(len(all_data.planes)):
	for j in range(len(all_data.planes[0].trials)):

		axs[i,j].imshow(np.mean(all_data.planes[i].trials[j].images, axis=0), vmin=0, vmax=500)
		axs[i,j].set_xticks([])
		axs[i,j].set_yticks([])

fig.set_size_inches(15, 35)
fig.subplots_adjust(hspace=0, wspace=0)

fig.savefig(path_home / fish_name / 'Summary of imaged planes.png')





# #!!!!!!
# trial = all_data.planes[0].trials[0]


# path_ = path_home / fish_name / 'test.h5'
# os.makedirs(os.path.dirname(path_), exist_ok=True)



# #!
# with pd.HDFStore(path_, complevel=4, complib='zlib') as store:

# 	store.append('behavior', trial.behavior, expectedrows=len(trial.behavior), append=False, chunksize=(trial.behavior.shape[0],trial.behavior.shape[1]))


# 	#! store.select(fish.dataset_key(), where=where_to_query)
# # pd.read_hdf(self._path, key=fish.dataset_key(), mode='r', complevel=self._compression_level, complib=self._compression_library)


# trial.images.to_netcdf(path_, mode='a', group='images', format='NETCDF4', engine='netcdf4', encoding={'images': {'zlib': True, 'complevel': 4, chunksize=(trial.images.shape[0],np.ceil(trial.images.shape[1]/8),np.ceil(trial.images.shape[2]/8))}})



# # Save the data as an HDF5 file
# h5_path = path_home / fish_name / (fish_name + '_before_motion_correction.h5')
# os.makedirs(os.path.dirname(h5_path), exist_ok=True)

# with h5py.File(h5_path, 'w') as h5_file:
# 	# Save anatomical stack images with compression
# 	h5_file.create_dataset('anatomical_stack_images', data=anatomical_stack_images, compression="gzip", compression_opts=4)
	
# 	planes_group =  h5_file.create_group(f'planes')

# 	# Save planes data with compression
# 	for plane_i, plane in enumerate(all_data.planes):
# 		plane_group = planes_group.create_group(f'plane_{plane_i}')
		
# 		# Save trials data with compression
# 		for trial_i, trial in enumerate(plane.trials):
# 			trial_group = plane_group.create_group(f'trial_{trial_i}')
# 			trial_group.create_dataset('protocol', data=trial.protocol, compression="gzip", compression_opts=4)
# 			trial_group.create_dataset('behavior', data=trial.behavior, compression="gzip", compression_opts=4, chunks=(trial.behavior.shape[0],trial.behavior.shape[1]))
# 			trial_group.create_dataset('images', data=trial.images, compression="gzip", compression_opts=4, chunks=(trial.images.shape[0],np.ceil(trial.images.shape[1]/8),np.ceil(trial.images.shape[2]/8)))

# # endregion
# 	# except:
# 	# 	pass

# # # Run the next script, where some plots are made.
# # exec(open('2. Motion correction.py').read())