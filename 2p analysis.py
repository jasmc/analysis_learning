import re
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from imagecodecs import imread
from PIL import Image, ImageSequence
from scipy import interpolate, signal
import tifffile
from tqdm import tqdm

import my_functions as f
from my_general_variables import *

pio.templates.default = "plotly_dark"
pd.options.mode.copy_on_write = True



light_percentage_increase_thr = 0.01
average_light_derivative_thr = 3

kernel_size = 3
ddepth = cv2.CV_16S




#! debug
nrows = None
# 100000000

number_rows_read = None


galvo_value_height_threshold = 0.5
galvo_value_distance_threshold = 100
galvo_value_width_threshold = 20


xy_movement_allowed = 0.15  # fraction of the real image


#! this is overwriting the one in my_general_variables.py
cols_to_use_orig = ['FrameID'] + ['x15'] + ['y15'] + ['angle15']
data_cols = ['X 14'] + ['Y 14'] + ['Angle (deg) 14']
angle_name = 'Angle (deg) 14'
angle_cols = [angle_name]

time_experiment_f = frame_id


#* Paths
path_home = Path(r'D:\2024 02_Delay 2p')


#* Definition of functions
#region Definition of functions
def get_bytes_header_and_image(images_path):
	
	byte_number = 0

	# byte_order = np.fromfile(images_path, dtype=np.uint16, count=1, offset=byte_number)[0]
	byte_number += 2

	# arbitrary = np.fromfile(images_path, dtype=np.uint16, count=1, offset=byte_number)[0]
	byte_number += 2

	# IGD1off = np.fromfile(images_path, dtype=np.uint32, count=1, offset=byte_number)[0]
	byte_number += 4

	number_fields = np.fromfile(images_path, dtype=np.uint16, count=1, offset=byte_number)[0].byteswap()
	byte_number += 2

	# tag = np.fromfile(images_path, dtype=np.uint16, count=1, offset=byte_number)[0]
	byte_number += 2

	# field_data = np.fromfile(images_path, dtype=np.uint8, count=6, offset=byte_number)
	byte_number += 6

	width = np.fromfile(images_path, dtype=np.uint32, count=1, offset=byte_number)[0].byteswap()
	byte_number += 4	

	# tag = np.fromfile(images_path, dtype=np.uint16, count=1, offset=byte_number)[0]
	byte_number += 2

	# field_data = np.fromfile(images_path, dtype=np.uint8, count=6, offset=byte_number)
	byte_number += 6

	height = np.fromfile(images_path, dtype=np.uint32, count=1, offset=byte_number)[0].byteswap()
	byte_number += 4

	for n in range(number_fields - 2):
		# np.fromfile(images_path, dtype=np.uint8, count=12, offset=byte_number)
		byte_number += 12

	# next_offset = np.fromfile(images_path, dtype=np.uint32, count=1, offset=byte_number)[0].byteswap()
	byte_number += 4

	# skip resolution data
	# np.fromfile(images_path, dtype=np.uint8, count=16, offset=byte_number)
	byte_number += 16

	# number_pixels = height * width

	# All bytes up to here subatracted by the 2 bytes used to store the byte order.
	bytes_header = number_fields*12+2+4+16

	# Because images are uint16.
	# bytes_image = number_pixels * 2

	# print('bytes_header', bytes_header)

	return bytes_header, height, width

def get_image_from_tiff(images_path, image_i, bytes_header, height, width):

	# 2 comes from the 2 bytes used to store the byte order.
	offset_image = 2 + (image_i) * (bytes_header + height * width * 2) + bytes_header

	image_data = np.fromfile(images_path, dtype=np.uint16, count=height*width, offset=offset_image).byteswap().reshape((height, width))

	return image_data

def get_number_images(images_path, bytes_header_and_image):

	total_tif_size = os.path.getsize(images_path)

	number_images = (total_tif_size - 2) // bytes_header_and_image

	return number_images

def find_plane_in_anatomical_stack(anatomical_stack_images, the_plane_mean_subset_last_images, plane_where_we_are=None):

	#* Handle to the multipage TIFF file with the plane being imaged.
	# the_plane_tiff = tifffile.TiffFile(the_plane_path)

	#! explain
	# the_plane_mean_of_last_images = the_plane_tiff.asarray(slice(-number_repetitions_of_the_plane_to_analyze-1,-1,step_between_repetitions_of_the_plane_to_analyze))
	the_plane_mean_subset_last_images = the_plane_mean_subset_last_images
	# [y_dim:-y_dim, x_dim:-x_dim]
	
	# if plane_where_we_are is not None:
		
	# 	anatomical_stack_images_ = anatomical_stack_images[plane_where_we_are - number_planes_around_the_plane : plane_where_we_are + number_planes_around_the_plane + 1]
		
	# 	first_plane_substack = plane_where_we_are - number_planes_around_the_plane
		
	# else:
	
	anatomical_stack_images_ = anatomical_stack_images

	first_plane_substack = 0
		
	template_matching_results = [cv2.matchTemplate(plane, the_plane_mean_subset_last_images, cv2.TM_CCOEFF_NORMED) for plane in anatomical_stack_images_]

	plane_i = np.argmax([x.max() for x in template_matching_results])

	xy_in_plane = np.argmax(template_matching_results[plane_i][0]), np.argmax(template_matching_results[plane_i][1])
	
	plane_i += first_plane_substack

	# Find the index of the maximum correlation value
	return plane_i, xy_in_plane

def read_camera(camera_path):

	try:
		# start = timer()
		
		# camera = pd.read_csv(str(camera_path), sep='\t', header=0, decimal='.', skiprows=[*range(1,number_frames_discard_beg)])
		camera = pd.read_csv(camera_path, engine='pyarrow', sep=' ', header=0, decimal='.')
		# , na_filter=False
		# dtype={time_experiment_f : 'int64', abs_time : 'int64', ela_time : 'float64'})
		# skipfooter=1
		camera = camera.iloc[:-1,:]

		camera.rename(columns={'FrameID' : frame_id, 'ID' : frame_id}, inplace=True)
		
		# print('Time to read cam.txt: {} (s)'.format(timer()-start))
		
		return camera

	except:

		print('Issues in the camera log file')
		
		return None

def framerate_and_reference_frame(camera):

	# first_frame_absolute_time = camera[abs_time].iloc[0]

	camera = camera.drop(columns=abs_time, errors='ignore')

	camera_diff = camera[ela_time].diff()

	print('Max IFI: {} ms'.format(camera_diff.max()))
	
	# First estimate of the interframe interval, using the median
	ifi = camera_diff.median()
	# camera_diff.iloc[number_frames_discard_beg : ].median()
	print('First estimate of IFI: {} ms'.format(ifi))


	camera_diff_index_correct_IFI = np.where(abs(camera_diff - ifi) <= max_interval_between_frames)[0]

	camera_diff_index_correct_IFI_diff = np.diff(camera_diff_index_correct_IFI)

	reference_frame_id = 0
	last_frame_id = 0

	#* Find a region at the beginning where the IFI from frame to frame does not vary significantly and is similar to the first estimate of the true IFI (ifi).
	for i in range(1, len(camera_diff_index_correct_IFI_diff)):

		if camera_diff_index_correct_IFI_diff[i-1] == 1 and camera_diff_index_correct_IFI_diff[i] == 1:

			reference_frame_id = camera[frame_id].iloc[camera_diff_index_correct_IFI[i] - 1]


			# # first_frame_absolute_time is not None when there is absolute time in the cam file.
			# if first_frame_absolute_time is not None:
			# 	reference_frame_time = first_frame_absolute_time + camera[ela_time].iloc[camera_diff_index_correct_IFI[i] - 1] - camera[ela_time].iloc[0]
			# else:
			# 	reference_frame_time = None

			break

	#* Find a similar region but at the end of the experiment.
	for i in range(len(camera_diff_index_correct_IFI_diff)-1, 0, -1):

		if camera_diff_index_correct_IFI_diff[i-1] == 1 and camera_diff_index_correct_IFI_diff[i] == 1:
			
			last_frame_id = camera[frame_id].iloc[camera_diff_index_correct_IFI[i] - 1]
			#last_frame_time = first_frame_absolute_time + camera[time].iloc[camera_diff_index_right_IFI[i] - 1] - camera[time].iloc[0]

			break


	#* Second estimate of the interframe interval, using the mean, and assuming there is no increasing accumulation of frames in the buffer during the experiment; Only the region between the two frames identified in the previous two for loops is considered.
	ifi = camera_diff.iloc[reference_frame_id - camera[frame_id].iloc[0] : last_frame_id - camera[frame_id].iloc[0]].mean()

	print('Second estimate of IFI: {} ms'.format(ifi))
	predicted_framerate = 1000 / ifi
	print('Estimated framerate: {} FPS'.format(predicted_framerate))


	# Lost_frames = lost_frames(camera, camera_diff, ifi, fish_name, fig_camera_name)

	return predicted_framerate, reference_frame_id

def read_tail_tracking_data(data_path):

	# Angles in data come in radians.

	try:
		
		# start = timer()
		
		data = pd.read_csv(data_path, engine='pyarrow', sep=' ', usecols=cols_to_use_orig, header=0, decimal='.', na_filter=False, names=[frame_id]+data_cols)
		# dtype=dict(zip(cols_to_use_orig, ['int64'] + ['float32']*len(cols_to_use_orig))))
		# skipfooter=1
		data = data.iloc[:-1,:]
		
		#* Right now, pyarrow engine ignores renaming when opening the csv.
		data.rename(columns=dict(zip(cols_to_use_orig, [frame_id] + data_cols)), inplace=True)

		# print('Time to read tail tracking .txt: {} (s)'.format(timer()-start))


		#? maybe before this was necessary because "decimal" in pd.read_csv was set to ",".
		#* Even if decimal separator is wrong, this will correct it.
		# data.iloc[:,1:] = data.iloc[:,1:].astype('float32')

		#* Convert tail tracking data from radian to degree
		data.loc[:,angle_cols] *= (180/np.pi)
		
		return data

	except:
		
		#TODO
		# f.save_info(protocol_info_path, self.metadata.name, 'Tail tracking might be corrupted!')
		print('Issues in tail tracking data')
		return None

def plot_behavior_overview(data, fish_name, fig_behavior_name):
	# data containing tail_angle.

	# mask_frames = np.ones(number_frames + round(60*framerate), dtype=bool)
	# mask_frames[:: round(framerate * 0.5)] = False
	# mask_frames[0] = False
	
	# rows_to_skip = np.arange(number_frames + round(60*framerate))
	# rows_to_skip = rows_to_skip[mask_frames]

	# start = timer()

	# overall_data = pd.read_csv(data, sep=' ', header=0, usecols=cols, skiprows=rows_to_skip, decimal=',')
	# overall_data = overall_data.astype('float32')

	# print(timer() - start)
	plt.figure(figsize=(30, 15))
	plt.plot(data[frame_id]/expected_framerate/60/60, data[tail_angle], 'black')
	plt.xlabel('Time (h)')
	plt.ylabel('Tail end angle (deg)')
	plt.suptitle('Behavior overview\n' + fish_name)
	# plt.show()
	# plt.legend(frameon=False, loc='upper center', ncol=2)
	plt.savefig(fig_behavior_name, dpi=100, bbox_inches='tight')
	plt.close()

def tracking_errors(data, single_point_tracking_error_thr = single_point_tracking_error_thr):

	if ((a := data.loc[:, angle_cols].abs().max()) > single_point_tracking_error_thr).any():
		print('Possible tracking error; max(abs(angle of individual point)): ')
		print(a)

		return True

	elif data.iloc[:,1:].isna().to_numpy().any():
		print('Possible tracking failures; there are NAs in data')

		return True

	else:
		return False

def merge_camera_with_data(data, camera):

	data = pd.merge_ordered(data, camera, on=frame_id, how='inner')

	data[frame_id] -= data[frame_id].iat[0]

	return data

def interpolate_data(data, predicted_framerate, expected_framerate=expected_framerate):
	# expected_framerate is the framerate to which data is interpolated. So, output data is as if it had been acquired at the expected_framerate (700 FPS when I wrote this).

	data_ = data.copy()

	#* Interpolate tail tracking data to the expected framerate.

	data_[frame_id] *= expected_framerate/predicted_framerate

	data_.rename(columns={frame_id : time_experiment_f}, inplace=True)

	interp_function = interpolate.interp1d(data_[time_experiment_f], data_.drop(columns=time_experiment_f), kind='slinear', axis=0, assume_sorted=True, bounds_error=False, fill_value="extrapolate")

	data = pd.DataFrame(np.arange(data_[time_experiment_f].iat[0], data_[time_experiment_f].iat[-1]), columns=[time_experiment_f])

	data[data_.drop(columns=time_experiment_f).columns] = interp_function(data[time_experiment_f])

	return data

def read_protocol(protocol_path):

	#* Read protocol file.
	if Path(protocol_path).exists():
		# protocol = pd.read_csv(str(protocol_path), sep=' ', header=0, names=['Type', beg, end], usecols=[0, 1, 2], index_col=0)
		protocol = pd.read_csv(protocol_path, engine='pyarrow', sep=' ', header=0, decimal='.', names=['Type', beg, end])
		# dtype={'Type' : 'str', beg : 'int', end : 'int'})

	else:
		print('The stim log file does not exist')
		return None

	#* Were the stimuli timings not saved?
	if protocol.empty:
		print('The stim log file is empty')
		return None

	if protocol.iloc[0,0] == 0:
		print('The stim log file is currupted')
		return None

	#* Right now, pyarrow engine ignores renaming when opening the csv.
	protocol.rename(columns={'Beg' : beg, 'End' : end}, inplace=True)
	protocol['Type'] = protocol['Type'].replace({'Cycle' : cs, 'Reinforcer' : us})
	protocol.sort_values(by=beg, inplace=True)
	protocol = protocol.set_index('Type')

	return protocol

def lost_stim(number_cycles, number_reinforcers, expected_number_cs_trials, expected_number_us_trials):

	if number_cycles < expected_number_cs_trials:

		# save_info(protocol_info_path, fish_name, 'Not all CS! Stopped at CS {} ({}).'.
		# format(number_cycles, id_debug))
		print('Not all CS!')

	if number_reinforcers < expected_number_us_trials:
		
		# save_info(protocol_info_path, fish_name, 'Not all US! Stopped at US {} ({}).'.format(number_reinforcers, id_debug))
		print('Not all US!')
			
	if number_cycles < expected_number_cs_trials or number_reinforcers < expected_number_us_trials:

		return True
	
	else:
		return False

def protocol_info(protocol):

	#* Count the number of cycles, trials, blocks and bouts.
	# Using len() just in case these is a single element.
	# number_cs = len(protocol.loc['Cycle', beg])

	if protocol.index.isin([us]).any():
		# number_us = len(protocol.loc[us, beg])

		us_beg = protocol.loc[us, beg]
		us_end = protocol.loc[us, end]
		us_dur = (us_end - us_beg).to_numpy() # in ms
		us_isi = (us_beg[1:] - us_end[:-1]).to_numpy() / 1000 / 60 # min
	else:
		# number_us = 0

		us_dur = None
		us_isi = None

	# habituation_duration = protocol.iloc[0,0] / 1000 / 60 # min

	cs_beg = protocol.loc[cs, beg]
	cs_end = protocol.loc[cs, end]
	cs_dur = (cs_end - cs_beg).to_numpy() # in ms
	cs_isi = (cs_beg[1:] - cs_end[:-1]).to_numpy() / 1000 / 60 # min


	return cs_dur, cs_isi, us_dur, us_isi

def plot_protocol(cs_dur, cs_isi, us_dur, us_isi, fish_name, fig_protocol_name):

	plt.figure(figsize=(14,14))
	plt.plot(np.arange(1, len(cs_isi) + 1), cs_isi, label='inter-cs interval\nmin int.=' + str(round(np.amin(cs_isi)*60,1)) + ' s\n' + 'cs min dur=' + str(round(np.amin(cs_dur)/1000,3)) + ' s\n' + 'cs max dur=' + str(round(np.amax(cs_dur)/1000,3)) + ' s')
	
	plt.plot(np.arange(5, 4+len(us_isi)+1), us_isi, label='inter-us interval\nmin int.=' + str(round(np.amin(us_isi)*60,1)) + ' s\n' + 'us min dur=' + str(round(np.amin(us_dur)/1000,3)) + 's\n' + 'us max dur='+ str(round(np.amax(us_dur)/1000,3)) + ' s')
	plt.xlabel('Trial number')
	plt.ylabel('ISI (min)')
	plt.ylim(0, 10)
	plt.legend(frameon=False, loc='upper center', ncol=2)
	plt.suptitle('Summary of protocol\n' + fish_name)
	plt.savefig(fig_protocol_name, dpi=100, bbox_inches='tight')
	plt.close()

def identify_trials(data, protocol):

	data[[cs, us]] = [0, 0]

	for cs_us in [cs, us]:

		protocol_sub = protocol.loc[cs_us, [beg, end]].to_numpy()

		for i, p in enumerate(protocol_sub):
			
			data.loc[data[abs_time].between(p[0], p[1]), cs_us] = i + 1

	data = data.set_index(abs_time)

	data.loc[:, data_cols] = data.loc[:, data_cols].interpolate(kind='slinear')

	#! data = data.reset_index(drop=True).dropna()
	data = data.reset_index().dropna()

	data[time_experiment_f] = data[time_experiment_f].astype('int64')

	data[[cs, us]] = data[[cs, us]].astype('Sparse[int16]')

	#* Fix dtypes.
	for cs_us in [cs, us]:

		data[cs_us] = data.loc[:, cs_us].astype(pd.CategoricalDtype(categories=data[cs_us].unique(), ordered=True))

	return data

#endregion			






fish_names = [folder.stem for folder in path_home.iterdir() if folder.is_dir()]
fish_names.remove('Behavior')


# for fish_name in fish_names:
# fish_name = r'20240228_01_delay_2p-1_mitfaMinusMinus,elavl3H2BCaMP6s_7dpf'

fish_name = r'20240314_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'
# "20240404_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_7dpf"
# '20240327_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_7dpf'

# '20240313_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_5dpf'
# '20240321_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'
# '20240311_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_5dpf'
# fish_names[-6]
#! '20240304_01_delay_2p-1_mitfaMinusMinus,Gal4elavl3GCaMP6s_6dpf' is good

behavior_path = Path(r'D:\2024 02_Delay 2p\Behavior')
imaging_path = Path(r'D:\2024 02_Delay 2p') / fish_name / 'Imaging'

# behavior_path = Path(r'E:\2024 03_Delay 2-P multiple planes\Behavior')
# imaging_path = Path(r'E:\2024 03_Delay 2-P multiple planes') / fish_name / 'Imaging'

protocol_path = behavior_path / (fish_name + '_stim control.txt')
camera_path = behavior_path / (fish_name + '_cam.txt')
tracking_path = behavior_path / (fish_name + '_mp tail tracking.txt')


galvo_path = imaging_path / 'signalsfeedback.xls'
images_path = imaging_path / (fish_name + '_green.tif')

anatomy_1_path = Path(r'D:\2024 02_Delay 2p') / fish_name / 'Anatomical stack 1.tif'



# data_folder = r"D:\2024 02_Delay 2p"
# data_folder = Path(data_folder) / fish_name





camera = read_camera(camera_path)
camera[abs_time] = camera[abs_time].astype('float64')


#* Estimate the true framerate.
predicted_framerate, reference_frame_id = framerate_and_reference_frame(camera)


camera = camera.drop(columns=ela_time)

#* Discard frames that will not be used (in camera and hence further down).
# The calculated interframe interval before the reference frame is variable. Discard what happens up to then (also achieved by using how='inner' in merge_camera_with_data).
camera = camera[camera[frame_id] >= reference_frame_id]

#! reverse data_cols to what we want
#! #* Open tail tracking data.
data = read_tail_tracking_data(tracking_path)

# if (data := read_tail_tracking_data(data_path)) is None: # type: ignore
# 	return None

# plot_behavior_overview(data, fish_name, fig_behavior_name)

# #* Look for possible tail tracking errors.
# if tracking_errors(data, single_point_tracking_error_thr):
# 	return None

#! #* Add information about the time of each frame to data.
data = merge_camera_with_data(data, camera)
# data = camera

#* Fix abs_time so that the time of each frame becomes closer to the time at which the frames were acquired by the data and not when they were caught by the computer.
# The delay between acquiring and catching the frame is unknown and therefore disregarded.
data[abs_time] = np.linspace(data[abs_time].iat[0], data[abs_time].iat[0] + len(data) * (1000 / predicted_framerate), len(data))

#! Need to join with galvo before doing this.
#! #* Interpolate data to the expected framerate.
# data = interpolate_data(data, predicted_framerate)

#* Open the stim log.
protocol = read_protocol(protocol_path)
	# return None

#TODO replace by exp_var.experiments_info[Experiment.name]
# if lost_stim(len(protocol.loc[cs,:]), len(protocol.loc[us,:]), Experiment.get_experiment_info(self.metadata.experiment, 'parts')['trials'][cs]['elements'][1], Experiment.get_experiment_info(self.metadata.experiment, 'parts')['trials'][us]['elements'][1]):
# 	print('Experiment did not run until the end')

# cs_dur, cs_isi, us_dur, us_isi = protocol_info(protocol)

# #* Plot overview of the experimental protocol actually run
# plot_protocol(cs_dur, cs_isi, us_dur, us_isi, fish_name, fig_protocol_name)

#* Identify the stimuli, trials of the experiment.
data = identify_trials(data, protocol)

# plt.plot(data[abs_time])








#* Read data.
galvo = pd.read_csv(galvo_path, sep='\t', decimal=',', usecols=[0,1], names=[abs_time, 'GalvoValue'], dtype={'GalvoValue':'float64'}, parse_dates=[abs_time], date_format=r'%d/%m/%Y  %H:%M:%S,%f', skip_blank_lines=True, skipinitialspace=True, nrows=nrows).dropna(axis=0)
galvo = galvo.reset_index(drop=True)
#* Convert the time in galvo to unixtime in ms
galvo[abs_time] = galvo[abs_time].astype('int64') / 10**6




plt.plot(galvo[abs_time])

plt.plot(data[abs_time])





# (first_timepoint_data - first_timepoint_galvo) / 60000



#* To align the galvo signals to the respective frames with need to consider at least from the beginning of the galvo signal.
# if (first_timepoint_galvo := galvo[abs_time].iat[0]) <= (first_timepoint_data := data[abs_time].iat[0]):

# 	galvo = galvo[galvo[abs_time] >= first_timepoint_data]
# 	first_timepoint_galvo = galvo[abs_time].iat[0]

# plt.plot(galvo[abs_time])

first_timepoint = galvo[abs_time].iat[0]

data[abs_time] -= first_timepoint
galvo[abs_time] -= first_timepoint

data = data[data[abs_time] >= 0]
galvo = galvo[galvo[abs_time] <= data[abs_time].iat[-1]]



# plt.plot(galvo[abs_time])
# plt.plot(data[abs_time])

# #region Estimate tracking framerate.
# first_frame_absolute_time = data[abs_time].iloc[0]

# data_ = data.drop(columns=abs_time, errors='ignore')

# camera_diff = data_[ela_time].diff()

# print('Max IFI: {} ms'.format(camera_diff.max()))

# # First estimate of the interframe interval, using the median
# ifi = camera_diff.median()
# # camera_diff.iloc[number_frames_discard_beg : ].median()
# print('First estimate of IFI: {} ms'.format(ifi))


# camera_diff_index_correct_IFI = np.where(abs(camera_diff - ifi) <= max_interval_between_frames)[0]

# camera_diff_index_correct_IFI_diff = np.diff(camera_diff_index_correct_IFI)

# reference_frame_id = 0

# #* Find a region at the beginning where the IFI from frame to frame does not vary significantly and is similar to the first estimate of the true IFI (ifi).
# for i in range(1, len(camera_diff_index_correct_IFI_diff)):

# 	if camera_diff_index_correct_IFI_diff[i-1] == 1 and camera_diff_index_correct_IFI_diff[i] == 1:

# 		reference_frame_id = data_['ID'].iloc[camera_diff_index_correct_IFI[i] - 1]


# 		# # first_frame_absolute_time is not None when there is absolute time in the cam file.
# 		# if first_frame_absolute_time is not None:
# 		# 	reference_frame_time = first_frame_absolute_time + data[ela_time].iloc[camera_diff_index_correct_IFI[i] - 1] - data[ela_time].iloc[0]
# 		# else:
# 		# 	reference_frame_time = None

# 		break

# #* Find a similar region but at the end of the experiment.
# for i in range(len(camera_diff_index_correct_IFI_diff)-1, 0, -1):

# 	if camera_diff_index_correct_IFI_diff[i-1] == 1 and camera_diff_index_correct_IFI_diff[i] == 1:
		
# 		last_frame_id = data_['ID'].iloc[camera_diff_index_correct_IFI[i] - 1]
# 		#last_frame_time = first_frame_absolute_time + data[time].iloc[camera_diff_index_right_IFI[i] - 1] - data[time].iloc[0]

# 		break


# #* Second estimate of the interframe interval, using the mean, and assuming there is no increasing accumulation of frames in the buffer during the experiment; Only the region between the two frames identified in the previous two for loops is considered.
# ifi = camera_diff.iloc[reference_frame_id - data_['ID'].iloc[0] : last_frame_id - data_['ID'].iloc[0]].mean()

# print('Second estimate of IFI: {} ms'.format(ifi))
# predicted_framerate = 1000 / ifi
# print('Estimated framerate: {} FPS'.format(predicted_framerate))
# #endregion




# #* Fix abs_time.
# reference_abs_time = data.loc[data['ID'] == reference_frame_id, abs_time].iat[0]
# a = data.loc[data['ID'] > reference_frame_id, abs_time]

# data.loc[data['ID'] > reference_frame_id, abs_time] = np.cumsum(np.ones(len(a)) / predicted_framerate) + reference_abs_time


# data[abs_time].diff().plot()
# data[ela_time].diff().plot()



# #* Merge protocol with data.
# protocol[[beg, end]] = protocol[[beg, end]].astype('float')

# for cs_us in [cs, us]:

# 	for beg_end in [beg, end]:

# 		beg_end_name = ' beg' if beg_end == beg else ' end'

# 		p = pd.DataFrame(protocol.loc[protocol['Type']==cs_us, beg_end]).rename(columns={beg_end : abs_time})

# 		p[cs_us + beg_end_name] = np.arange(1, 1+len(p))

# 		data = pd.merge_ordered(data, p, on=abs_time, how='outer').drop_duplicates(abs_time, keep='first')

# # del protocol_, p






#* Remove consecutive duplicates...
galvo = galvo[galvo[galvo_value].ne(galvo[galvo_value].shift())]


galvo = galvo.reset_index(drop=True)


# plt.plot(galvo[galvo_value])


#* Find the beginning of the frames.
beg_first_image = signal.find_peaks(galvo[galvo_value], height=0.5)[0][0]
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



# GALVO = galvo.copy()
# DATA = data.copy()

# data=DATA.copy()
# galvo=GALVO.copy()








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









#* Get the images and align them to data.
#! Do not forget to discard the first images.
#!!!!! images = np.array([get_image_from_tiff(images_path, image_i, bytes_header, height, width) for image_i in range(number_images_before_first_image_to_consider, len(peaks))])
# images_subset_mean = [np.mean(image[-30:-10][-30:-10]).astype('float32') for image in images]
images = np.array([get_image_from_tiff(images_path, image_i, bytes_header, height, width).astype('float32') for image_i in range(number_images)])

images.shape















images_subset_mean = [image.mean() for image in images]

images_subset_mean_top = [image[ : round(image.shape[0]/2)] [ -round(image.shape[1]/2) : ].mean() for image in images]
images_subset_mean_bottom = [image[ : round(image.shape[0]/2) ] [ : round(image.shape[1]/2) ].mean() for image in images]


plt.plot(images_subset_mean_top)
# plt.plot(images_subset_mean_bottom)



# images_subset_mean.flatten()
# images_subset_mean.shape







data.loc[data['Frame beg'].notna(),:]
number_images






#* Merge galvo with data.
# data = pd.merge_ordered(data, galvo.drop(columns=galvo_value), on=abs_time, how='outer')
data = pd.merge_ordered(data, galvo, on=abs_time, how='outer')

# del galvo
# data = pd.merge_ordered(data, galvo, on=abs_time, how='outer')



#!!!!!!!!!!!!!!!!! Remove all colummns where there is no tracking data and no frame started.
data = data.dropna(subset=[frame_id, 'Frame beg', cs, us], how='all')

# data[[frame_id, 'Frame beg']] = data[[frame_id, 'Frame beg']].fillna(0)

plt.plot(data[abs_time])


# DATA_ = data.copy()




data['Image mean'] = np.nan

data.loc[data['Frame beg'].notna(), 'Image mean'] = images_subset_mean[:len(data[data['Frame beg'].notna()])]
# data.loc[data['Frame beg'].notna(), 'Image mean'] = images_subset_mean_top[:len(data[data['Frame beg'].notna()])]

plt.plot(data.loc[data['Frame beg'].notna(), 'Image mean'])
# len(peaks)
# len(images_subset_mean_top)
# number_images


data = data.reset_index(drop=True)



#!
data[[cs, us]] = data[[cs, us]].fillna(0)

data[cs].cat.remove_unused_categories()
data[us].cat.remove_unused_categories()




data[us].unique()

x0 = 25000
x1 = 25000

# np.diff(protocol.loc[us, beg].to_numpy())

stim_numbers = data.loc[data[us] > 0, us].unique()

# data = data.dropna(subset=[cs, us], how='any')

# np.diff(protocol.loc[us, :].to_numpy()[-1])


# data[(data[us] == stim_number)]





# data[(data[us].astype('float').diff() > 0)]

# data[(data[us].astype('float').diff().shift(-1) < 0)]




# data[(data[us] == stim_number) & (data[us].astype('float').diff() > 0)]

# data[(data[us] == stim_number) & (data[us].astype('float').diff().shift(-1) < 0)]

# plt.plot(data[galvo_value], 'k.')



fig, axs = plt.subplots(stim_numbers.size-1, 1, figsize=(10, 50), sharex=True, sharey=True)

# fig, axs = plt.subplots(3, 1, figsize=(10, 50))

# fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)

for stim_number_i, stim_number in enumerate(stim_numbers[1:]):

	us_beg_ = data.loc[(data[us] == stim_number) & (data[us].astype('float').diff() > 0), abs_time].to_numpy()[0]

	us_end_ = data.loc[(data[us] == stim_number) & (data[us].astype('float').diff().shift(-1) < 0), abs_time].to_numpy()[0]


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



import h5py
with h5py.File(r"I:\20240314_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf\Data.hdf", "w") as file:
	file.create_dataset("data", data=data.drop(columns=data.columns[[2,3,4]]), compression="gzip")








# Save images as a multi-page TIFF
tifffile.imwrite(r"I:\20240314_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf\After removing dark images at the end.tif", images[:len(data.loc[data['Frame beg'].notna(),:])])

images_=images[:len(data.loc[data['Frame beg'].notna(),:])]
import h5py
with h5py.File(r"I:\20240314_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf\After removing dark images at the end.hdf", "w") as file:
	file.create_dataset("images", data=images_, compression="gzip")



# images_ = tifffile.imread(r"I:\20240314_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf\After removing dark images at the end.tif")


# with pd.HDFStore(r"I:\20240314_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf\After removing dark images at the end.hdf", complevel=4, complib: str='zlib') as store:
	
# 	store.append('data', images_, expectedrows=images_.shape, append=False)

# 	store.get_storer(fish.dataset_key()).attrs['metadata'] = fish.metadata._asdict()

# images_.shape


plt.plot(data[abs_time], data[abs_time])



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# images = np.array([get_image_from_tiff(images_path, image_i, bytes_header, height, width).astype('float32') for image_i in range(100, 30000)])


#! Piece of 3 min
images_subset = images[5000:5000+180*3]








#! def



top = np.mean(np.mean(images_subset[:, 0:50, :], axis=1), axis=1)
bottom = np.mean(np.mean(images_subset[:, -50:, :], axis=1), axis=1)
front = np.mean(np.mean(images_subset[:, :, -100:], axis=1), axis=1)
back = np.mean(np.mean(images_subset[:, :, 0:100], axis=1), axis=1)

all = np.mean(np.mean(images_subset, axis=1), axis=1)

light_percentage_increase = np.abs(all - np.mean(all))/np.mean(all)

# plt.plot(top)
# plt.plot(bottom)
# plt.plot(front)
# plt.plot(back)
# plt.plot(all)
# plt.plot(light_percentage_increase)

# plt.plot(np.abs(np.diff(top)))
# plt.plot(np.diff(bottom))
# plt.plot(np.diff(front))
# plt.plot(np.diff(back))
# plt.plot(np.diff(all))



#* Discard based on overall light (too low or too high)
mask_good_frames = light_percentage_increase < light_percentage_increase_thr

#* And also discard based on the derivative
mask_good_frames = mask_good_frames & ([False] + list((np.abs(np.diff(top)) < average_light_derivative_thr) & (np.abs(np.diff(bottom)) < average_light_derivative_thr) & (np.abs(np.diff(front)) < average_light_derivative_thr) & (np.abs(np.diff(back)) < average_light_derivative_thr) & (np.abs(np.diff(all) < average_light_derivative_thr))))

images_subset_corrected = images_subset[mask_good_frames,:,:]
plt.imshow(np.mean(images_subset_corrected, axis=0))

images_subset_corrected_average = np.mean(images_subset_corrected, axis=0)

# top = np.mean(np.mean(images_subset_corrected[:, 0:50, :], axis=1), axis=1)
# bottom = np.mean(np.mean(images_subset_corrected[:, -50:, :], axis=1), axis=1)
# front = np.mean(np.mean(images_subset_corrected[:, :, -100:], axis=1), axis=1)
# back = np.mean(np.mean(images_subset_corrected[:, :, 0:100], axis=1), axis=1)

# all = np.mean(np.mean(images_subset_corrected, axis=1), axis=1)




#! def
images_subset_corrected_average = images_subset_corrected_average.astype('uint8')


#* Remove noise by blurring with a Gaussian filter
src = cv2.GaussianBlur(images_subset_corrected_average, (3, 3), 0)
src.dtype

#* Apply Laplace function
dst = cv2.Laplacian(src, ddepth, ksize=kernel_size)


plt.imshow(dst)












images = images_original[::10]


#* Anatomy
# original_anatomy=np.sum(images,0)

anatomical_stack_images = tifffile.imread(anatomy_1_path)
# anatomical_stack_images.shape


#* Calculate the dimensions of the imaged plane to crop out before template-matching.
_, y_dim, x_dim = np.array(anatomical_stack_images.shape)

x_dim = int(x_dim * xy_movement_allowed/2)
y_dim = int(y_dim * xy_movement_allowed/2)



#* Find the plane in the anatomical stack
initial_imaging_plane = np.mean(images[10:110], axis=0).astype('float32')

# anatomical_stack_images[initial_plane].max()
# initial_imaging_plane.max()


initial_plane, xy_in_plane = find_plane_in_anatomical_stack(anatomical_stack_images, initial_imaging_plane[y_dim:-y_dim, x_dim:-x_dim])

plt.imshow(anatomical_stack_images[initial_plane], cmap='gray')
plt.imshow(initial_imaging_plane, cmap='gray')







#* Phase cross-correlation to measure motion of each frame
original_anatomy = anatomical_stack_images[initial_plane]
frames = images

from skimage.registration import phase_cross_correlation
import math


Xs=np.zeros(np.shape(images)[0])
Ys=np.zeros(np.shape(images)[0])
total_motion=np.zeros(np.shape(images)[0])
for i in tqdm(range(frames.shape[0])):
	X=phase_cross_correlation(original_anatomy, frames[i,:,:], upsample_factor=10, space='real')
	Xs[i]=X[0][0]
	Ys[i]=X[0][1]
	total_motion[i]=math.sqrt(Xs[i]*Xs[i]+Ys[i]*Ys[i])



plt.figure()
plt.subplot(1,2,1)
plt.plot(total_motion)
plt.subplot(1,2,2)
plt.scatter(Xs-0.01+0.02*np.random.rand(Xs.shape[0]),Ys-0.01+0.02*np.random.rand(Xs.shape[0]),s=0.5)
plt.show()





#! Joaquim
# Discard frames with too much motion
images[np.where(total_motion > 100)[0]] = np.zeros(images.shape[1:])





from scipy.ndimage import shift

aligned_frames=np.zeros(frames.shape)
for i in tqdm(range(frames.shape[0])):
	aligned_frames[i,:,:]=shift(frames[i,:,:], (Xs[i],Ys[i]), output=None, order=3, mode='constant', cval=0.0, prefilter=True)
#   the commented lines below check that the shift was performed in the correct direction	
#   X=phase_cross_correlation(anatomy, frames[i,:,:] ,upsample_factor=10, space='real')
#   print(X)
#   Y=phase_cross_correlation(anatomy, aligned_frames[i,:,:] ,upsample_factor=10, space='real')
#   print(Y)

aligned_anatomy=np.sum(aligned_frames,0)
plt.figure()
plt.imshow(aligned_anatomy, cmap='gray')







from skimage.registration import phase_cross_correlation
import math
Xs2=np.zeros(np.shape(images)[0])
Ys2=np.zeros(np.shape(images)[0])
total_motion2=np.zeros(np.shape(images)[0])
for i in tqdm(range(frames.shape[0])):
    X=phase_cross_correlation(aligned_anatomy, frames[i,:,:] ,upsample_factor=10, space='real')
    Xs2[i]=X[0][0]
    Ys2[i]=X[0][1]
    total_motion2[i]=math.sqrt(Xs2[i]*Xs2[i]+Ys2[i]*Ys2[i])

plt.figure()
plt.plot(total_motion)
plt.plot(total_motion2)
plt.show()








from scipy.ndimage import shift
final_frames=np.zeros(frames.shape)
for i in tqdm(range(frames.shape[0])):
    final_frames[i,:,:]=shift(frames[i,:,:], (Xs2[i],Ys2[i]), output=None, order=3, mode='constant', cval=0.0, prefilter=True)
final_anatomy=np.sum(final_frames,0)


plt.figure()
plt.subplot(3,3,1)
plt.imshow(original_anatomy, cmap='gray')
plt.subplot(3,3,2)
plt.imshow(aligned_anatomy, cmap='gray')
plt.subplot(3,3,3)
plt.imshow(final_anatomy)
plt.subplot(3,3,4)
plt.imshow(original_anatomy[10:-10,10:-10], cmap='gray')
plt.subplot(3,3,5)
plt.imshow(aligned_anatomy[10:-10,10:-10], cmap='gray')
plt.subplot(3,3,6)
plt.imshow(final_anatomy[10:-10,10:-10])
plt.subplot(3,3,7)
plt.imshow(original_anatomy[130:150,110:130], cmap='gray')
plt.subplot(3,3,8)
plt.imshow(original_anatomy[130:150,110:130]-final_anatomy[130:150,110:130], cmap='gray')
plt.subplot(3,3,9)
plt.imshow(final_anatomy[130:150,110:130], cmap='gray')









from scipy.stats import pearsonr

correlation_map=np.zeros(final_anatomy.shape)
for i in tqdm(range(final_anatomy.shape[0])):
    if i>0 and i<(final_anatomy.shape[0]-1):
        for j in range(final_anatomy.shape[1]):
            if j>0 and j<(final_anatomy.shape[1]-1):
                this_pixel=np.squeeze(final_frames[:,i,j])
                surr_pixels=np.squeeze(np.sum(np.sum(np.squeeze(final_frames[:,i-1:i+2,j-1:j+2]),2),1))-this_pixel
                C, _ = pearsonr(this_pixel, surr_pixels)
                correlation_map[i,j]=C
original_correlation_map=np.copy(correlation_map)   

plt.figure()
plt.subplot(1,2,1)
plt.imshow(final_anatomy)
plt.subplot(1,2,2)
plt.imshow(correlation_map)
plt.show()













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
    from skimage import morphology

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







Nrois=300
all_traces=np.zeros((Nrois,aligned_frames.shape[0]))
all_rois=np.zeros(original_correlation_map.shape)
used_pixels=np.zeros(original_correlation_map.shape)
original_correlation_map[:5,:]=0
original_correlation_map[:,:5]=0
original_correlation_map[-5:,:]=0
original_correlation_map[:,-5:]=0
correlation_map=np.copy(original_correlation_map)


for i in tqdm(range(Nrois)):
    this_roi3,this_roi_trace,N,this_correlation_map=next_roi(correlation_map, final_frames, 0.4,150)
    all_traces[i,:]=this_roi_trace
    all_rois=all_rois+(i+1)*this_roi3
    used_pixels=used_pixels+this_roi3
    correlation_map[all_rois>0]=0


from scipy.stats import zscore

fig,(ax1,ax2,ax3,ax4)= plt.subplots(1,4)

ax1 = plt.subplot(121)
img=ax1.imshow(zscore(all_traces, 1), aspect="auto", vmin=-3, vmax=3, cmap="RdBu_r")
ax1.set_ylabel("trace ROI number")
ax1.set_xlabel("frame number")
fig.colorbar(img,ax=ax1)
ax2 = plt.subplot(322)
ax2.imshow(all_rois)
ax3 = plt.subplot(324)
ax3.imshow(correlation_map)
ax4 = plt.subplot(326)
ax4.imshow(original_correlation_map)
plt.show()
fig.tight_layout()






#* CS average




images = np.array(images)

# images[:10].shape
# np.mean(images[:10], axis=0).shape

images_subset_mean_cs = np.mean(images[:10], axis=0).astype('float32')




stim_number = 50


us_beg_index = data[data[cs_beg] == stim_number].index[0]
us_end_index = data[data[cs_end] == stim_number].index[0]


x0 = 5000
x1 = 5000

data_CS_bef = data.iloc[us_beg_index-10000:us_beg_index]
data_CS_aft = data.iloc[us_beg_index:us_end_index-2000]


image_indices_CS_bef = data_CS_bef.loc[data_CS_bef['Frame beg']!=0, 'Frame beg'].to_numpy()
image_indices_CS_aft = data_CS_aft.loc[data_CS_aft['Frame beg']!=0, 'Frame beg'].to_numpy()


image_bef_mean = np.mean(images[image_indices_CS_bef], axis=0).astype('float32')
image_aft_mean = np.mean(images[image_indices_CS_aft], axis=0).astype('float32')

plt.imshow(image_bef_mean)
		#	, cmap='gray')
plt.imshow(image_aft_mean)


plt.imshow(image_aft_mean - image_bef_mean, cmap='gray')


plt.imshow(images[-1])
images.shape

# USbeg = data__.loc[data__[us_beg]==stim_number, abs_time].to_numpy()[0]
# USend = data__.loc[data__[us_end]==stim_number, abs_time].to_numpy()[0]


# USend - USbeg

time = data_plot[abs_time].to_numpy()

# plt.plot(time, data__[galvo_value]], 'k')
# plt.plot(time, data__['Frame beg'], 'bo')
plt.plot(time, data_plot['Image mean'], 'k.')
plt.axvline(x=USbeg, color='g', linestyle='--')
plt.axvline(x=USend, color='r', linestyle='--')
# plt.plot(time, data__[us_beg], 'yo')
# plt.plot(time, data__[us_end], 'mo')


# data__.plot(x=abs_time, y=['Frame beg', 'Image mean', us_beg, us_end], ls='.')













plt.plot(images_subset_mean)

images.shape
# images = np.mean(images, axis=0).astype('float32')
images.shape
plt.imshow(images, cmap='gray')

# the_plane_mean_subset_last_images = get_the_plane_mean_subset_last_images(images_path, number_images, number_repetitions_the_plane, step_between_repetitions_of_the_plane)





























#* Open the images and take the mean
# images_paths = [*Path(images_path).glob('*tiff')]

# number_images = len(images_paths)

# images_mean = [0 for _ in images_paths]

# for image_i, image in enumerate(images_paths):
	
# 	images_mean[image_i] = np.sum(np.array(Image.open(image)))
# 	# [240:260, 240:260])


# images_mean = np.array(images_mean, dtype='int')















data.dtypes

data['ID'].max()


x0 = 10
x1 = 10

for i in data[data[us_beg] > 0]:
	# break

	i


	time = data.loc[i-x0:i+x1, abs_time].to_numpy()
	gv = data.loc[i-x0:i+x1, galvo_value].to_numpy()
	usb = data.loc[i-x0:i+x1, us_beg].to_numpy()
	use = data.loc[i-x0:i+x1, us_end].to_numpy()

	# use[use > 0]

	plt.plot(time, gv, 'k')
	plt.plot(time, usb, 'ro')
	plt.plot(time, use, 'bo')

	break







data[[us_beg, 'Frame beg']].plot()



x0=-10000
x1=-1

a = interframe_interval.iloc[x0:x1].median()

# plt.plot(galvo[abs_time].iloc[x0:x1], galvo[galvo_value].iloc[x0:x1], 'k')
plt.plot(galvo[abs_time].iloc[x0:x1], galvo[galvo_value].iloc[x0:x1], 'k')
plt.plot(galvo[abs_time].iloc[x0:x1], galvo['Frame beg'].iloc[x0:x1], 'ro')
# plt.plot(galvo[abs_time].iloc[x0:x1], interframe_interval.iloc[x0:x1] - a, 'y.')







fig = go.Figure()
fig.add_trace(go.Scattergl(x=galvo[abs_time], y=galvo[galvo_value].to_numpy()))
# fig.add_trace(go.Scattergl(x=galvo[::2][abs_time], y=galvo[galvo_value][::2].to_numpy()))
# fig.add_trace(go.Scattergl(x=galvo[abs_time], y=galvo['1diff'].to_numpy()))
# fig.add_trace(go.Scattergl(x=galvo[abs_time], y=galvo['2diff'].to_numpy()))
# fig.add_trace(go.Scattergl(x=galvo[abs_time], y=galvo['3'].to_numpy()))

# fig.add_trace(go.Scattergl(x=galvo[abs_time][:-1], y=galvo['2diff'].to_numpy()[1:]))
# fig.add_trace(go.Scattergl(x=galvo[abs_time], y=galvo['2diff'].diff().to_numpy()))
fig.add_trace(go.Scattergl(x=galvo[abs_time], y=galvo['beg'].to_numpy()))





galvo['1diff'].median()



galvo.loc[index - space : index + space]['beg'].dropna().index



space = 1000

# index=815563

for i, index in enumerate(galvo[galvo['beg'].notna()].index):

	fig = go.Figure()
	fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space][galvo_value].to_numpy()))
	# fig.add_trace(go.Scattergl(x=galvo[::2].loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space][galvo_value][::2].to_numpy()))
	# fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space]['1diff'].to_numpy()))
	# fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space]['2diff'].to_numpy()))
	# fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space]['3'].to_numpy()))

	# fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time][:-1], y=galvo.loc[index - space : index + space]['2diff'].to_numpy()[1:]))
	# fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space]['2diff'].diff().to_numpy()))
	fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space]['beg'].to_numpy()))

	fig.show()

	break

	# plt.plot(galvo.loc[index - space : index + space][abs_time], galvo.loc[index - space : index + space][galvo_value], 'k.')
	# plt.plot(galvo.loc[index - space : index + space][abs_time], galvo.loc[index - space : index + space]['1diff'], 'g.')
	# plt.plot(galvo.loc[index - space : index + space][abs_time], galvo.loc[index - space : index + space]['2diff'], 'r.')

	# plt.plot(galvo.loc[index - space : index + space][abs_time], galvo.loc[index - space : index + space]['beg'], 'bo')

	# plt.show()







#!
DATA = data.copy()

# plt.plot(data[us_beg], 'ko')

data = pd.merge_ordered(data, galvo, on=abs_time, how='outer').drop_duplicates(abs_time, keep='first')

# data.loc[data[pmt_off_beg].notna(), pmt_off_beg] = 1
# data.loc[data[pmt_off_end].notna(), pmt_off_end] = 1

data.loc[data[us_beg].notna(), us_beg] = 1
data.loc[data[us_end].notna(), us_end] = 1




interp_function = interpolate.interp1d(galvo[abs_time], galvo[galvo_value], kind='slinear', axis=0, assume_sorted=True, bounds_error=False)

data[galvo_value] = interp_function(data[abs_time])










# galvo_peaks = galvo['GalvoValue'].diff()

# # galvo_peaks[galvo_peaks>1.5]


# galvo_peaks = galvo_peaks.to_numpy()


# galvo_peaks = np.where(galvo_peaks>1.5)

# galvo_peaks = np.where(galvo_peaks>1.5, 5, 0)



# galvo.plot(abs_time, 'GalvoValue')

# galvo = galvo.drop(columns='GalvoValue')

# galvo = galvo.rename(columns={'AblationValue' : 'GalvoValue'})

# galvo = galvo.rename(columns={'FrameID' : 'ID'})

# galvo = galvo.drop(columns=['ElapsedTime', 'AbsoluteTime'])




		# np.median(np.diff(galvo_peaks))




# fig = go.Figure()
# fig.add_trace(go.Scattergl(x=np.arange(len(galvo)), y=galvo['GalvoValue'].to_numpy()))
# fig.add_trace(go.Scattergl(x=np.arange(len(galvo)), y=galvo_peaks))
# fig.add_trace(go.Scattergl(x=np.arange(len(galvo)), y=galvo['GalvoValue'].to_numpy()))


# fig.write_html(r"C:\Users\joaqc\Desktop\test.html")






# galvo['GalvoValue'].iloc[50000:52000].plot()
# galvo_peaks.iloc[50000:52000].plot()

# galvo['GalvoValue'][galvo_peaks>2]



	#! this is for when the galvo signal is saved throgh LabView
	# galvo = pd.read_csv(galvo_path, engine='pyarrow', sep='\t', header=4, decimal='.', na_filter=False)

	# galvo.rename(columns={'time' : abs_time, 'Dev1/ai0' : galvo_value}, inplace=True)

	# galvo = galvo.iloc[:,[0,1]]

	# galvo[abs_time] = galvo[abs_time].astype('datetime64[ns]') - pd.Timedelta('1h')

	# # Calculate unixtime in ms
	# galvo[abs_time] = galvo[abs_time].astype('int64') / 10**6
	# # galvo = (galvo[abs_time] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
	# # galvo = galvo[abs_time].map(pd.Timestamp.timestamp)



# data = f.highlight_stim_in_data(data, protocol)



# del galvo


		# index_pmt_beg = np.zeros(len(data[data[pmt_off_beg].notna()]))
		# # len(index_pmt_beg)


		# for index in data.loc[data[pmt_off_beg].notna(), abs_time]:

		# 	data = data.loc[data[abs_time].between(index - 1000, index + 3000)]
			
			
		# 	plt.plot(data[abs_time], data[pmt_off_beg] + 1, 'ko')
		# 	plt.plot(data[abs_time], data[pmt_off_end] + 1, 'go')
		# 	plt.plot(data[abs_time], data[us_beg] + 1, '.')
		# 	plt.plot(data[abs_time], data[us_end] + 1, '.')
		# 	plt.plot(data[abs_time], data[galvo_value])



		# 	print(data.loc[data[pmt_off_end].notna(), abs_time].to_numpy() - data.loc[data[pmt_off_beg].notna(), abs_time].to_numpy(), data.loc[data[us_beg].notna(), abs_time].to_numpy() - data.loc[data[pmt_off_beg].notna(), abs_time].to_numpy(), data.loc[data[pmt_off_end].notna(), abs_time].to_numpy() - data.loc[data[us_end].notna(), abs_time].to_numpy())

		# 	break






A = data[galvo_value].diff()



A[A>2]


data.iloc[5000:][galvo_value].plot()


len(images_mean)


plt.plot(A.iloc[5000:10000], 'k.')
plt.plot(data[galvo_value].iloc[5000:10000], 'g.')



plt.plot(images_mean)


#* Pad the image paths
# List all files in the folder
# files = os.listdir(folder_path)
images_paths = [*Path(images_path).glob('*tiff')]

# Regex pattern to find all integer numbers in the file names
pattern = re.compile(r'(\d+)')

# Iterate through each file and rename it
for images_name in images_paths:

	new_image_name = re.sub(pattern, lambda x: x.group(1).zfill(10), str(images_name.stem))
	
	images_name.rename(Path(images_path).joinpath(new_image_name + '.tiff'))


import multipagetiff as mtif
s = mtif.read_stack(images_path, units='um')
mtif.plot_flatten(s)
pages = s.pages
pages.shape
mtif.Stack(pages)



tif = TiffFile(images_path)
len(tif.pages)  # number of pages in the file

np.mean(tif.pages[1000])


imread(images_path, key=-100:)

imread()






#! images_paths = images_path + r"\test_1green.tif"

im = Image.open(images_path)

images_mean = []

ImageSequence.all_frames(im, np.mean)

try:
	for frame in ImageSequence.Iterator(im):
		
		images_mean.append(np.sum(frame))
except:
	pass

plt.plot(images_mean)



images_mean = np.array(images_mean, dtype='int')




len(images_mean)



data['image'] = 0




B = data.loc[data['beg'].notna(), 'image'].iloc[::2].index

data.loc[B, 'image'] = images_mean[1:-1]



len(images_mean[1:-1])














data.loc[:,[cs_beg, cs_end, us_beg, us_end, 'PMT_OFF beg', 'PMT_OFF end']] = data[[cs_beg, cs_end, us_beg, us_end, 'PMT_OFF beg', 'PMT_OFF end']].fillna(0)


plt.plot(data[abs_time], data['PMT_OFF beg'], 'ko')
plt.plot(data[abs_time], data[galvo_value])

# data = data.dropna(subset='ID')




data.dtypes






#* Fix dtypes.
data[cols_stim + ['PMT_OFF beg', 'PMT_OFF end']] = data[cols_stim + ['PMT_OFF beg', 'PMT_OFF end']].astype('Sparse[int16]')

data['ID'] = data['ID'].astype('int')









plt.plot(data['image'])
plt.plot(data['PMT_OFF end'])
plt.plot(images_mean*(-1/20000))

plt.plot(images_mean)
plt.plot(A)


A = np.diff(images_mean)
# *(-1/20000)

len(A[A<-50000])


plt.plot(data[abs_time], y=data[galvo_value])
plt.plot(data[abs_time], y=data[pmt_off_beg])
plt.plot(data[abs_time], y=data[pmt_off_end])


space = 10000

for index in data[data[us_beg]>0].index:
	
	data = data.loc[index-space:index+space]

	fig = go.Figure()
	fig.add_trace(go.Scattergl(x=data[abs_time], y=data[galvo_value].to_numpy()))
	fig.add_trace(go.Scattergl(x=data[abs_time], y=data['image'].to_numpy()))
	fig.add_trace(go.Scattergl(x=data[abs_time], y=data[pmt_off_beg].to_numpy()))
	fig.add_trace(go.Scattergl(x=data[abs_time], y=data[pmt_off_end].to_numpy()))


	break

fig.write_html(r"C:\Users\joaqc\Desktop\test.html")






#region

need to find the beginning of each imaging frame









# B = data['GalvoValue'] / data[abs_time].diff()
# A = data['GalvoValue'] / data[ela_time].diff()

# fig = go.Figure()
# fig.add_trace(go.Scattergl(x=np.arange(len(A)), y=A.to_numpy()))
# fig.add_trace(go.Scattergl(x=np.arange(len(B)), y=B.to_numpy()))

# A = A.dropna()

# A.median()

data = data.reset_index(drop=True)

galvo_peaks = data['GalvoValue'].diff()

# galvo_peaks[galvo_peaks>1.5]

galvo_peaks = galvo_peaks.to_numpy()

# galvo_peaks.where(galvo_peaks > 1.5, False)

galvo_peaks_index = np.where(galvo_peaks>1.5)[0]

#* Interval between frames (only works for a lot of frames).
interval_between_frames = np.median(np.diff(galvo_peaks_index))

mask = np.diff(galvo_peaks_index) > 600
# (np.diff(galvo_peaks_index) > 340) & (np.diff(galvo_peaks_index) < 360)

C = data.loc[galvo_peaks_index[1:][mask][0]:]
# data.loc[data.index >= galvo_peaks_index[1:][mask][0]]



fig = go.Figure()
fig.add_trace(go.Scattergl(x=data.index, y=galvo_peaks))
fig.add_trace(go.Scattergl(x=data.index, y=data['GalvoValue'].to_numpy()))
fig.add_trace(go.Scattergl(x=C.index, y=C['GalvoValue'].to_numpy()))



# fig.add_trace(go.Scattergl(x=np.arange(len(data)), y=galvo_peaks))
fig.add_trace(go.Scattergl(x=np.arange(len(data)), y=data['GalvoValue'].to_numpy()))


fig.write_html(r"C:\Users\joaqc\Desktop\test.html")



#! Improve this part
galvo_peaks = C['GalvoValue'].diff()
# galvo_peaks[galvo_peaks>1.5]
galvo_peaks = galvo_peaks.to_numpy()
# galvo_peaks.where(galvo_peaks > 1.5, False)
galvo_peaks_index = np.where(galvo_peaks>1.5)[0]

#? Fix the time of all peaks?

D = C.iloc[galvo_peaks_index]

E = np.zeros(len(D))
E[:len(images_mean)] = images_mean

D['Mean of images'] = E


data['Mean of images'] = 0

data.loc[D.index] = D


x = data[abs_time].to_numpy()

fig = go.Figure()
# fig.add_trace(go.Scattergl(x=x, y=D['GalvoValue'].to_numpy()))
fig.add_trace(go.Scattergl(x=x, y=data['Mean of images'].to_numpy()))

fig.add_trace(go.Scattergl(x=x, y=data['PMT_OFF beg'].to_numpy()*200))





# data[abs_time] -= data[abs_time].iloc[0]

# fig = go.Figure()
# fig.add_trace(go.Scattergl(x=data[abs_time].to_numpy(), y=data['GalvoValue'].to_numpy()))
# # fig.add_trace(go.Scattergl(x=np.arange(len(data)), y=galvo_peaks))
# fig.add_trace(go.Scattergl(x=np.arange(len(data)), y=data['GalvoValue'].to_numpy()))


#endregion











fig = go.Figure()
fig.add_trace(go.Scatter(x=data[abs_time], y=data['GalvoValue'].to_numpy()))
fig.add_trace(go.Scatter(x=data[abs_time], y=galvo_peaks))












galvo.plot(y='GalvoValue')

galvo.iloc[15000:20000].plot(y='GalvoValue')





data = data.iloc[20000:80000]



data['GalvoValue'].iloc[20000:22000].plot()

data.plot(x='ID', y=abs_time)










plt.plot(images_mean)




#! PLOTS



data['ID'] -= data['ID'].iat[0]




# with open(galvo_path, 'rb') as file:
#	 binary_string = file.read()

# # print(binary_string)

# # with open(r'E:\\data\\test\\zscan\\conversionNEW.txt', 'w') as new_file:
# # 	new_file.write(binary_string)
	

# text_string = binary_string.decode('')
# print(text_string)






#! Shape of the galvoValue
data.plot('ID', 'GalvoValue')

data.plot(x='ID', y=['GalvoValue', us_beg, us_end] )


plt.plot(data[us_beg], 'green')
plt.plot(data[us_beg], 'red')


data = data[data['ID'].between(8000,45000)]
data.plot(x='ID', y=[us_beg, 'PMT_OFF beg'], )


data.plot(x='ID', y=[us_beg, us_end, cs_beg, cs_end], )


data.plot(x='ID', y=[ 'PMT_OFF beg', 'PMT_OFF end'])

# data[abs_time].diff().plot()

data.plot(x='ID', y=[us_beg, us_end],)



protocol[end] - protocol[beg]











data.dtypes



data.plot()






import struct
import tifffile
import h5py

galvo_path = Path(r"C:\Users\joaqc\Desktop\monacoshutterSat, Oct 28, 2023 6-04-58 PM.dat")

# galvo_bits = open(galvo_path, 'rb')
# galvo_bits = galvo_bits.read()
# struct.unpack('>i', galvo_bits)





# Reading and decoding data from the file
with open(str(galvo_path), 'rb') as f:
	binary_data = f.read()
	decoded_data = binary_data.decode('ascii', 'ignore')
	print(decoded_data)
