from pathlib import Path
from scipy import ndimage
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import plotly.io as pio
import tifffile
from scipy import interpolate, signal
from tqdm import tqdm
from dataclasses import dataclass

from scipy.stats import pearsonr
from skimage.registration import phase_cross_correlation
import math
from scipy.ndimage import shift

import my_functions as f
from my_general_variables import *

pio.templates.default = "plotly_dark"

pd.set_option("mode.copy_on_write", True)
pd.set_option("compute.use_numba", True)
pd.set_option("compute.use_numexpr", True)
pd.set_option("compute.use_bottleneck", True)


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

# kernel_size = 3
# ddepth = cv2.CV_16S


total_motion_thr = 5


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

extra_border = 50  # pixels


#! this is overwriting the one in my_general_variables.py
# cols_to_use_orig = ['FrameID'] + ['x15'] + ['y15'] + ['angle15']
# data_cols = ['X 14'] + ['Y 14'] + ['Angle (deg) 14']
# angle_name = 'Angle (deg) 14'
# angle_cols = [angle_name]

time_experiment_f = frame_id
#endregion



#region Paths
path_home = Path(r'E:\2024 03_Delay 2-P multiple planes')


# fish_names = [folder.stem for folder in path_home.iterdir() if folder.is_dir()]
# fish_names.remove('Behavior')


# for fish_name in fish_names:
# fish_name = r'20240228_01_delay_2p-1_mitfaMinusMinus,elavl3H2BCaMP6s_7dpf'

fish_name = r'20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20240314_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'
# "20240404_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_7dpf"
# '20240327_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_7dpf'

# '20240313_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_5dpf'
# '20240321_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'
# '20240311_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_5dpf'

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



#region Functions
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




def find_plane_in_anatomical_stack(anatomical_stack_images, the_plane_mean_subset_last_images, plane_where_we_are, x_dim, y_dim):

	#* Handle to the multipage TIFF file with the plane being imaged.
	# the_plane_tiff = tifffile.TiffFile(the_plane_path)

#! explain
	# the_plane_mean_of_last_images = the_plane_tiff.asarray(slice(-number_repetitions_of_the_plane_to_analyze-1,-1,step_between_repetitions_of_the_plane_to_analyze))
	the_plane_mean_subset_last_images = the_plane_mean_subset_last_images[y_dim:-y_dim, x_dim:-x_dim]
	
	if plane_where_we_are is not None:
		
		if (first_plane_substack := plane_where_we_are - number_planes_around_the_plane) < 0:
			
			first_plane_substack = 0

		if (last_plane_substack := plane_where_we_are + number_planes_around_the_plane + 1) > len(anatomical_stack_images):
			
			last_plane_substack = len(anatomical_stack_images)

		anatomical_stack_images_ = anatomical_stack_images[first_plane_substack : last_plane_substack]
		
	else:
		
		anatomical_stack_images_ = anatomical_stack_images

		first_plane_substack = 0
		
	template_matching_results = [cv2.matchTemplate(plane, the_plane_mean_subset_last_images, cv2.TM_CCOEFF_NORMED) for plane in anatomical_stack_images_]

	# b = [np.array(x.flatten())[np.argpartition(x.flatten(), 3)[:3]] for x in template_matching_results]

	# # [print(ii) for ii in b]

	# a = np.mean(b, axis=1)

	plane_i = np.argmax([x.max() for x in template_matching_results])
	# np.argmax(a)

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

	data = pd.merge_ordered(data, camera, on=frame_id, how='outer')

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

	try:
		data.loc[:, data_cols] = data.loc[:, data_cols].interpolate(kind='slinear')
	except:
		print('HERE. FIX THIS')

	#! data = data.reset_index(drop=True).dropna()
	data = data.reset_index().dropna()

	data[time_experiment_f] = data[time_experiment_f].astype('int64')

	data[[cs, us]] = data[[cs, us]].astype('Sparse[int16]')

	#* Fix dtypes.
	# for cs_us in [cs, us]:

	# 	data[cs_us] = data.loc[:, cs_us].astype(pd.CategoricalDtype(categories=data[cs_us].unique(), ordered=True))

	return data




def get_good_images_indices(images_subset):

	top = np.mean(np.mean(images_subset[:, :top_bottom_frame_slice, :], axis=1), axis=1)
	bottom = np.mean(np.mean(images_subset[:, -top_bottom_frame_slice:, :], axis=1), axis=1)
	front = np.mean(np.mean(images_subset[:, :, -front_back_frame_slice:], axis=1), axis=1)
	back = np.mean(np.mean(images_subset[:, :, :front_back_frame_slice], axis=1), axis=1)

	all = np.mean(np.mean(images_subset, axis=1), axis=1)
	all_mean = np.mean(all)

	light_percentage_change = (np.abs(all - all_mean) / all_mean) * 100

	#* Discard based on overall light (too low or too high)
	mask_good_images = light_percentage_change < light_percentage_increase_thr

	#* And also discard based on the derivative
	mask_good_images = mask_good_images & ([True] + list((np.abs(np.diff(top)) < average_light_derivative_thr) & (np.abs(np.diff(bottom)) < average_light_derivative_thr) & (np.abs(np.diff(front)) < average_light_derivative_thr) & (np.abs(np.diff(back)) < average_light_derivative_thr) & (np.abs(np.diff(all) < average_light_derivative_thr))))

	plt.plot(top-np.median(top))
	plt.plot(bottom-np.median(bottom))
	plt.plot(front-np.median(front))
	plt.plot(back-np.median(back))
	plt.plot(all-np.median(all))
	plt.plot(light_percentage_change)
	plt.plot(np.where(mask_good_images, mask_good_images, np.nan)*(-10), lw=3)
	plt.legend(['Top', 'Bottom', 'Front', 'Back', 'Whole', r'Whole % change', r'Good images'], loc='center left', bbox_to_anchor=(1, 0.5))
	plt.show()
	
	return mask_good_images


def get_fixed_number_good_last_images(images_subset):

	mask_good_images = get_good_images_indices(images_subset)

	#* Discard the first and last frames
	mask_good_images[:3] = False
	mask_good_images[-3:] = False

	#* Find consecutive True regions
	new_mask = np.zeros_like(mask_good_images, dtype=bool)
	consecutive_count = 0
	for i, value in enumerate(mask_good_images[::-1]):

		if value:
			consecutive_count += 1
			if consecutive_count >= number_repetitions_the_plane_consecutively_stable:
				new_mask[i - number_repetitions_the_plane_consecutively_stable + 1 : i + 1] = True
				break
		else:
			consecutive_count = 0
	mask_good_images = new_mask[::-1]
	

	images_subset = images_subset[mask_good_images,:,:]

	# plt.plot(mask_good_images)
	# # plt.axvspan(np.diff(mask_good_images)==1, np.diff(mask_good_images)==-1)
	# plt.legend([r'Good images'], loc='center left', bbox_to_anchor=(1, 0.5))
	# plt.show()

	return images_subset


def get_maximum_number_good_last_images(images_subset):

	mask_good_images = get_good_images_indices(images_subset)

	#* Discard the first and last frames
	mask_good_images[:3] = False
	mask_good_images[-3:] = False

	#* Find consecutive True regions
	new_mask = np.zeros_like(mask_good_images, dtype=bool)
	consecutive_count = 0
	for i, value in enumerate(mask_good_images[::-1]):

		if value:
			consecutive_count += 1
			# if consecutive_count >= number_repetitions_the_plane_consecutively_stable:
			# 	new_mask[i - number_repetitions_the_plane_consecutively_stable + 1 : i + 1] = True
			# 	break
		else:
			if consecutive_count > 0 and consecutive_count > number_repetitions_the_plane_consecutively_stable:
				new_mask[i - consecutive_count : i] = True
				break
			
	mask_good_images = new_mask[::-1]
	

	images_subset = images_subset[mask_good_images,:,:]

	plt.plot(mask_good_images)
	# plt.axvspan(np.diff(mask_good_images)==1, np.diff(mask_good_images)==-1)
	plt.legend([r'Good images'], loc='center left', bbox_to_anchor=(1, 0.5))
	plt.show()

	return images_subset


def get_template_image(frames):

	template_image = ndimage.median_filter(np.nanmean(frames, axis=0), size=median_filter_kernel)

	plt.imshow(template_image)
	plt.title('Anatomy')
	plt.show()

	return template_image



def measure_motion(frames, anatomy):

	x_motion=np.zeros(np.shape(frames)[0])
	y_motion=np.zeros(np.shape(frames)[0])
	for j in range(frames.shape[0]):
		X=phase_cross_correlation(anatomy, frames[j,:,:], upsample_factor=10, space='real')
		x_motion[j]=X[0][0]
		y_motion[j]=X[0][1]

	return np.column_stack([x_motion, y_motion])


def get_total_motion(motion):
	# total_motion=np.zeros(np.shape(frames)[0])
	total_motion = np.linalg.norm(motion, axis=1)

	plt.show()
	fig, axs = plt.subplots(1, 2)
	axs[0].plot(total_motion)
	axs[0].set_title('Motion of each frame')
	axs[1].scatter(motion[:,0]-0.01+0.02*np.random.rand(motion[:,0].shape[0]),motion[:,1]-0.01+0.02*np.random.rand(motion[:,1].shape[0]),s=0.5)
	# fig.show()
	plt.show()

	return total_motion



def align_frames(frames, motion, total_motion_thr=total_motion_thr):

	total_motion = get_total_motion(motion)

	##* Discard frames with too much motion.
	frames_indices_ignore = np.where(total_motion > total_motion_thr)[0]
	
	aligned_frames=np.zeros(frames.shape)

	for j in range(frames.shape[0]):
		if j not in frames_indices_ignore:
			aligned_frames[j,:,:]=shift(frames[j,:,:], motion[j], output=None, order=3, mode='constant', cval=0.0, prefilter=True)
		#   the commented lines below check that the shift was performed in the correct direction	
			# X=phase_cross_correlation(original_anatomy, frames[j,:,:] ,upsample_factor=10, space='real')
			# print(X)
			# Y=phase_cross_correlation(original_anatomy, aligned_frames[j,:,:] ,upsample_factor=10, space='real')
			# print(Y)
   
	return aligned_frames








#endregion			




#! flag summer time
if (date := int(fish_name.split('_')[0][4:6])) >= 4 and date <= 10:
	Summer_time = True
else:
	Summer_time = False




#region Behavior camera

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




#region Stim log and merge it with the behavior camera data

#* Open the stim log.
protocol = read_protocol(protocol_path)


# protocol.iloc[:,1] - protocol.iloc[:,0]

#* Identify the stimuli, trials of the experiment.
data_cols = []
data = identify_trials(data, protocol)

# plt.plot(data[abs_time])
# data[cs].unique()
#endregion

# AA = data.loc[data[cs] == 10, :]
# (data.loc[data[cs] == 10, abs_time].iloc[-1] - data.loc[data[cs] == 10, abs_time].iloc[0])

# plt.plot(AA[abs_time], AA[cs])

# AA[cs].min()



#region Galvo signal
#* Read the galvo signal.
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

























#region Image data


#endregion


#region Merge galvo signal, stim log and behavior camera data


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


#* Get the images and align them to data.
#! Do not forget to discard the first images.
#!!!!! images = np.array([get_image_from_tiff(images_path, image_i, bytes_header, height, width) for image_i in range(number_images_before_first_image_to_consider, len(peaks))])
# images_subset_mean = [np.mean(image[-30:-10][-30:-10]).astype('float32') for image in images]
images = np.array([get_image_from_tiff(images_path, image_i, bytes_header, height, width).astype('float32') for image_i in tqdm(range(number_images))])

images.shape

images_mean = [image.mean() for image in images]







#* Merge galvo with data.
# data = pd.merge_ordered(data, galvo.drop(columns=galvo_value), on=abs_time, how='outer')
data = pd.merge_ordered(data, galvo, on=abs_time, how='outer')

del galvo, protocol


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

#endregion


#region Check whether the different pieces of data are aligned.

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



#endregion


# data.loc[data['Frame beg'].notna()]

data.drop(columns=['Image mean', 'GalvoValue'], inplace=True)





#!!!!!!!!!!!! read the behavior here


#! reverse data_cols to what we want
#! #* Open tail tracking data.
data_cols = x_cols + y_cols + angle_cols
behavior = read_tail_tracking_data(tracking_path).astype('float32')

# behavior.dtypes
# if (tail := read_tail_tracking_data(data_path)) is None: # type: ignore
# 	return None

# plot_behavior_overview(tail, fish_name, fig_behavior_name)

# #* Look for possible tail tracking errors.
# if tracking_errors(tail, single_point_tracking_error_thr):
# 	return None









# Cannot use pd.merge_ordered because of NANs
# data = pd.merge_ordered(data, behavior, on=frame_id, how='left')
all_data = data.copy()
data = all_data.copy()

data = pd.merge(data, behavior, on=frame_id, how='left')

data.reset_index(drop=True, inplace=True)
data.rename(columns={abs_time : 'Time (ms)'}, inplace=True)
abs_time = 'Time (ms)'
data[abs_time] -= data[abs_time].iat[0]



protocol = data[[abs_time, cs, us]].copy()
protocol = protocol[((protocol[cs]!=0) | (protocol[us]!=0))]

behavior = data[[abs_time] + data_cols].dropna().rename(columns={frame_id : 'Frame number (behavior)'}).copy()
# behavior_array = xr.DataArray(behavior, coords=[('time', all_data.loc[all_data['Frame number'].notna(), :].index), ('parameters', behavior.columns)], dims=['time', 'parameters'])
# behavior_array.to_dataframe()

imaging = xr.DataArray(images, coords={'index': ('time', data.loc[data['Frame beg'].notna(), :].index), 'time': data.loc[data['Frame beg'].notna(), abs_time].to_numpy(), 'x': range(images.shape[1]), 'y': range(images.shape[2])}, dims=['time', 'x', 'y'])

imaging.name = 'Imaging data'

















#region Label the planes.

#* Read the anatomical stack.
anatomical_stack_images = tifffile.imread(anatomy_1_path).astype('float32')
# anatomical_stack_images = tifffile.imread(anatomy_1_filtered_path).astype('float32')

anatomical_stack_images = ndimage.median_filter(anatomical_stack_images, size=median_filter_kernel, axes=(1,2))


# anatomical_stack_images.shape


#ToDo this should involve the pixel spacing!!!


_, y_dim, x_dim = np.array(anatomical_stack_images.shape)

x_dim = int(x_dim * xy_movement_allowed/2)
y_dim = int(y_dim * xy_movement_allowed/2)




@dataclass
class Trial:
	
	trial_number : int

	# position_anatomical_stack : int
	# reference_image : np.ndarray

	protocol : pd.DataFrame
	behavior : pd.DataFrame
	images : xr.DataArray


@dataclass
class Plane:

	trials : list[Trial]

	# reference_image_position_anatomical_stack : int
	# reference_image : np.ndarray

	# order_planes_sequence : int

	def get_reference_position(self):

		return round(np.median([trial.position_anatomical_stack for trial in self.trials]))

@dataclass
class Imaging:
	
	plane : list[Plane]










cs_onset_index = np.array([protocol.loc[protocol[cs] == relevant_cs[i], :].index[0] for i in range(len(relevant_cs))])

index_list = [[i, i+1, i+2*number_imaged_planes, i+2*number_imaged_planes+1] for i in range(0, number_reps_plane_consective * number_imaged_planes, 2)]

planes_cs_onset_indices = cs_onset_index

planes_cs_onset_indices = [cs_onset_index[[j for j in i]] for i in index_list]

del cs_onset_index, index_list


trials_list = [0 for _ in range(len(planes_cs_onset_indices[0]))]
planes_list = [0 for _ in range(len(planes_cs_onset_indices))]

i = 0
for plane_i, plane_cs_onset_indices in tqdm(enumerate(planes_cs_onset_indices)):

	for trial_i, trial_cs_onset_index in enumerate(plane_cs_onset_indices):

		time_start = protocol.loc[trial_cs_onset_index, abs_time] - 45000
		time_end = protocol.loc[trial_cs_onset_index, abs_time] + 15000

		# index = protocol[protocol[abs_time].between(time_start, time_end)].index

		images_trial = imaging.loc[time_start : time_end,:,:]

		# images_average = ndimage.median_filter(np.mean(get_good_last_images(images_trial), axis=0), size=median_filter_kernel)
		# # images_average = ndimage.median_filter(np.mean(images_trial[i], axis=0), size=median_filter_kernel)

		# if images_trial[i] is not None:

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


		trials_list[trial_i] = Trial(i, protocol[protocol[abs_time].between(time_start, time_end)], behavior[behavior[abs_time].between(time_start, time_end)], images_trial)

		i += 1

	planes_list[plane_i] = Plane(trials_list)

del trials_list



for plane in tqdm(planes_list):
	for trial in plane.trials:

	# 	break
	# break





	images_trial = trial.images.to_numpy()


	#* First iteration of motion correction relative to trials average.
	
	##* Discard bad frames due to motion, gating of the PMT or plane change when making a template image for the trial.
#? USE ONLY THE LAST GOOD IMAGES???
#? Filter?
	template_image = get_template_image(get_maximum_number_good_last_images(images_trial))
	
	##* Measure motion of each frame using phase cross-correlation.
	motion = measure_motion(images_trial, template_image)
	# total_motion = get_total_motion(motion)

	##* Align the frames to their average.
	aligned_frames = align_frames(images_trial, motion, total_motion_thr)



	#* Second iteration of motion correction relative to trials average.
	
	##* Measure motion of each frame using phase cross-correlation.
	total_motion = get_total_motion(motion)
	template_image = get_template_image(aligned_frames[np.where(total_motion <= total_motion_thr)[0]])
	del aligned_frames

	##* Measure motion of each frame using phase cross-correlation.
	motion = measure_motion(images_trial, template_image)

	##* Align the frames to their average.
	images_trial = align_frames(images_trial, motion, total_motion_thr)



	#* Final template image.
	motion = measure_motion(images_trial, template_image)
	total_motion = get_total_motion(motion)
	template_image = get_template_image(images_trial[np.where(total_motion <= total_motion_thr)[0]])



	#* Identify the plane number of the trial.
	plane_number, motion = find_plane_in_anatomical_stack(anatomical_stack_images, template_image.astype('float32'), None, x_dim, y_dim)
	plt.imshow(anatomical_stack_images[plane_number])






the_plane_mean_subset_last_images = template_image.astype('float32')

plt.imshow(the_plane_mean_subset_last_images)


# the_plane_mean_of_last_images = the_plane_tiff.asarray(slice(-number_repetitions_of_the_plane_to_analyze-1,-1,step_between_repetitions_of_the_plane_to_analyze))
the_plane_mean_subset_last_images = the_plane_mean_subset_last_images[y_dim:-y_dim, x_dim:-x_dim]

anatomical_stack_images_ = anatomical_stack_images

first_plane_substack = 0
	
template_matching_results = [cv2.matchTemplate(plane, the_plane_mean_subset_last_images, cv2.TM_CCOEFF_NORMED) for plane in anatomical_stack_images_]

# b = [np.array(x.flatten())[np.argpartition(x.flatten(), 3)[:3]] for x in template_matching_results]

# # [print(ii) for ii in b]

# a = np.mean(b, axis=1)

plane_i = np.argmax([x.max() for x in template_matching_results])
# np.argmax(a)

xy_in_plane = np.argmax(template_matching_results[plane_i][0]), np.argmax(template_matching_results[plane_i][1])



# Assume `template` is your template and `image` is your image
template = the_plane_mean_subset_last_images
image = anatomical_stack_images[plane_number]

# Perform template matching
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# # Perform template matching
# result = template_matching_results[plane_i]

# Find the location of the best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

result.shape

# Calculate the shift needed to align the template with the image
shift_x = max_loc[0] - template.shape[1] // 2
shift_y = max_loc[1] - template.shape[0] // 2


encontrar o 0,0 da recortada

shift_by  = np.array([x_dim, y_dim])-np.array(max_loc)








shifted_image = shift(template, shift_by[::-1], mode='constant', cval=0.0)

plt.imshow(anatomical_stack_images[plane_number] - shifted_image)
plt.imshow(shifted_image)



w, h = template.shape[::-1]




template_matching_results[plane_i].shape
template_matching_results[0].shape
import cv2 as cv

_,_, min_loc, max_loc = cv.minMaxLoc(template_matching_results[plane_i])


image = np.mean(images_trial, axis=0)

plt.imshow(image)


plt.imshow(anatomical_stack_images[plane_number] - image)


new_image = shift(image, np.array(motion)[::-1], output=None, order=3, mode='constant', cval=0.0, prefilter=True)

# plt.imshow(new_image)

# np.array([image[crop:-crop,crop:-crop] for image in images_trial])

plt.imshow(anatomical_stack_images[plane_number] - new_image)



plt.imshow(anatomical_stack_images[plane_number])





	crop = 25
	# int(extra_border/2)

	template_image = anatomical_stack_images[plane_number][crop:-crop,crop:-crop]
	# template_image = np.pad(anatomical_stack_images[plane_number][crop:-crop,crop:-crop], pad_width=extra_border, mode='constant', constant_values=0).astype('float32')

	plt.imshow(template_image)

	new_images_trial = np.array([image[crop:-crop,crop:-crop] for image in images_trial])
	# new_images_trial = np.array([np.pad(image[crop:-crop,crop:-crop], pad_width=extra_border, mode='constant', constant_values=0).astype('float32') for image in images_trial])

	plt.imshow(np.mean(new_images_trial, axis=0))


	##* Measure motion of each frame using phase cross-correlation.
	motion = measure_motion(new_images_trial, template_image)
	total_motion = get_total_motion(motion)

	##* Align the frames to their average.
	aligned_frames = align_frames(new_images_trial, motion, 1000)

	plt.imshow(np.mean(aligned_frames, axis=0))


images_trial.max()






	total_motion = get_total_motion(motion)
	template_image = get_template_image(get_maximum_number_good_last_images(aligned_frames[np.where(total_motion <= total_motion_thr)[0]]))
	del aligned_frames

	##* Measure motion of each frame using phase cross-correlation.
	motion = measure_motion(images_trial, template_image)

	##* Align the frames to their average.
	images_trial = align_frames(images_trial, motion, template_image, total_motion_thr)






plt.imshow(anatomical_stack_images[plane_number])
plt.imshow(np.mean(aligned_frames, axis=0))
plt.imshow(anatomical_stack_images[plane_number]-np.mean(aligned_frames, axis=0))
plt.imshow(anatomical_stack_images[plane_number]-template_image)






	trial.images.values = images_trial










planes_numbers = np.zeros(len(relevant_cs), dtype='int32')
reference_images = np.zeros((relevant_cs.shape[0], anatomical_stack_images.shape[1], anatomical_stack_images.shape[2]), dtype='float32')


for plane in tqdm(planes_list):
	for trial in plane.trials:

	# 	break
	# break

	images_average = ndimage.median_filter(np.nanmean(get_maximum_number_good_last_images(frames), axis=0), size=median_filter_kernel)

	#* Identify the plane number of the trial.
	planes_numbers[i] = find_plane_in_anatomical_stack(anatomical_stack_images, images_average.astype('float32'), None, x_dim, y_dim)[0]
















plt.plot(planes_numbers)
plt.show()

plt.plot(planes_numbers[:int(len(planes_numbers)/2)])
plt.plot(planes_numbers[int(len(planes_numbers)/2):])
plt.show()

a = planes_numbers[int(len(planes_numbers)/2):] - planes_numbers[:int(len(planes_numbers)/2)]
plt.plot(a)
plt.show()



plane = planes_list[0]
plt.plot(plane.get_reference_position())
plt.show()




#!!!!!!!!!!!!!!!!!!!!! MOTION CORRECTION


SAVE TO HDF







for plane in tqdm(planes_list):
	for trial in plane.trials:
	# 	break
	# break
	
	# trial = trials_list[]


	#* Drift correction relative to the average of the frames.

	frames = trial.images.to_numpy()


#TODO discard bad frames due to motion, gating of the PMT or plane change.




	##* Measure motion of each frame using phase cross-correlation.
	template_image = ndimage.median_filter(np.nanmean(frames, axis=0), size=median_filter_kernel)
	# np.nanmean([ndimage.median_filter(frame, size=median_filter_kernel) for frame in frames], axis=0)
	# np.nanmean(frames, axis=0)

	plt.figure('1')
	plt.imshow(template_image, cmap='gray')
	plt.title('Anatomy 1')
	plt.show()

	x_motion=np.zeros(np.shape(frames)[0])
	y_motion=np.zeros(np.shape(frames)[0])
	total_motion=np.zeros(np.shape(frames)[0])
	for j in range(frames.shape[0]):
		X=phase_cross_correlation(template_image, frames[j,:,:], upsample_factor=10, space='real')
		x_motion[j]=X[0][0]
		y_motion[j]=X[0][1]
		total_motion[j]=math.sqrt(x_motion[j]*x_motion[j]+y_motion[j]*y_motion[j])

	plt.figure()
	# plt.title('1. Motion of each frame. CS {}'.format(i))
	# plt.subplot(1,2,1)
	plt.plot(total_motion)
	# plt.subplot(1,2,2)
	# plt.scatter(x_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),y_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),s=0.5)
	plt.show()


	##* Align the frames to their average.
	aligned_frames=np.zeros(frames.shape)

	for j in range(frames.shape[0]):
		aligned_frames[j,:,:]=shift(frames[j,:,:], (x_motion[j],y_motion[j]), output=None, order=3, mode='constant', cval=0.0, prefilter=True)
	#   the commented lines below check that the shift was performed in the correct direction	
		# X=phase_cross_correlation(original_anatomy, frames[j,:,:] ,upsample_factor=10, space='real')
		# print(X)
		# Y=phase_cross_correlation(original_anatomy, aligned_frames[j,:,:] ,upsample_factor=10, space='real')
		# print(Y)

	# aligned_anatomy=np.sum(aligned_frames,0)
	# plt.figure()
	# plt.imshow(aligned_anatomy)


	##* Phase cross-correlation to measure motion of each frame after aligning them to their average.
	frames = aligned_frames

	x_motion=np.zeros(np.shape(frames)[0])
	y_motion=np.zeros(np.shape(frames)[0])
	total_motion=np.zeros(np.shape(frames)[0])
	for j in range(frames.shape[0]):
		X=phase_cross_correlation(template_image, frames[j,:,:], upsample_factor=10, space='real')
		x_motion[j]=X[0][0]
		y_motion[j]=X[0][1]
		total_motion[j]=math.sqrt(x_motion[j]*x_motion[j]+y_motion[j]*y_motion[j])

	plt.figure()
	# plt.title('2. Motion of each frame. CS {}'.format(i))
	# plt.subplot(1,2,1)
	plt.plot(total_motion)
	# plt.subplot(1,2,2)
	# plt.scatter(x_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),y_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),s=0.5)
	plt.show()






	##* Discard frames with too much motion.
	frames[np.where(total_motion > total_motion_thr)[0],:,:] = np.empty(frames.shape[1:])
	# * np.nan








	

	#* Second iteration of the drift correction, now relative to the new average of the frames.

	##* Phase cross-correlation to measure motion of each frame.
	template_image = ndimage.median_filter(np.nanmean(frames, axis=0), size=median_filter_kernel)
	# np.nanmean([ndimage.median_filter(frame, size=median_filter_kernel) for frame in frames], axis=0)
	# np.nanmean(frames, axis=0)
 
	plt.figure('2')
	plt.title('Anatomy 2')
	plt.imshow(template_image, cmap='gray')
	plt.show()

	x_motion=np.zeros(np.shape(frames)[0])
	y_motion=np.zeros(np.shape(frames)[0])
	total_motion=np.zeros(np.shape(frames)[0])
	for j in range(frames.shape[0]):
		X=phase_cross_correlation(template_image, frames[j,:,:], upsample_factor=10, space='real')
		x_motion[j]=X[0][0]
		y_motion[j]=X[0][1]
		total_motion[j]=math.sqrt(x_motion[j]*x_motion[j]+y_motion[j]*y_motion[j])

	plt.figure()
	# plt.title('1. Motion of each frame. CS {}'.format(i))
	# plt.subplot(1,2,1)
	plt.plot(total_motion)
	# plt.subplot(1,2,2)
	# plt.scatter(x_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),y_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),s=0.5)
	plt.show()


	##* Align the frames to their average.
	aligned_frames=np.zeros(frames.shape)

	for j in range(frames.shape[0]):
		aligned_frames[j,:,:]=shift(frames[j,:,:], (x_motion[j],y_motion[j]), output=None, order=3, mode='constant', cval=0.0, prefilter=True)
	#   the commented lines below check that the shift was performed in the correct direction	
		# X=phase_cross_correlation(original_anatomy, frames[j,:,:] ,upsample_factor=10, space='real')
		# print(X)
		# Y=phase_cross_correlation(original_anatomy, aligned_frames[j,:,:] ,upsample_factor=10, space='real')
		# print(Y)

	# aligned_anatomy=np.sum(aligned_frames,0)
	# plt.figure()
	# plt.imshow(aligned_anatomy)

	##* Phase cross-correlation to measure motion of each frame after aligning them to their average.
	frames = aligned_frames

	x_motion=np.zeros(np.shape(frames)[0])
	y_motion=np.zeros(np.shape(frames)[0])
	total_motion=np.zeros(np.shape(frames)[0])
	for j in range(frames.shape[0]):
		X=phase_cross_correlation(template_image, frames[j,:,:], upsample_factor=10, space='real')
		x_motion[j]=X[0][0]
		y_motion[j]=X[0][1]
		total_motion[j]=math.sqrt(x_motion[j]*x_motion[j]+y_motion[j]*y_motion[j])

	plt.figure()
	# plt.title('2. Motion of each frame. CS {}'.format(i))
	# plt.subplot(1,2,1)
	plt.plot(total_motion)
	# plt.subplot(1,2,2)
	# plt.scatter(x_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),y_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),s=0.5)
	plt.show()


	##* Discard frames with too much motion.
	# frames[np.where(total_motion > total_motion_thr)[0],:,:] = np.empty(frames.shape[1:])
	# * np.nan

	template_image = ndimage.median_filter(np.nanmean(frames, axis=0), size=median_filter_kernel)
	# np.nanmean([ndimage.median_filter(frame, size=median_filter_kernel) for frame in frames], axis=0)
	# np.nanmean(frames, axis=0)
	plt.figure('3')
	plt.title('Anatomy 3')
	plt.imshow(template_image, cmap='gray')
	plt.show()












#!!!!!!!! then, motion correction with the mean image

	#* Phase cross-correlation to measure motion of each frame
	template_image = frames.mean(axis=0)
	
	x_motion=np.zeros(np.shape(frames)[0])
	y_motion=np.zeros(np.shape(frames)[0])
	total_motion=np.zeros(np.shape(frames)[0])
	for j in range(frames.shape[0]):
		X=phase_cross_correlation(template_image, frames[j,:,:], upsample_factor=10, space='real')
		x_motion[j]=X[0][0]
		y_motion[j]=X[0][1]
		total_motion[j]=math.sqrt(x_motion[j]*x_motion[j]+y_motion[j]*y_motion[j])

		# if j == 100:
		# 	break

	# plt.figure()
	# plt.title('1. Motion of each frame. CS {}'.format(i))
	# plt.subplot(1,2,1)
	# plt.plot(total_motion)
	# plt.subplot(1,2,2)
	# plt.scatter(x_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),y_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),s=0.5)
	# plt.show()





	aligned_frames=np.zeros(frames.shape)

	for j in range(frames.shape[0]):
		aligned_frames[j,:,:]=shift(frames[j,:,:], (x_motion[j],y_motion[j]), output=None, order=3, mode='constant', cval=0.0, prefilter=True)
	#   the commented lines below check that the shift was performed in the correct direction	
		# X=phase_cross_correlation(original_anatomy, frames[j,:,:] ,upsample_factor=10, space='real')
		# print(X)
		# Y=phase_cross_correlation(original_anatomy, aligned_frames[j,:,:] ,upsample_factor=10, space='real')
		# print(Y)

	# aligned_anatomy=np.sum(aligned_frames,0)
	# plt.figure()
	# plt.imshow(aligned_anatomy)





	frames = aligned_frames

	x_motion=np.zeros(np.shape(frames)[0])
	y_motion=np.zeros(np.shape(frames)[0])
	total_motion=np.zeros(np.shape(frames)[0])
	for j in range(frames.shape[0]):


#!!!!!!!!!!!!!!!
		# frames[j,:,:][np.isnan(frames[j,:,:])] = 0

		X=phase_cross_correlation(template_image, frames[j,:,:], upsample_factor=10, space='real')
		x_motion[j]=X[0][0]
		y_motion[j]=X[0][1]
		total_motion[j]=math.sqrt(x_motion[j]*x_motion[j]+y_motion[j]*y_motion[j])




	# plt.figure()
	# plt.title('2. Motion of each frame. CS {}'.format(i))
	# plt.subplot(1,2,1)
	# plt.subplot(1,2,2)
	# plt.scatter(x_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),y_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),s=0.5)
	# plt.show()

	plt.plot(total_motion)
	plt.show()

	trials_list[i].images.values = frames


	#!!!!!* Discard frames with too much motion
	# frames[np.where(total_motion > total_motion_thr)[0]] = np.empty(frames.shape[1:]) * np.nan
	
	
	aligned_anatomy=np.nansum(frames,0)
	plt.figure()
	plt.imshow(aligned_anatomy)

	# break






for i, trial in enumerate(trials_list):
	
	print(i)

	frames = trial.images.to_numpy().copy()

	template_image = frames.mean(axis=0)

	#* Calculate total motion of each frame to then discard bad frames.
	x_motion=np.zeros(np.shape(frames)[0])
	y_motion=np.zeros(np.shape(frames)[0])
	total_motion=np.zeros(np.shape(frames)[0])
	for j in range(frames.shape[0]):
		X=phase_cross_correlation(template_image, frames[j,:,:], upsample_factor=10, space='real')
		x_motion[j]=X[0][0]
		y_motion[j]=X[0][1]
		total_motion[j]=math.sqrt(x_motion[j]*x_motion[j]+y_motion[j]*y_motion[j])

	plt.plot(total_motion)

	#* Discard bad frames.
	frames[np.where(total_motion > total_motion_thr)[0]] = np.empty(frames.shape[1:]) * np.nan

	final_anatomy = np.nanmean(frames, 0)

	plt.imshow(final_anatomy)
	plt.show()

# trial.__dict__.keys()
# trial.position_anatomical_stack
	final_anatomy = final_anatomy[25:375,50:300]
	final_frames = frames[:,25:375,50:300].copy()
	# np.array([ndimage.median_filter(frame, size=median_filter_kernel) for frame in frames[:,25:375,50:300]])


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

	break










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




aligned_frames = final_frames.copy()


Nrois=50
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
# plt.show()
fig.tight_layout()


















#! group trials in planes




planes_numbers = [planes_numbers[i:i+2] for i in range(0, len(planes_numbers), 2)]

planes_numbers = np.array(planes_numbers).flatten()


reference_positions = [round(np.median([np.concatenate([planes_numbers[i : i+2], planes_numbers[i+int(planes_numbers.shape[0]/2) : i+int(planes_numbers.shape[0]/2) + 2]])])) for i in range(0, int(planes_numbers.shape[0]/2), 2)]



plt.plot(reference_positions)
plt.show()




planes_list = [Plane(trials_list[i:i+2] + trials_list[i+int(planes_numbers.shape[0]/2) : i+int(planes_numbers.shape[0]/2) + 2], reference_images[i], reference_positions[i]) for i in range(0, int(planes_numbers.shape[0]/2), 2)]


[Plane(trials_list[i:i+2] + trials_list[i+int(planes_numbers.shape[0]/2) : i+int(planes_numbers.shape[0]/2) + 2], reference_positions[i], reference_images[i], int(i/2)) for i in range(0, int(planes_numbers.shape[0]/2), 2)]

len(reference_images)

i = 28





for i, cs_onset_index in tqdm(enumerate(relevant_cs_onset[int(len(relevant_cs)/2):])):
	# break
# for i, cs_onset in tqdm(enumerate(zip(relevant_cs_onset - 60000, relevant_cs_onset + 15000))):

	# data_sub = data.loc[data[abs_time].between(exp_period[0], exp_period[1]), :]

	# images_sub = images[data_sub['Frame number (imaging)'].iat[0] : data_sub['Frame number (imaging)'].iat[-1]].copy()
	
	
	#!!!!!!!!!!!!! REMOVE?
	# images_sub = ndimage.median_filter(images_sub, size=median_filter_kernel)

	# images_sub.shape
 
	i += 30


	for l in [0,1]:

		# break

		#* Phase cross-correlation to measure motion of each frame
		template_image = reference_images[i].copy()
		

		
		#!!!!!!!!!!!!!!!!!!!!!!!!!!
		frames = ndimage.median_filter(images_sub[i][l], size=median_filter_kernel).copy()
		
		x_motion=np.zeros(np.shape(frames)[0])
		y_motion=np.zeros(np.shape(frames)[0])
		total_motion=np.zeros(np.shape(frames)[0])
		for j in range(frames.shape[0]):
			X=phase_cross_correlation(template_image, frames[j,:,:], upsample_factor=10, space='real')
			x_motion[j]=X[0][0]
			y_motion[j]=X[0][1]
			total_motion[j]=math.sqrt(x_motion[j]*x_motion[j]+y_motion[j]*y_motion[j])

			# if j == 100:
			# 	break

		# plt.figure()
		# plt.title('1. Motion of each frame. CS {}'.format(i))
		# plt.subplot(1,2,1)
		# plt.plot(total_motion)
		# plt.subplot(1,2,2)
		# plt.scatter(x_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),y_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),s=0.5)
		# plt.show()





		aligned_frames=np.zeros(frames.shape)

		for j in range(frames.shape[0]):
			aligned_frames[j,:,:]=shift(frames[j,:,:], (x_motion[j],y_motion[j]), output=None, order=3, mode='constant', cval=0.0, prefilter=True)
		#   the commented lines below check that the shift was performed in the correct direction	
			# X=phase_cross_correlation(original_anatomy, frames[j,:,:] ,upsample_factor=10, space='real')
			# print(X)
			# Y=phase_cross_correlation(original_anatomy, aligned_frames[j,:,:] ,upsample_factor=10, space='real')
			# print(Y)

		# aligned_anatomy=np.sum(aligned_frames,0)
		# plt.figure()
		# plt.imshow(aligned_anatomy)





#!!!!!!!!!!! need to implement this part
		frames = aligned_frames

		x_motion=np.zeros(np.shape(frames)[0])
		y_motion=np.zeros(np.shape(frames)[0])
		total_motion=np.zeros(np.shape(frames)[0])
		for j in range(frames.shape[0]):


#!!!!!!!!!!!!!!!
			# frames[j,:,:][np.isnan(frames[j,:,:])] = 0

			X=phase_cross_correlation(template_image, frames[j,:,:], upsample_factor=10, space='real')
			x_motion[j]=X[0][0]
			y_motion[j]=X[0][1]
			total_motion[j]=math.sqrt(x_motion[j]*x_motion[j]+y_motion[j]*y_motion[j])




		# plt.figure()
		# plt.title('2. Motion of each frame. CS {}'.format(i))
		# plt.subplot(1,2,1)
		# plt.plot(total_motion)
		# plt.subplot(1,2,2)
		# plt.scatter(x_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),y_motion-0.01+0.02*np.random.rand(x_motion.shape[0]),s=0.5)
		# plt.show()




		aligned_frames[np.isnan(aligned_frames)] = 0
		images_sub[i][l] = aligned_frames.copy()

		
		

		#* Discard frames with too much motion
		images_sub[i][l][np.where(total_motion > 10)[0]] = np.empty(images_sub[i][l].shape[1:]) * np.nan
		



	# break

	print(i)



	# plt.imshow(np.nanmean(np.concatenate(images_sub[i], axis=0), axis=0))
	# plt.imshow(np.nanmean(images_sub[i][1], axis=0))
	# plt.show()


	A = np.nanmean(images_sub[i][0][-18:], axis=0)
	B = np.nanmean(images_sub[i][1][:18], axis=0)

	
	# plt.imshow(B)
	plt.imshow((B-A)/(A+B))
	plt.show()
	plt.close()

	# break






















for i, cs_ in enumerate(relevant_cs_onset):
	# break


	a = (data.loc[data['Frame beg'], abs_time] < cs_)
	
	index_beg = a[a][:-120].index
	index_end = a[a][-120:].index

	a.loc[index_beg] = False
	
	index_images_group_in_data[i] = a[a].index.to_numpy()
	
	# data.loc[a, :]

	a = a.to_numpy()
	
	
	images_ = images[:len(a)][a]

	get_good_images_indices(images_)


	#* Find the plane number for each group of images.
	planes_numbers1 = find_plane_in_anatomical_stack(anatomical_stack_images, images_.astype('float32'), None, x_dim, y_dim)[0]


	






first_relevant_cs = 5

relevant_cs_onset = data.loc[data[cs] == first_relevant_cs, abs_time].iat[0]

time_beg = relevant_cs_onset - 60000

number_images_not_consider = (data.loc[data['Frame beg'], abs_time] < time_beg).sum()



number_images_not_consider = number_images_not_consider - number_images_not_consider % images_bin_size





#! Do not like this approach. Should be smarter. We do not care about all of these images.
images_considered = images[:images.shape[0]-im+-0ages.shape[0] % images_bin_size]

# images_considered.shape


#* Split the images array into groups of frames.
images_groups_frames = np.array([images[i:i+images_bin_size] for i in range(number_images_not_consider, images_considered.shape[0], images_bin_size)])

#! images_groups_frames_ = images_groups_frames.copy()
#! images_groups_frames = images_groups_frames_.copy()
# images_groups_frames.shape

#* Find the groups of images where there was no movement.
images_groups_frames = [get_good_images_indices(image) for image in images_groups_frames]

# images_groups_frames_mean[0].shape

#* Calculate the mean of each group of frames.
#* Filter the images with a median filter.
for i, images_group in tqdm(enumerate(images_groups_frames)):

	if images_group is not None:
		images_groups_frames[i] = ndimage.median_filter(np.mean(images_group, axis=0), size=median_filter_kernel)

# images_groups_frames[i].shape
# len(images_groups_frames)




#* Read the anatomical stack.
anatomical_stack_images = tifffile.imread(anatomy_1_path).astype('float32')
# anatomical_stack_images = tifffile.imread(anatomy_1_filtered_path).astype('float32')

# anatomical_stack_images.shape


xy_movement_allowed = 0.1  # fraction of the real image


_, y_dim, x_dim = np.array(anatomical_stack_images.shape)

x_dim = int(x_dim * xy_movement_allowed/2)
y_dim = int(y_dim * xy_movement_allowed/2)




# ONLY DO THIS FOR THE IMAGES THAT MATTER

planes_numbers = np.array([find_plane_in_anatomical_stack(anatomical_stack_images, image.astype('float32'), None, x_dim, y_dim)[0] if image is not None else np.nan for image in tqdm(images_groups_frames)])


# planes_numbers = [plane[0] if plane is not None else np.nan for plane in planes_numbers]

# planes_numbers_ = [0 for _ in images_groups_frames]

# for i, image in tqdm(enumerate(images_groups_frames)):
# 	if image is not None:
# 		plane_number, _ = find_plane_in_anatomical_stack(anatomical_stack_images, image.astype('float32'), None, x_dim, y_dim)
# 		planes_numbers_[i] = plane_number

# 		# break
# 	else:
# 		planes_numbers_[i] = None


planes_numbers_ = [plane for plane in planes_numbers[~np.isnan(planes_numbers)]]
edges = np.linspace(min(planes_numbers_), max(planes_numbers_), 15, endpoint=True)
edges[-1] += 0.00000000001


planes_numbers__ = np.digitize(planes_numbers, edges).astype('float')

planes_numbers__[np.isnan(planes_numbers)] = np.nan

plt.plot(planes_numbers__)

# planes_numbers[~np.isnan(planes_numbers)]
















#* Add the plane number to data.
data['Plane'] = np.nan

index = data.loc[data['Frame beg'], 'Plane'].index



data.loc[index[number_images_not_consider : len(images_considered)], 'Plane'] = np.array([np.ones(images_bin_size)*plane for plane in planes_numbers__]).flatten()



plt.plot(data['Plane'], '.')





#* Fill in the gaps in the plane numbers in data, when the edges of the gaps are the same.
A = data['Plane'].fillna(method='ffill')
B = data['Plane'].fillna(method='bfill')

#TODO set a limit...
mask = A == B

data.loc[mask, 'Plane'] = A[mask]

plt.plot(data['Plane'].to_numpy(), '.')
plt.plot(np.digitize(planes_numbers__, edges))






#endregion

#!!!!!!!!!!!!!!!!!!!!!!!!!




#TODO

NEED TO DO THE MOTION CORRECTION... FOR SIMILAR PLANES










#TODO plot where CS and position in the anatomical stack










#TODO in data, select the regions of interest based on the protocol


bef_stim_onset = 25000
aft_stim_offset = 25000




# cs_numbers = data.loc[data[us] != 0, us].unique()
# data.loc[data[cs] != 0, cs].unique()
cs_numbers = np.arange(5,75)
# cs_numbers = data[us].unique()



#!!!!!!!!!!!!!!!!!!!!!!!!!                 I THINK THE CS TIMES ARE WRONG



fig, axs = plt.subplots(cs_numbers.size, 1, figsize=(10, 90), sharex=True, sharey=True)

# fig, axs = plt.subplots(3, 1, figsize=(10, 50))

# fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)

for stim_number_i, stim_number in enumerate(cs_numbers):


	data_ = data.loc[data[cs] == stim_number, abs_time]

	cs_beg_ = data_.iat[0]
	cs_end_ = data_.iat[-1]

	#!!!!!!!!!! print(stim_number_i, cs_end_ - cs_beg_)


	data_plot = data.loc[data[abs_time].between(cs_beg_-bef_stim_onset, cs_end_+aft_stim_offset)]


	axs[stim_number_i].plot(data_plot[abs_time].to_numpy() - cs_beg_, data_plot['Plane'].to_numpy(), 'k.')
	# axs[stim_number_i].plot(data_plot[abs_time].to_numpy() - cs_beg_, data_plot['Image mean'].to_numpy(), 'k.')
	# axs[stim_number_i].plot(data_plot[abs_time].to_numpy() - cs_beg_, data_plot['Angle (deg) 14'].diff().to_numpy(), 'k.')
	# axs[stim_number_i].plot(data_plot[abs_time].to_numpy() - cs_beg_, data_plot[galvo_value].to_numpy() + 100, 'bo')
	#! axs[stim_number_i].plot(data__[abs_time].to_numpy() - cs_beg_, data__[galvo_value].to_numpy(), 'm')

	axs[stim_number_i].axvline(x=cs_beg_ - cs_beg_, color='g', linestyle='-')
	axs[stim_number_i].axvline(x=cs_end_ - cs_beg_, color='r', linestyle='-')
	axs[stim_number_i].set_title(f"Stimulus Number: {stim_number}")
	# plt.plot(time, data__[galvo_value]], 'k')
	# plt.plot(time, data__['Frame beg'], 'bo')
	# plt.plot(time, data__[cs], 'yo')
	# plt.plot(time, data__[us_end], 'mo')

	# data__.plot(x=abs_time, y=['Frame beg', 'Image mean', cs, us_end], ls='.')

fig.show()

del data_, data_plot











plt.plot(planes_numbers)
# plt.plot(planes_numbers_)


a = np.array([plane for plane in planes_numbers if plane is not np.nan])

plt.plot(a)
plt.plot(a_diff)

a_diff = np.diff(a)

len(np.where(abs(a_diff) >= 3)[0])
ind = np.where(abs(a_diff) >= 3)[0]
plt.plot(a[ind])
a
# planes_numbers__ = planes_numbers_.copy()
# planes_numbers_ = planes_numbers__.copy()


#TODO 

planes_numbers_ = np.array([np.nan if plane is None else plane for plane in planes_numbers])






NOW FILL IN THE GAPS

planes_numbers




















planes_numbers_ = planes_numbers_.fillna(method='ffill')

arr = planes_numbers_


mask = np.isnan(arr)

idx = np.where(~mask, np.arange(len(mask)), 0)



arr[mask] = arr[np.nonzero(mask)[0], idx[mask]]




plt.plot(np.diff(planes_numbers_))


np.diff(planes_numbers_)

np.where(np.diff(planes_numbers_) > 3 )[0]


































save tiff 
set to correct dtype








import h5py
import datetime

with h5py.File(r"I:\20240314_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf\Data.hdf", "w") as file:
	file.create_dataset("data", data=data, compression="gzip")