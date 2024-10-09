import math
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from importlib import reload
from pathlib import Path
from timeit import default_timer as timer

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.io as pio
import scipy.ndimage as ndimage
import seaborn as sns
import tifffile
import xarray as xr
from pandas.api.types import CategoricalDtype
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from scipy import interpolate, signal
from scipy.ndimage import shift
from scipy.stats import pearsonr, zscore
from skimage import morphology
from skimage.measure import block_reduce
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

import my_classes as c
import my_functions as f
import my_parameters as p
from my_general_variables import *

# from my_experiment_specific_variables import expected_number_cs

plt.style.use('classic')





def create_folders(path_home):

	path_lost_frames = path_home / 'Lost frames'
	path_lost_frames.mkdir(parents=True, exist_ok=True)

	path_summary_exp = path_home / 'Summary of protocol actually run'
	path_summary_exp.mkdir(parents=True, exist_ok=True)

	path_summary_beh = path_home / 'Summary of behavior'
	path_summary_beh.mkdir(parents=True, exist_ok=True)


	#* Path to save processed data; create folder to save processed data if it does not exist yet.
	path_processed_data = path_home / 'Processed data'
	path_processed_data.mkdir(parents=True, exist_ok=True)


	path_cropped_exp_with_bout_detection = path_processed_data / '1. summary of exp.'
	path_cropped_exp_with_bout_detection.mkdir(parents=True, exist_ok=True)

	path_tail_angle_fig_cs = path_processed_data / '2. single fish_tail angle' / 'aligned to CS'
	path_tail_angle_fig_cs.mkdir(parents=True, exist_ok=True)

	path_tail_angle_fig_us = path_processed_data / '2. single fish_tail angle' / 'aligned to US'
	path_tail_angle_fig_us.mkdir(parents=True, exist_ok=True)

	path_raw_vigor_fig_cs = path_processed_data / '3. single fish_raw vigor heatmap' / 'aligned to CS'
	path_raw_vigor_fig_cs.mkdir(parents=True, exist_ok=True)
	
	path_raw_vigor_fig_us = path_processed_data / '3. single fish_raw vigor heatmap' / 'aligned to US'
	path_raw_vigor_fig_us.mkdir(parents=True, exist_ok=True)

	path_scaled_vigor_fig_cs = path_processed_data / '4. single fish_scaled vigor heatmap' / 'aligned to CS'
	path_scaled_vigor_fig_cs.mkdir(parents=True, exist_ok=True)
	
	path_scaled_vigor_fig_us = path_processed_data / '4. single fish_scaled vigor heatmap' / 'aligned to US'
	path_scaled_vigor_fig_us.mkdir(parents=True, exist_ok=True)

	path_suppression_ratio_fig_cs = path_processed_data / '5. single fish_suppression ratio vigor trial' / 'aligned to CS'
	path_suppression_ratio_fig_cs.mkdir(parents=True, exist_ok=True)

	path_suppression_ratio_fig_us = path_processed_data / '5. single fish_suppression ratio vigor trial' / 'aligned to US'
	path_suppression_ratio_fig_us.mkdir(parents=True, exist_ok=True)
	
	path_pooled_vigor_fig = path_processed_data / 'All fish'
	path_pooled_vigor_fig.mkdir(parents=True, exist_ok=True)

	path_analysis_protocols = path_processed_data / 'Analysis of protocols'
	path_analysis_protocols.mkdir(parents=True, exist_ok=True)

	path_pkl = path_processed_data / 'pkl files'
	path_pkl.mkdir(parents=True, exist_ok=True)


	path_orig_pkl = path_pkl / '1. Original'
	path_orig_pkl.mkdir(parents=True, exist_ok=True)

	path_all_fish = path_pkl / '2. All fish by condition'
	path_all_fish.mkdir(parents=True, exist_ok=True)

	path_pooled = path_pkl / '3. Pooled data'
	path_pooled.mkdir(parents=True, exist_ok=True)

	return path_lost_frames, path_summary_exp, path_summary_beh, path_processed_data, path_cropped_exp_with_bout_detection, path_tail_angle_fig_cs, path_tail_angle_fig_us, path_raw_vigor_fig_cs, path_raw_vigor_fig_us, path_scaled_vigor_fig_cs, path_scaled_vigor_fig_us, path_suppression_ratio_fig_cs, path_suppression_ratio_fig_us, path_pooled_vigor_fig, path_analysis_protocols, path_orig_pkl, path_all_fish, path_pooled





def msg(stem_fish_path_orig, message):
	
	if type(message) is list:
		message = '\t'.join([str(i) for i in message])

	return [stem_fish_path_orig] + ['\t' + message + '\n']

	# return [stem_fish_path_orig + '\t' + message + '\n']

def save_info(protocol_info_path, stem_fish_path_orig, message):

	message = msg(stem_fish_path_orig, message)
	print(message)

	with open(protocol_info_path, 'a') as file:
		file.writelines(message)

def fish_id(stem_path):
	# Info about a specific fish.
	
	stem_fish_path = stem_path.lower()
	stem_fish_path = stem_fish_path.split('_')
	day = stem_fish_path[0]

	# strain = stem_fish_path[1]
	# age = stem_fish_path[2].replace('dpf', '')
	# exp_type = stem_fish_path[3]
	# rig = stem_fish_path[4]
	# fish_number = stem_fish_path[5].replace('fish', '')

	fish_number = stem_fish_path[1]
	exp_type = stem_fish_path[2]
	rig = stem_fish_path[3]
	strain = stem_fish_path[4]
	age = stem_fish_path[5].replace('dpf', '')
	

	return day, strain, age, exp_type, rig, fish_number

def read_initial_abs_time(camera_path):
	# Read the absolute time at the beginning of the experiment.

	try:
		with open(camera_path, 'r') as f:
			f.readline()
		# Previous version.
			# first_frame_absolute_time = int(float(f.readline().strip('\n').split('\t')[2]))

		# return first_frame_absolute_time

			return int(float(f.readline().strip('\n').split('\t')[2]))

	except:
		print('No absolute time in cam file.')

		return None

# def read_camera(camera_path) -> pd.DataFrame:

# 	try:
# 		start = timer()
		
# 		# camera = pd.read_csv(str(camera_path), sep='\t', header=0, decimal='.', skiprows=[*range(1,number_frames_discard_beg)])
# 		camera = pd.read_csv(str(camera_path), sep=' ', header=0, decimal='.', skiprows=[*range(1,number_frames_discard_beg)])
# 		# {'FrameID' : 'int', ela_time : 'float', abs_time : 'int'}
# 		print('Time to read cam.txt: {} (s)'.format(timer()-start))

# 		camera.rename(columns={'TotalTime' : ela_time}, inplace=True)
# 		camera.rename(columns={'ID' : frame_id}, inplace=True)
		
# 		return camera

# 	except:

# 		print('Cannot read camera file.')
		
# 		return None


def read_sync_reader(sync_reader_path):

	try:
		start = timer()
		
		sync_reader = pd.read_csv(str(sync_reader_path), sep=' ', header=0, decimal='.')
		
		print('Time to read scape sync reader.txt: {} (s)'.format(timer()-start))

		# sync_reader.rename(columns={'Time' : ela_time}, inplace=True)
		
		return sync_reader

	except:
		# print('Cannot read scape sync file.')

		return None


# def framerate_and_reference_frame(camera, first_frame_absolute_time, protocol_info_path, stem_fish_path_orig, fig_camera_name):
	
# 	camera.loc[:,[ela_time]] = camera.loc[:,[ela_time]].astype('float')
	
# 	camera_diff = camera[ela_time].diff()

# 	print('Max IFI: {} ms'.format(camera_diff.max()))
	
# 	# First estimate of the interframe interval, using the median
# 	ifi = camera_diff.median()
# 	# camera_diff.iloc[number_frames_discard_beg : ].median()
# 	print('First estimate of IFI: {} ms'.format(ifi))


# 	camera_diff_index_right_IFI = np.where(abs(camera_diff - ifi) <= max_interval_between_frames)[0]

# 	camera_diff_index_right_IFI_diff = np.diff(camera_diff_index_right_IFI)

# 	#* Find a region at the beginning where the IFI from frame to frame does not vary significantly and is similar to the first estimate of the true IFI (ifi).
# 	for i in range(1, len(camera_diff_index_right_IFI_diff)):

# 		if camera_diff_index_right_IFI_diff[i-1] == 1 and camera_diff_index_right_IFI_diff[i] == 1:

# 			reference_frame_id = camera[frame_id].iloc[camera_diff_index_right_IFI[i] - 1]

# 			# first_frame_absolute_time is not None when there is absolute time in the cam file.
# 			if first_frame_absolute_time is not None:
# 				reference_frame_time = first_frame_absolute_time + camera[ela_time].iloc[camera_diff_index_right_IFI[i] - 1] - camera[ela_time].iloc[0]
# 			else:
# 				reference_frame_time = None

# 			break

# 	#* Find a similar region but at the end of the experiment.
# 	for i in range(len(camera_diff_index_right_IFI_diff)-1, 0, -1):

# 		if camera_diff_index_right_IFI_diff[i-1] == 1 and camera_diff_index_right_IFI_diff[i] == 1:
			
# 			last_frame_id = camera[frame_id].iloc[camera_diff_index_right_IFI[i] - 1]
# 			#last_frame_time = first_frame_absolute_time + camera[time].iloc[camera_diff_index_right_IFI[i] - 1] - camera[time].iloc[0]

# 			break


# 	#* Second estimate of the interframe interval, using the mean, and assuming there is no increasing accumulation of frames in the buffer during the experiment; Only the region between the two frames identified in the previous two for loops is considered.
# 	ifi = camera_diff.iloc[reference_frame_id - camera[frame_id].iloc[0] : last_frame_id - camera[frame_id].iloc[0]].mean()

# 	print('Second estimate of IFI: {} ms'.format(ifi))
# 	predicted_framerate = 1000 / ifi
# 	print('Estimated framerate: {} FPS'.format(predicted_framerate))


# 	def lost_frames(camera, camera_diff, ifi, protocol_info_path, stem_fish_path_orig, fig_camera_name):
		
# 		return False

# 		# # Delay to capture frames by the computer
# 		# delay = (camera_diff - ifi).cumsum().to_numpy()


# 		# # Number of lost frames
# 		# # More than one frame might be lost and number_frames_lost sometimes is not monotonically crescent (can go down when some ms are 'recovered').

# 		# print(delay)
# 		# print(ifi)
# 		# number_frames_lost = np.floor(delay / (ifi * buffer_size))
# 		# #TODO use this to speed up
# 		# # number_frames_lost = np.max(number_frames_lost, 0, axis)
# 		# number_frames_lost = np.where(number_frames_lost>=0, number_frames_lost, 0)

# 		# number_frames_lost_diff = np.floor(np.diff(number_frames_lost))
# 		# number_frames_lost_diff = np.where(number_frames_lost_diff>=0,number_frames_lost_diff,0)


# 		# # Indices where frames were potentially lost
# 		# where_frames_lost = np.where(number_frames_lost_diff > 0)[0]


# 		# # Total number of missed frames
# 		# if (Lost_frames := len(where_frames_lost) > 0):
# 		# 	print('Total number of lost frames: ', len(where_frames_lost))
# 		# 	print('Where: ', where_frames_lost)
# 		# 	save_info(protocol_info_path, stem_fish_path_orig, 'Lost frames.')
# 		# else:
# 		# 	print('No frames were lost.')

# 		# fig, axs = plt.subplots(5, 1, sharex=True, facecolor='white', figsize=(20, 40), constrained_layout=True)

# 		# axs[0].plot(camera.iloc[:,1],'k')
# 		# axs[0].set_ylabel('Elapsed time (ms)')
# 		# axs[0].set_title('Estimated IFI: {} ms.    Estimated framerate: {} FPS'.format(round(ifi, 3), round(predicted_framerate, 3)))

# 		# axs[1].plot(camera_diff,'k')
# 		# axs[1].set_ylabel('IFI (ms)')

# 		# axs[2].plot(delay,'k')
# 		# axs[2].set_ylabel('Delay (ms)')

# 		# axs[3].plot(number_frames_lost_diff.cumsum(),'k')
# 		# axs[3].set_ylabel('Cumulative number of lost frames')

# 		# axs[4].plot(number_frames_lost_diff,'black')
# 		# # axs[4].set_xlabel('frame number')
# 		# axs[4].set_ylabel('Lost frames')

# 		# fig.supxlabel('Frame number')
# 		# plt.suptitle('Analysis of lost frames\n' + stem_fish_path_orig)

# 		# fig.savefig(fig_camera_name, dpi=100, facecolor='white')
# 		# plt.close(fig)


# 		# # Correct frame IDs in camera dataframe.
# 		# # correctedID = np.zeros(len(camera))

# 		# # for i in tqdm(where_frames_lost):
# 		# # 	correctedID[i:number_frames_diff] += 1 # And not correctedID[i:] += 1 because, when the buffer is full, the Mako U29-B camera keeps what is already in the buffer and does not receive any new frames while the buffer is full.

# 		# # del where_frames_lost, number_frames_lost_diff

# 		# # camera['Corrected ID'] = camera['ID'] + correctedID
# 		# # camera['Corrected ID'] = camera['Corrected ID'].astype('int')

# 		# # # Second estimate of the interframe interval, using the median, and after estimating where there are missing frames 
# 		# # camera_diff = camera.loc[:,'ElapsedTime'].diff()
# 		# # ifi = camera_diff.iloc[number_frames_discard_beg : -number_frames_discard_beg].median()
# 		# # print('\nFirst estimate of IFI: {} ms'.format(ifi))

# 		# return Lost_frames


# 	Lost_frames = lost_frames(camera, camera_diff, ifi, protocol_info_path, stem_fish_path_orig, fig_camera_name)




	# return predicted_framerate, reference_frame_id, reference_frame_time, Lost_frames

# def read_protocol(protocol_path:Path, reference_frame_time_or_id, protocol_info_path, stem_fish_path_orig) -> pd.DataFrame:


# 	#* Read protocol file.
# 	if Path(protocol_path).exists():
# 		# Discarding the last column, which contains the cumulative number of bouts identified in C#.
# 		protocol = pd.read_csv(str(protocol_path), sep=' ', header=0, names=[experiment_type, beg, end], usecols=[0, 1, 2], index_col=0)

# 	else:
# 		save_info(protocol_info_path, stem_fish_path_orig, 'stim control file does not exist.')
		
# 		return None

# 	#* Were the stimuli timings not saved?
# 	if protocol.empty:
# 		save_info(protocol_info_path, stem_fish_path_orig, 'stim control file is empty.')
		
# 		return None


# 	if protocol.iloc[0,0] == 0:
		
# 		return None
	
# 	# #* Is the first stimulus fake? This happened at some point. There was sometimes a line in protocol file in excess.
# 	# if protocol.loc[:,beg].iloc[0] == 0 and len(protocol.loc[protocol.index.get_level_values(experiment_type) == 'Cycle&Bout']) == expected_number_cs+1:
# 	# 	save_info(protocol_info_path, stem_fish_path_orig, 'Lost beginning of first cycle.')
		
# 	# 	return None
	
# 	# protocol.rename(index={'Cycle&Bout': 'Cycle'}, inplace=True)
# 	protocol.sort_values(by=beg, inplace=True)

# 	# if reference_frame_time is not None:
# 		# Getting here means that there is absolute time in the cam file.
# 		# protocol = protocol - reference_frame_time
# 	protocol = protocol - reference_frame_time_or_id

# 	return protocol

# def protocol_info(protocol):

# 	#* Count the number of cycles, trials, blocks and bouts.
# 	# Using len() just in case these is a single element.
# 	number_cycles = len(protocol.loc['Cycle', beg])

# 	if protocol.index.isin(['Session']).any():
# 		number_blocks = len(protocol.loc['Session', beg])
# 	else:
# 		number_blocks = number_cycles

# 	if protocol.index.isin([trial]).any():
# 		number_trials = len(protocol.loc[trial, beg])
# 	else:
# 		number_trials = number_cycles

# 	if protocol.index.isin([bout]).any():
# 		number_bouts = len(protocol.loc[bout, beg])	  
# 	else:
# 		number_bouts = 0
		
# 	if protocol.index.isin(['Reinforcer']).any():
# 		number_reinforcers = len(protocol.loc['Reinforcer', beg])

# 		us_beg = protocol.loc['Reinforcer', beg]
# 		us_end = protocol.loc['Reinforcer', end]
# 		us_dur = (us_end - us_beg).to_numpy() # in ms
# 		us_isi = (us_beg[1:] - us_end[:-1]).to_numpy() / 1000 / 60 # min
# 	else:
# 		number_reinforcers = 0

# 		us_dur = None
# 		us_isi = None

# 	habituation_duration = protocol.iloc[0,0] / 1000 / 60 # min

# 	cs_beg = protocol.loc['Cycle', beg]
# 	cs_end = protocol.loc['Cycle', end]
# 	cs_dur = (cs_end - cs_beg).to_numpy() # in ms
# 	cs_isi = (cs_beg[1:] - cs_end[:-1]).to_numpy() / 1000 / 60 # min


# 	return number_cycles, number_reinforcers, number_trials, number_blocks, number_bouts, habituation_duration, cs_dur, cs_isi, us_dur, us_isi

def map_abs_time_to_elapsed_time(camera, protocol):
	
	stimuli = protocol.index.unique()

	camera[abs_time] = camera[abs_time].astype('float')
	
	camera[ela_time] = camera[ela_time].astype('float')
	
	for beg_end in [beg, end]:

		protocol_ = protocol.loc[:,beg_end].reset_index().rename(columns={beg_end : abs_time})

		camera_protocol = pd.merge_ordered(camera, protocol_).set_index(abs_time).interpolate(kind='slinear').reset_index()

		#* Because here I am relying on "absolute time" (UNIX time, which has ms-resolution), some rows in the original camera dataframe may have the same value of absolute time.
		camera_protocol = camera_protocol.drop_duplicates(abs_time, keep='first')

		# protocol.loc['Cycle',beg_end] = camera_protocol[camera_protocol[experiment_type]=='Cycle'].set_index(experiment_type).loc[:,ela_time]
		# protocol.loc['Reinforcer',beg_end] = camera_protocol[camera_protocol[experiment_type]=='Reinforcer'].set_index(experiment_type).loc[:,ela_time]
		# protocol.loc[:,beg_end] = camera_protocol[camera_protocol[experiment_type].notna()].set_index(experiment_type).loc[:,ela_time].to_numpy()

		for stim in stimuli:

			if len(camera_protocol[camera_protocol[experiment_type]==stim].set_index(experiment_type).loc[:,ela_time]) == 1:
				
				protocol.loc[stim,beg_end] = camera_protocol[camera_protocol[experiment_type]==stim].set_index(experiment_type).loc[:,ela_time].to_numpy()[0]
				
			else:
				
				protocol.loc[stim,beg_end] = camera_protocol[camera_protocol[experiment_type]==stim].set_index(experiment_type).loc[:,ela_time]

	return protocol[protocol.notna().all(axis=1)]


# def lost_stim(number_cycles, number_reinforcers, min_number_cs_trials, min_number_us_trials, protocol_info_path, stem_fish_path_orig, id_debug):

# 	if number_cycles < min_number_cs_trials:

# 		save_info(protocol_info_path, stem_fish_path_orig, 'Not all CS! Stopped at CS {} ({}).'.format(number_cycles, id_debug))

# 		return True

# 	elif number_reinforcers < min_number_us_trials:
		
# 		save_info(protocol_info_path, stem_fish_path_orig, 'Not all US! Stopped at US {} ({}).'.format(number_reinforcers, id_debug))

# 		return True
# 	else:
# 		return False

# def plot_protocol(cs_dur, cs_isi, us_dur, us_isi, stem_fish_path_orig, fig_protocol_name):

# 	plt.figure(figsize=(14,14))
# 	plt.plot(np.arange(1, len(cs_isi) + 1), cs_isi, label='inter-cs interval\nmin int.=' + str(round(np.amin(cs_isi)*60,1)) + ' s\n' + 'cs min dur=' + str(round(np.amin(cs_dur)/1000,3)) + ' s\n' + 'cs max dur=' + str(round(np.amax(cs_dur)/1000,3)) + ' s')
	
# 	plt.plot(np.arange(5, 4+len(us_isi)+1), us_isi, label='inter-us interval\nmin int.=' + str(round(np.amin(us_isi)*60,1)) + ' s\n' + 'us min dur=' + str(round(np.amin(us_dur)/1000,3)) + 's\n' + 'us max dur='+ str(round(np.amax(us_dur)/1000,3)) + ' s')
# 	plt.xlabel('Trial number')
# 	plt.ylabel('ISI (min)')
# 	plt.ylim(0, 10)
# 	plt.legend(frameon=False, loc='upper center', ncol=2)
# 	plt.suptitle('Summary of protocol\n' + stem_fish_path_orig)
# 	plt.savefig(fig_protocol_name, dpi=100, bbox_inches='tight')
# 	plt.close()

def number_frames_discard(data_path, reference_frame_id):
	# Consider the experiment starts only whith the first frame whose ID is both in tail tracking and camera files.

	with open(data_path, 'r') as f:
		f.readline()
		tracking_frames_to_discard = 0
		while reference_frame_id != int(f.readline().split(' ')[0]):
			tracking_frames_to_discard += 1

	return tracking_frames_to_discard

#def readTailTracking(data_path, protocol_frame, tracking_frames_to_discard, time_bcf_window, time_max_window, time_min_window, time_bef_frame, time_aft_frame):
	#	# protocol in number of frames


	#	start = timer()
	#	extra_time_window = np.max([time_bcf_window, time_max_window, time_min_window])

	#	protocol_frame += tracking_frames_to_discard
	#	protocol_frame[beg] = protocol_frame[beg] + time_bef_frame- 2*extra_time_window
	#	protocol_frame[end] = protocol_frame[end] + time_aft_frame + 2*extra_time_window

	#	number_frames = protocol_frame[end].max()
	#	rows_to_skip = []
	#	number_rows = None
	#	for i in range(len(protocol_frame)):
	#		if i == 0:
	#			# 1 is required to avoid removing the names of the columns
	#			rows_to_skip.extend(np.arange(1, protocol_frame.iat[i,0]))
	#		else:
	#			if (b:= protocol_frame[beg].iat[i]) - (a:= protocol_frame[end].iloc[:i].max()) > 0:
	#				rows_to_skip.extend(np.arange(a, b))

	#	# frames = np.arange(reference_frame_id - tracking_frames_to_discard, last_frame)
	#	# # frames = pd.read_csv(data, sep=' ', usecols=[0], decimal=',', dtype='int64', engine='c', squeeze=True)
	#	# # frames -= reference_frame_id

	#	# mask_frames = np.zeros(len(frames), dtype=bool)

	#	# for i in range(len(protocol)):
	#	# 	mask_frames |= ((frames >= protocol.iat[i,0]) & (frames <= protocol.iat[i,1]))

	#	# rows_to_skip = np.arange(len(mask_frames))[~mask_frames]+1
	#	number_rows = number_frames - len(rows_to_skip)
		
	#	data = pd.read_csv(data_path, sep=' ', header=0, usecols=cols_to_use_orig, nrows=number_rows, skiprows=rows_to_skip, decimal=',')
		
	#	print(timer()-start)
		
	#	return data



# def read_tail_tracking_data(data_path, reference_frame_id):
# # number_frames
# 	# Angles in data come in radians.

# 	start = timer()

# 	try:

# 		# na_filter=False to speed up.
# 		# skipfooter=1 because in one file there were NaN's in, and only in, the last line.
# 		data = pd.read_csv(data_path, sep=' ', header=0, usecols=cols_to_use_orig, decimal='.', engine='pyarrow') #, skipfooter=1)
# 		# na_filter=False
# 		# nrows=number_frames
# 		data = data.iloc[:-1,:]

# 		#! To correct a corrupted file (20220503_Tu_6dpf_delay_test_3-black_fish8).
# 		# data = pd.read_csv(data_path, sep=' ', header=0, usecols=cols_to_use_orig, decimal='.', na_filter=False)
# 		# data = data.iloc[:-1,:]
# 		# for col in data.columns[1:]:
# 		# 	data.loc[:,col] = data.loc[:,col].str.replace(',','.')
# 		# data.iloc[:,1:] = data.iloc[:,1:].astype('float32')

# 	except:
# 		try:
			
# 			data = pd.read_csv(data_path, sep=' ', header=0, usecols=cols_to_use_orig, decimal=',', engine='pyarrow')

# 		except:
# 			return None

# 	print('Time to read tail tracking .txt: {} (s)'.format(timer()-start))

# 	# data.iloc[:,0] = data.iloc[:,0].astype('int')
# 	data.loc[:,frame_id] = data.loc[:, frame_id] - reference_frame_id
# 	data = data[data[frame_id] >= 0]

# 	#? maybe before this was necessary because "decimal" in pd.read_csv was set to ",".
# 	#* Even if decimal separator is wrong, this will correct it.
# 	data.iloc[:,1:] = data.iloc[:,1:].astype('float32')

# 	# Convert tail tracking data from radian to degree
# 	data.iloc[:,1:] = data.iloc[:,1:] * (180/np.pi)
	
# 	# Rename columns
# 	data.rename(columns=dict(zip(cols_to_use_orig[1:], cols[1:])), inplace=True)

# 	return data






# def tracking_errors(data, single_point_tracking_error_thr):

# 	errors = False

# 	if ((a := data.iloc[:,1:].abs().max()) > single_point_tracking_error_thr).any():
# 		print('Possible tracking error! Max(abs(angle of individual point)):')
# 		print(a)

# 		errors = True

# 	if data.iloc[:,1:].isna().to_numpy().any():
# 		print('Possible tracking failures. There are NAs in data!')

# 		errors = True

# 	return errors

# def interpolate_data(data, expected_framerate, predicted_framerate):
# 	# expected_framerate is the framerate to which data is interpolated. So, output data is as if it had been acquired at the expected_framerate (700 FPS when I wrote this).


# 	data.iloc[:,0] = data.iloc[:,0] * expected_framerate/predicted_framerate
	
# 	interp_function = interpolate.interp1d(data.iloc[:,0], data.iloc[:,1:], kind='slinear', axis=0, assume_sorted=True, bounds_error=False, fill_value="extrapolate")

# 	data_ = pd.DataFrame(np.arange(data.iat[0,0], data.iat[-1,0]), columns=['Time (frame) [{} FPS]'.format(expected_framerate)])
# 	data_[data.columns[1:]] = interp_function(data_.iloc[:,0])

# 	# Old
# 		# # data_['Original'] = True

# 		# # # data.iloc[:,0] = (data.iloc[:,0] * 1000/predicted_framerate)  # ms

# 		# # Create a dataframe with what is going to be the index for interpolation.
# 		# data_ = pd.DataFrame(np.nan((len(timepoints_at_expected_framerate), data.shape[1])), columns=data.columns.to_numpy(), dtype='float32')
# 		# data_.iloc[:,0] = timepoints_at_expected_framerate
# 		# data_['Original'] = False

# 		# print('!!!!!!!!!!', data)
# 		# pd.concat([data.set_index(data.columns[0]), data_.set_index(data.columns[0])], axis=0, join='outer', copy=False).sort_index().reset_index().drop_duplicates(subset=[data.columns[0]], inplace=True)
# 		# print('????????????', data)


# 		# data.interpolate(method='slinear', axis=0, inplace=True, copy=False, assume_sorted=True)


# 		# data.iloc[:,0] = np.arange(0, len(data))

# 		#s Rename column with time.
# 		# # data.rename(columns={data.columns[0]: 'Time (frame) [{} FPS]'.format(expected_framerate)}, inplace=True)

# 	return data_

def rolling_window(a, window):

	#* Alexandre Laborde confirmed this.
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def filter_data(data, space_bcf_window, time_bcf_window):

	#* Select just the part of data to change.
	data_ = data.loc[:, [data.columns[0]] + cols[1:]]


#! I think it does not make sense to do this
	#* Calculate the cumulative sum in space
	# This works as a filter in space
	data_.iloc[:, 1:] = data_.iloc[:, 1:].cumsum(axis=1)

	#* Filter with a rolling average in space
	#* Alexandre Laborde confirmed this.
	# Not using pandas rolling mean beacause over columns it takes a lot of time (confirmed that with this way the result is the same)
	# The fact that here we are using the cumsum means that when averaging more importance is given to the first points
	data_.iloc[:, 2:-1] = np.mean(rolling_window(data_.iloc[:, 1:].to_numpy(), space_bcf_window), axis=2)
	data_.iloc[:, 1] = data_.iloc[:, 1:3].mean(axis=1)
	data_.iloc[:, -1] = data_.iloc[:, -2:].mean(axis=1)

	# This does not work with data_ in float32. Might be a bug of Pandas.
	# Too slow. Use alternative above.
	# data_.iloc[:, 1:] = data_.astype('float').iloc[:, 1:].rolling(window=space_bcf_window, center=True, axis=1).mean()

	#* Filter with a rolling average in time
	data_.iloc[:, 1:] = data_.iloc[:, 1:].rolling(window=time_bcf_window, center=True, axis=0).mean()
	
	# Filter eye tracking data_
	# if eyedata_:	
	# 	data_.loc[:, right_eye_angle] = data_.loc[:, right_eye_angle].rolling(window=time_bcf_window, center=center_window).mean().astype('float32')
	# 	data_.loc[:, left_eye_angle] = data_.loc[:, left_eye_angle].rolling(window=time_bcf_window, center=center_window).mean().astype('float32')

	#* Update data with the values changed in data_.
	data.loc[:, [data.columns[0]] + cols[1:]] = data_
	
	data = data.dropna()

	return data

def vigor_for_bout_detection(data, chosen_tail_point, time_min_window, time_max_window):
	# Calculate 'vigor_bout_detection' (deg/ms)
	#! JUST TRY THIS
	#! data.loc[:, vigor_bout_detection] = (data.iloc[:,1:2+chosen_tail_point].diff(axis=1).diff().rolling(window=7, center=True, axis=0).mean() * expected_framerate / 1000).pow(2).sum(axis=1)

	#* Calculate the cumulative sum of the angular velocity over space.
	#* This allows to take into account movement in any segment with a single scalar value.
	data.loc[:, vigor_bout_detection] = data.iloc[:,1:2+chosen_tail_point].diff().abs().sum(axis=1) * (expected_framerate / 1000) # deg/ms
	
	#* Calculate the abstract measure defined by me as 'vigor_bout_detection'.
	data.loc[:, vigor_bout_detection] = data.loc[:, vigor_bout_detection].rolling(window=time_max_window, center=True, axis=0).max() - data.loc[:, vigor_bout_detection].rolling(window=time_min_window, center=True, axis=0).min()
	
	data.dropna(inplace=True)

	return data

def stim_in_data(data, protocol):
	# , number_cycles, number_reinforcers, exp_type):

	#region Merge protocol with data
	data[cols_stim[:4]] = 0


	# TODO case of Operant Conditioning.
	#if exp_type == 'OC':

		#data['Trial beg'] = 0
		#data['Trial end'] = 0
		#data['Reinforcer beg'] = 0
		#data['Reinforcer end'] = 0
	
		#if protocol.index.isin(['Session']).any():
		
		#	# For operant conditioning experiments
		#	data['Session beg'] = 0
		#	data['Session end'] = 0

		#	for i in range(number_blocks):
		#		data.loc[data.iloc[:,0] == protocol.loc['Session', beg].iat[i], 'Session beg'] = int(i + 1)
		#		data.loc[data.iloc[:,0] == protocol.loc['Session', end].iat[i], 'Session end'] = int(i + 1)
	
		#if protocol.index.isin(['Trial']).any():
		
		#	# For operant conditioning experiments	
		#	for i in range(number_trials):
		#		data.loc[data.iloc[:,0] == protocol.loc[trial, beg].iat[i], 'Trial beg'] = int(i + 1)
		#		data.loc[data.iloc[:,0] == protocol.loc[trial, end].iat[i], 'Trial end'] = int(i + 1)

		#	data['Cycle beg'] = 0
		#	data['Cycle end'] = 0

		#	for i in range(number_cycles):
			
		#		data.loc[data.iloc[:,0] == protocol.loc['Cycle', beg].iat[i], 'Cycle beg'] = int(i + 1)
		#		data.loc[data.iloc[:,0] == protocol.loc['Cycle', end].iat[i], 'Cycle end'] = int(i + 1)
		
		#else:
		#	for i in range(number_cycles):
		#		data.loc[data.iloc[:,0] == protocol.loc['Cycle', beg].iat[i], 'Trial beg'] = int(i + 1)
		#		data.loc[data.iloc[:,0] == protocol.loc['Cycle', end].iat[i], 'Trial end'] = int(i + 1)
		
		#if protocol.index.isin(['Reinforcer']).any():
		#	for i in range(number_reinforcers):
		#		data.loc[data.iloc[:,0] == protocol.loc['Reinforcer', beg].iat[i], 'Reinforcer beg'] = int(i + 1)
		#		data.loc[data.iloc[:,0] == protocol.loc['Reinforcer', end].iat[i], 'Reinforcer end'] = int(i + 1)
	
	# 	data['Session beg', 'Trial beg', 'Cycle beg', 'Reinforcer beg', 'Session end', 'Trial end', 'Cycle end', 'Reinforcer end'] = data['Session', trial, 'Cycle', 'Reinforcer'].astype('category')

	#if exp_type != 'OC':	

	for i, [cs_b, cs_e] in enumerate(protocol.loc['Cycle', [beg, end]].to_numpy()):
		# print('hi')
		data.loc[data.iloc[:,0].astype('int') == cs_b, cs_beg] = int(i + 1)
		data.loc[data.iloc[:,0].astype('int') == cs_e, cs_end] = int(i + 1)
		
	if protocol.index.isin(['Reinforcer']).any():

		for i, [us_b, us_e] in enumerate(protocol.loc['Reinforcer', [beg, end]].to_numpy()):
			data.loc[data.iloc[:,0].astype('int') == us_b, us_beg] = int(i + 1)
			data.loc[data.iloc[:,0].astype('int') == us_e, us_end] = int(i + 1)


	# data.loc[:,cols_stim[:4]] = data.loc[:,cols_stim[:4]].astype('category')
	data.loc[:, cs_beg] = data.loc[:, cs_beg].astype(CategoricalDtype(categories=data[cs_beg].unique().sort(), ordered=True))		
	data.loc[:, us_beg] = data.loc[:, us_beg].astype(CategoricalDtype(categories=data[us_beg].unique().sort(), ordered=True))
	data.loc[:, cs_end] = data.loc[:, cs_end].astype(CategoricalDtype(categories=data[cs_end].unique().sort(), ordered=True))
	data.loc[:, us_end] = data.loc[:, us_end].astype(CategoricalDtype(categories=data[us_end].unique().sort(), ordered=True))

	return data

# def plot_behavior_overview(data, stem_fish_path_orig, fig_behavior_name):
# 	# data containing tail_angle.

# 	# mask_frames = np.ones(number_frames + round(60*framerate), dtype=bool)
# 	# mask_frames[:: round(framerate * 0.5)] = False
# 	# mask_frames[0] = False
	
# 	# rows_to_skip = np.arange(number_frames + round(60*framerate))
# 	# rows_to_skip = rows_to_skip[mask_frames]

# 	# start = timer()

# 	# overall_data = pd.read_csv(data, sep=' ', header=0, usecols=cols, skiprows=rows_to_skip, decimal=',')
# 	# overall_data = overall_data.astype('float32')

# 	# print(timer() - start)
# 	plt.figure(figsize=(28, 14))
# 	plt.plot(data.iloc[:,0]/expected_framerate/60/60, data.iloc[:,1	+chosen_tail_point], 'black')
# 	plt.xlabel('Time (h)')
# 	plt.ylabel('Tail end angle (deg)')
# 	plt.suptitle('Behavior overview\n' + stem_fish_path_orig)
# 	# plt.show()
# 	# plt.legend(frameon=False, loc='upper center', ncol=2)
# 	plt.savefig(fig_behavior_name, dpi=100, bbox_inches='tight')
# 	plt.close()

def extract_data_around_stimuli(data, protocol_frame, time_bef_frame, time_aft_frame, time_bcf_window, time_max_window, time_min_window):

	# protocol_frame is the protocol in number of frames.
	# protocol_frame sorted by beg and in number of frames.
	# Only save data around the stimuli.


	extra_time_window = np.max([time_bcf_window, time_max_window, time_min_window])

	protocol_frame[beg] = protocol_frame[beg] + time_bef_frame- 2*extra_time_window
	protocol_frame[end] = protocol_frame[end] + time_aft_frame + 2*extra_time_window

	# rows_to_skip contains the line numbers with data belonging to frames between trials and not within each trial time span (-time_bef to time_aft referenced to stim).
	rows_to_skip = np.arange(protocol_frame.iat[0,0]).tolist()
	# For each stimulus, check if the beginning of the trial of stimulus 'i' happens after or before of the end of the previous trials. Remember that protocol contains the stimuli order by their beginning.
	for i in range(1, len(protocol_frame)):
		if (b:= protocol_frame[beg].iat[i]) - (a:= protocol_frame[end].iloc[:i].max()) > 0:
			rows_to_skip.extend(np.arange(a, b).tolist())

	# errors='ignore' to ignore if rows_to_skip includes line numbers of lines that are already not present in data, instead of showing an error.
	data.drop(index=rows_to_skip, errors='ignore', inplace=True)

	return data

def find_beg_and_end_of_bouts(data, bout_detection_thr_1, min_bout_duration, min_interbout_time, bout_detection_thr_2):

	#* Use the derivative to find the beginning and end of bouts.
	def bouts_beg_and_end(bouts):
		bouts_beg = np.where(np.diff(bouts) > 0)[0] + 1
		bouts_end = np.where(np.diff(bouts) < 0)[0]
		return bouts_beg, bouts_end


	#* For each timepoint, bouts indicates whether it belongs to a bout or not.
	# It cannot be initialized to an array of nan because of the derivative calculated below.
	bouts = np.zeros(len(data.loc[:, vigor_bout_detection]))


	# bouts[0] and bouts[-1] = 0 to account for cases when the period under analysis starts in the middle of a bout or finishes in the middle of a bout.
	bouts[1:-1][data[vigor_bout_detection].iloc[1:-1] >= bout_detection_thr_1] = 1


	bouts_beg, bouts_end = bouts_beg_and_end(bouts)


	# In principle, the line where we used bouts[1:-1] does not allow to enter in the else part.
	if len(bouts_beg) == len(bouts_end):
		bouts_interval = bouts_beg[1:] - bouts_end[:-1]
	else:
		print('bouts_beg and end have diff len')


	#* Join bouts close in time after finding the interbout intervals too short.
	for short_interval_bout in reversed(np.where(bouts_interval < min_interbout_time)[0]):
		# if short_interval_bout < len(bouts) - 1:
		bouts[bouts_end[short_interval_bout] + 1 : bouts_beg[short_interval_bout + 1]] = 1


	bouts_beg, bouts_end = bouts_beg_and_end(bouts)


	#* Find bouts too short and remove them.
	for short_bouts in np.where(bouts_end - bouts_beg < min_bout_duration)[0]:

		bouts[bouts_beg[short_bouts] : bouts_end[short_bouts] + 1] = 0


	bouts_beg, bouts_end = bouts_beg_and_end(bouts)


	#* Filter by maximum tail angle of each tail movement.
	# bouts_max = np.zeros_like(bouts_beg)

	for bout_b, bout_e in zip(bouts_beg, bouts_end):
	
		# Angular velocity is converted to deg/ms.
		if data.iloc[bout_b : bout_e + 1, data.columns.get_loc(tail_angle)].diff().abs().max() * (expected_framerate / 1000) < bout_detection_thr_2:
			bouts[bout_b : bout_e + 1] = 0


	# Previous version
			# for bout in range(len(bouts_beg)):
			
			# 	# Find the maximum of each bout.		
			# 	bouts_max[bout] = data.iloc[bouts_beg[bout] : bouts_end[bout] + 1, data.columns.get_loc(tail_angle)].diff().abs().max()

			# too_weak_bouts = np.where(bouts_max < bout_detection_thr_2)[0]
			
			# for weak_bouts in too_weak_bouts:

			# 	bouts[bouts_beg[weak_bouts] : bouts_end[weak_bouts] + 1] = 0

	data[bout] = bouts
	data[bout_beg] = data[bout].diff() > 0
	data[bout_end] = data[bout].diff() < 0

	# data[cols_bout] = data[cols_bout].astype(pd.SparseDtype('bool'))

	# # Create a column in data with the beginning and end of bouts.
	# bouts_beg, bouts_end = bouts_beg_and_end(bouts)
	
	# data.iloc[bouts_beg, data.columns.get_loc(bout_beg)] = True
	# data.iloc[bouts_end, data.columns.get_loc(bout_end)] = True
	
	# bouts_beg = data.iloc[bouts_beg,0].to_numpy()
	# bouts_end = data.iloc[:,0].iloc[bouts_end].to_numpy()
	# bouts_beg = data.iloc[:,0].iloc[np.where(data['Bout beg'])[0]].to_numpy()
	# bouts_end = data.iloc[:,0].iloc[np.where(data['Bout end'])[0]].to_numpy()

	return data
	# , bouts_beg, bouts_end



def findStim(data):

	def correct_stim_array(stim_beg, stim_end):

		if len(stim_beg) > 0 or len(stim_end) > 0:

			if (len(stim_beg) > 0 and len(stim_end) == 0):
				stim_end = np.append(stim_end, data.loc[data.index[-1], time_trial_s])

			if (len(stim_beg) == 0 and len(stim_end) > 0):
				stim_beg = np.append(data.loc[data.index[0], time_trial_s], stim_beg)

			if (stim_end[0] < stim_beg[0]):
				stim_beg = np.append(data.loc[data.index[0], time_trial_s], stim_beg)
							
			if (stim_end[-1] < stim_beg[-1]):
				stim_end = np.append(stim_end, data.loc[data.index[-1], time_trial_s])

		return stim_beg, stim_end


	# Time needs to be in data's first column.

	cs_beg_array = data.loc[data[cs_beg] != 0, data.columns[0]].to_numpy()
	cs_end_array = data.loc[data[cs_end] != 0, data.columns[0]].to_numpy()

	us_beg_array = data.loc[data[us_beg] != 0, data.columns[0]].to_numpy()
	us_end_array = data.loc[data[us_end] != 0, data.columns[0]].to_numpy()



	#* Correct when the beg or end of a block happens while there is a stim going on.
	result = [correct_stim_array(stim_beg, stim_end) for stim_beg, stim_end in [(cs_beg_array, cs_end_array), (us_beg_array, us_end_array)]]

	return result[0][0], result[0][1], result[1][0], result[1][1]


def findEvents(data, event_beg, event_end):

	def correctEventArray(e_beg, e_end):

		if len(e_beg) > 0 or len(e_end) > 0:

			if (len(e_beg) > 0 and len(e_end) == 0):
				e_end = np.append(e_end, data.loc[data.index[-1], data.columns[0]])

			if (len(e_beg) == 0 and len(e_end) > 0):
				e_beg = np.append(data.loc[data.index[0], data.columns[0]], e_beg)

			if (e_end[0] < e_beg[0]):
				e_beg = np.append(data.loc[data.index[0], data.columns[0]], e_beg)
							
			if (e_end[-1] < e_beg[-1]):
				e_end = np.append(e_end, data.loc[data.index[-1], data.columns[0]])

		return e_beg, e_end


	# Time needs to be in data's first column.

	e_beg_array = data.loc[data[event_beg] != 0, data.columns[0]].to_numpy()
	e_end_array = data.loc[data[event_end] != 0, data.columns[0]].to_numpy()

	return correctEventArray(e_beg_array, e_end_array)




def plot_cropped_experiment(data, expected_framerate, bout_detection_thr_1, bout_detection_thr_2, downsampling_step, stem_fish_path_orig, fig_cropped_exp_with_bout_detection_name):

	data = data.copy()

	# Convert time to s.
	data.iloc[:,0] = data.iloc[:,0] / expected_framerate


	# # Stimuli beg and end need to be read from data as there were a few changes to data after applying stim_in_data function.
	# cs_beg_array = data.loc[data[cs_beg] != 0, data.columns[0]].to_numpy()
	# cs_end_array = data.loc[data[cs_end] != 0, data.columns[0]].to_numpy()

	# us_beg_array = data.loc[data[us_beg] != 0, data.columns[0]].to_numpy()
	# us_end_array = data.loc[data[us_end] != 0, data.columns[0]].to_numpy()

	# if len(cs_end_array) < len(cs_beg_array):
	# 	cs_end_array.extend([data.iat[-1,0]])

	# if len(us_end_array) < len(us_beg_array):
	# 	us_end_array.extend([data.iat[-1,0]])

#!!!!!!!!!!
	cs_beg_array, cs_end_array, us_beg_array, us_end_array = findStim(data)


	bouts_beg_array = data.loc[data[bout_beg], data.columns[0]].to_numpy()
	bouts_end_array = data.loc[data[bout_end], data.columns[0]].to_numpy()

	trial_transition = np.where(np.diff(data.index) > 1)[0]
	# np.where(data.iloc[:,0].diff() > 1)[0]
	trial_transition = data.iloc[trial_transition-1,0].to_numpy() # / expected_framerate

	# data_plot = deepcopy(data.iloc[::downsampling_step, :])
	data = data.iloc[::downsampling_step, :]


	x = data.iloc[:,0] #/ expected_framerate

	fig = make_subplots(specs=[[{'secondary_y': True}]])

	fig.add_scatter(x=x, y=data.loc[:, tail_angle], name='bcf-time bcf-space cumsum angle [point {}]'.format(chosen_tail_point), mode='lines', line_color='black', opacity=0.7, secondary_y=True,)

	fig.add_scatter(x=x, y=data.loc[:, tail_angle].diff().abs() * expected_framerate/1000, name='Abs velocity [point {}]'.format(chosen_tail_point), mode='lines', line_color='rgb'+str(tuple(us_color)), opacity=0.7, visible='legendonly')

	fig.add_scatter(x=x, y=data.loc[:, vigor_bout_detection], name='Vigour', mode='lines', line_color='blue', opacity=0.7, visible='legendonly', legendgroup='Vigour')

	if camera_value in data.columns:

		fig.add_scatter(x=x, y=data.loc[:, camera_value], name='Camera', mode='lines', line_color='red', opacity=0.7, visible='legendonly')
	
	if galvo_value in data.columns:

		fig.add_scatter(x=x, y=data.loc[:, galvo_value], name='Galvo', mode='lines', line_color='purple', opacity=0.7, visible='legendonly')

	if photodiode_value in data.columns:

		fig.add_scatter(x=x, y=data.loc[:, photodiode_value]*150, name='Photodiode', mode='lines', line_color='brown', opacity=0.7, visible='legendonly')

	if arduino_value in data.columns:

		fig.add_scatter(x=x, y=data.loc[:, arduino_value].diff(), name='Arduino', mode='lines', line_color='darkyellow', opacity=0.7, visible='legendonly')



	# Make shapes for the plots
	shapes = [go.layout.Shape(type='line', xref='x', x0=trial, x1=trial, yref='paper', y0=0, y1=1, opacity=0.7, line_width=1, fillcolor='gray') for trial in trial_transition]


	# if not data[data['Bout beg']].empty:
	# for bout in range(len(bouts_beg)):
	for bout_b, bout_e in zip(bouts_beg_array, bouts_end_array):

		shapes.append(go.layout.Shape(type='rect', xref='x', x0=bout_b, x1=bout_e, yref='paper', y0=0, y1=1, opacity=0.3, line_width=0, fillcolor='lightgray'))
		# fig.add_vrect(x0=bouts_beg[bout], x1=bouts_end[bout], opacity=0.3, line_width=0, fillcolor='lightgray')

	shapes.append(go.layout.Shape(type='line', xref='paper', x0=0, x1=1, yref='y', y0=bout_detection_thr_1, y1=bout_detection_thr_1,
	opacity=0.5, line_width=1, fillcolor='gray', line_dash='dash'))

	shapes.append(go.layout.Shape(type='line', xref='paper', x0=0, x1=1, yref='y', y0=bout_detection_thr_2, y1=bout_detection_thr_2,
	opacity=0.5, line_width=1, fillcolor='gray', line_dash='dot'))

	# fig.add_hline(y=bout_detection_thr_1, opacity=0.5, line_width=1, fillcolor='gray', line_dash='dash')
	# fig.add_hline(y=bout_detection_thr_2, opacity=0.5, line_width=1, fillcolor='gray', line_dash='dot')

	for cs_b, cs_e in zip(cs_beg_array, cs_end_array):

		shapes.append(go.layout.Shape(type='rect', xref='x', x0=cs_b, x1=cs_e, yref='paper', y0=0, y1=1, opacity=0.4, line_width=0, fillcolor='rgb' + str(tuple(cs_color))))
		# fig.add_vrect(x0=cs_beg_, x1=cs_end_, opacity=0.4, line_width=0, fillcolor=cs_color)

	# for stim in range(len(cs_beg_array)):

		# cs_beg_ = cs_beg_array[stim]
		# # cs_end_array = data.loc[data['cs'] == t, data.columns[0]].iloc[-1]
		# cs_end_ = cs_end_array[stim]

		# shapes.append(go.layout.Shape(type='rect', xref='x', x0=cs_beg_, x1=cs_end_, yref='paper', y0=0, y1=1, opacity=0.4, line_width=0, fillcolor=cs_color))
		# # fig.add_vrect(x0=cs_beg_, x1=cs_end_, opacity=0.4, line_width=0, fillcolor=cs_color)

	for us_b, us_e in zip(us_beg_array, us_end_array):

		shapes.append(go.layout.Shape(type='rect', xref='x', x0=us_b, x1=us_e, yref='paper', y0=0, y1=1, opacity=0.4, line_width=0, fillcolor='rgb' + str(tuple(us_color))))
		# fig.add_vrect(x0=us_beg_, x1=us_end_, opacity=0.4, line_width=0, fillcolor=us_color)	

	# for stim in range(len(us_beg_array)):

		# us_beg_ = us_beg_array[stim]
		# # us_end_array = data.loc[data['us'] == t, data.columns[0]].iloc[-1]
		# us_end_ = us_end_array[stim]

		# shapes.append(go.layout.Shape(type='rect', xref='x', x0=us_beg_, x1=us_end_, yref='paper', y0=0, y1=1, opacity=0.4, line_width=0, fillcolor=us_color))
		# # fig.add_vrect(x0=us_beg_, x1=us_end_, opacity=0.4, line_width=0, fillcolor=us_color)

	fig.update_layout(height=1000, width=2000, showlegend=True, plot_bgcolor='rgba(0,0,0,0)', title_text='Behavior before cleaning data, downsampled 5X        ' + stem_fish_path_orig, legend=dict(yanchor='top',y=1,xanchor='left',x=0, bgcolor='white'), shapes=shapes)

	# paper_bgcolor='rgba(0,0,0,0)',

	fig.update_xaxes(title='t (s)', showgrid=False, automargin=True,)

	fig.update_yaxes(title='Velocity or vigor (deg/ms)', showgrid=False, zeroline=False, zerolinecolor='black', automargin=False, range=[0, 20], secondary_y=False,)	

	fig.update_yaxes(title='Angle (deg)', showgrid=False, zeroline=False, zerolinecolor='black', automargin=False, range=[-200, 200], secondary_y=True,)

	py.io.write_html(fig=fig, file=fig_cropped_exp_with_bout_detection_name, auto_open=False)

def clean_data(data):

	# We should not set the angles to 0 deg because of subsequent steps.
	data.loc[~data[bout], data.columns[1:2+chosen_tail_point]] = np.nan

	# Previous version

			# mask_with_lines_to_keep_bout = np.array([False] * len(data))

			# bouts_beg = data.iloc[:,0].iloc[np.where(data['Bout beg'])[0]].to_numpy()
			# bouts_end = data.iloc[:,0].iloc[np.where(data['Bout end'])[0]].to_numpy()


			# if not data[data['Bout beg']].empty:
			# 	for bout in range(len(bouts_beg)):
					
			# 		mask_with_lines_to_keep_bout += ((data.iloc[:,0] >= bouts_beg[bout]) & (data.iloc[:,0] <= bouts_end[bout])).to_numpy()
	

			# # We should not set the angles to 0 deg...
			# data.loc[~mask_with_lines_to_keep_bout, cols[1:]] = np.nan

	data.drop(columns=vigor_bout_detection, inplace=True)

	# data[cols[1:]] = data[cols[1:]].astype(pd.SparseDtype('float32', np.nan))
	# # Need to do this again as the previous operation seems to change the dtype to int32.
	# data[cols_stim] = data[cols_stim].astype(pd.SparseDtype('int8', 0))

	return data

# def calculate_tail_vigor(data, cols, chosen_tail_point, expected_framerate):
	
	# data[vigor] = data[cols[1:1+chosen_tail_point]].diff().abs().sum(axis=1) * (expected_framerate / 1000) # deg/ms
	# # data[vigor] = data[vigor].astype(pd.SparseDtype('float32', 0))
	
	# # Discard the columns with the angle data used to calculate the vigor.
	# # data.drop(cols[1:], axis=1, inplace=True)

	# # data[bout] = data[vigor] > 0
	# # data.loc[data[vigor] > 0, bout] = True

	# return data

# def identify_trials(data, time_bef_frame, time_aft_frame):

# 	trials_list = []

# 	for cs_us in ['CS', 'US']:

# 		cs_us_beg = cs_us + ' beg'

# 		trials_csus = data.loc[data[cs_us_beg] != 0, cs_us_beg].unique()
# 		# trials_csus = data.loc[data[cs_us_beg] > 0, cs_us_beg].unique()

# 		for t in trials_csus:

# 			# trial_beg = trial_reference + time_bef_frame# time_bef_ms / 1000
# 			# # trial_end in relation to cs_us_beg also because stimuli duration may slightly differ from number_trial to number_trial.
# 			# trial_end = trial_reference + time_aft_frame # time_aft_ms / 1000 
# 			trial_reference = data.loc[data[cs_us_beg] == t, data.columns[0]].to_numpy()[0]
			
# 			trial = data.loc[(data.iloc[:,0] >= trial_reference + time_bef_frame) & (data.iloc[:,0] <= trial_reference + time_aft_frame), :]
			
# 			# trial[time_trial_f] is not given by np.arange(time_bef_frame, time_aft_frame + 1) because there may be "incomplete' trials at the end (stopped before trial_reference + time_aft_frame).
# 			# trial[[type_trial_csus, number_trial, time_trial_f]] = cs_us, str(t), np.arange(time_bef_frame, len(trial) + time_bef_frame)	
# 			trial[[type_trial_csus, number_trial]] = cs_us, str(t)
# 			trial[time_trial_f] = 0
# 			trial[time_trial_f] = np.arange(time_bef_frame, len(trial) + time_bef_frame)
# 			# 1000/expected_framerate

# 			trials_list.append(trial)
		
# 	data = pd.concat(trials_list)

# 	# data[vigor] = data[vigor].astype(pd.SparseDtype('float32', 0))
# 	# data.loc[ : , number_trial] = data.loc[ : , number_trial].astype('category')
# 	data[type_trial_csus] = data[type_trial_csus].astype('category')
# 	# data.loc[ : , time_trial_f] = data.loc[ : , time_trial_f].astype('float32')

# 	data.drop(data.columns[0], axis=1, inplace=True)

# 		# To discard automatically fish.
# 			#zero_bouts_trials = 0
			
# 			# trial = data.loc[data[number_trial] == t, :]

# 			# Check that fish beats the tail before the us at least every few trials.
# 			# if csus == 'us':
# 			# 	if trial.loc[(trial[time_trial_f] > -numb_seconds_before_us*expected_framerate) & (trial[time_trial_f] < numb_seconds_after_us*expected_framerate) & (trial[vigor] > 0),:].empty:
# 			# 		zero_bouts_trials += 1
# 			# 		if zero_bouts_trials == max_numb_trials_no_bout_bef:
# 			# 			print('!!! Quiet fish before and after us !!!  trial: ', t)
# 			# 			lines.append(stem_fish_path + '\n\t' ' Quiet fish before and after cs ({} consecutive trials). last trial: {}\n'.format(max_numb_trials_no_bout_bef, t))

# 			# 			skip = True
# 			# 			break
# 			# 	else:
# 			# 		if zero_bouts_trials > 0:
# 			# 			zero_bouts_trials = 0

# 			# Check that fish always beats the tail after the us.
# 			# else:
# 			# 	if trial.loc[(trial[time_trial_f] > 0) & (trial[time_trial_f] < numb_seconds_after_us*expected_framerate) & (trial[vigor] > 0),:].empty:
# 			# 		print('!!! Fish inactive after us !!!  trial: ', t)
# 			# 		lines.append(stem_fish_path + '\n\t' ' Fish inactive after us. trial: {}\n'.format(t))

# 			# 		skip = True
# 			# 		break


# 	return data



def identify_blocks_trials(data, blocks_dict):

	data[block_name] = ''

	for csus in [cs,us]:

		blocks_csus = blocks_dict[blocks_10t][csus][trials_blocks]

		for s_i, trials_in_s in enumerate(blocks_csus):

			# if type(trials_in_s) is list:
			data.loc[(data[type_trial_csus]==csus) & (data[number_trial].astype('int').isin([t for t in trials_in_s])), block_name] = blocks_dict[blocks_10t][csus][names_blocks][s_i]
			# s_i + 1

			# In case of single trials and blocks_csus entries being scalars and not lists with a single entry.
			# else:

			# 	data.loc[data[number_trial] == str(trials_in_s), name_block] = s_i + 1

#!!!!! NEEDS TO BE FIXED FOR CSUS = US
		data[block_name] = data[block_name].astype(CategoricalDtype(categories=blocks_dict[blocks_10t][cs][names_blocks], ordered=True))


	return data


# vigor_digested: Final = 'delta Vigour / Vigour (AU)'


def calculate_digested_vigor(data):

	data[vigor_digested] = 0

	for stim in [cs, us]:

		data_stim = deepcopy(data.loc[data[type_trial_csus]==stim])

		for t in data_stim[number_trial].unique():

			data_trial = data_stim.loc[data_stim[number_trial] == t]

			mean_vigor_baseline_window = data_trial.loc[data_trial[time_trial_f].between(-baseline_window*expected_framerate, 0), vigor_raw].mean()


			#* Kind of deltaF/F.
			data_trial[vigor_digested] = (data_trial[vigor_raw] - mean_vigor_baseline_window) / mean_vigor_baseline_window


			data_stim.loc[data_stim[number_trial] == t] = data_trial

		data.loc[data[type_trial_csus]==stim] = data_stim

	return data





# def heatmapDataframe(data, downsampling_step):

# 	data_heatmap = data.drop(cols_stim[:4]+[block_name], axis=1).pivot(index=time_trial_s, columns=number_trial).reset_index()


# 	#* Downsample.
# 	data_heatmap = 	data_heatmap.iloc[::downsampling_step]


# 	return data_heatmap.set_index(time_trial_s).droplevel(0,axis=1).T








def convert_time_from_frame_to_s(data):

	data[time_trial_f] = data[time_trial_f] / expected_framerate # s
	
	return data.rename(columns={time_trial_f : time_trial_s})


def convert_time_from_s_to_frame(data):

	data[time_trial_s] = data[time_trial_s] * expected_framerate # frame
	
	data[time_trial_s] = data[time_trial_s].astype('int')
	
	return data.rename(columns={time_trial_s : time_trial_f})








def suppression_ratio_pooled(data, metric, baseline_window, cr_window, segments_analysis, groups, csus):

	#!!!!!!! Now only implemented for csus==cs.


#!!!!!!!!!

	# data[metric].fillna(0, inplace=True)
 

	if csus == cs:

		trials_bef_onset = data.loc[data[time_trial_s].between(-baseline_window, 0), :].groupby(groups, observed=True)[metric].agg('mean')
		# trials_bef_and_aft_onset = data.loc[data[time_trial_s].between(-baseline_window, cr_window), :].groupby(groups, observed=True)[metric].agg('mean')

		trials_aft_onset = data.loc[data[time_trial_s].between(0, cr_window), :].groupby(groups, observed=True)[metric].agg('mean')
		
	
	else:
		pass

		trials_bef_onset = data.loc[data[time_trial_s].between(-baseline_window-cr_window, -cr_window), :].groupby(groups, observed=True)[metric].agg('mean')

		trials_aft_onset = data.loc[data[time_trial_s].between(-cr_window, 0), :].groupby(groups, observed=True)[metric].agg('mean')
 		




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# trials_bef_and_aft_onset.fillna(0, inplace=True)
	# trials_aft_onset.fillna(0, inplace=True)



	data_division_at_onset = pd.concat([trials_bef_onset, trials_aft_onset], axis=1)
	# data_division_at_onset = pd.concat([trials_bef_and_aft_onset, trials_aft_onset], axis=1)
	# data_division_at_onset = trials_aft_onset / trials_bef_and_aft_onset

#!
	# data_division_at_onset.fillna(0, inplace=True)




	data_division_at_onset[segments_analysis[2]] = data_division_at_onset.iloc[:,1] / (data_division_at_onset.iloc[:,0] + data_division_at_onset.iloc[:,1])
	# data_division_at_onset[segments_analysis[2]] = data_division_at_onset.iloc[:,1] / data_division_at_onset.iloc[:,0]
	
	# data_division_at_onset[segments_analysis[2]] = (data_division_at_onset.iloc[:,1] - data_division_at_onset.iloc[:,0]) / data_division_at_onset.iloc[:,0]
	


#!
	# data_division_at_onset[segments_analysis[2]] = data_division_at_onset[segments_analysis[2]].fillna(0.5)
	



	data_division_at_onset.columns = segments_analysis

	# data_division_at_onset = pd.concat([data_division_at_onset, data], axis=1).reset_index(number_trial)

	return data_division_at_onset.reset_index(groups).sort_index()





# def plotVigorHeatmap(data_heatmap, downsampling_step, csus, stim_dur, window_data_plot, interval_between_xticks):


# 	if csus == cs:

# 		color = cs_color

# 		# fig, axs = plt.subplots(1, 1, facecolor='white')

# 		# sns.heatmap(data_heatmap, cbar=False, robust=True, xticklabels=int(15*expected_framerate/downsampling_step), yticklabels=False, ax=axs, clip_on=False)


# 	elif csus == us:

# 		color = us_color

# 	fig, axs = plt.subplots(1, 1, facecolor='white')

# 	sns.heatmap(data_heatmap, cbar=False, robust=True, xticklabels=int(interval_between_xticks/downsampling_step), yticklabels=False, ax=axs, clip_on=False)

# 	xlims = axs.get_xlim()
# 	middle = np.mean(xlims)
# 	factor = (xlims[-1] - xlims[0]) / (2*window_data_plot)

# 	axs.axvline(middle, color=color, alpha=0.95, linewidth=1, linestyle='-')
# 	axs.axvline(middle + stim_dur * factor, color=color, alpha=0.95, linewidth=1, linestyle='-')
	
# 	# axs.set_xbound(-40,40)
# 	# axs.set_xticks(ticks=axs.get_xticks(), labels=np.arange(-baseline_window, baseline_window+1, interval_between_xticks))

# 	axs.set_xlabel('Time relative to {} onset (s)'.format(csus))
# 	axs.tick_params(axis='both', which='both', bottom=True, top=False, right=False, direction='out')
# 	# axs.set_title(csus, color=color, fontsize=14)

# 	return fig, axs



def setDtypesAndSortIndex(data):


	#* Set the columns' dtypes.
	data = data.astype({
		time_trial_f:'int32',
		cs_beg:	CategoricalDtype(categories=np.sort(data[cs_beg].unique()).astype('int64'), ordered=True),
		cs_end:	CategoricalDtype(categories=np.sort(data[cs_end].unique()).astype('int64'), ordered=True),
		us_beg:	CategoricalDtype(categories=np.sort(data[us_beg].unique()).astype('int64'), ordered=True),
		us_end:	CategoricalDtype(categories=np.sort(data[us_end].unique()).astype('int64'), ordered=True),
		number_trial:	CategoricalDtype(categories=np.sort(data[number_trial].unique()).astype('int64'), ordered=True),
		# tail_angle:'float32',
		vigor_raw:'float32',
		# vigor_digested:'float32',
		bout:'bool',
		# bout_beg:'bool',
		# bout_end:'bool'
		}, copy=False)
	

	ind_list = []

	for ind in data.index.names:
		
		ind_list.append(data.index.get_level_values(ind).astype('category'))

	data.index = ind_list

		# astype(CategoricalDtype(categories=np.sort(data[col_s].unique()), ordered=True))


	data.sort_index(inplace=True)

	return data



def firstPrep(data):

	data[experiment] = data[experiment].astype(CategoricalDtype(categories=data[experiment].unique(), ordered=True))
	data[fish] = data[fish].astype(CategoricalDtype(categories=data[fish].unique(), ordered=True))

	return data


def prepareData(data):

	data[fish] = ['_'.join(i)  for i in data.index]
	data.reset_index(experiment, inplace=True)
	# data.reset_index(drop=True, inplace=True)
	# data.loc[:, experiment] = data.loc[:, experiment]
	
	data = setDtypesAndSortIndex(data)

	return data



def change_block_names (data, blocks_csus, blocks_csus_names):

	data.drop(columns=block_name,inplace=True)

	data[block_name] = ''

	for s_i, trials_in_s in enumerate(blocks_csus):

		data.loc[(data[number_trial].astype('int').isin(trials_in_s)), block_name] = blocks_csus_names[s_i]

	data[block_name] = data[block_name].astype(CategoricalDtype(categories=blocks_csus_names, ordered=True))

	return data





#* Functions
# %%
# region Functions
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
		
		if (first_plane_substack := plane_where_we_are - p.number_planes_around_the_plane) < 0:
			
			first_plane_substack = 0

		if (last_plane_substack := plane_where_we_are + p.number_planes_around_the_plane + 1) > len(anatomical_stack_images):
			
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

	# try:
	# 	data.loc[:, data_cols] = data.loc[:, data_cols].interpolate(kind='slinear')
	# except:
	# 	print('HERE. FIX THIS')

	#! data = data.reset_index(drop=True).dropna()
	data = data.reset_index().dropna()

	data['Frame number'] = data['Frame number'].astype('int64')

	data[[cs, us]] = data[[cs, us]].astype('Sparse[int16]')

	#* Fix dtypes.
	# for cs_us in [cs, us]:

	# 	data[cs_us] = data.loc[:, cs_us].astype(pd.CategoricalDtype(categories=data[cs_us].unique(), ordered=True))

	return data




def get_good_images_indices(images_subset):


	# top = np.nanmean(images_subset[:, :p.top_bottom_frame_slice, :], axis=(1,2))
	# bottom = np.nanmean(images_subset[:, -p.top_bottom_frame_slice:, :], axis=(1,2))
	# front = np.nanmean(images_subset[:, :, -p.front_back_frame_slice:], axis=(1,2))
	# back = np.nanmean(images_subset[:, :, :p.front_back_frame_slice], axis=(1,2))

	all = np.nanmean(images_subset, axis=(1,2))
	all_median = np.nanmedian(all)

	light_percentage_change = (np.abs(all - all_median) / all_median * 100)

	#* Discard based on overall light (too low or too high)
	mask_good_images = light_percentage_change < p.light_percentage_increase_thr

	#* And also discard based on the derivative
	# mask_good_images = mask_good_images & ([True] + list((np.abs(np.diff(top)) < p.average_light_derivative_thr) & (np.abs(np.diff(bottom)) < p.average_light_derivative_thr) & (np.abs(np.diff(front)) < p.average_light_derivative_thr) & (np.abs(np.diff(back)) < p.average_light_derivative_thr) & (np.abs(np.diff(all) < p.average_light_derivative_thr))))

	# plt.plot(top-np.nanmedian(top))
	# plt.plot(bottom-np.nanmedian(bottom))
	# plt.plot(front-np.nanmedian(front))
	# plt.plot(back-np.nanmedian(back))
	# plt.plot(all-np.nanmedian(all))
	# plt.plot(light_percentage_change)
	# plt.plot(np.where(mask_good_images, mask_good_images, np.nan)*(-10), lw=3)
	# plt.legend(['Top', 'Bottom', 'Front', 'Back', 'Whole', r'Whole % change', r'Good images'], loc='center left', bbox_to_anchor=(1, 0.5))
	# plt.show()
	
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
			if consecutive_count >= p.number_repetitions_the_plane_consecutively_stable:
				new_mask[i - p.number_repetitions_the_plane_consecutively_stable + 1 : i + 1] = True
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
			if consecutive_count >= p.number_repetitions_the_plane_consecutively_stable + 2:
				new_mask[i - consecutive_count + 1 : i-1] = True
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


def get_template_image(frames):

	template_image = np.mean(ndimage.median_filter(frames, size=p.median_filter_kernel, axes=(1,2)), axis=0)
	# ndimage.median_filter(np.nanmean(frames, axis=0), size=p.median_filter_kernel)

	# plt.figure(figsize=(10, 6))
	# plt.imshow(template_image)
	# plt.colorbar(shrink=0.5)
	# plt.title('Anatomy')
	# plt.show()

	return template_image



def measure_motion(frames, anatomy, normalization=None):

	x_motion=np.zeros(np.shape(frames)[0])
	y_motion=np.zeros(np.shape(frames)[0])
	for j in range(frames.shape[0]):
		X=phase_cross_correlation(anatomy, frames[j,:,:], upsample_factor=10, space='real', normalization=normalization, overlap_ratio=0.9)
		x_motion[j]=X[0][0]
		y_motion[j]=X[0][1]

	return np.column_stack([x_motion, y_motion])


def get_total_motion(motion):
	# total_motion=np.zeros(np.shape(frames)[0])
	total_motion = np.linalg.norm(motion, axis=1)
	
	# fig, axs = plt.subplots(1, 2)
	# fig.suptitle('Motion of each frame')
	# axs[0].plot(total_motion, 'k.')
	# axs[1].scatter(motion[:,0]-0.01+0.02*np.random.rand(motion[:,0].shape[0]),motion[:,1]-0.01+0.02*np.random.rand(motion[:,1].shape[0]),s=0.5)
	# fig.show()

	return total_motion



def align_frames(frames, motion, total_motion, total_motion_thr=None):

	# total_motion = get_total_motion(motion)

	##* Discard frames with too much motion.
	if total_motion_thr is not None:
		frames_indices_ignore = np.where(total_motion > total_motion_thr)[0]
	else:
		frames_indices_ignore = []
	
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


def correct_motion_within_trial(trial, anatomical_stack_images, x_dim, y_dim, number_iterations=5):

	images_trial_ = trial.images.to_numpy()

	template_image_ = get_template_image(get_maximum_number_good_last_images(images_trial_))

	motion_thr = 5

	for _ in range(number_iterations):

		motion_thr = int(np.ceil(motion_thr))

		#! does this make sense?
		motion_thr = motion_thr if motion_thr > 5 else 5
		
		#* Measure the motion of each frame using phase cross-correlation.
		motion_ = measure_motion(images_trial_[:, motion_thr:-motion_thr, motion_thr:-motion_thr], template_image_[motion_thr:-motion_thr, motion_thr:-motion_thr], normalization=None)

		#* Get the total motion.
		total_motion = get_total_motion(motion_)
		# Use half of the frames to get the template image.
		motion_thr = np.median(total_motion)

		#* Align the frames to their average.
		aligned_frames = align_frames(images_trial_, motion_, total_motion, 5)
		
		#* Motion correction relative to trials average.
		template_image_ = get_template_image(aligned_frames[np.where(total_motion <= motion_thr)[0]])

	# fig, axs = plt.subplots(1, 2)
	# axs[0].imshow(ndimage.median_filter(np.mean(aligned_frames, axis=0), size=p.median_filter_kernel))
	# axs[1].imshow(np.mean(ndimage.median_filter(aligned_frames, size=p.median_filter_kernel, axes=(1,2)), axis=0))
	# fig.show()

	#* Identify the plane number of the trial.
	plane_number_, _ = find_plane_in_anatomical_stack(anatomical_stack_images, template_image_.astype('float32'), None, x_dim, y_dim)

	return motion_, template_image_, plane_number_



def correct_motion_across_trials():

	return




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




def get_ROIs(Nrois, correlation_map, images, threshold, max_pixels):
	all_traces = np.zeros((Nrois, images.shape[0]))
	all_rois = np.zeros(correlation_map.shape)
	used_pixels = np.zeros(correlation_map.shape)
	correlation_map[:5, :] = 0
	correlation_map[:, :5] = 0
	correlation_map[-5:, :] = 0
	correlation_map[:, -5:] = 0

	correlation_map_ = np.copy(correlation_map)
	images = images.copy()

	for i in tqdm(range(Nrois)):
		this_roi3, this_roi_trace, N, this_correlation_map = next_roi(correlation_map_, images, threshold, max_pixels)
		all_traces[i, :] = this_roi_trace
		all_rois = all_rois + (i + 1) * this_roi3
		used_pixels = used_pixels + this_roi3
		correlation_map_[all_rois > 0] = 0

	return all_traces, all_rois, used_pixels, correlation_map_

#endregion