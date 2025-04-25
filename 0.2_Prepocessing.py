# 1. Preprocess data, using absolute time; plot summary of each experiment, showing cropped data; clean data after identifying swim movements.

# %%
import gc
from importlib import reload
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from tqdm import tqdm

import my_functions_behavior as fb
from my_experiment_specific_variables import *
from my_general_variables import *

pd.set_option("mode.copy_on_write", True)
pd.options.mode.chained_assignment = None

Over_write = False


protocol_info_path = str(path_save / 'Protocol summary.txt')


#* Write header of protocol_info
fb.save_info(protocol_info_path, 'Fish ID', ['notes', 'habituation (min)', 'min cs-cs inter. (min)', 'min cs dur. (ms)', 'max cs dur.(ms)', 'min us-us inter. (min)', 'min us dur. (ms)', 'max us dur.(ms)', 'number of CS', 'number of US'])

#* Get all fish raw data paths.
all_fish_raw_data_paths = [*Path(path_home).glob('*mp tail tracking.txt')]

# for fish_path in tqdm(reversed(all_fish_raw_data_paths)):

# 	try:

#* To free up memory.
gc.collect()

fish_path = Path(r"C:\Users\joaqc\Desktop\WIP\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf\Behavior\20240910_02_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_6dpf_mp tail tracking.txt")

stem_fish_path_orig = fish_path.stem.replace('mp tail tracking', '').lower()

#! pkl_name = str(path_orig_pkl / stem_fish_path_orig) + '.pkl'
pkl_name = r"C:\Users\joaqc\Desktop\WIP\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf_behavior.pkl"


# #* Do nothing if pkl file already exists.
# if not Over_write and Path(pkl_name).exists():
# 	print('Pkl with data already exists.')
# 	continue

# if '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'.lower() not in stem_fish_path_orig:
# 	continue

print('\n\n' + stem_fish_path_orig)
#* Paths with the names of different images to be saved.
fig_camera_name = str(path_lost_frames / stem_fish_path_orig)  + '_camera.png'
fig_protocol_name = str(path_summary_exp / stem_fish_path_orig) + '_protocol.png'
fig_behavior_name = str(path_summary_beh / stem_fish_path_orig) + '_behavior.png'
fig_cropped_exp_with_bout_detection_name = str(path_cropped_exp_with_bout_detection / stem_fish_path_orig) + '.html'


#* Paths of different pieces of data.
data_path = str(fish_path)
protocol_path = data_path.replace('mp tail tracking', 'stim control')
camera_path = data_path.replace('mp tail tracking', 'cam')
sync_reader_path = data_path.replace('mp tail tracking', 'scape sync reader')


#* Get fish info from the name of the file with the tracking data.
day, strain, age, exp_type, rig, fish_number = fb.fish_id(stem_fish_path_orig)


#* Read the cam file.
#! cam file contains number and the absolute time at which frames were caught by the computer. This is what allows to link the tail tracking data to when the stimuli happened.
camera = fb.read_camera(camera_path)

try:
	#* Set the initial timepoint to 0.
	first_frame_absolute_time = camera[abs_time].iloc[0]
except:
	pass
	# continue
camera.loc[:,[ela_time, abs_time]] -= camera.loc[:,[ela_time, abs_time]].iloc[0]


#* Look into the elapsed time column.
print('Looking into the ElapsedTime column:')
#* Check if any frames were not caught by the computer.
# This is complicated. It calculates the true framerate of the camera, estimates the accumulation of lag to capture the frames, checks if any frames were not caught by the computer based on that.
# It also returns a reference frame at the beginning of the experiment, where the computer was catching the frames at the expected time (interval between frames similar to what would be if recording exactly at 700 frames per second).
# predicted_framerate should be almost the expected framerate (700 frames per second).
predicted_framerate, reference_frame_id, reference_frame_time, lost_f = fb.framerate_and_reference_frame(camera.drop(columns=abs_time, errors='ignore'), first_frame_absolute_time, protocol_info_path, stem_fish_path_orig, fig_camera_name)


#* If any frames were not caught by the computer, ignore this fish data and go to the next one.
if lost_f:
	del camera
	gc.collect()
	# continue


#* Discard frames that will not be used in the camera dataframe.
camera[frame_id] -= reference_frame_id
camera = camera[camera[frame_id] >= 0]


#* Read the stim control file.
#! stim control file contains the times of the stimuli (stim log) in miliseconds.
# If there is any issue with that file, ignore this fish data and move to the next one.
if (protocol := fb.read_protocol(protocol_path, reference_frame_time if reference_frame_time is not None else reference_frame_id, protocol_info_path, stem_fish_path_orig)) is None:
	print('fix protocol')
	# continue


#* Discard the first stimulus.
# The first stimulus is an optovin stimulus which is not relevant.
# Need to do this because of the number of frames discarded at the beginning of the experiment and the first stimulus happening 1 min after the start. If I do not do this, then the time window of the first stimulus starts at negative frame number.
mask = ((protocol[beg] < 0) | (protocol[end] < 0))
protocol = protocol[~mask]


#* Map stimuli timings of protocol (in unixtime) to ElapsedTime in camera.
# Sometimes, the unixtime of the PC where the experiments are run gets updated during the experiment, creating a shift in the two ways of measuring time.
if camera.iloc[1:,2].notna().any():

	protocol = fb.map_abs_time_to_elapsed_time(camera, protocol)

elif (camera[abs_time].diff() - 1000 / predicted_framerate).max() >= (buffer_size * 1000 / predicted_framerate):

	print('Cannot use these data because unixtime was updated during the experiment and only the absolute time of the first frame was saved.')
	# continue

# number_cycles, number_reinforcers, _, _, _, habituation_duration, cs_dur, cs_isi, us_dur, us_isi = f.protocol_info(protocol)
# if f.lost_stim(number_cycles, number_reinforcers, min_number_cs_trials, min_number_us_trials, protocol_info_path, stem_fish_path_orig, 1):
# 	continue
# f.save_info(protocol_info_path, stem_fish_path_orig, ['', habituation_duration, np.min(cs_isi), np.min(cs_dur), np.max(cs_dur), np.min(us_isi), np.min(us_dur), np.max(us_dur), number_cycles, number_reinforcers])

#* Plot overview of the experimental protocol actually run
# f.plot_protocol(cs_dur, cs_isi, us_dur, us_isi, stem_fish_path_orig, fig_protocol_name)


#* Read the tail tracking file.
#! mp tail tracking file contains the frame numbers, the X and Y coordinates, and the angles of the tail at different points along the tail (calculated online from X and Y).
# If there is any issue with that file, ignore this fish data and move to the next one.
if (data := fb.read_tail_tracking_data(data_path, reference_frame_id)) is None:
	
	fb.save_info(protocol_info_path, stem_fish_path_orig, 'Tail tracking might be corrupted!')
	# continue

#* Look for possible tail tracking errors
if fb.tracking_errors(data, single_point_tracking_error_thr):
	pass
	# continue


#* Merge data with camera dataframes.
data = pd.merge_ordered(data, camera)


# Due to bug in Pandas?!
data = data.astype('float64')

data = data.dropna()


# #* Interpolate data to expeted_framerate (700 FPS).
# # Cameras record at a similar framerate, but not exactly the same. This function interpolates the data to the expected framerate so that later we can pool data from different fish.
# data = f.interpolate_data(data, expected_framerate, predicted_framerate)

# del predicted_framerate, reference_frame_id, reference_frame_time


#* Calculate the cumulative sum of the angles of the tail segments.
# This works as a filter in space.
data.loc[:, cols[1:]] = data.loc[:, cols[1:]].cumsum(axis=1)


#* Filter the tail tracking data in space (using the multiple points along the tail) and time.
data = fb.filter_data(data, space_bcf_window, time_bcf_window)


#? Doubt the angles are correct. Plus and minus almost 300 degrees?! This can happen during the response to the optovin stimulus (US). We do not believe the tracking is accurate. That is fine because we do not care so much about this short period.
print('Max tail angle at the chosen point: {} deg'.format(round(data.loc[:,tail_angle].max())))
fb.save_info(protocol_info_path, stem_fish_path_orig, 'Max tail angle at the chosen point: {} deg'.format(round(data.loc[:,tail_angle].max())))






#* Calculate tail vigor.
#* Calculate the cumulative sum of the angular velocity over space.
#* This allows to take into account movement in any segment with a single scalar value.
#! After filtering in space, calculate the angles of each segment relative to the previous one again.
# data[vigor_raw] = data.loc[:, cols[1:]].diff(axis=1).drop(columns='Angle of point 0 (deg)').abs().sum(axis=1) * (expected_framerate / 1000) # deg/ms
# data[vigor_raw] = data[tail_angle].diff().abs() * (expected_framerate / 1000) # deg/ms
# data[vigor_raw] = data.iloc[:,1:2+chosen_tail_point].diff().abs().sum(axis=1) * (expected_framerate / 1000) # deg/ms
# data[vigor_raw] = (data.iloc[:,1:2+chosen_tail_point].diff(axis=1).diff() * expected_framerate / 1000).diff().abs().sum(axis=1) # deg/ms^2


# A = []
# for i in range(1, 15):
# 	A.append(np.linalg.norm([*zip(data['x' + str(i)].diff(), data['y' + str(i)].diff())], axis=1))
# A = np.array(A).T
# data[vigor_raw] = np.sum(A, axis=1)


# A = np.linalg.norm([*zip(data['x14'].diff(), data['y14'].diff())], axis=1)
# data[vigor_raw] = A

# cols_angle = ['Angle of point 0 (deg)',
# 	'Angle of point 1 (deg)', 'Angle of point 2 (deg)',
# 	'Angle of point 3 (deg)', 'Angle of point 4 (deg)',
# 	'Angle of point 5 (deg)', 'Angle of point 6 (deg)',
# 	'Angle of point 7 (deg)', 'Angle of point 8 (deg)',
# 	'Angle of point 9 (deg)', 'Angle of point 10 (deg)',
# 	'Angle of point 11 (deg)', 'Angle of point 12 (deg)',
# 	'Angle of point 13 (deg)']

# A = (data[cols_angle] - data[cols_angle].rolling(window=round(50 * 1/0.7), center=True, axis=0).mean()) / data[cols_angle].rolling(window=round(50 * 1/0.7), center=True, axis=0).std()

# data[vigor_raw] = data[cols_angle].diff().abs().sum(skipna=False, axis=1)

# data[vigor_raw] = A.sum(skipna=False, axis=1)

# data[vigor_raw] = data[tail_angle].rolling(window=round(50 * 1/0.7), center=True, axis=0).std()


data[vigor_raw] = data[cols[1:]].diff().abs().sum(skipna=False, axis=1) * (expected_framerate / 1000) # deg/ms



#* Convert times in protocol from ms to number of frames and adjust time to next be able to merge with the tracking data (which was interpolated).
# This is not the real time at which the stimuli happened, but the time of the stimuli if the framerate had been the expected_framerate.
protocol = (protocol * expected_framerate/1000).astype('int')


#* Merge data (tail tracking data) with protocol (stimuli information).
# Can do this because time in both dataframes is in the same units (frames) and was interpolated to the same framerate.
data = fb.stim_in_data(data, protocol)


#* Make a quick plot of the behavior.
# This is useful to discard fish that did not move at all over a long time (usually more than 2 h) and that were obviously dead.
# fb.plot_behavior_overview(data, stem_fish_path_orig, fig_behavior_name)

# if f.lost_stim(len(data[data[cs_beg]!=0]), len(data[data[us_beg]!=0]), min_number_cs_trials, min_number_us_trials, protocol_info_path, stem_fish_path_orig, 2):
# 	print(data[data[cs_beg]!=0])
# 	print(data[data[us_beg]!=0])
# 	continue




time_bef_frame = int(np.ceil(time_bef_ms * predicted_framerate/1000)) # frames
time_aft_frame = int(np.ceil(time_aft_ms * predicted_framerate/1000)) # frames







#* Segment the data into trials.
# time_bef_frame and time_aft_frame are for expected_framerate (700 FPS).
data = fb.extract_data_around_stimuli(data, protocol, time_bef_frame, time_aft_frame, time_bcf_window, time_max_window, time_min_window)


#* Set the first timepoint to 0.
data.iloc[:,0] = data.iloc[:,0] - data.iat[0,0]








#!!!!!!!!!!!

#* Calculate 'vigor_bout_detection'.
# This was the way I developed to then identify tail movements, inspired by the work of João.
# data = f.vigor_for_bout_detection(data, chosen_tail_point, time_min_window, time_max_window)
#* Calculate the abstract measure defined by me as 'vigor_bout_detection'.
data.loc[:, vigor_bout_detection] = data.loc[:, vigor_raw].rolling(window=time_max_window, center=True, axis=0).max() - data.loc[:, vigor_raw].rolling(window=time_min_window, center=True, axis=0).min()

data.dropna(inplace=True)


#* Find the beginning and end of bouts (tail movements).
data = fb.find_beg_and_end_of_bouts(data, bout_detection_thr_1, min_bout_duration, min_interbout_time, bout_detection_thr_2)


#* Make an interactive plot of all trial data.
# try:
# fb.plot_cropped_experiment(data, expected_framerate, bout_detection_thr_1, bout_detection_thr_2, downsampling_step, stem_fish_path_orig, fig_cropped_exp_with_bout_detection_name)
# except:
# 	pass

#* Drop the metric used to detect bouts because it is not needed anymore and it is different from the vigor calculated next.
data.drop(columns=vigor_bout_detection, inplace=True)


#!!!!!!!!!!!




#* Identify the trials.
data = fb.identify_trials(data, time_bef_frame, time_aft_frame)


#* Identify blocks of trials.
data = fb.identify_blocks_trials(data, blocks_dict)


#* Order the columns.
data = data[cols_ordered]


#* Set the columns' dtypes.
data['Original frame number'] = data['Original frame number'].astype('int')
data[cols[1:]] = data[cols[1:]].astype('float32')
data[vigor_raw] = data[vigor_raw].astype('float32')
data[time_trial] = data[time_trial].astype('int')
data[cols_bout] = data[cols_bout].astype(pd.SparseDtype('bool'))

for col_s in [cs_beg_time, cs_end, us_beg_time, us_end, number_trial]:
	data[col_s] = data[col_s].astype('int')
	data[col_s] = data[col_s].astype(CategoricalDtype(categories=np.sort(data[col_s].unique()), ordered=True))

# ignore
#! give numbers to the setups instead AND ADD ANOTHER COL WITH THE CS COLOR

#* Prepare to set the index.
data['Exp.'] = exp_type
data['ProtocolRig'] = rig
data['Age (dpf)'] = age
data['Day'] = day
data['Fish no.'] = fish_number
data['Strain'] = strain


#* Change to a categorical dtype.
data[['Exp.', 'ProtocolRig', 'Age (dpf)', 'Day', 'Fish no.', 'Strain']] = data[['Exp.', 'ProtocolRig', 'Age (dpf)', 'Day', 'Fish no.', 'Strain']].astype('category')


#* Set the index.
data.set_index(keys=['Strain', 'Age (dpf)', 'Exp.', 'ProtocolRig', 'Day', 'Fish no.'], inplace=True)


#* Save as a pickle file.
data.to_pickle(pkl_name)




	# 	gc.collect()
	# except:
	# 	print('\n\nerror!\n\n')
	# 	pass

	# break

print('FINISHED')

# Run the next script, where some plots are made.
# exec(open('A.1.1_Single fish_Single trials_angle, vigor, SR.py').read())