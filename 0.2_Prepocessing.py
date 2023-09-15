data = pd.read_pickle(r"E:\Results (paper)\2022 11_Basic delay and (increasing) trace CC paradigm\Processed data\pkl files\1. Original\20221115_6dpf_black-1_mitfaminusminus,elavl3gff,10uasgcamp6fef05_04_delay.pkl")

# 1. Preprocess data, using absolute time; plot summary of each experiment, showing cropped data; clean data after identifying swim movements.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from tqdm import tqdm
import gc

from my_general_variables import *
from my_classes import *
import my_functions as f
	
# pd.options.mode.chained_assignment = None

# from importlib import reload
# reload(f)

Over_write = False

# protocol_info_path = str(path_save / 'Protocol summary.txt')


#* Write header of protocol_info
# f.save_info(protocol_info_path, 'Fish ID', ['notes', 'habituation (min)', 'min cs-cs inter. (min)', 'min cs dur. (ms)', 'max cs dur.(ms)', 'min us-us inter. (min)', 'min us dur. (ms)', 'max us dur.(ms)', 'number of CS', 'number of US'])



experiment = Experiment('original')




all_fish_raw_data_paths = [*Path(experiment.path_home).glob('*mp tail tracking.txt')]

for fish_path in tqdm(all_fish_raw_data_paths):

	stem_fish_path_orig = fish_path.stem.replace('mp tail tracking', '').lower()

	pkl_name = str(experiment.path_orig_pkl / stem_fish_path_orig) + '.pkl'


	# break


	#* Do nothing if pkl file already exists.
	if not Over_write and Path(pkl_name).exists():
		print('Pkl with data already exists.')
		continue


	print('\n\n' + stem_fish_path_orig)


	fig_camera_name = str(experiment.path_lost_frames / stem_fish_path_orig)  + '_camera.png'
	fig_protocol_name = str(experiment.path_summary_exp / stem_fish_path_orig) + '_protocol.png'
	fig_behavior_name = str(experiment.path_summary_beh / stem_fish_path_orig) + '_behavior.png'
	fig_cropped_exp_with_bout_detection_name = str(experiment.path_cropped_exp_with_bout_detection / stem_fish_path_orig) + '.html'


	#* Paths of different pieces of data.
	data_path = str(fish_path)
	protocol_path = data_path.replace('mp tail tracking', 'stim control')
	camera_path = data_path.replace('mp tail tracking', 'cam')
	sync_reader_path = data_path.replace('mp tail tracking', 'scape sync reader')


	day, strain, age, exp_type, rig, fish_number = f.fish_id(stem_fish_path_orig)


	if ((camera := f.read_sync_reader(sync_reader_path)) is not None):


		#* Normalize 'CameraValue', 'GalvoValue', 'PhotodiodeValue' columns.
		camera[[camera_value, galvo_value, photodiode_value]] /= camera[[camera_value, galvo_value, photodiode_value]].max()
		camera[[camera_value, galvo_value]] -= camera[[camera_value, galvo_value]].min()


		#TODO binarize CameraValue

		#! Need to do this because we sampled the DAQ every FEW readings.
		camera = pd.merge_ordered(camera.drop(columns=[ela_time, abs_time]), f.read_camera(camera_path), on = [frame_id]).dropna(subset=[ela_time, abs_time])

	else:

		camera = f.read_camera(camera_path)


	first_frame_absolute_time = camera[abs_time].iloc[0]

	camera.loc[:,[ela_time, abs_time]] -= camera.loc[:,[ela_time, abs_time]].iloc[0]


	#* Look into the elapsed time column.
	print('Looking into the ElapsedTime column:')
	predicted_framerate, reference_frame_id, reference_frame_time, lost_f = f.framerate_and_reference_frame(camera.drop(columns=abs_time, errors='ignore'), first_frame_absolute_time, protocol_info_path, stem_fish_path_orig, fig_camera_name)



	if lost_f:
		# del camera
		# gc.collect()
		continue



	#* Discard frames that will not be used in camera.
	camera[frame_id] -= reference_frame_id
	camera = camera[camera[frame_id] >= 0]


	if (protocol := f.read_protocol(protocol_path, reference_frame_time if reference_frame_time is not None else reference_frame_id, experiment.protocol_info_path, stem_fish_path_orig)) is None:

		print('Issues in stim log')
		continue

	
	#! Need to do this because of the number of frames discarded at the beginning of the experiment and in some experiments the first stimulus happening in the period discarded. If I do not do this, then the time window of the first stimulus starts at negative frame number...
	mask = ((protocol[beg] > 0) & (protocol[end] > 0))
	protocol = protocol[mask]


	#* Map stimuli timings of protocol (in unixtime) to ElapsedTime in camera. (Sometimes, the unixtime of the PC where the experiments are run gets updated during the experiment, creating a shift in the two ways of measuring time.)
	if camera.iloc[1:,2].notna().any():

		protocol = f.map_abs_time_to_elapsed_time(camera, protocol)

	elif (camera[abs_time].diff() - 1000 / predicted_framerate).max() >= (buffer_size * 1000 / predicted_framerate):

		print('Cannot use these data because unixtime was updated during the experiment and only the absolute time of the first frame was saved.')
		continue


#TODO need to improve this part to make it more useful
	number_cycles, number_reinforcers, _, _, _, habituation_duration, cs_dur, cs_isi, us_dur, us_isi = f.protocol_info(protocol)
	
	if f.lost_stim(number_cycles, number_reinforcers, experiment.expected_number_cs, experiment.expected_number_us, experiment.protocol_info_path, stem_fish_path_orig, 1):
		continue


#TODO need to improve this part to make it more useful
	f.save_info(experiment.protocol_info_path, stem_fish_path_orig, ['', habituation_duration, np.min(cs_isi), np.min(cs_dur), np.max(cs_dur), np.min(us_isi), np.min(us_dur), np.max(us_dur), number_cycles, number_reinforcers])


	#* Plot overview of the experimental protocol actually run
	f.plot_protocol(cs_dur, cs_isi, us_dur, us_isi, stem_fish_path_orig, fig_protocol_name)


	gc.collect()

#TODO try engine='pyarrow'
	if (data := f.read_tail_tracking_data(data_path, reference_frame_id)) is None:

		f.save_info(experiment.protocol_info_path, stem_fish_path_orig, 'Tail tracking might be corrupted!')
		continue


	#* Look for possible tail tracking errors
	if f.tracking_errors(data, single_point_tracking_error_thr):
		continue


	#* Merge data with camera (due to bug in Pandas (?) have to force casting to the right type)
	data = pd.merge_ordered(data, camera).astype('float64')
	
	
	if all(x in data.columns for x in [camera_value, galvo_value, photodiode_value]):
		
		data[[camera_value, galvo_value, photodiode_value]] = data[[camera_value, galvo_value, photodiode_value]].interpolate(method='slinear', axis=0)

	data = data.dropna()


	#* Interpolate data to the expected framerate
	data = f.interpolate_data(data, expected_framerate, predicted_framerate)


	#* Filter tail tracking data
	data = f.filter_data(data, space_bcf_window, time_bcf_window)


	print('Max tail angle at the chosen point: {} deg'.format(round(data.loc[:,tail_angle].max())))
	f.save_info(experiment.protocol_info_path, stem_fish_path_orig, 'Max tail angle at the chosen point: {} deg'.format(round(data.loc[:,tail_angle].max())))


	#* Segment bouts
	data = f.vigor_for_bout_detection(data, chosen_tail_point, time_min_window, time_max_window)
	data = f.identify_bouts(data, bout_detection_thr_1, min_bout_duration, min_interbout_time, bout_detection_thr_2)


	#* Convert protocol from ms to number of frames.
	#* This is not the real time at which the stimuli happened, but the time of the stimuli if the framerate had been the expected_framerate.
	protocol = (protocol * expected_framerate/1000).astype('int')


	data = f.stim_in_data(data, protocol)

	f.plot_behavior_overview(data, stem_fish_path_orig, fig_behavior_name)





#! save this pkl


	if f.lost_stim(len(data[data[cs_beg]!=0]), len(data[data[us_beg]!=0]), experiment.expected_number_cs, experiment.expected_number_us, experiment.protocol_info_path, stem_fish_path_orig, 2):

		print(data[data[cs_beg]!=0])
		print(data[data[us_beg]!=0])
		continue



	#* time_bef_frame and time_aft_frame are for expected_framerate (700 FPS).
	data = f.extract_data_around_stimuli(data, protocol, time_bef_frame, time_aft_frame, time_bcf_window, time_max_window, time_min_window)


	data.iloc[:,0] = data.iloc[:,0] - data.iat[0,0]


	f.plot_cropped_experiment(data, expected_framerate, bout_detection_thr_1, bout_detection_thr_2, downsampling_step, stem_fish_path_orig, fig_cropped_exp_with_bout_detection_name)


	data.drop(columns=vigor_bout_detection, inplace=True)



#! Do not nned to do this
	#* Calculate tail vigor.
	data[vigor_raw] = data.iloc[:,1:2+chosen_tail_point].diff().abs().sum(axis=1) * (expected_framerate / 1000) # deg/ms
	# data[vigor_raw] = (data.iloc[:,1:2+chosen_tail_point].diff(axis=1).diff() * expected_framerate / 1000).diff().abs().sum(axis=1) # deg/ms^2



	data = f.identify_trials(data, time_bef_frame, time_aft_frame)


	#* Identify blocks of trials.
	data = f.identify_blocks_trials(data, blocks_dict)


	#* Calculate scaled vigor.
	# data = f.calculate_digested_vigor(data)


#! Remove from cols_ordered some of the things
	#* Order the columns.
	data = data[cols_ordered]


	#* Set the columns' dtypes.
	data[cols[1:]] = data[cols[1:]].astype('float32')
	data[vigor_raw] = data[vigor_raw].astype('float32')
	# data[vigor_digested] = data[vigor_digested].astype('float32')
	data[time_trial] = data[time_trial].astype('int')
	data[cols_bout] = data[cols_bout].astype(pd.SparseDtype('bool'))



	for col_s in [cs_beg, cs_end, us_beg, us_end, number_trial]:
		
		data[col_s] = data[col_s].astype('int')
		data[col_s] = data[col_s].astype(CategoricalDtype(categories=np.sort(data[col_s].unique()), ordered=True))


#! ADD ANOTHER COL WITH THE CS COLOR #! give numbers to the setupes instead AND 







	#* Set the index.
	data['Exp.'] = exp_type
	data['ProtocolRig'] = rig
	data['Age (dpf)'] = age
	data['Day'] = day
	data['Fish no.'] = fish_number
	data['Strain'] = strain

	data['Rig color'] = 'red'
	if data['ProtocolRig'].isin(['orange', 'brown']):
		data['Rig color'] = 'white'


	#* Change to a categorical index.
	data[['Exp.', 'ProtocolRig', 'Rig color', 'Age (dpf)', 'Day', 'Fish no.', 'Strain', type_trial_csus]] = data[['Exp.', 'ProtocolRig', 'Rig color', 'Age (dpf)', 'Day', 'Fish no.', 'Strain', type_trial_csus]].astype('category')


	data.set_index(keys=['Strain', 'Age (dpf)', 'Exp.', 'ProtocolRig', 'Day', 'Fish no.', type_trial_csus], inplace=True)

	data_cs = data.xs



#! Split the dataframe into 2: CS-alinged and US-aligned data


#! Add 
# 'Alignment'
	# data_cs = data[data[type_trial_csus] == cs]
	# data_us = data[data[type_trial_csus] == us]




	data.set_index(keys=['Strain', 'Age (dpf)', 'Exp.', 'ProtocolRig', 'Day', 'Fish no.'], inplace=True)



#! Use parquet instead
	#* Save as a pickle file.
	data.to_pickle(pkl_name)


	gc.collect()

	# break

print('FINISHED')
exec(open('A.1.1_Single fish_Single trials_angle, vigor, SR.py').read())