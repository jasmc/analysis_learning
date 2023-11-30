from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

from my_general_variables import *
import my_experiment_specific_variables as exp_var
import my_functions as f
# import Main as m


# 	# # #TODO return an iterable of the experiments
# 	# def experiments_list(self, experiments_list):
		
# 	# 	# return [self.__dict__.get(x) for x in experiments_list]
# 	# 	return [attr for attr in self.__dict__.keys()]

# Experiment('Delay_increasingTrace').path_home
# __dict__

class Experiment:

	def __init__(self, experiment):

		self.name = experiment

		# # In case I want to read everything from the dict
		# for key, value in exp_var.experiments_info[experiment].items():
		# 	setattr(self, key, value)


	#! Set parts of the experiment, specific part chosen
	def define_parts_experiment(self, part_type, cs_us):
		
		self.part_type = self.Part(part_type, cs_us)

	#* Instantiate the conditions of the experiment.
	def define_conditions(self):
		for cond in self.get_conditions():

			setattr(self, cond, self.Condition(name=cond['name'],
							  	name_in_path=cond['name_in_path'],
								color=cond['color'],
								cr_window=cond['cr_window']))

		#* This overwrites what was initially read from the module exp_var
		self.conditions = self.get_conditions()

	@staticmethod
	def get_experiment_info(experiment, variable):
		return exp_var.experiments_info[experiment][variable]

	# def get_path(self, path):
		return self.get_experiment_info(self.name, path)

	@staticmethod
	def get_conditions(experiment):
		return Experiment.get_experiment_info(experiment, 'conditions').keys()
	
	@staticmethod
	def get_conditions_info(experiment, variable):
		return [Experiment.get_experiment_info(experiment, 'conditions')[cond][variable] for cond in Experiment.get_conditions(experiment)]
	
	class Parts:

		def __init__(self, name, cs_us):

			self.elements, self.names_elements = exp_var.experiments_info[Experiment.name]['parts'][name][cs_us]

		@staticmethod
		def get_number_elements(experiment, part_type, cs_us):
			
			if cs_us in [cs, us]:
				return exp_var.experiments_info[experiment]['parts'][part_type][cs_us]
			
			elif cs_us == 'plot':
				return max(len(exp_var.experiments_info[experiment]['parts'][part_type][cs]), len(exp_var.experiments_info[experiment]['parts'][part_type][us]))

	class Condition:
		
		def __init__(self, name, name_in_path=None, color='black', cr_window=(0, cs_duration)):
				
			#! This will be usefull to change the legends.
			self.name = name

			self.name_in_path = name_in_path
			self.color = color

			self.cr_window : cr_window


		def get_us_latency(self, us_latency_after_cs_onset=None, number_reinforced_trials=None, min_us_latency_trace=None, max_us_latency_trace=None, min_trace_interval_stable_numb_trials=None, max_trace_interval_stable_numb_trials=None):

			if us_latency_after_cs_onset == 'increasing':

				number_us_trials_increasing_trace = number_reinforced_trials - (min_trace_interval_stable_numb_trials + max_trace_interval_stable_numb_trials)

				return cs_duration + np.array([min_us_latency_trace] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(min_us_latency_trace, max_us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [max_us_latency_trace] * max_trace_interval_stable_numb_trials)

			elif us_latency_after_cs_onset == 'stable':
				
				return [min_us_latency_trace] * number_reinforced_trials
			else:
				return None


		def set_color(self, color):

			self.color = color

		def set_name_in_path(self, name_in_path):

			self.name_in_path = name_in_path

		def set_cr_window(self, cr_window):

			self.cr_window = cr_window

		#! def
		# have access to all fish from the condition
			# get all fish in AllFishInfo


#! To be finished
	def preprocess_data(self, Overwrite=False):

		all_fish_raw_data_paths = [*Path(self.path_home).glob('*mp tail tracking.txt')]
		
		for fish_raw_path in tqdm(all_fish_raw_data_paths):

			# self.fish_name = fish_raw_path.stem.replace('mp tail tracking', '').lower()

			fish = Fish(self.experiment).preprocess(Overwrite, fish_raw_path)

#! the rest with Fish class
			# call Fish methods
			# return None, then return None


	#! def
	# have access to all fish from the experiment
		# get all fish in AllFishInfo

#TODO cls.conditions_order = [cond.lower() for cond in condition_dict.keys()]


	@staticmethod
	def path_first_analysis(experiment, fish_name, type_analysis, frmt):
		
		path = Experiment.get_experiment_info(experiment, 'path_save') / type_analysis
		path.mkdir(parents=True, exist_ok=True)
		
		return str(path / fish_name) + '.' + frmt



	def path_processed_data(cls, type_analysis, alignment, name_fish, frmt):
		pass
#!!!!! HOW DOES IT WORK WITH ALIGNMENT SET TO NONE??

		path = cls.path_save / 'Processed data' / type_analysis / alignment
		path.mkdir(parents=True, exist_ok=True)
		
		return str(path / self.fish_name) + '_' + name_fish + '.' + frmt




class AllFishInfo:

	def add_fish(self, fish):
		# SORT TABLE
		pass

	def remove_fish(self, fish):
		pass

	def sort_table(self):
		pass

#! method to get data (or just the relative paths in the HDF5) according to all possible parameters (exp., condition, age, day, etc)
	# def 
		#! should call Fish(fish).get_data()


	#! CREATE method to get the corresponding relative path inside the huge HDF5

	#! read parquet files from single fish, seleting the required cols and concat
		# get all fish in AllFishInfo



class Fish:

	def __init__(self, experiment=None, fish_name=r'20180315_6dpf_black-1_mitfaminusminus,elavl3gff,10uasgcamp6fef05_01_delay', fish_raw_path=None):

		if fish_raw_path is None:
			self.experiment = experiment
			self.name = fish_name.lower()
			# self.fish_raw_path = self.get_path('Raw')

		else:
			self.experiment, self.name = self.fish_name_and_experiment_from_path(fish_raw_path)
			self.fish_raw_path = fish_raw_path


	@classmethod
	def from_path(cls, fish_raw_path):

		experiment, fish_name = cls.fish_name_and_experiment_from_path(fish_raw_path)
		
		fish_raw_path = Path(fish_raw_path)
		parent, stem, extension = fish_raw_path.parent, fish_raw_path.stem, fish_raw_path.suffix 
		stem = stem.replace('cam', 'mp tail tracking').replace('stim control', 'mp tail tracking').replace('scape sync reader', 'mp tail tracking')
		fish_raw_path = fish_raw_path.joinpath(parent, stem).with_suffix(extension)

		return cls(experiment, fish_name, fish_raw_path)

	@staticmethod
	def fish_name_and_experiment_from_path(fish_path):

		fish_path = Path(fish_path)
		fish_name = fish_path.stem.replace('mp tail tracking', '').replace('cam', '').replace('stim control', '').replace('scape sync reader', '').lower()
		experiment = exp_var.map_folder_to_experiment[fish_path.parts[-3]]
	
		return experiment, fish_name
	
		
#!!! USE PROPERTY STUFF HERE
	def fish_info(self):

		info = self.name.split('_')
		day = info[0]

		# strain = info[1]
		# age = info[2].replace('dpf', '')
		# condition = info[3]
		# rig = info[4]
		# fish_number = info[5].replace('fish', '')

		fish_number = info[1]
		condition = info[2]	
		rig_name, protocol_number = info[3].split('-')

		if rig_name in ['orange', 'brown']:
			rig_cs_color = 'white'
		elif rig_name in ['blue', 'black']:
			rig_cs_color = 'red'
		
		strain = info[4]
		age = info[5].replace('dpf', '')

		return {'Strain' : strain, 'Day' : day, 'Fish no.' : fish_number, 'Age (dpf)' : age, 'Experiment' : self.experiment, 'Condition' : condition, 'Rig name' : rig_name, 'Protocol number' : protocol_number, 'CS color' : rig_cs_color}

		# return strain, day, fish_number, age, self.experiment, condition, rig_name, protocol_number, rig_cs_color


#! method to read fish data from HDF5. has to call the All_Fish class in principle
	def get_data(self, cols):
		#! cols are the columns to read from the dataframe
		pass

#TODO
	# def get_path(self, alignment):
		
	# 	if alignment == 'Raw':
	# 		return Experiment.get_experiment_info(self.experiment).path_home / 'Raw data' / self.name
		
	# 	elif alignment in [cs, us, 'Whole processed']:
	# 		return Experiment.get_experiment_info(self.experiment).path_save / 'Processed data' / 'parquet files' / '1. Original' / alignment / self.name / 'pkl'

	def get_path(self, dataset_type):
MISSES ARGUMENT FOR LOCATION OF HDF
AND CONDITION NEEDS TO COME FROM SOMEWHERE ELSE...
		if dataset_type == 'Raw':
			return Experiment.get_experiment_info(self.experiment).path_home / 'Raw data' / self.name
		
		elif dataset_type == 'HDF':
			return Experiment.get_experiment_info(self.experiment) / self.condition / self.name / 'h5'



	def preprocess(self, Overwrite):

		if self.fish_raw_path is None:
			self.fish_raw_path = self.get_path(self, 'Raw')

#TODO change to parquet format
		pkl_name = str(self.get_path('Whole'))


		#* Do nothing if pkl file already exists.
		if not Overwrite and Path(pkl_name).exists():
			print('Pkl with data already exists.')
			return None


		print(self.name + '\n\n')



		data_path = str(self.fish_raw_path)
		protocol_path = data_path.replace('mp tail tracking', 'stim control')
		camera_path = data_path.replace('mp tail tracking', 'cam')
		sync_reader_path = data_path.replace('mp tail tracking', 'scape sync reader')



		camera = f.read_camera(camera_path)


		if (protocol := f.read_protocol(protocol_path)) is None:

			print('Fix protocol.')

			return None




		protocol = f.map_abs_time_to_elapsed_time(camera, protocol)






#! At some point, make summary of the protocol



#TODO at some point, I might want to change the format in a smart way, without running the whole thing again
		fig_camera_name = Experiment.path_first_analysis(self.experiment, self.name, 'Lost frames', 'png')
		fig_protocol_name = Experiment.path_first_analysis(self.experiment, self.name, 'Summary of protocol actually run', 'png')
		fig_behavior_name = Experiment.path_first_analysis(self.experiment, self.name, 'Summary of behavior', 'png')
		fig_cropped_exp_with_bout_detection_name = Experiment.path_first_analysis(self.experiment, self.name, 'Summary of experiment/Processed data', 'html')
		

		protocol_info_path = str(Experiment.get_experiment_info(self.experiment, 'path_save') / 'Protocol summary.txt')


		# strain, day, fish_number, age, self.experiment, condition, rig, rig_color = self.fish_info()

		#* For data from SCAPE.
		if ((camera := f.read_sync_reader(sync_reader_path)) is not None):

			#* Normalize 'CameraValue', 'GalvoValue', 'PhotodiodeValue' columns.
			camera[[camera_value, galvo_value, photodiode_value]] /= camera[[camera_value, galvo_value, photodiode_value]].max()
			camera[[camera_value, galvo_value]] -= camera[[camera_value, galvo_value]].min()
			#TODO binarize CameraValue

			#TODO #! Need to do this because we sampled the DAQ every FEW readings.
			camera = pd.merge_ordered(camera.drop(columns=[ela_time, abs_time]), f.read_camera(camera_path), on = [frame_id]).dropna(subset=[ela_time, abs_time])

		else:
			camera = f.read_camera(camera_path)





		first_frame_absolute_time = camera[abs_time].iloc[0]

		camera.loc[:,[ela_time, abs_time]] -= camera.loc[:,[ela_time, abs_time]].iloc[0]


		#* Look into the elapsed time column.
		print('Looking into the ElapsedTime column:')
		predicted_framerate, reference_frame_id, reference_frame_time, Lost_frames = f.framerate_and_reference_frame(camera.drop(columns=abs_time, errors='ignore'), first_frame_absolute_time, protocol_info_path, self.name, fig_camera_name)


		if Lost_frames:
			return None


		#* Discard frames that will not be used in camera.
		camera[frame_id] -= reference_frame_id
		camera = camera[camera[frame_id] >= 0]


		if (protocol := f.read_protocol(protocol_path, reference_frame_time if reference_frame_time is not None else reference_frame_id, protocol_info_path, self.name)) is None:

			print('FIX PROTOCOL')

			return None

		
		#* For most experiments: Discard the first stimulus. (The first stimulus is an optovin stimulus which is not relevant.)
		# Need to do this because of the number of frames discarded at the beginning of the experiment and the first stimulus happening 1 min after the start (in most experiments). If I do not do this, then the the first stimulus appears to start at negative frame number...
		mask = ((protocol[beg] < 0) | (protocol[end] < 0))
		protocol = protocol[~mask]


		#* Map stimuli timings of protocol (in unixtime) to ElapsedTime in camera. (Sometimes, the unixtime of the PC where the experiments are run gets updated during the experiment, creating a shift in the two ways of measuring time.)
		if camera.iloc[1:,2].notna().any():
			protocol = f.map_abs_time_to_elapsed_time(camera, protocol)

		elif (camera[abs_time].diff() - 1000 / predicted_framerate).max() >= (buffer_size * 1000 / predicted_framerate):

			print('Cannot use these data because unixtime was updated during the experiment and only the absolute time of the first frame was saved.')
			return None

		number_cycles, number_reinforcers, _, _, _, habituation_duration, cs_dur, cs_isi, us_dur, us_isi = f.protocol_info(protocol)
		
		if f.lost_stim(number_cycles, number_reinforcers, Experiment.Parts.get_number_elements(self.experiment, trials, cs), Experiment.Parts.get_number_elements(self.experiment, trials, us), protocol_info_path, self.name, 1):
			return None

		f.save_info(protocol_info_path, self.name, ['', habituation_duration, np.min(cs_isi), np.min(cs_dur), np.max(cs_dur), np.min(us_isi), np.min(us_dur), np.max(us_dur), number_cycles, number_reinforcers])


		#* Plot overview of the experimental protocol actually run.
		f.plot_protocol(cs_dur, cs_isi, us_dur, us_isi, self.name, fig_protocol_name)




#! Read also the coordinates of the tail points

#TODO try engine='pyarrow'
#! Try different engines
		if (data := f.read_tail_tracking_data(data_path, reference_frame_id)) is None:

			f.save_info(protocol_info_path, self.name, 'Tail tracking might be corrupted!')
			return None






		#* Look for possible tail tracking errors.
		if f.tracking_errors(data, single_point_tracking_error_thr):
			return None


		#* Merge data with camera. #! last part due to bug in Pandas (?)
		data = pd.merge_ordered(data, camera).astype('float64')


		#* For data from SCAPE.
		if all(x in data.columns for x in [camera_value, galvo_value, photodiode_value]):
			data[[camera_value, galvo_value, photodiode_value]] = data[[camera_value, galvo_value, photodiode_value]].interpolate(method='slinear', axis=0)

		data = data.dropna()







		#* Interpolate data to expeted_framerate (700 FPS).
		data = f.interpolate_data(data, expected_framerate, predicted_framerate)


		#* Filter tail tracking data.
		data = f.filter_data(data, space_bcf_window, time_bcf_window)






		print('Max tail angle at the chosen point: {} deg'.format(round(data.loc[:,tail_angle].max())))
		f.save_info(protocol_info_path, self.name, 'Max tail angle at the chosen point: {} deg'.format(round(data.loc[:,tail_angle].max())))





#! MERGE stim_in_data, crop_exp, identify_trials, ...



		#* Segment bouts.
		data = f.vigor_for_bout_detection(data, chosen_tail_point, time_min_window, time_max_window)
		data = f.identify_bouts(data, bout_detection_thr_1, min_bout_duration, min_interbout_time, bout_detection_thr_2)




		#* Convert protocol from ms to number of frames.
		#* This is not the real time at which the stimuli happened, but the time of the stimuli if the framerate had been the expected_framerate.
		protocol = (protocol * expected_framerate/1000).astype('int')


		#* Add the information when the stimuli happened to the dataframe with the data.
		data = f.stim_in_data(data, protocol)


		f.plot_behavior_overview(data, self.name, fig_behavior_name)


		if f.lost_stim(len(data[data[cs_beg]!=0]), len(data[data[us_beg]!=0]), Experiment.get_experiment_info(self.experiment).expected_number_cs, Experiment.get_experiment_info(self.experiment).expected_number_us, protocol_info_path, self.name, 2):
			
			print(data[data[cs_beg]!=0])
			print(data[data[us_beg]!=0])
			
			return None


		#* Save to a parquet file.
		data.to_pickle(pkl_name)
#! try gzip

#! SAVE ONLY ONCE THE ENTIRE EXPERIMENT
#! EVERYTHING THERE: STIM IDENTIFIED, BOUTS, ...
#! THEN USE MASKS TO GET CS DATA, US DATA. THESE CAN BE SAVED AS NUMPY ARRAYS OR RETRIEVED FROM THE DATAFRAME DIRECTLY
		data.to_parquet(pkl_name, engine='auto', compression=None, index=None, partition_cols=None)























		#* time_bef_frame and time_aft_frame are for expected_framerate (700 FPS).
		data = f.extract_data_around_stimuli(data, protocol, time_bef_frame, time_aft_frame, time_bcf_window, time_max_window, time_min_window)


		data.iloc[:,0] = data.iloc[:,0] - data.iat[0,0]


		f.plot_cropped_experiment(data, expected_framerate, bout_detection_thr_1, bout_detection_thr_2, downsampling_step, self.name, fig_cropped_exp_with_bout_detection_name)


		data.drop(columns=vigor_bout_detection, inplace=True)



	#! Do not nned to do this
		#* Calculate tail vigor.
		data[vigor_raw] = data.iloc[:,1:2+chosen_tail_point].diff().abs().sum(axis=1) * (expected_framerate / 1000) # deg/ms
		# data[vigor_raw] = (data.iloc[:,1:2+chosen_tail_point].diff(axis=1).diff() * expected_framerate / 1000).diff().abs().sum(axis=1) # deg/ms^2



		data = f.identify_trials(data, time_bef_frame, time_aft_frame)

#!!!!!! todo
		#* Identify blocks of trials.
		# data = f.identify_blocks_trials(data, blocks_dict)


		#* Calculate scaled vigor.
		# data = f.calculate_digested_vigor(data)


	#! Remove from cols_ordered some of the things
		#* Order the columns.
		# data = data[cols_ordered]


		#* Set the columns' dtypes.
		# data[cols[1:]] = data[cols[1:]].astype('float32')
		data[vigor_raw] = data[vigor_raw].astype('float32')
		# data[vigor_digested] = data[vigor_digested].astype('float32')
		data[time_trial_f] = data[time_trial_f].astype('int')
		data[cols_bout] = data[cols_bout].astype(pd.SparseDtype('bool'))



		for col_s in [cs_beg, cs_end, us_beg, us_end, number_trial]:
			
			data[col_s] = data[col_s].astype('int')
			data[col_s] = data[col_s].astype(pd.api.types.CategoricalDtype(categories=np.sort(data[col_s].unique()), ordered=True))



		strain, day, fish_number, age, experiment, condition, rig_name, protocol_number, rig_cs_color = self.fish_info()


#! check if this works with parquet
		data.attrs = {'Strain' : strain,
					'Day' : day,
					'Fish no.' : fish_number,
					'Age (dpf)' : age,
					'Experiment' : experiment,
					'Condition' : condition,
					'Rig name' : rig_name,
					'Protocol number' : protocol_number,
					'CS color' : rig_cs_color}


		# #* Set the index.
		# data['Strain'] = strain
		# data['Day'] = day
		# data['Fish no.'] = fish_number
		# data['Age (dpf)'] = age
		# data['Experiment'] = experiment
		# data['Condition'] = condition
		# data['Rig name'] = rig_name
		# data['Protocol number'] = protocol_number
		# data['CS color'] = rig_cs_color


		# #* Change to a categorical index.
		# data[['Exp.', 'ProtocolRig', 'Rig color', 'Age (dpf)', 'Day', 'Fish no.', 'Strain', type_trial_csus]] = data[['Exp.', 'ProtocolRig', 'Rig color', 'Age (dpf)', 'Day', 'Fish no.', 'Strain', type_trial_csus]].astype('category')


		# data.set_index(keys=['Strain', 'Age (dpf)', 'Exp.', 'ProtocolRig', 'Day', 'Fish no.', type_trial_csus], inplace=True)

		data_cs = data.xs



	#! Split the dataframe into 2: CS-alinged and US-aligned data


	#! Add 
	# 'Alignment'
		# data_cs = data[data[type_trial_csus] == cs]
		# data_us = data[data[type_trial_csus] == us]




		# data.set_index(keys=['Strain', 'Age (dpf)', 'Exp.', 'ProtocolRig', 'Day', 'Fish no.'], inplace=True)


#!!!!!!! append to hdf5
	#! Use parquet instead
		#* Save as a pickle file.
		data.to_pickle(pkl_name)
		#! HDF5   !!!!!!!!!
		#!  organize hierarchically in experiment/condition and then all the corresponding fish
		#! save all fish in the same HDF5


	def preprocess_2(self, Overwrite):

		if self.fish_raw_path is None:
			self.fish_raw_path = self.get_path(self, 'Raw')

#TODO change to parquet format
		pkl_name = str(self.get_path('Whole'))


		#* Do nothing if pkl file already exists.
		if not Overwrite and Path(pkl_name).exists():
			print('Pkl with data already exists.')
			return None


		print(self.name + '\n\n')


#TODO at some point, I might want to change the format in a smart way, without running the whole thing again
		fig_camera_name = Experiment.path_first_analysis(self.experiment, self.name, 'Lost frames', 'png')
		fig_protocol_name = Experiment.path_first_analysis(self.experiment, self.name, 'Summary of protocol actually run', 'png')
		fig_behavior_name = Experiment.path_first_analysis(self.experiment, self.name, 'Summary of behavior', 'png')
		fig_cropped_exp_with_bout_detection_name = Experiment.path_first_analysis(self.experiment, self.name, 'Summary of experiment/Processed data', 'html')
		
		data_path = str(self.fish_raw_path)
		protocol_path = data_path.replace('mp tail tracking', 'stim control')
		camera_path = data_path.replace('mp tail tracking', 'cam')
		sync_reader_path = data_path.replace('mp tail tracking', 'scape sync reader')

		protocol_info_path = str(Experiment.get_experiment_info(self.experiment, 'path_save') / 'Protocol summary.txt')


		# strain, day, fish_number, age, self.experiment, condition, rig, rig_color = self.fish_info()

		print(camera_path)


		#* For data from SCAPE.
		if ((camera := f.read_sync_reader(sync_reader_path)) is not None):

			#* Normalize 'CameraValue', 'GalvoValue', 'PhotodiodeValue' columns.
			camera[[camera_value, galvo_value, photodiode_value]] /= camera[[camera_value, galvo_value, photodiode_value]].max()
			camera[[camera_value, galvo_value]] -= camera[[camera_value, galvo_value]].min()
			#TODO binarize CameraValue

			#TODO #! Need to do this because we sampled the DAQ every FEW readings.
			camera = pd.merge_ordered(camera.drop(columns=[ela_time, abs_time]), f.read_camera(camera_path), on = [frame_id]).dropna(subset=[ela_time, abs_time])

		else:
			camera = f.read_camera(camera_path)

		first_frame_absolute_time = camera[abs_time].iloc[0]

		camera.loc[:,[ela_time, abs_time]] -= camera.loc[:,[ela_time, abs_time]].iloc[0]


		#* Look into the elapsed time column.
		print('Looking into the ElapsedTime column:')
		predicted_framerate, reference_frame_id, reference_frame_time, Lost_frames = f.framerate_and_reference_frame(camera.drop(columns=abs_time, errors='ignore'), first_frame_absolute_time, protocol_info_path, self.name, fig_camera_name)


		if Lost_frames:
			return None


		#* Discard frames that will not be used in camera.
		camera[frame_id] -= reference_frame_id
		camera = camera[camera[frame_id] >= 0]


		if (protocol := f.read_protocol(protocol_path, reference_frame_time if reference_frame_time is not None else reference_frame_id, protocol_info_path, self.name)) is None:

			print('FIX PROTOCOL')

			return None

		
		#* For most experiments: Discard the first stimulus. (The first stimulus is an optovin stimulus which is not relevant.)
		# Need to do this because of the number of frames discarded at the beginning of the experiment and the first stimulus happening 1 min after the start (in most experiments). If I do not do this, then the the first stimulus appears to start at negative frame number...
		mask = ((protocol[beg] < 0) | (protocol[end] < 0))
		protocol = protocol[~mask]


		#* Map stimuli timings of protocol (in unixtime) to ElapsedTime in camera. (Sometimes, the unixtime of the PC where the experiments are run gets updated during the experiment, creating a shift in the two ways of measuring time.)
		if camera.iloc[1:,2].notna().any():
			protocol = f.map_abs_time_to_elapsed_time(camera, protocol)

		elif (camera[abs_time].diff() - 1000 / predicted_framerate).max() >= (buffer_size * 1000 / predicted_framerate):

			print('Cannot use these data because unixtime was updated during the experiment and only the absolute time of the first frame was saved.')
			return None

		number_cycles, number_reinforcers, _, _, _, habituation_duration, cs_dur, cs_isi, us_dur, us_isi = f.protocol_info(protocol)
		
		if f.lost_stim(number_cycles, number_reinforcers, Experiment.Parts.get_number_elements(self.experiment, trials, cs), Experiment.Parts.get_number_elements(self.experiment, trials, us), protocol_info_path, self.name, 1):
			return None

		f.save_info(protocol_info_path, self.name, ['', habituation_duration, np.min(cs_isi), np.min(cs_dur), np.max(cs_dur), np.min(us_isi), np.min(us_dur), np.max(us_dur), number_cycles, number_reinforcers])


		#* Plot overview of the experimental protocol actually run.
		f.plot_protocol(cs_dur, cs_isi, us_dur, us_isi, self.name, fig_protocol_name)




#! Read also the coordinates of the tail points

#TODO try engine='pyarrow'
#! Try different engines
		if (data := f.read_tail_tracking_data(data_path, reference_frame_id)) is None:

			f.save_info(protocol_info_path, self.name, 'Tail tracking might be corrupted!')
			return None






		#* Look for possible tail tracking errors.
		if f.tracking_errors(data, single_point_tracking_error_thr):
			return None


		#* Merge data with camera. #! last part due to bug in Pandas (?)
		data = pd.merge_ordered(data, camera).astype('float64')


		#* For data from SCAPE.
		if all(x in data.columns for x in [camera_value, galvo_value, photodiode_value]):
			data[[camera_value, galvo_value, photodiode_value]] = data[[camera_value, galvo_value, photodiode_value]].interpolate(method='slinear', axis=0)

		data = data.dropna()


		#* Interpolate data to expeted_framerate (700 FPS).
		data = f.interpolate_data(data, expected_framerate, predicted_framerate)


		#* Filter tail tracking data.
		data = f.filter_data(data, space_bcf_window, time_bcf_window)


		print('Max tail angle at the chosen point: {} deg'.format(round(data.loc[:,tail_angle].max())))
		f.save_info(protocol_info_path, self.name, 'Max tail angle at the chosen point: {} deg'.format(round(data.loc[:,tail_angle].max())))





#! MERGE stim_in_data, crop_exp, identify_trials, ...



		#* Segment bouts.
		data = f.vigor_for_bout_detection(data, chosen_tail_point, time_min_window, time_max_window)
		data = f.identify_bouts(data, bout_detection_thr_1, min_bout_duration, min_interbout_time, bout_detection_thr_2)




		#* Convert protocol from ms to number of frames.
		#* This is not the real time at which the stimuli happened, but the time of the stimuli if the framerate had been the expected_framerate.
		protocol = (protocol * expected_framerate/1000).astype('int')


		#* Add the information when the stimuli happened to the dataframe with the data.
		data = f.stim_in_data(data, protocol)


		f.plot_behavior_overview(data, self.name, fig_behavior_name)


		if f.lost_stim(len(data[data[cs_beg]!=0]), len(data[data[us_beg]!=0]), Experiment.get_experiment_info(self.experiment).expected_number_cs, Experiment.get_experiment_info(self.experiment).expected_number_us, protocol_info_path, self.name, 2):
			
			print(data[data[cs_beg]!=0])
			print(data[data[us_beg]!=0])
			
			return None


		#* Save to a parquet file.
		data.to_pickle(pkl_name)
#! try gzip

#! SAVE ONLY ONCE THE ENTIRE EXPERIMENT
#! EVERYTHING THERE: STIM IDENTIFIED, BOUTS, ...
#! THEN USE MASKS TO GET CS DATA, US DATA. THESE CAN BE SAVED AS NUMPY ARRAYS OR RETRIEVED FROM THE DATAFRAME DIRECTLY
		data.to_parquet(pkl_name, engine='auto', compression=None, index=None, partition_cols=None)























		#* time_bef_frame and time_aft_frame are for expected_framerate (700 FPS).
		data = f.extract_data_around_stimuli(data, protocol, time_bef_frame, time_aft_frame, time_bcf_window, time_max_window, time_min_window)


		data.iloc[:,0] = data.iloc[:,0] - data.iat[0,0]


		f.plot_cropped_experiment(data, expected_framerate, bout_detection_thr_1, bout_detection_thr_2, downsampling_step, self.name, fig_cropped_exp_with_bout_detection_name)


		data.drop(columns=vigor_bout_detection, inplace=True)



	#! Do not nned to do this
		#* Calculate tail vigor.
		data[vigor_raw] = data.iloc[:,1:2+chosen_tail_point].diff().abs().sum(axis=1) * (expected_framerate / 1000) # deg/ms
		# data[vigor_raw] = (data.iloc[:,1:2+chosen_tail_point].diff(axis=1).diff() * expected_framerate / 1000).diff().abs().sum(axis=1) # deg/ms^2



		data = f.identify_trials(data, time_bef_frame, time_aft_frame)

#!!!!!! todo
		#* Identify blocks of trials.
		# data = f.identify_blocks_trials(data, blocks_dict)


		#* Calculate scaled vigor.
		# data = f.calculate_digested_vigor(data)


	#! Remove from cols_ordered some of the things
		#* Order the columns.
		# data = data[cols_ordered]


		#* Set the columns' dtypes.
		# data[cols[1:]] = data[cols[1:]].astype('float32')
		data[vigor_raw] = data[vigor_raw].astype('float32')
		# data[vigor_digested] = data[vigor_digested].astype('float32')
		data[time_trial_f] = data[time_trial_f].astype('int')
		data[cols_bout] = data[cols_bout].astype(pd.SparseDtype('bool'))



		for col_s in [cs_beg, cs_end, us_beg, us_end, number_trial]:
			
			data[col_s] = data[col_s].astype('int')
			data[col_s] = data[col_s].astype(pd.api.types.CategoricalDtype(categories=np.sort(data[col_s].unique()), ordered=True))



		strain, day, fish_number, age, experiment, condition, rig_name, protocol_number, rig_cs_color = self.fish_info()


#! check if this works with parquet
		data.attrs = {'Strain' : strain,
					'Day' : day,
					'Fish no.' : fish_number,
					'Age (dpf)' : age,
					'Experiment' : experiment,
					'Condition' : condition,
					'Rig name' : rig_name,
					'Protocol number' : protocol_number,
					'CS color' : rig_cs_color}


		# #* Set the index.
		# data['Strain'] = strain
		# data['Day'] = day
		# data['Fish no.'] = fish_number
		# data['Age (dpf)'] = age
		# data['Experiment'] = experiment
		# data['Condition'] = condition
		# data['Rig name'] = rig_name
		# data['Protocol number'] = protocol_number
		# data['CS color'] = rig_cs_color


		# #* Change to a categorical index.
		# data[['Exp.', 'ProtocolRig', 'Rig color', 'Age (dpf)', 'Day', 'Fish no.', 'Strain', type_trial_csus]] = data[['Exp.', 'ProtocolRig', 'Rig color', 'Age (dpf)', 'Day', 'Fish no.', 'Strain', type_trial_csus]].astype('category')


		# data.set_index(keys=['Strain', 'Age (dpf)', 'Exp.', 'ProtocolRig', 'Day', 'Fish no.', type_trial_csus], inplace=True)

		data_cs = data.xs



	#! Split the dataframe into 2: CS-alinged and US-aligned data


	#! Add 
	# 'Alignment'
		# data_cs = data[data[type_trial_csus] == cs]
		# data_us = data[data[type_trial_csus] == us]




		# data.set_index(keys=['Strain', 'Age (dpf)', 'Exp.', 'ProtocolRig', 'Day', 'Fish no.'], inplace=True)


#!!!!!!! append to hdf5
	#! Use parquet instead
		#* Save as a pickle file.
		data.to_pickle(pkl_name)
		#! HDF5   !!!!!!!!!
		#!  organize hierarchically in experiment/condition and then all the corresponding fish
		#! save all fish in the same HDF5