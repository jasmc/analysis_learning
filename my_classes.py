from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import pandas as pd

from my_general_variables import *
import my_experiment_specific_variables as exp_var
# import my_functions as f
# import Main as m

# experiment_ = 'original'
# m.experiment
# 'longTermSpacedVsMassed-all'


# 'traceMK801'

# 'stableVsIncreasingTrace'
# '10sTrace'
# 'tracePuromycin'
# 'tracePartialReinforcement'
# 'traceSpaced'

# all_experiments = ['original', 'stableVsIncreasingTrace', '10sTrace']

# MyLife(all_experiments).experiments_list(['original', 'stableVsIncreasingTrace'])


# class MyLife:

# 	def __init__(self, all_experiments):

# 		for experiment in all_experiments:
			
# 			match experiment:

# 				#! November 2022 delay and increasing trace experiment
# 				case 'original':

# 					self.Delay_increasingTrace = self.Exp_original()

# 				# case 'stableVsIncreasingTrace':

# 				# 	self.stableTrace_increasingTrace = self.Experiment(experiment)

# 				# case '10sTrace':

# 				# 	self.shortTrace_longTrace = self.Experiment(experiment)
				
# 				# case 'traceMK801':
					
# 				# 	self.mk801Trace = self.Experiment(experiment)

# 				# case 'tracePuromycin':
					
# 				# 	self.puromycinTrace = self.Experiment(experiment)
				


# 	# # #TODO return an iterable of the experiments
# 	# def experiments_list(self, experiments_list):
		
# 	# 	# return [self.__dict__.get(x) for x in experiments_list]
# 	# 	return [attr for attr in self.__dict__.keys()]

# Experiment('Delay_increasingTrace').path_home
# __dict__

class Experiment:

	def __init__(self, experiment):

		# match experiment:
		# if experiment is not None:
		for key, value in exp_var.experiments_dictionary[experiment].items():

			self.name = experiment

			setattr(self, key, value)


		Experiment.exp_parts(self.expected_number_cs, self.expected_number_us,
					   self.trials_blocks_cs, self.trials_blocks_us,
					   self.names_blocks_cs, self.names_blocks_us,
					   self.trials_phases_cs, self.trials_phases_us,
					   self.names_phases_cs, self.names_phases_us)

		del self.trials_blocks_cs, self.trials_blocks_us, self.names_blocks_cs, self.names_blocks_us, self.trials_phases_cs, self.trials_phases_us

		self.conditions = {control : Condition('Control').set_color(blue).set_id_in_path('control').get_us_latency(None),
						delay : Condition('Delay').set_color(magenta).set_id_in_path('control').get_us_latency(us_latency_after_cs_onset='stable', number_reinforced_trials=46, min_us_latency_trace=3),
						trace : Condition('Trace').set_color(yellow).set_id_in_path('control').get_us_latency(us_latency_after_cs_onset=9, number_reinforced_trials=46, min_us_latency_trace=0.5, max_us_latency_trace=3, min_trace_interval_stable_numb_trials=10, max_trace_interval_stable_numb_trials=10)}
				# self.control = Condition('Control').set_color(blue).set_id_in_path('control')
				# self.delay = Condition('Delay').set_color(magenta).set_id_in_path('control')
				# self.trace = Condition('Trace').set_color(yellow).set_id_in_path('control')

	@classmethod
	def exp_parts(cls,
				   expected_number_cs, expected_number_us,
				   trials_blocks_cs, trials_blocks_us,
				   names_blocks_cs, names_blocks_us,
				   trials_phases_cs, trials_phases_us,
				   names_phases_cs, names_phases_us):
		
		number_trials_plot = max(expected_number_cs, expected_number_us)
		number_blocks_plot = max(len(trials_blocks_cs), len(trials_blocks_us))
		number_phases_plot = max(len(trials_phases_cs), len(trials_phases_us))

		cls.experiment_parts = {trials :
									{cs :
	   									{number_elements : range(1,expected_number_cs+1)},
									us :
										{number_elements : range(1,expected_number_us+1)},
									number_cols_or_rows : number_trials_plot},
								blocks :
									{cs :
	   									{number_elements : trials_blocks_cs, names_trials_blocks_phases : names_blocks_cs},
									us :
										{number_elements : trials_blocks_us, names_trials_blocks_phases : names_blocks_us},
									number_cols_or_rows : number_blocks_plot},
								phases :
									{cs : {
										number_elements : trials_phases_cs, names_trials_blocks_phases : names_phases_cs},
									us : {number_elements : trials_phases_us, names_trials_blocks_phases : names_phases_us},
									number_cols_or_rows : number_phases_plot}}
	
	@classmethod
	def get_conditions(cls):
		return [cls.conditions.keys()]
	
	@classmethod
	def get_conditions_id_in_path(cls):
		return [cls.conditions[k].id_in_path for k in cls.conditions.keys()]
	
	@classmethod
	def get_color_palette(cls):
		return [cls.conditions[k].color for k in cls.conditions.keys()]


	def preprocess_data(self, Overwrite=False):

		all_fish_raw_data_paths = [*Path(self.path_home).glob('*mp tail tracking.txt')]
		
		for fish_raw_path in tqdm(all_fish_raw_data_paths):

			# self.fish_name = fish_raw_path.stem.replace('mp tail tracking', '').lower()

			fish = Fish(self.experiment).preprocess(Overwrite, fish_raw_path)

#! the rest to Fish class
			# call Fish methods
			# return None, then return None





	#! def
	# have access to all fish from the experiment
		# get all fish in AllFishInfo

#TODO cls.conditions_order = [cond.lower() for cond in condition_dict.keys()]

#TODO
	def path_first_analysis(self, type_analysis, fish_name, frmt):
		
		path = self.path_save / type_analysis
		path.mkdir(parents=True, exist_ok=True)
		
		return str(path / fish_name) + '.' + frmt



	def path_processed_data(cls, type_analysis, alignment, name_fish, frmt):

#!!!!! HOW DOES IT WORK WITH ALIGNMENT SET TO NONE??

		path = cls.path_save / 'Processed data' / type_analysis / alignment
		path.mkdir(parents=True, exist_ok=True)
		
		return str(path / self.fish_name) + '_' + name_fish + '.' + frmt





	
# WORK HERE
class Condition:
	
	def __init__(self, name):
			
		#! This will be usefull to change the legends.
		self.name = name

		self.id_in_path = None
		self.color = None

		self.cr_window : (0, cs_duration)  # s


	def get_us_latency(self, us_latency_after_cs_onset, number_reinforced_trials=None, min_us_latency_trace=None, max_us_latency_trace=None, min_trace_interval_stable_numb_trials=None, max_trace_interval_stable_numb_trials=None):

		if us_latency_after_cs_onset == 'increasing':

			number_us_trials_increasing_trace = number_reinforced_trials - (min_trace_interval_stable_numb_trials + max_trace_interval_stable_numb_trials)

			return cs_duration + np.array([min_us_latency_trace] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(min_us_latency_trace, max_us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [max_us_latency_trace] * max_trace_interval_stable_numb_trials)

		elif us_latency_after_cs_onset == 'stable':
			
			return [min_us_latency_trace] * number_reinforced_trials


	def set_color(self, color):

		self.color = color

	def set_id_in_path(self, id_in_path):

		self.id_in_path = id_in_path

	def set_cr_window(self, cr_window):

		self.cr_window = cr_window

	#! def
	# have access to all fish from the condition
		# get all fish in AllFishInfo




class AllFishInfo:

	def add_fish(self):
		# SORT TABLE
		pass

	def remove_fish(self):

		pass

	#! read parquet files from single fish, seleting the required cols and concat
		# get all fish in AllFishInfo



#!!!!!!!!!!!!!!!!
#! Experiment(self.experiment)
# have to be turned into a 

class Fish:

	def __init__(self, experiment=None, fish_name='Anonymous'):

		self.fish_name = fish_name.lower()
		# self.fish_name = Path(self.fish_name).stem.replace('mp tail tracking', '').lower()
		self.experiment = experiment


#!!! USE PROPERTY STUFF HERE
	def fish_info(self):

		info = self.fish_name.split('_')
		day = info[0]

		# strain = info[1]
		# age = info[2].replace('dpf', '')
		# condition = info[3]
		# rig = info[4]
		# fish_number = info[5].replace('fish', '')

		fish_number = info[1]
		condition = info[2]	
		rig = info[3]

		if (rig_color := rig.split('-')[0] in ['orange', 'brown']):
			rig_color = 'white'
		elif rig.split('-')[0] in ['blue', 'black']:
			rig_color = 'red'
		
		strain = info[4]
		age = info[5].replace('dpf', '')

		return strain, day, fish_number, age, self.experiment, condition, rig, rig_color


#TODO
	def get_path(self, alignment):
		
		if alignment == 'Raw':
			return exp_var.experiments_dictionary[self.experiment].path_home / 'Raw data' / self.fish_name
		
		elif alignment in [cs, us, 'Whole processed']:
			return exp_var.experiments_dictionary[self.experiment].path_save / 'Processed data' / 'parquet files' / '1. Original' / alignment / self.fish_name / 'pkl'
		

	def preprocess(self, Overwrite, fish_raw_path):

		# if self.fish_name == 'Anonymous':
		self.fish_name = Path(fish_raw_path).stem.replace('mp tail tracking', '').lower()

#TODO change to parquet format
		pkl_name = str(fish.get_path('Whole'))


		#* Do nothing if pkl file already exists.
		if not Overwrite and Path(pkl_name).exists():
			print('Pkl with data already exists.')
			return None


		print(self.fish_name + '\n\n')


#TODO at some point, I might want to change the format in a smart way, without running the whole thing again
		fig_camera_name = self.path_first_analysis('Lost frames', self.fish_name, 'png')
		fig_protocol_name = self.path_first_analysis('Summary of protocol actually run', self.fish_name, 'png')
		fig_behavior_name = self.path_first_analysis('Summary of behavior', self.fish_name, 'png')
		fig_cropped_exp_with_bout_detection_name = self.path_first_analysis('Summary of experiment/Processed data', self.fish_name, 'html')
		
		data_path = str(fish_raw_path)
		protocol_path = data_path.replace('mp tail tracking', 'stim control')
		camera_path = data_path.replace('mp tail tracking', 'cam')
		sync_reader_path = data_path.replace('mp tail tracking', 'scape sync reader')

		protocol_info_path = str(exp_var.experiments_dictionary[self.experiment].path_save / 'Protocol summary.txt')



		strain, day, fish_number, age, self.experiment, condition, rig, rig_color = self.fish_info()


		if ((camera := f.read_sync_reader(sync_reader_path)) is not None):


			#* Normalize 'CameraValue', 'GalvoValue', 'PhotodiodeValue' columns.
			camera[[camera_value, galvo_value, photodiode_value]] /= camera[[camera_value, galvo_value, photodiode_value]].max()
			camera[[camera_value, galvo_value]] -= camera[[camera_value, galvo_value]].min()



	# camera = pd.read_csv(str(camera_path), sep=' ', header=0, decimal='.', skiprows=[*range(1,number_frames_discard_beg)])
	# camera.astype('float')

	#TODO binarize CameraValue


	#TODO 
	#! Need to do this because we sampled the DAQ every FEW readings.
			camera = pd.merge_ordered(camera.drop(columns=[ela_time, abs_time]), f.read_camera(camera_path), on = [frame_id]).dropna(subset=[ela_time, abs_time])

		else:

			camera = f.read_camera(camera_path)

		first_frame_absolute_time = camera[abs_time].iloc[0]

		camera.loc[:,[ela_time, abs_time]] -= camera.loc[:,[ela_time, abs_time]].iloc[0]


		#* Look into the elapsed time column.
		print('Looking into the ElapsedTime column:')
		predicted_framerate, reference_frame_id, reference_frame_time, Lost_frames = f.framerate_and_reference_frame(camera.drop(columns=abs_time, errors='ignore'), first_frame_absolute_time, protocol_info_path, self.fish_name, fig_camera_name)


		if Lost_frames:
			return None
		



		#* Discard frames that will not be used in camera.
		camera[frame_id] -= reference_frame_id
		camera = camera[camera[frame_id] >= 0]

		# break

		if (protocol := f.read_protocol(protocol_path, reference_frame_time if reference_frame_time is not None else reference_frame_id, protocol_info_path, self.fish_name)) is None:

			print('FIX PROTOCOL')

			return None

		
		#* Discard the first stimulus. (The first stimulus is an optovin stimulus which is not relevant.)
		#! Need to do this because of the number of frames discarded at the beginning of the experiment and the first stimulus happening 1 min after the start. If I do not do this, then the time window of the first stimulus starts at negative frame number...
		mask = ((protocol[beg] < 0) | (protocol[end] < 0))
		protocol = protocol[~mask]

		# protocol = pd.concat([protocol.reset_index(), protocol.iloc[-2:,:]], axis=1)


		#* Map stimuli timings of protocol (in unixtime) to ElapsedTime in camera. (Sometimes, the unixtime of the PC where the experiments are run gets updated during the experiment, creating a shift in the two ways of measuring time.)
		if camera.iloc[1:,2].notna().any():

			protocol = f.map_abs_time_to_elapsed_time(camera, protocol)

		elif (camera[abs_time].diff() - 1000 / predicted_framerate).max() >= (buffer_size * 1000 / predicted_framerate):

			print('Cannot use these data because unixtime was updated during the experiment and only the absolute time of the first frame was saved.')
			return None

		number_cycles, number_reinforcers, _, _, _, habituation_duration, cs_dur, cs_isi, us_dur, us_isi = f.protocol_info(protocol)
		
		# if f.lost_stim(number_cycles, number_reinforcers, min_number_cs_trials, min_number_us_trials, protocol_info_path, self.fish_name, 1):
		# 	return None

		f.save_info(protocol_info_path, self.fish_name, ['', habituation_duration, np.min(cs_isi), np.min(cs_dur), np.max(cs_dur), np.min(us_isi), np.min(us_dur), np.max(us_dur), number_cycles, number_reinforcers])


		#* Plot overview of the experimental protocol actually run
		f.plot_protocol(cs_dur, cs_isi, us_dur, us_isi, self.fish_name, fig_protocol_name)




#! Read also the positions
#TODO try engine='pyarrow'
#! Try different engines
		if (data := f.read_tail_tracking_data(data_path, reference_frame_id)) is None:

			f.save_info(protocol_info_path, self.fish_name, 'Tail tracking might be corrupted!')
			return None






		#* Look for possible tail tracking errors
		if f.tracking_errors(data, single_point_tracking_error_thr):
			return None



		#* Merge data with camera
		#! last part due to bug in Pandas (?)
		data = pd.merge_ordered(data, camera).astype('float64').astype('float64')


		if all(x in data.columns for x in [camera_value, galvo_value, photodiode_value]):
			
			data[[camera_value, galvo_value, photodiode_value]] = data[[camera_value, galvo_value, photodiode_value]].interpolate(method='slinear', axis=0)

		data = data.dropna()


		#* Interpolate data to expeted_framerate (700 FPS).
		data = f.interpolate_data(data, expected_framerate, predicted_framerate)

		#* Filter tail tracking data
		data = f.filter_data(data, space_bcf_window, time_bcf_window)






		#! Doubt the angles are correct. Plus and minus almost 300 degrees?!?!?!?
		print('Max tail angle at the chosen point: {} deg'.format(round(data.loc[:,tail_angle].max())))
		f.save_info(protocol_info_path, self.fish_name, 'Max tail angle at the chosen point: {} deg'.format(round(data.loc[:,tail_angle].max())))


		#* Calculate 'vigor_bout_detection'.
		data = f.vigor_for_bout_detection(data, chosen_tail_point, time_min_window, time_max_window)


		#* Convert protocol from ms to number of frames.
		#* This is not the real time at which the stimuli happened, but the time of the stimuli if the framerate had been the expected_framerate.
		protocol = (protocol * expected_framerate/1000).astype('int')


		data = f.stim_in_data(data, protocol)

		f.plot_behavior_overview(data, self.fish_name, fig_behavior_name)


		if f.lost_stim(len(data[data[cs_beg]!=0]), len(data[data[us_beg]!=0]), exp_var.experiments_dictionary[self.experiment].expected_number_cs, exp_var.experiments_dictionary[self.experiment].expected_number_us, protocol_info_path, self.fish_name, 2):

			print(data[data[cs_beg]!=0])
			print(data[data[us_beg]!=0])
			
			return None






# COMMITT