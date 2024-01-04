from collections import namedtuple
from dataclasses import dataclass
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
		
		for fish_raw_data_path in tqdm(all_fish_raw_data_paths):

			# self.fish_name = fish_raw_data_path.stem.replace('mp tail tracking', '').lower()

			fish = Fish(self.experiment).preprocess(Overwrite, fish_raw_data_path)

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






class Fish:

	def __init__(self, fish_metadata_arg: str, fish_raw_data_path=None):

		fish_metadata = fish_metadata_arg.split('_')[:9]

		Metadata = namedtuple('metadata', ['experiment', 'condition', 'strain', 'age', 'day', 'fish_number', 'rig_name', 'rig_cs_color', 'protocol_number'])
		
		self.metadata = Metadata._make(fish_metadata)

		if fish_raw_data_path is not None:
			self.fish_raw_data_path = fish_raw_data_path


	@classmethod
	def from_raw_data_txt(cls, fish_raw_data_path):

		fish_raw_data_path = Path(fish_raw_data_path.replace('mp tail tracking', '').replace('cam', '').replace('stim control', '').replace('scape sync reader', '').replace('.txt', ''))
		
		experiment = exp_var.map_folder_to_experiment[fish_raw_data_path.parts[-3]]

		fish_metadata = str(fish_raw_data_path.stem).lower().split('_')

		condition = fish_metadata[2]
		day = fish_metadata[0]
		fish_number = fish_metadata[1]

		rig_name, protocol_number = fish_metadata[3].split('-')

		# print(rig_name, protocol_number)

		if rig_name in ['orange', 'brown']:
			rig_cs_color = 'white'
		elif rig_name in ['blue', 'black']:
			rig_cs_color = 'red'
		
		strain = fish_metadata[4]
		age = fish_metadata[5].replace('dpf', '')

		fish_metadata = ('_').join([experiment, condition, age, strain, day, fish_number, rig_name, rig_cs_color, protocol_number])

		return cls(fish_metadata, str(fish_raw_data_path))


	def fish_name(self):

		return '_'.join(self.metadata)


	def dataset_key(self):

		# return '//'.join([self.metadata.experiment, self.metadata.condition, '_'.join(self.metadata[2:])])
		return '//'.join([self.metadata.experiment, self.metadata.condition, self.fish_name()])


	def get_path(self, dataset_type, store=None):
		
		if dataset_type == 'txt raw data' and store is None:
		
			return Experiment.get_experiment_info(self.metadata.experiment, 
			'path_home') / '_'.join([fish.metadata.day, fish.metadata.fish_number, fish.metadata.condition, fish.metadata.rig_name, fish.metadata.strain, fish.metadata.age])

		elif store is not None:
		
			if dataset_type == 'HDF raw data':

				return self.dataset_key() + '.h5'

			elif dataset_type == 'HDF processed data':

				return Experiment.get_experiment_info(self.metadata.experiment) / self.	metadata.condition / self.fish_name() + '.h5'


	def preprocess(self, store: 'AllRawData', Overwrite: bool=True):

		if not Overwrite or store.fish_is_in_store(self):

			self.data_raw = store.get_fish_raw_data(self)
			# return None

		else:

			if self.fish_raw_data_path is None:
				self.fish_raw_data_path = self.get_path(self, 'Raw')

			data_path = str(self.fish_raw_data_path) + 'mp tail tracking.txt'
			protocol_path = str(self.fish_raw_data_path) + 'stim control.txt'
			camera_path = str(self.fish_raw_data_path) + 'cam.txt'
			# sync_reader_path = data_path.replace('mp tail tracking', 'scape sync reader')

			fig_camera_name = Experiment.path_first_analysis(self.metadata.experiment, '_'.join(self.metadata), 'Lost frames', 'png')
			fig_protocol_name = Experiment.path_first_analysis(self.metadata.experiment, '_'.join(self.metadata), 'Summary of protocol actually run', 'png')

			#* Open the file with information about the time of each frame.
			camera = f.read_camera(camera_path)

			#* Estimate the true framerate.
			predicted_framerate, reference_frame_id, Lost_frames = f.framerate_and_reference_frame(camera, name, fig_camera_name)
			

		#TODO
			if Lost_frames:
				return None

			camera = camera.drop(columns=ela_time)


			
			#* Discard frames that will not be used (in camera and hence further down).
			# The calculated interframe interval before the reference frame is variable. Discard what happens up to then (also achieved by using how='inner' in merge_camera_with_data).
			camera = camera[camera[frame_id] >= reference_frame_id]


			#* Open tail tracking data.
			data = f.read_tail_tracking_data(data_path)


		#TODO
			# if (data := f.read_tail_tracking_data(data_path, reference_frame_id)) is None:

			# 	f.save_info(protocol_info_path, self.metadata.name, 'Tail tracking might be corrupted!')
			# 	return None

			# #* Look for possible tail tracking errors.
			# if f.tracking_errors(data, single_point_tracking_error_thr):
			# 	return None

			#* Add information about the time of each frame to data.
			data = f.merge_camera_with_data(data, camera)

			#* Fix abs_time so that the time of each frame becomes closer to the time at which the frames were acquired by the camera and not when they were caught by the computer.
			# The delay between acquiring and catching the frame is unknown and therefore disregarded.
			data[abs_time] = np.linspace(data[abs_time].iat[0], data[abs_time].iat[0] + len(data) * (1000 / predicted_framerate), len(data))

			#* Interpolate data to the expected framerate.
			data = f.interpolate_data(data, predicted_framerate)

			#* Open the stim log.
			protocol = f.read_protocol(protocol_path)


			#* Identify the stimuli, trials and blocks of the experiment.
		#TODO replace by exp_var.experiments_info[Experiment.name]
			data = f.identify_trials(data, protocol)


		#TODO
				# if f.lost_stim(len(data[data[cs_beg]!=0]), len(data[data[us_beg]!=0]), experiment.expected_number_cs, experiment.expected_number_us, experiment.protocol_info_path, stem_fish_path_orig, 2):

				# 	print(data[data[cs_beg]!=0])
				# 	print(data[data[us_beg]!=0])
				# 	continue


			self.data_raw = data
			
			store.add_fish_raw_data(self)


	def filter_data(self):

		# TODO clean up also coordinates of the tail
		#! Only filtering the angle data so far.
			#* Filter tail tracking data.
		# TODO might want to test Adrien's way of filtering data (from Megabouts)
		data = f.filter_data(data)

		self.data = data
			# return data


	def get_data(self, store: 'AllRawData', cols):

		#! cols are the columns to read from the dataframe

		store.get_fish_raw_data()

		pass

class AllRawData:

	def __init__(self, hdf_store_path: str, compression_level=4, compression_library: str="zlib"):

		self._path = Path(hdf_store_path)
		self._compression_level = compression_level
		self._compression_library = compression_library
		# self.store = pd.HDFStore(hdf_store_path, complevel=compression_level, complib="zlib")

	@property
	def compression_level(self): return self._compression_level

	@property
	def compression_library(self): return self._compression_library

	@property
	def path(self): return self._path


	def add_fish_raw_data(self, fish: Fish):

		with pd.HDFStore(self._path, complevel=self._compression_level, complib=self._compression_library) as store:
			
	#! CONFIRM THE DATA_COLS
			store.append(fish.dataset_key(), fish.data_raw, data_columns=[cs, us], expectedrows=len(fish.data_raw), append=False)

			store.get_storer(fish.dataset_key()).attrs['metadata'] = fish.metadata._asdict()
			
		# fish.data_raw.to_hdf(self._path, fish.dataset_key(), data_columns=[cs, us], expectedrows=len(fish.data_raw), mode='w', complevel=self._compression_level, complib=self._compression_library)



#!!!!!!!!!!!!!!!!     overload: Experiment=...    , Condition=...,    cols to query
	def get_fish_raw_data(self, fish: Fish):

		with pd.HDFStore(self._path, complevel=self._compression_level, complib=self._compression_library) as store:

			return store.select(fish.dataset_key())
		# pd.read_hdf(self._path, key=fish.dataset_key(), mode='r', complevel=self._compression_level, complib=self._compression_library)

#?			
			# Metadata = namedtuple('metadata', ['experiment', 'condition', 'strain', 'age', 'day', 'fish_number', 'rig_name', 'rig_cs_color', 'protocol_number'])
			# fish.metadata = Metadata(**store.get_storer(fish.dataset_key()).attrs['metadata'])


	def get_fish_metadata(self, fish: Fish):
		
		with pd.HDFStore(self._path, complevel=self._compression_level, complib=self._compression_library) as store:
			
			return store.get_storer(fish.dataset_key()).attrs['metadata']


	def remove_fish(self, fish: Fish):
		
		with pd.HDFStore(self._path, complevel=self._compression_level, complib=self._compression_library) as store:
			
			store.remove(fish.dataset_key())


	def fish_is_in_store(self, fish: Fish):

		with pd.HDFStore(self._path, complevel=self._compression_level, complib=self._compression_library) as store:
		
			return fish.dataset_key() in store


#! def get_experiment_data and so on


#! method to get data (or just the relative paths in the HDF5) according to all possible parameters (exp., condition, age, day, etc)
	# def 
		#! should call Fish(fish).get_data()

# 	def add_fish_processed_data(self, fish: Fish):
		
# 		# store = pd.HDFStore(r"C:\Users\joaqc\Desktop\2000 01_Test\all_data.h5", complevel=compression_level, complib="zlib")
# 		# try:
# #TODO use put to save raw data in fixed format


# 		self.store.get_storer(self.name).attrs['metadata'] = fish.fish_info()
# 		self.store.close()
# 		# except:
# 		# 	print('Need to preprocess fish data!')

#! SORT TABLE
	
	#! read HDF files from single fish, seleting the required cols and concat
		# get all fish in AllFishInfo