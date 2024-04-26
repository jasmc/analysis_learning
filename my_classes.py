from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm import tqdm

import my_experiment_specific_variables as exp_var
import my_functions as f
from my_general_variables import *

# import Main as m


# 	# # #TODO return an iterable of the experiments
# 	# def experiments_list(self, experiments_list):
		
# 	# 	# return [self.__dict__.get(x) for x in experiments_list]
# 	# 	return [attr for attr in self.__dict__.keys()]

# Experiment('Delay_increasingTrace').path_home
# __dict__

class  Experiment:

	def __init__(self, experiment):

		self.name = experiment

		# # In case I want to read everything from the dict
		# for key, value in exp_var.experiments_info[experiment].items():
		# 	setattr(self, key, value)


	#! Set parts of the experiment, specific part chosen
	def define_parts_experiment(self, part_type, cs_us):
		
		self.part_type = self.Parts(part_type, cs_us)

	#* Instantiate the conditions of the experiment.
	def define_conditions(self):
		for cond in Experiment.get_conditions(self.name):

			setattr(self, cond, self.Condition(name=cond['name'],
							  	name_in_path=cond['name_in_path'],
								color=cond['color'],
								cr_window=cond['cr_window']))

		#* This overwrites what was initially read from the module exp_var
		self.conditions = self.get_conditions(self.name)

	@staticmethod
	def get_experiment_info(experiment, variable):
		return exp_var.experiments_info[experiment][variable]

	def get_path(self):
		return self.get_experiment_info(self.name, 'path_home')

	@staticmethod
	def get_conditions(experiment):
		return Experiment.get_experiment_info(experiment, 'conditions').keys()
	
	@staticmethod
	def get_conditions_info(experiment, variable):
		return [Experiment.get_experiment_info(experiment, 'conditions')[cond][variable] for cond in Experiment.get_conditions(experiment)]
	
	class Parts:

		def __init__(self, name_part, cs_us):
			self.elements, self.names_elements = exp_var.experiments_info[Experiment.__name__]['parts'][name_part][cs_us]

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


	def preprocess(self, store: 'AllRawData', Overwrite: bool=True):

		"""
		Preprocesses all fish from the experiment.
		"""


		all_fish_raw_data_paths = [*Path(self.get_path()).glob('*mp tail tracking.txt')]
		
		for fish_raw_data_path in tqdm(all_fish_raw_data_paths):

			fish = Fish.from_raw_data_txt(fish_raw_data_path)

			fish.preprocess(store, Overwrite)


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

		fish_metadata = ('_').join([experiment, condition, strain, age, day, fish_number, rig_name, rig_cs_color, protocol_number])

		return cls(fish_metadata, str(fish_raw_data_path))


	def name(self):

		return '_'.join(self.metadata)


	def dataset_key(self):

		# return '//'.join([self.metadata.experiment, self.metadata.condition, '_'.join(self.metadata[2:])])
		return '//'.join([self.metadata.experiment, self.metadata.condition, self.name()])


	def get_path(self, dataset_type, store=None):
		
		if dataset_type == 'txt raw data' and store is None:
		
			return Experiment.get_experiment_info(self.metadata.experiment, 
			'path_home') / '_'.join([self.metadata.day, self.metadata.fish_number, self.metadata.condition, self.metadata.rig_name, self.metadata.strain, self.metadata.age])

		elif store is not None:
		
			if dataset_type == 'HDF raw data':

				return self.dataset_key() + '.h5'

			elif dataset_type == 'HDF processed data':

				return Experiment.get_experiment_info(self.metadata.experiment) / self.	metadata.condition / fish_name + '.h5'


	def preprocess(self, store: 'AllRawData', Overwrite: bool=True):
		"""
		gives all raw data
		not possible to select cols
		if you want to select cols, use Fish.get_raw_data()
		"""

		#* Functions used in this method.

		def read_camera(camera_path):

			try:
				start = timer()
				
				# camera = pd.read_csv(str(camera_path), sep='\t', header=0, decimal='.', skiprows=[*range(1,number_frames_discard_beg)])
				camera = pd.read_csv(camera_path, engine='pyarrow', sep=' ', header=0, decimal='.')
				# , na_filter=False
				# dtype={time_experiment_f : 'int64', abs_time : 'int64', ela_time : 'float64'})
				# skipfooter=1
				camera = camera.iloc[:-1,:]

				camera.rename(columns={'FrameID' : frame_id}, inplace=True)
				
				print('Time to read cam.txt: {} (s)'.format(timer()-start))
				
				return camera

			except:

				print('Issues in the camera log file')
				
				return None

		def framerate_and_reference_frame(camera, fish_name, fig_camera_name):

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


			def lost_frames(camera, camera_diff, ifi, fish_name, fig_camera_name):


				# Delay to capture frames by the computer
				delay = (camera_diff - ifi).cumsum().to_numpy()


				# Number of lost frames
				# More than one frame might be lost and number_frames_lost sometimes is not monotonically crescent (can go down when some ms are 'recovered').
				number_frames_lost = np.floor(delay / (ifi * buffer_size))
				#TODO use this to speed up
				# number_frames_lost = np.max(number_frames_lost, 0, axis)
				number_frames_lost = np.where(number_frames_lost>=0, number_frames_lost, 0)

				number_frames_lost_diff = np.floor(np.diff(number_frames_lost))
				number_frames_lost_diff = np.where(number_frames_lost_diff>=0,number_frames_lost_diff,0)


				# Indices where frames were potentially lost
				where_frames_lost = np.where(number_frames_lost_diff > 0)[0]


				# Total number of missed frames
				if (Lost_frames := len(where_frames_lost) > 0):
					print('Total number of lost frames: ', len(where_frames_lost))
					print('Where: ', where_frames_lost)
					# save_info(protocol_info_path, fish_name, 'Lost frames.')
				else:
					print('No frames were lost')

				fig, axs = plt.subplots(5, 1, sharex=True, facecolor='white', figsize=(20, 40), constrained_layout=True)

				axs[0].plot(camera.iloc[:,1],'k')
				axs[0].set_ylabel('Elapsed time (ms)')
				axs[0].set_title('Estimated IFI: {} ms.    Estimated framerate: {} FPS'.format(round(ifi, 3), round(predicted_framerate, 3)))

				axs[1].plot(camera_diff,'k')
				axs[1].set_ylabel('IFI (ms)')

				axs[2].plot(delay,'k')
				axs[2].set_ylabel('Delay (ms)')

				axs[3].plot(number_frames_lost_diff.cumsum(),'k')
				axs[3].set_ylabel('Cumulative number of lost frames')

				axs[4].plot(number_frames_lost_diff,'black')
				# axs[4].set_xlabel('frame number')
				axs[4].set_ylabel('Lost frames')

				fig.supxlabel('Frame number')
				plt.suptitle('Analysis of lost frames\n' + fish_name)

				fig.savefig(fig_camera_name, dpi=100, facecolor='white')
				plt.close(fig)


				# Correct frame IDs in camera dataframe.
				# correctedID = np.zeros(len(camera))

				# for i in tqdm(where_frames_lost):
				# 	correctedID[i:number_frames_diff] += 1 # And not correctedID[i:] += 1 because, when the buffer is full, the Mako U29-B camera keeps what is already in the buffer and does not receive any new frames while the buffer is full.

				# del where_frames_lost, number_frames_lost_diff

				# camera['Corrected ID'] = camera['ID'] + correctedID
				# camera['Corrected ID'] = camera['Corrected ID'].astype('int')

				# # Second estimate of the interframe interval, using the median, and after estimating where there are missing frames 
				# camera_diff = camera.loc[:,'ElapsedTime'].diff()
				# ifi = camera_diff.iloc[number_frames_discard_beg : -number_frames_discard_beg].median()
				# print('\nFirst estimate of IFI: {} ms'.format(ifi))

				return Lost_frames


			Lost_frames = lost_frames(camera, camera_diff, ifi, fish_name, fig_camera_name)

			return predicted_framerate, reference_frame_id, Lost_frames

		def read_tail_tracking_data(data_path):

			# Angles in data come in radians.

			try:
				
				start = timer()
				
				data = pd.read_csv(data_path, engine='pyarrow', sep=' ', usecols=cols_to_use_orig, header=0, decimal='.', na_filter=False, names=[frame_id]+data_cols)
				# dtype=dict(zip(cols_to_use_orig, ['int64'] + ['float32']*len(cols_to_use_orig))))
				# skipfooter=1
				data = data.iloc[:-1,:]
				
				#* Right now, pyarrow engine ignores renaming when opening the csv.
				data.rename(columns=dict(zip(cols_to_use_orig, [frame_id] + data_cols)), inplace=True)

				print('Time to read tail tracking .txt: {} (s)'.format(timer()-start))


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

			data = data.reset_index(drop=True).dropna()

			data[time_experiment_f] = data[time_experiment_f].astype('int64')

			data[[cs, us]] = data[[cs, us]].astype('Sparse[int16]')

			#* Fix dtypes.
			for cs_us in [cs, us]:

				data[cs_us] = data.loc[:, cs_us].astype(pd.CategoricalDtype(categories=data[cs_us].unique(), ordered=True))

			return data
		


		if not Overwrite and store.fish_is_in_store(self):

			self.raw_data = store.get_raw_data(self)

		else:

			if self.fish_raw_data_path is None:
				self.fish_raw_data_path = self.get_path(self, 'Raw')

			fish_name = self.name()

			print('Preprocessing fish ', fish_name)

			data_path = str(self.fish_raw_data_path) + 'mp tail tracking.txt'
			protocol_path = str(self.fish_raw_data_path) + 'stim control.txt'
			camera_path = str(self.fish_raw_data_path) + 'cam.txt'

			fig_camera_name = Experiment.path_first_analysis(self.metadata.experiment, fish_name, 'Camera log', 'png')
			fig_behavior_name = Experiment.path_first_analysis(self.metadata.experiment, fish_name, 'Behavior overview', 'png')
			fig_protocol_name = Experiment.path_first_analysis(self.metadata.experiment, fish_name, 'Stim log', 'png')

			#* Open the file with information about the time of each frame.
			if (camera := read_camera(camera_path)) is None:
				return None

			#* Estimate the true framerate.
			predicted_framerate, reference_frame_id, Lost_frames = framerate_and_reference_frame(camera, name, fig_camera_name)

			if Lost_frames:
				return None

			camera = camera.drop(columns=ela_time)

			#* Discard frames that will not be used (in camera and hence further down).
			# The calculated interframe interval before the reference frame is variable. Discard what happens up to then (also achieved by using how='inner' in merge_camera_with_data).
			camera = camera[camera[frame_id] >= reference_frame_id]

			#* Open tail tracking data.
			if (data := read_tail_tracking_data(data_path)) is None: # type: ignore
				return None

			plot_behavior_overview(data, fish_name, fig_behavior_name)

			#* Look for possible tail tracking errors.
			if tracking_errors(data, single_point_tracking_error_thr):
				return None

			#* Add information about the time of each frame to data.
			data = merge_camera_with_data(data, camera)

			#* Fix abs_time so that the time of each frame becomes closer to the time at which the frames were acquired by the camera and not when they were caught by the computer.
			# The delay between acquiring and catching the frame is unknown and therefore disregarded.
			data[abs_time] = np.linspace(data[abs_time].iat[0], data[abs_time].iat[0] + len(data) * (1000 / predicted_framerate), len(data))

			#* Interpolate data to the expected framerate.
			data = interpolate_data(data, predicted_framerate)

			#* Open the stim log.
			if (protocol := read_protocol(protocol_path)) is None:
				return None

#TODO replace by exp_var.experiments_info[Experiment.name]
			if lost_stim(len(protocol.loc[cs,:]), len(protocol.loc[us,:]), Experiment.get_experiment_info(self.metadata.experiment, 'parts')['trials'][cs]['elements'][1], Experiment.get_experiment_info(self.metadata.experiment, 'parts')['trials'][us]['elements'][1]):
				print('Experiment did not run until the end')

			cs_dur, cs_isi, us_dur, us_isi = protocol_info(protocol)

			#* Plot overview of the experimental protocol actually run
			plot_protocol(cs_dur, cs_isi, us_dur, us_isi, fish_name, fig_protocol_name)

			#* Identify the stimuli, trials of the experiment.
			data = identify_trials(data, protocol)

			print('\n\n')

			self.raw_data = data
			
			store.add_fish_raw_data(self)


	def filter_data(self):


		# def filter_data(data, space_bcf_window=space_bcf_window, time_bcf_window=time_bcf_window):


		# 	data_ = data[angle_cols].copy()

		# 	#* Calculate the cumulative sum in space.
		# 	data_[angle_cols] = data_[angle_cols].cumsum(axis=1)

		# 	#* Filter with a rolling average in space.
		# 	# This option is too slow. Using the transpose is also slow.
		# 	# data_[angle_cols] = data_[angle_cols].rolling(window=space_bcf_window, center=True, axis=1).mean()
		# 	#! Alexandre Laborde confirmed this.
		# 	# Not using pandas rolling mean beacause over columns it takes a lot of time (confirmed that with this way the result is the same)
		# 	# The fact that here we are using the cumsum means that when averaging more importance is given to the first points
		# 	data_[angle_cols[1:-1]] = np.mean(rolling_window(data_[angle_cols].to_numpy(), space_bcf_window), axis=2)
		# 	# data_.iloc[:, 2:-1] = np.mean(rolling_window(data_.iloc[:, 1:].to_numpy(), space_bcf_window), axis=2)
		# 	data_[angle_cols[0]] = data_[angle_cols[:3]].mean(axis=1)
		# 	# data_.iloc[:, 1] = data_.iloc[:, 1:3].mean(axis=1)
		# 	data_[angle_cols[-1]] = data_[angle_cols[-2:]].mean(axis=1)
		# 	# data_.iloc[:, -1] = data_.iloc[:, -2:].mean(axis=1)

		# 	#* Filter with a rolling average in time.
		# 	data_[angle_cols] = data_[angle_cols].rolling(window=time_bcf_window, center=True, axis=0).mean()

		# 	#* Update data with the values changed in data_.
		# 	data[angle_cols] = data_

		# 	data = data.dropna()

		# 	data[time_experiment_f] -= data[time_experiment_f].iat[0]

		# 	data = data.set_index(time_experiment_f)

		# 	print('Max tail angle at the chosen point: {} deg'.format(round(data.loc[:,tail_angle].max())))


			
		# # TODO might want to test Adrien's way of filtering data (from Megabouts)

		# 	# data[angle_cols] = data[angle_cols].cumsum(axis=1)

		# 	# from sklearn.decomposition import PCA
		# 	# from scipy.signal import savgol_filter

		# 	# pca = PCA(n_components=4)
		# 	# low_D = pca.fit_transform(data[angle_cols])
		# 	# data[angle_cols] = pca.inverse_transform(low_D)

		# 	# data[angle_cols] = savgol_filter(data[angle_cols], window_length=11, polyorder=2, deriv=0, delta=1.0, axis=0, mode='interp', cval=0.0)


		# 	return data

		# TODO clean up also coordinates of the tail
		#! Only filtering the angle data so far.
			#* Filter tail tracking data.
		# TODO might want to test Adrien's way of filtering data (from Megabouts)
		data = f.filter_data(data)

		self.data = data
			# return data


	def get_data(self, store: 'AllRawData', cols):

		#! cols are the columns to read from the dataframe

		# store.get_raw_data()

		pass







class AllRawData:

	def __init__(self, hdf_store_path: str, compression_level=4, compression_library: str='zlib'):

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
			store.append(fish.dataset_key(), fish.raw_data, data_columns=[cs, us], expectedrows=len(fish.raw_data), append=False)

			store.get_storer(fish.dataset_key()).attrs['metadata'] = fish.metadata._asdict()
			
		# fish.data_raw.to_hdf(self._path, fish.dataset_key(), data_columns=[cs, us], expectedrows=len(fish.data_raw), mode='w', complevel=self._compression_level, complib=self._compression_library)




	def get_raw_data(self, fish: Fish, where_to_query: str | None = None):
		# Optional[list[str]]

		with pd.HDFStore(self._path, complevel=self._compression_level, complib=self._compression_library) as store:

			return store.select(fish.dataset_key(), where=where_to_query)
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