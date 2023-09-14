from pathlib import Path
from dataclasses import dataclass

import numpy as np

from my_general_variables import *
import my_functions as f
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

all_experiments = ['original', 'stableVsIncreasingTrace', '10sTrace']

# MyLife(all_experiments).experiments_list(['original', 'stableVsIncreasingTrace'])


class MyLife:

	def __init__(self, all_experiments):

		for experiment in all_experiments:
			
			match experiment:

				#! November 2022 delay and increasing trace experiment
				case 'original':

					self.Delay_increasingTrace = self.Exp_original()

				# case 'stableVsIncreasingTrace':

				# 	self.stableTrace_increasingTrace = self.Experiment(experiment)

				# case '10sTrace':

				# 	self.shortTrace_longTrace = self.Experiment(experiment)
				
				# case 'traceMK801':
					
				# 	self.mk801Trace = self.Experiment(experiment)

				# case 'tracePuromycin':
					
				# 	self.puromycinTrace = self.Experiment(experiment)
				


# 	# # #TODO return an iterable of the experiments
# 	# def experiments_list(self, experiments_list):
		
# 	# 	# return [self.__dict__.get(x) for x in experiments_list]
# 	# 	return [attr for attr in self.__dict__.keys()]

class Experiment:

	def __init__(self, experiment):

		match experiment:

			#! November 2022 delay and increasing trace experiment
			case 'original':

				# self.Delay_increasingTrace = self.Exp_original()
	
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
	
	#! def
	# have access to all fish from the experiment
		# get all fish in AllFishInfo

#TODO cls.conditions_order = [cond.lower() for cond in condition_dict.keys()]

#TODO
	def path_first_analysis(cls, type_analysis, stem_fish_path_orig,  name_exp, frmt):
		
		path = cls.path_save / type_analysis
		path.mkdir(parents=True, exist_ok=True)
		
		return str(path / stem_fish_path_orig) + '_' + name_exp + '.' + frmt



	def path_processed_data(cls, type_analysis, alignment, name_fish, frmt):

#!!!!! HOW DOES IT WORK WITH ALIGNMENT SET TO NONE??

		path = cls.path_save / 'Processed data' / type_analysis / alignment
		path.mkdir(parents=True, exist_ok=True)
		
		return str(path / stem_fish_path_orig) + '_' + name_fish + '.' + frmt

	#!This could be moved to a JSON
	class Exp_original():

		def __init__(self):
			#* Path where the raw data is.
			self.path_home = Path(r'D:\2022 11_Basic delay and (increasing) trace CC paradigm\Raw data')
			self.path_save = Path(r'E:\Results (paper)\2022 11_Basic delay and (increasing) trace CC paradigm')

		#! Exception
		#! Might be moved to condition
			# self.cr_window = (0.5, 9)  # s

			self.expected_number_cs = 94
			self.expected_number_us = 78

			catch_trials_train = [25, 39, 53, 59]
			# catch_trials_plot = list(np.arange(5,15)) + catch_trials_train + list(np.arange(65,95))


			self.trials_blocks_cs = (range(5,15),
								range(15,25),
								range(25,35),
								range(35,45),
								range(45,55),
								range(55,65),
								range(65,75),
								range(75,85),
								range(85,95))

			# Starts at 18 because the first US is discarded in the analysis.
			self.trials_blocks_us = (range(18,28),
								range(28,37),
								range(37,46),
								range(46,55),
								range(55,64))

			self.names_blocks_cs = ('Pre-train',
								'Train 1',
								'Train 2',
								'Train 3',
								'Train 4',
								'Train 5',
								'Post-train 1',
								'Post-train 2',
								'Post-train 3')

			self.names_blocks_us = ('Train 1',
								'Train 2',
								'Train 3',
								'Train 4',
								'Train 5')

			self.trials_phases_cs = (range(5,15),
				   			range(15,65),
							range(65,95))

			self.trials_phases_us = (range(15,65))

			self.names_phases_cs = ('Pre-train',
							  'Train',
							  'Test')

			self.names_phases_us = ('Train',)

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
		.



class Fish:

	def __init__(self, name):

		self.name = name.lower()


USE PROPERTY STUFF HERE
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
		rig = info[3]

		if (rig_color := rig.split('-')[0] in ['orange', 'brown']):
			rig_color = 'white'
		elif rig.split('-')[0] in ['blue', 'black']:
			rig_color = 'red'
		
		strain = info[4]
		age = info[5].replace('dpf', '')

		return day, strain, age, condition, rig, rig_color, fish_number

	def get_path(self, alignment):
		
		

	
	PREPROCESSING FUNCTIONS

COMMITT