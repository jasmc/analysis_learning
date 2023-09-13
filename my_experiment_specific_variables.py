from typing import Final
from pathlib import Path
import numpy as np

from my_general_variables import *
import my_functions as f
# import Main as m



experiment_ = 'original'
# m.experiment
# 'longTermSpacedVsMassed-all'


# 'traceMK801'

# 'fixedVsIncreasingTrace'
# '10sTrace'
# 'tracePuromycin'
# 'tracePartialReinforcement'
# 'traceSpaced'


class Experiment:

	def __init__(self, experiment_):

		self.name = experiment_
	
		match experiment_:

			case 'original':

			#! November 2022 basic delay and trace experiment

				#* Path where the raw data is.
				self.path_home = Path(r'D:\2022 11_Basic delay and (increasing) trace CC paradigm\Raw data')
				self.path_save = Path(r'E:\Results (paper)\2022 11_Basic delay and (increasing) trace CC paradigm')

				cond0 = 'control'
				cond1 = 'delay'
				cond2 = 'trace'

				min_trace_interval_stable_numb_trials = 10
				max_trace_interval_stable_numb_trials = 10

				us_latency_trace_min = 0.5 # s
				us_latency_trace_max = 3  # s

				#! Careful with the number of catch trials!
				number_us_trials_increasing_trace = 26

				# us_latency_conditioning_list = cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)/1000


				#TODO add more stuff to this dict.
				condition_dict = {
					cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control',
			  			us_latency : None},
					cond1 : {color :  [216, 31, 98], name : 'Delay CC', id_in_path : 'delay',
			  			us_latency : [9]*46},
						  cond2 : {color : [226, 166, 14], name : 'Trace CC', id_in_path : 'trace',
				 			us_latency : cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}}

#! Exception
				self.cr_window = 9  # s

				# Minimum number of trials to not discard an experiment.
				min_number_cs_trials = 94
				min_number_us_trials = 78

				# To estimate maximum duration of the experiment.
				self.expected_number_cs = 94
				self.expected_number_us = 78
				# time_aft_last_trial = 1 # min


#! add this to the other experiments
				catch_trials_training = [25, 39, 53, 59]

				# catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



				# To plot pooled trials.

				# 9*10 trials
				trials_cs_blocks_10 = [[*range(5,15)],
										[*range(15,25)],
										[*range(25,35)],
										[*range(35,45)],
										[*range(45,55)],
										[*range(55,65)],
										[*range(65,75)],
										[*range(75,85)],
										[*range(85,95)]]

				# Starts at 18 because the first US is discarded in the analysis.
				trials_us_blocks_10 = [[*range(18,28)],
										[*range(28,37)],
										[*range(37,46)],
										[*range(46,55)],
										[*range(55,64)]]

				names_cs_blocks_10 = [
					'Pre-train',
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5',
					'Post-train 1',
					'Post-train 2',
					'Post-train 3']

				names_us_blocks_10 = [
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5']



				trials_cs_blocks_phases = [np.arange(5,15),
							   np.arange(15,65),
							   np.arange(65,95)]

				trials_us_blocks_phases = [np.arange(15,65)]

				names_cs_blocks_phases = [
					'Pre-train',
					'Train',
					'Test']

				names_us_blocks_phases = [
					'Train']
				

			case 'fixedVsIncreasingTrace':

			#! End February-March 2023 comparison between 3-s fixed trace and 3-s increasing trace

				#* Path where the raw data is.
				self.path_home = Path(r'D:\2023 02-03_Fixed vs increasing trace (3 s)\Raw data')
				self.path_save = Path(r'E:\Results (paper)\2023 02-03_Fixed vs increasing trace (3 s)')

				cond0 = 'control'
				cond1 = 'fixedTrace'
				cond2 = 'increasingTrace'

				min_trace_interval_stable_numb_trials = 10
				max_trace_interval_stable_numb_trials = 10

				us_latency_trace_min = 0.5  # s
				us_latency_trace_max = 3  # s

				#! Careful with the number of catch trials!
				number_us_trials_increasing_trace = 26


				#TODO add more stuff to this dict.
				condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
						  cond1 : {color :  [245, 130, 48], name : 'Trace CC fixed', id_in_path : 'fixedTrace', us_latency : [9]*46},
						  cond2 : {color : [226, 166, 14], name : 'Trace CC increasing', id_in_path : 'increasingTrace', us_latency : 
			       cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}
						  }


				cr_window = 10  # s


				# Minimum number of trials to not discard an experiment.
				min_number_cs_trials = 94
				min_number_us_trials = 78

				# To estimate maximum duration of the experiment.
				expected_number_cs = 94
				expected_number_us = 78
				time_aft_last_trial = 1 # min

				catch_trials_plot = list(np.arange(5,15)) + [25, 39, 53, 59] + list(np.arange(65,95))



				# To plot pooled trials.

				# 9*10 trials
				trials_cs_blocks_10 = [[*range(5,15)],
										[*range(15,25)],
										[*range(25,35)],
										[*range(35,45)],
										[*range(45,55)],
										[*range(55,65)],
										[*range(65,75)],
										[*range(75,85)],
										[*range(85,95)],
										]

				# Starts at 18 because the first US is discarded in the analysis.
				trials_us_blocks_10 = [[*range(18,28)],
										[*range(28,37)],
										[*range(37,46)],
										[*range(46,55)],
										[*range(55,64)]]

				names_cs_blocks_10 = [
					'Pre-train',
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5',
					'Post-train 1',
					'Post-train 2',
					'Post-train 3']

				names_us_blocks_10 = [
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5'
				]



				trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

				trials_us_blocks_phases = [np.arange(15,65)]

				names_cs_blocks_phases = [
					'Pre',
					'Train',
					'Post'
				]

				names_us_blocks_phases = [
					'Train'
				]

			case '10sTrace':

			#! March 2023 3-s trace vs 10-s trace experiment

				#* Path where the raw data is.
				self.path_home = Path(r'D:\2023 03_3-s vs 10-s fixed trace\Raw data')
				self.path_save = Path(r'E:\Results (paper)\2023 03_3-s vs 10-s fixed trace')

				
				cond0 = 'control'
				cond1 = '3sFixedTrace'
				cond2 = '10sFixedTrace'


				#* Colors for data in the plots.

				#TODO add more stuff to this dict.
				condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
						  cond1 : {color : [245, 130, 48], name : 'Trace 3 s fixed', id_in_path : '3sFixedTrace', us_latency : [13]*46},
						  cond2 : {color : [139,69,19], name : 'Trace 10 s fixed', id_in_path : '10sFixedTrace', us_latency : [20]*46}
						  }

				cr_window = 10  # s

				# Minimum number of trials to not discard an experiment.
				min_number_cs_trials = 94
				min_number_us_trials = 78

				# To estimate maximum duration of the experiment.
				expected_number_cs = 94
				expected_number_us = 78
				time_aft_last_trial = 1 # min

				catch_trials_plot = list(np.arange(5,15)) + [25, 39, 53, 59] + list(np.arange(65,95))



				# To plot pooled trials.

				# 9*10 trials
				trials_cs_blocks_10 = [[*range(5,15)],
										[*range(15,25)],
										[*range(25,35)],
										[*range(35,45)],
										[*range(45,55)],
										[*range(55,65)],
										[*range(65,75)],
										[*range(75,85)],
										[*range(85,95)],
										]

				# Starts at 18 because the first US is discarded in the analysis.
				trials_us_blocks_10 = [[*range(18,28)],
										[*range(28,37)],
										[*range(37,46)],
										[*range(46,55)],
										[*range(55,64)]]

				names_cs_blocks_10 = [
					'Pre-train',
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5',
					'Post-train 1',
					'Post-train 2',
					'Post-train 3']

				names_us_blocks_10 = [
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5'
				]



				trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

				trials_us_blocks_phases = [np.arange(15,65)]

				names_cs_blocks_phases = [
					'Pre',
					'Train',
					'Post'
				]

				names_us_blocks_phases = [
					'Train'
				]

			case 'traceMK801':
				
			#! April 2023 trace with MK 801 experiment

				#* Path where the raw data is.
				self.path_home = Path(r'D:\2023 04_Fixed 3-s trace with MK 801 (100 μM)\Raw data')
				self.path_save = Path(r'E:\Results (paper)\2023 04_Fixed 3-s trace with MK 801 (100 μM)')


				cond0 = 'controlNoMK801'
				cond1 = 'controlMK801'
				cond2 = 'traceNoMK801'
				cond3 = 'traceMK801'

				min_trace_interval_stable_numb_trials = 10
				max_trace_interval_stable_numb_trials = 10

				# us_latency_trace = 3  # s

				#! Careful with the number of catch trials!
				number_us_trials_increasing_trace = 26


				#TODO add more stuff to this dict.
				condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control No MK801', id_in_path : 'controlNoMK801'},

						  cond1 : {color : [245, 130, 48], name : 'Trace No MK801', id_in_path : 'traceNoMK801',
			       us_latency : [13]*46
				#    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
				   },

						  cond2 : {color :  [0, 0, 139], name : 'Control MK801', id_in_path : 'controlMK801'},

		   	              cond3 : {color : [246, 190, 0], name : 'Trace MK801', id_in_path : 'traceMK801',
			       us_latency : [13]*46
				#    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
				   },
					   }


				cr_window = 10  # s


				# Minimum number of trials to not discard an experiment.
				min_number_cs_trials = 94
				min_number_us_trials = 78

				# To estimate maximum duration of the experiment.
				expected_number_cs = 94
				expected_number_us = 78
				time_aft_last_trial = 1 # min

				catch_trials_plot = list(np.arange(5,15)) + [25, 39, 53, 59] + list(np.arange(65,95))



				# To plot pooled trials.

				# 9*10 trials
				trials_cs_blocks_10 = [[*range(5,15)],
										[*range(15,25)],
										[*range(25,35)],
										[*range(35,45)],
										[*range(45,55)],
										[*range(55,65)],
										[*range(65,75)],
										[*range(75,85)],
										[*range(85,95)],
										]

				# Starts at 18 because the first US is discarded in the analysis.
				trials_us_blocks_10 = [[*range(18,28)],
										[*range(28,37)],
										[*range(37,46)],
										[*range(46,55)],
										[*range(55,64)]]

				names_cs_blocks_10 = [
					'Pre-train',
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5',
					'Post-train 1',
					'Post-train 2',
					'Post-train 3']

				names_us_blocks_10 = [
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5'
				]



				trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

				trials_us_blocks_phases = [np.arange(15,65)]

				names_cs_blocks_phases = [
					'Pre',
					'Train',
					'Post'
				]

				names_us_blocks_phases = [
					'Train'
				]

			case 'tracePuromycin':
				
			#! May 2023 trace with MK 801 experiment

				#* Path where the raw data is.
				self.path_home = Path(r'D:\2023 05 & 08_Fixed 3-s trace with Puromycin (5 mg per L)\Raw data')
				self.path_save = Path(r'E:\Results (paper)\2023 05 & 08_Fixed 3-s trace with Puromycin (5 mg per L)')

				cond0 = 'controlNoPuromycin'
				cond1 = 'controlPuromycin'
				cond2 = 'traceNoPuromycin'
				cond3 = 'tracePuromycin'

				min_trace_interval_stable_numb_trials = 10
				max_trace_interval_stable_numb_trials = 10

				# us_latency_trace_min = 0.5 # s
				# us_latency_trace = 3  # s

				#! Careful with the number of catch trials!
				number_us_trials_increasing_trace = 26

				# us_latency_conditioning_list = cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)/1000


				#TODO add more stuff to this dict.
				condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control No Puromycin', id_in_path : 'controlNoPuromycin'},

						  cond1 : {color : [245, 130, 48], name : 'Trace No Puromycin', id_in_path : 'traceNoPuromycin',
			       us_latency : [13]*46
				#    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
				   },

						  cond2 : {color :  [0, 0, 139], name : 'Control Puromycin', id_in_path : 'controlPuromycin'},

		   	              cond3 : {color : [246, 190, 0], name : 'Trace Puromycin', id_in_path : 'tracePuromycin',
			       us_latency : [13]*46
				#    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
				   },
					   }


				cr_window = 10  # s


				# Minimum number of trials to not discard an experiment.
				min_number_cs_trials = 94
				min_number_us_trials = 78

				# To estimate maximum duration of the experiment.
				expected_number_cs = 94
				expected_number_us = 78
				time_aft_last_trial = 1 # min

				catch_trials_plot = list(np.arange(5,15)) + [25, 39, 53, 59] + list(np.arange(65,95))



				# To plot pooled trials.

				# 9*10 trials
				trials_cs_blocks_10 = [[*range(5,15)],
										[*range(15,25)],
										[*range(25,35)],
										[*range(35,45)],
										[*range(45,55)],
										[*range(55,65)],
										[*range(65,75)],
										[*range(75,85)],
										[*range(85,95)],
										]

				# Starts at 18 because the first US is discarded in the analysis.
				trials_us_blocks_10 = [[*range(18,28)],
										[*range(28,37)],
										[*range(37,46)],
										[*range(46,55)],
										[*range(55,64)]]

				names_cs_blocks_10 = [
					'Pre-train',
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5',
					'Post-train 1',
					'Post-train 2',
					'Post-train 3']

				names_us_blocks_10 = [
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5'
				]



				trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

				trials_us_blocks_phases = [np.arange(15,65)]

				names_cs_blocks_phases = [
					'Pre',
					'Train',
					'Post'
				]

				names_us_blocks_phases = [
					'Train'
				]

			case 'tracePartialReinforcement':

			#! August 2023 trace with partial and full reinforcement

				#* Path where the raw data is.
				# self.path_home = Path(r'I:\Joaquim\data')
				self.path_home = Path(r'D:\2023 08_Fully reinforced fixed 3-s trace vs partially reinforced\3\Raw data')
				self.path_save = Path(r'E:\Results (paper)\2023 08_Fully reinforced fixed 3-s trace vs partially reinforced\3')

				cond0 = 'control'
				cond1 = 'tracePartiallyReinforced'
				cond2 = 'traceFullyReinforced'

				us_latency_trace_min = 0.5 # s
				us_latency_trace = 3  # s


				#TODO add more stuff to this dict.
				condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
						  cond1 : {color :  [216, 31, 98], name : 'Trace Partially Reinforced', id_in_path : 'tracePartiallyReinforced'},
						  cond2 : {color : [226, 166, 14], name : 'Trace Fully Reinforced', id_in_path : 'traceFullyReinforced'}}

		#! Exception
				cr_window = 13  # s

				# Minimum number of trials to not discard an experiment.
				min_number_cs_trials = 94
				min_number_us_trials = 78

				# To estimate maximum duration of the experiment.
				expected_number_cs = 94
				expected_number_us = 78
				time_aft_last_trial = 1 # min


		#TODO add this to the other experiments
				catch_trials_training = [25, 39, 53, 59]

				catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

				# To plot pooled trials.

				# 9*10 trials
				trials_cs_blocks_10 = [[*range(5,15)],
										[*range(15,25)],
										[*range(25,35)],
										[*range(35,45)],
										[*range(45,55)],
										[*range(55,65)],
										[*range(65,75)],
										[*range(75,85)],
										[*range(85,95)],
										]

				# Starts at 18 because the first US is discarded in the analysis.
				trials_us_blocks_10 = [[*range(18,28)],
										[*range(28,37)],
										[*range(37,46)],
										[*range(46,55)],
										[*range(55,64)]]

				names_cs_blocks_10 = [
					'Pre-train',
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5',
					'Post-train 1',
					'Post-train 2',
					'Post-train 3']

				names_us_blocks_10 = [
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5'
				]



				trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

				trials_us_blocks_phases = [np.arange(18,65)]

				names_cs_blocks_phases = [
					'Pre',
					'Train',
					'Post'
				]

				names_us_blocks_phases = [
					'Train'
				]

			case 'traceSpaced':


			#! August 2023 trace with massed or spaced schedule

				#* Path where the raw data is.
				# self.path_home = Path(r'I:\Joaquim\data')
				self.path_home = Path(r'D:\2023 08_Spaced fixed 3-s trace\Raw data')
				self.path_save = Path(r'E:\Results (paper)\2023 08_Spaced fixed 3-s trace')

				cond0 = 'control'
				cond1 = 'traceSpaced'
				# cond2 = 'traceMassed'

				us_latency_trace = 3  # s


				#TODO add more stuff to this dict.
				condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
						  cond1 : {color :  [216, 31, 98], name : 'Trace Spaced', id_in_path : 'traceSpaced'},
						#   cond2 : {color : [226, 166, 14], name : 'Trace Massed', id_in_path : 'traceMassed'}
						  }

				cr_window = 13  # s

				# Minimum number of trials to not discard an experiment.
				min_number_cs_trials = 69
				min_number_us_trials = 66

				# To estimate maximum duration of the experiment.
				expected_number_cs = 69
				expected_number_us = 66
				time_aft_last_trial = 1 # min


		#TODO add this to the other experiments
				# catch_trials_training = [25, 39, 53, 59]

				# catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

				# To plot pooled trials.

				# 9*10 trials
				trials_cs_blocks_10 = [[*range(5,15)],
										[*range(15,25)],
										[*range(25,35)],
										[*range(35,45)],
										[*range(45,55)],
										[*range(55,65)],
										[*range(65,70)],
										# [*range(75,85)],
										# [*range(85,95)],
										]

				# Starts at 18 because the first US is discarded in the analysis.
				trials_us_blocks_10 = [[*range(18,28)],
										[*range(28,37)],
										[*range(37,46)],
										[*range(46,55)],
										[*range(55,64)]]

				names_cs_blocks_10 = [
					'Pre-train',
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5',
					'Post-train 1',
					# 'Post-train 2',
					# 'Post-train 3'
					]

				names_us_blocks_10 = [
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5'
				]



				trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,70)]

				trials_us_blocks_phases = [np.arange(15,65)]

				names_cs_blocks_phases = [
					'Pre',
					'Train',
					'Post'
				]

				names_us_blocks_phases = [
					'Train'
				]

			case 'longTermSpacedVsMassed-1':

			#! August 2023 trace with massed or spaced schedule

				#* Path where the raw data is.
				# self.path_home = Path(r'I:\Joaquim\data')
				self.path_home = Path(r'D:\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\2\Block 1\Raw data')
				self.path_save = Path(r'E:\Results (paper)\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\2\Block 1')

				cond0 = 'controlMassed'
				cond1 = 'controlSpaced'
				cond2 = 'traceMassed'
				cond3 = 'traceSpaced'

				us_latency_trace = 3  # s


				#TODO add more stuff to this dict.
				condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control Massed', id_in_path : 'controlMassed'},
						  cond1 : {color :  [245, 130, 48], name : 'Control Spaced', id_in_path : 'controlSpaced'},
						  cond2 : {color : [0, 0, 139], name : 'Trace Massed', id_in_path : 'traceMassed'},
						  cond3 : {color : [246, 190, 0], name : 'Trace Spaced', id_in_path : 'traceSpaced'}
						  }

				cr_window = 13  # s

				# Minimum number of trials to not discard an experiment.
				min_number_cs_trials = 64
				min_number_us_trials = 63

				# To estimate maximum duration of the experiment.
				expected_number_cs = 64
				expected_number_us = 63
				# time_aft_last_trial = 1 # min


		#TODO add this to the other experiments
				# catch_trials_training = [25, 39, 53, 59]

				# catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

				# To plot pooled trials.

				# 9*10 trials
				trials_cs_blocks_10 = [[*range(5,15)],
										[*range(15,25)],
										[*range(25,35)],
										[*range(35,45)],
										[*range(45,55)],
										[*range(55,65)],
										# [*range(65,70)],
										# [*range(75,85)],
										# [*range(85,95)],
										]

				# Starts at 18 because the first US is discarded in the analysis.
				trials_us_blocks_10 = [[*range(18,28)],
										[*range(28,37)],
										[*range(37,46)],
										[*range(46,55)],
										[*range(55,64)]]

				names_cs_blocks_10 = [
					'Pre-train',
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5',
				]

				names_us_blocks_10 = [
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5'
				]



				trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65)]

				trials_us_blocks_phases = [np.arange(15,65)]

				names_cs_blocks_phases = [
					'Pre',
					'Train',
				]

				names_us_blocks_phases = [
					'Train'
				]

			case 'old longTermSpacedVsMassed-2':

			#! August 2023 trace with massed or spaced schedule

				#* Path where the raw data is.
				# self.path_home = Path(r'I:\Joaquim\data')
				self.path_home = Path(r'D:\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\4\Block 2\Raw data')
				self.path_save = Path(r'E:\Results (paper)\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\4\Block 2')

				cond0 = 'controlMassed'
				cond1 = 'controlSpaced'
				cond2 = 'traceMassed'
				cond3 = 'traceSpaced'

				us_latency_trace = 3  # s


				#TODO add more stuff to this dict.
				condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control Massed', id_in_path : 'controlMassed'},
						  cond1 : {color :  [245, 130, 48], name : 'Control Spaced', id_in_path : 'controlSpaced'},
						  cond2 : {color : [0, 0, 139], name : 'Trace Massed', id_in_path : 'traceMassed'},
						  cond3 : {color : [246, 190, 0], name : 'Trace Spaced', id_in_path : 'traceSpaced'}
						  }

				cr_window = 13  # s

				# Minimum number of trials to not discard an experiment.
				min_number_cs_trials = 10
				min_number_us_trials = 2

				# To estimate maximum duration of the experiment.
				expected_number_cs = 10
				expected_number_us = 2
				# time_aft_last_trial = 1 # min


		#TODO add this to the other experiments
				# catch_trials_training = [25, 39, 53, 59]

				# catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

				# To plot pooled trials.

				# 9*10 trials
				trials_cs_blocks_10 = [[*range(64,74)]]

				# Starts at 18 because the first US is discarded in the analysis.
				trials_us_blocks_10 = []
				names_cs_blocks_10 = ['Test']

				names_us_blocks_10 = []



				trials_cs_blocks_phases = [np.arange(64,74)]

				trials_us_blocks_phases = []

				names_cs_blocks_phases = ['Test']

				names_us_blocks_phases = []

			case 'longTermSpacedVsMassed-2':

			#! August 2023 trace with massed or spaced schedule

				#* Path where the raw data is.
				# self.path_home = Path(r'I:\Joaquim\data')
				self.path_home = Path(r'D:\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\3\Block 2\Raw data')
				self.path_save = Path(r'E:\Results (paper)\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\3\Block 2')

				cond0 = 'controlMassed'
				cond1 = 'controlSpaced'
				cond2 = 'traceMassed'
				cond3 = 'traceSpaced'

				us_latency_trace = 3  # s


				#TODO add more stuff to this dict.
				condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control Massed', id_in_path : 'controlMassed'},
						  cond1 : {color :  [245, 130, 48], name : 'Control Spaced', id_in_path : 'controlSpaced'},
						  cond2 : {color : [0, 0, 139], name : 'Trace Massed', id_in_path : 'traceMassed'},
						  cond3 : {color : [246, 190, 0], name : 'Trace Spaced', id_in_path : 'traceSpaced'}
						  }

				cr_window = 13  # s

				# Minimum number of trials to not discard an experiment.
				min_number_cs_trials = 110
				min_number_us_trials = 74

				# To estimate maximum duration of the experiment.
				expected_number_cs = 110
				expected_number_us = 74
				# time_aft_last_trial = 1 # min


		#TODO add this to the other experiments
				catch_trials_training = [25, 39, 53, 59]

				# catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

				# To plot pooled trials.

				# 9*10 trials
				trials_cs_blocks_10 = [[*range(1,11)],
										[*range(11,21)],
										[*range(21,31)],
										[*range(31,41)],
										[*range(41,51)],
										[*range(51,61)],
										[*range(61,71)],
										[*range(71,81)],
										[*range(81,91)],
										[*range(91,101)],
										[*range(101,111)]
										]

				# Starts at 18 because the first US is discarded in the analysis.
				trials_us_blocks_10 = [[*range(15,25)],
								 		[*range(25,34)],
										[*range(34,43)],
										[*range(43,52)],
										[*range(52,61)]
										]

				names_cs_blocks_10 = [
								'Test 1',
								'Test 2',
								'Test 3',
								'Retrain 1',
								'Retrain 2',
								'Retrain 3',
								'Retrain 4',
								'Retrain 5',
								'Post-retrain 1',
								'Post-retrain 2',
								'Post-retrain 3'
								]

				names_us_blocks_10 = [
								'Retrain 1',
								'Retrain 2',
								'Retrain 3',
								'Retrain 4',
								'Retrain 5'
								]



				trials_cs_blocks_phases = [np.arange(1,31), np.arange(31,81), np.arange(81,111)]

				trials_us_blocks_phases = [np.arange(14,60)]

				names_cs_blocks_phases = ['Test',
									'Retrain',
									'Post-retrain']

				names_us_blocks_phases = ['Retrain']



		#!!!!!!!!!!!!!!!!!!! do this
			case 'longTermSpacedVsMassed-all':

			#! August 2023 trace with massed or spaced schedule

				#* Path where the raw data is.
				# self.path_home = Path(r'I:\Joaquim\data')
				self.path_home = Path(r'D:\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\3\All blocks\Raw data')
				self.path_save = Path(r'E:\Results (paper)\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\3\All blocks')

				cond0 = 'controlMassed'
				cond1 = 'controlSpaced'
				cond2 = 'traceMassed'
				cond3 = 'traceSpaced'

				us_latency_trace = 3  # s


				#TODO add more stuff to this dict.
				condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control Massed', id_in_path : 'controlMassed'},
						  cond1 : {color :  [245, 130, 48], name : 'Control Spaced', id_in_path : 'controlSpaced'},
						  cond2 : {color : [0, 0, 139], name : 'Trace Massed', id_in_path : 'traceMassed'},
						  cond3 : {color : [246, 190, 0], name : 'Trace Spaced', id_in_path : 'traceSpaced'}
						  }

				cr_window = 13  # s

				# Minimum number of trials to not discard an experiment.
				min_number_cs_trials = 64 + 110
				min_number_us_trials = 63 + 74

				# To estimate maximum duration of the experiment.
				expected_number_cs = 64 + 110
				expected_number_us = 63 + 74
				# time_aft_last_trial = 1 # min


		#TODO add this to the other experiments
				catch_trials_training = [25, 39, 53, 59] + [25 + 64, 39 + 64, 53 + 64, 59 + 64]

				# catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

				# To plot pooled trials.

				# 9*10 trials
				trials_cs_blocks_10 = [[*range(5,15)],
										[*range(15,25)],
										[*range(25,35)],
										[*range(35,45)],
										[*range(45,55)],
										[*range(55,65)],
										
										[*range(65,75)],
										[*range(75,85)],
										[*range(85,95)],
										[*range(95,105)],
										[*range(105,115)],
										[*range(115,125)],
										[*range(125,135)],
										[*range(135,145)],
										[*range(145,155)],
										[*range(155,165)],
										[*range(165,175)]
										]

				# Starts at 18 because the first US is discarded in the analysis.
				trials_us_blocks_10 = [[*range(18,28)],
										[*range(28,37)],
										[*range(37,46)],
										[*range(46,55)],
										[*range(55,64)],

										[*range(79,89)],
								 		[*range(89,98)],
										[*range(98,107)],
										[*range(107,116)],
										[*range(116,117)]
										]

				names_cs_blocks_10 = [
					'Pre-train',
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5',
					'Test 1',
					'Test 2',
					'Test 3',
					'Retrain 1',
					'Retrain 2',
					'Retrain 3',
					'Retrain 4',
					'Retrain 5',
					'Post-retrain 1',
					'Post-retrain 2',
					'Post-retrain 3'
				]

				names_us_blocks_10 = [
					'Train 1',
					'Train 2',
					'Train 3',
					'Train 4',
					'Train 5',
					'Retrain 1',
					'Retrain 2',
					'Retrain 3',
					'Retrain 4',
					'Retrain 5'
				]



				trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65, 95), np.arange(95, 145), np.arange(145, 175)]

				trials_us_blocks_phases = [np.arange(15,65), np.arange(79, 117)]



				names_cs_blocks_phases = [
					'Pre',
					'Train',
					'Test',
					'Retrain',
					'Post-retrain']

				names_us_blocks_phases = [
					'Train',
					'Retrain']





		path_lost_frames, path_summary_exp, path_summary_beh, path_processed_data, path_cropped_exp_with_bout_detection, path_tail_angle_fig_cs, path_tail_angle_fig_us, path_raw_vigor_fig_cs, path_raw_vigor_fig_us, path_scaled_vigor_fig_cs, path_scaled_vigor_fig_us, path_suppression_ratio_fig_cs, path_suppression_ratio_fig_us, path_pooled_vigor_fig, path_analysis_protocols, path_orig_pkl, path_all_fish, path_pooled = f.create_folders(self.path_save)




		self.exp_types = [*condition_dict.keys()]
		self.exp_types_names = [condition_dict[k][id_in_path] for k in [*condition_dict.keys()]]
		self.exp_types_order = [cond.lower() for cond in condition_dict.keys()]

		self.color_palette = [condition_dict[x][color] for x in condition_dict.keys()]

		for c in condition_dict.keys():
			condition_dict[c][color][:3] = [i / 255 for i in condition_dict[c][color][:3]]



		# To plot single trials.
		blocks_cs_single_trials = [[x] for x in np.arange(1, expected_number_cs+1)]

		blocks_us_single_trials = [[x] for x in np.arange(1, expected_number_us+1)]

		number_rows_single_trials = max(len(blocks_cs_single_trials), len(blocks_us_single_trials))


		number_rows_blocks_10 = max(len(trials_cs_blocks_10), len(trials_us_blocks_10))

		number_rows_blocks_phases = max(len(trials_cs_blocks_phases), len(trials_us_blocks_phases))




		blocks_dict = {

			blocks_1t : {
				cs : {
					trials_blocks : blocks_cs_single_trials,
					names_blocks : [str(x[0]) for x in blocks_cs_single_trials]
					},
				us : {
					trials_blocks : blocks_us_single_trials,
					names_blocks : [str(x[0]) for x in blocks_us_single_trials]
					},
				number_cols_or_rows : expected_number_cs,
				horizontal_fig : False,
				fig_size : (15,2*expected_number_cs/3)
				},
				
			blocks_10t : {
				cs : {
					trials_blocks : trials_cs_blocks_10,
					names_blocks : names_cs_blocks_10
					},
				us : {
					trials_blocks : trials_us_blocks_10,
					names_blocks : names_us_blocks_10
					},
				number_cols_or_rows : number_rows_blocks_10,
				horizontal_fig : True,
				fig_size : (60*number_rows_blocks_10/4,15)
				},
			
			blocks_phases : {
				cs : {
					trials_blocks : trials_cs_blocks_phases,
					names_blocks : names_cs_blocks_phases
					},
				us : {
					trials_blocks : trials_us_blocks_phases,
					names_blocks : names_us_blocks_phases
					},
				number_cols_or_rows : number_rows_blocks_phases,
				}
			}