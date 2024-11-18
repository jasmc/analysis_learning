# from typing import Final
from pathlib import Path

import numpy as np
import seaborn as sns

import my_functions_behavior as f
from my_general_variables import *

experiment_ = '2-P multiple planes top'

# '2-P multiple planes zoom in'
# 'delayMK801'
# 'longTermDelaySpacedPuromycin-all'
# '2-P multiple planes bottom'
# '2-P single plane'
# '2-P multiple planes ca8'

# '3sTraceFrom2DiffExp'
# 'original'
# 'traceMK801'






# '10sTrace'
# 'fixedVsIncreasingTrace'




# 'tracePuromycin'



# 'cfos'
# 'ablation'

# 'delayDanionella'
# 'longTermDelaySpacedPuromycin-all'
# 'longTermSpacedVsMassed-all'
# 'campari'
# 'tracePartialReinforcement'

match experiment_:

	case 'ablation':

	#! November 2022 basic delay and trace experiment
	
		#* Path where the raw data is.
		path_home = Path(r'D:\2024 06_Delay with Pkj cells ablated')
		path_save = Path(r'E:\Results (paper)\2024 06_Delay with Pkj cells ablated')

		cond0 = 'delaygcampnomtz'
		cond1 = 'delaygcampwithmtz'
		cond2 ='delayntrnomtz'
		cond3 = 'delayntrwithmtz'

		condition_dict = {cond0 : {color : [58, 129, 195], name : 'DelayGCaMPNoMTZ', id_in_path : 'delaygcampnomtz'},
		cond1 : {color :  [216, 31, 98], name : 'DelayGCaMPWithMTZ', id_in_path : 'delaygcampwithmtz'},
		cond2 : {color : [185,160,255], name : 'DelayNTRNoMTZ', id_in_path : 'delayntrnomtz'},
		cond3 : {color : [255, 0, 0], name : 'DelayNTRWithMTZ', id_in_path : 'delayntrwithmtz'}}

#! Exception
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 94
		min_number_us_trials = 78

		# To estimate maximum duration of the experiment.
		expected_number_cs = 94
		expected_number_us = 78
		time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		catch_trials_training = [25, 39, 53, 59]

		catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
		# list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



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
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Post-conditioning 1',
			'Post-conditioning 2',
			'Post-conditioning 3']

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5']



		trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

		trials_us_blocks_phases = [np.arange(15,65)]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
			'Post']

		names_us_blocks_phases = [
			'Train']




	case 'cfos':
		
		#* Path where the raw data is.
		path_home = Path(r'D:\2024 05_Delay c-Fos')
		path_save = Path(r'E:\Results (paper)\2024 05_Delay c-Fos')

		cond0 = 'control'
		cond1 = 'delay'

		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
						cond1 : {color :  [216, 31, 98], name : 'Delay CC', id_in_path : 'delay', us_latency : [9]*46}}

		#! Exception
		cr_window = 9  # s


		#! FIX ALL

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 64
		min_number_us_trials = 64

		# To estimate maximum duration of the experiment.
		expected_number_cs = 94
		expected_number_us = 78
		time_aft_last_trial = 1 # min

		catch_trials_training = [25, 39, 53, 59]

		catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
		# list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



		# To plot pooled trials.

		# 9*10 trials
		trials_cs_blocks_10 = [[*range(5,15)],
								[*range(15,25)],
								[*range(25,35)],
								[*range(35,45)],
								[*range(45,55)],
								[*range(55,65)]]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [[*range(18,28)],
								[*range(28,37)],
								[*range(37,46)],
								[*range(46,55)],
								[*range(55,64)]]

		names_cs_blocks_10 = ['Pre-conditioning',
							'Conditioning 1',
							'Conditioning 2',
							'Conditioning 3',
							'Conditioning 4',
							'Conditioning 5']

		names_us_blocks_10 = ['Conditioning 1',
							'Conditioning 2',
							'Conditioning 3',
							'Conditioning 4',
							'Conditioning 5']


		trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65)]

		trials_us_blocks_phases = [np.arange(15,65)]

		names_cs_blocks_phases = ['Pre', 'Train']

		names_us_blocks_phases = ['Train']


	case 'longTermDelaySpacedPuromycinNew-1':

	#! July 2024 trace with spaced schedule

		#* Path where the raw data is.
		path_home = Path(r'D:\2024 07_Long-term memory control, delay spaced training with Puromycin (25 mg per L)\2+3\Block 1')
		path_save = Path(r'E:\Results (paper)\2024 07_Long-term memory control, delay spaced training with Puromycin (25 mg per L)\2+3\Block 1')


		cond0 = 'controlSpacedNoPuromycin'
		cond1 = 'delaySpacedNoPuromycin'
		cond2 = 'controlSpacedWithPuromycin'
		cond3 = 'delaySpacedWithPuromycin'

		us_latency_delay = -1  # s


		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : 'lightblue', name : 'Control Spaced No Puromycin', id_in_path : 'controlSpacedNoPuromycin'},
				  cond1 : {color :  'pink', name : 'Control Spaced Puromycin', id_in_path : 'controlSpacedPuromycin'},
				  cond2 : {color : 'blue', name : 'Delay Spaced No Puromycin', id_in_path : 'delaySpacedNoPuromycin'},
				  cond3 : {color : 'red', name : 'Delay Spaced Puromycin', id_in_path : 'delaySpacedPuromycin'}
				  }
		
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 64
		min_number_us_trials = 59

		# To estimate maximum duration of the experiment.
		expected_number_cs = 64
		expected_number_us = 59
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
								]


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

		trials_us_blocks_phases = [np.arange(18,64)]
		# [np.arange(15,65)]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
		]

		names_us_blocks_phases = [
			'Train'
		]

	case 'longTermDelaySpacedPuromycinNew-2':

		#* Path where the raw data is.
		path_home = Path(r'D:\2024 07_Long-term memory control, delay spaced training with Puromycin (25 mg per L)\2+3\Block 2')
		path_save = Path(r'E:\Results (paper)\2024 07_Long-term memory control, delay spaced training with Puromycin (25 mg per L)\2+3\Block 2')

		cond0 = 'controlSpacedNoPuromycin'
		cond1 = 'delaySpacedNoPuromycin'
		cond2 = 'controlSpacedWithPuromycin'
		cond3 = 'delaySpacedWithPuromycin'

		us_latency_delay = -1  # s


		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : 'lightblue', name : 'Control Spaced No Puromycin', id_in_path : 'controlSpacedNoPuromycin'},
				  cond1 : {color :  'pink', name : 'Control Spaced Puromycin', id_in_path : 'controlSpacedPuromycin'},
				  cond2 : {color : 'blue', name : 'Delay Spaced No Puromycin', id_in_path : 'delaySpacedNoPuromycin'},
				  cond3 : {color : 'red', name : 'Delay Spaced Puromycin', id_in_path : 'delaySpacedPuromycin'}
				  }

		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 70
		min_number_us_trials = 52

		# To estimate maximum duration of the experiment.
		expected_number_cs = 70
		expected_number_us = 52
		# time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		# catch_trials_training = [25, 39, 53, 59]

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
								]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [
								[*range(16,26)],
								[*range(26,35)],
								[*range(35,44)],
								[*range(44,53)],
								[*range(53,62)],
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
						]

		names_us_blocks_10 = [
						'Retrain 1',
						'Retrain 2',
						'Retrain 3',
						'Retrain 4',
						'Retrain 5',
						]

		trials_cs_blocks_phases = [np.arange(1,31), np.arange(31,51), np.arange(51,81)]

		trials_us_blocks_phases = [np.arange(15,34), np.arange(34,62)]

		names_cs_blocks_phases = ['Test 1', 'Retrain 1', 'Retrain 2']

		names_us_blocks_phases = ['Retrain 1', 'Retrain 2']

	case 'longTermDelaySpacedPuromycinNew-3':

		#* Path where the raw data is.
		path_home = Path(r'D:\2024 07_Long-term memory control, delay spaced training with Puromycin (25 mg per L)\2+3\Block 3')
		path_save = Path(r'E:\Results (paper)\2024 07_Long-term memory control, delay spaced training with Puromycin (25 mg per L)\2+3\Block 3')

		cond0 = 'controlSpacedNoPuromycin'
		cond1 = 'delaySpacedNoPuromycin'
		cond2 = 'controlSpacedWithPuromycin'
		cond3 = 'delaySpacedWithPuromycin'

		us_latency_delay = -1  # s

		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : 'lightblue', name : 'Control Spaced No Puromycin', id_in_path : 'controlSpacedNoPuromycin'},
				  cond1 : {color :  'pink', name : 'Control Spaced Puromycin', id_in_path : 'controlSpacedPuromycin'},
				  cond2 : {color : 'blue', name : 'Delay Spaced No Puromycin', id_in_path : 'delaySpacedNoPuromycin'},
				  cond3 : {color : 'red', name : 'Delay Spaced Puromycin', id_in_path : 'delaySpacedPuromycin'}
				  }

		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 50
		min_number_us_trials = 35

		# To estimate maximum duration of the experiment.
		expected_number_cs = 50
		expected_number_us = 35
		# time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		# catch_trials_training = [25, 39, 53, 59]

		# catch_trials_plot = list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))

		# To plot pooled trials.

		# 9*10 trials
		trials_cs_blocks_10 = [[*range(1,11)],
								[*range(11,21)],
								[*range(21,31)],
								[*range(31,41)],
								[*range(41,51)],
								]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [
								[*range(16,26)],
								[*range(26,35)],
								]

		names_cs_blocks_10 = [
						'Test 4',
						'Test 5',
						'Test 6',
						'Retrain 6',
						'Retrain 7',
						]

		names_us_blocks_10 = [
						'Retrain 6',
						'Retrain 7',
						]



		trials_cs_blocks_phases = [np.arange(1,31), np.arange(31,51)]

		trials_us_blocks_phases = [np.arange(16,34)]

		names_cs_blocks_phases = ['Test 2', 'Retrain 3']

		names_us_blocks_phases = ['Retrain 3']

	case 'longTermDelaySpacedPuromycinNew-all':

	#! August 2023 delay with massed or spaced schedule

		#* Path where the raw data is.
		path_home = Path(r'D:\2024 07_Long-term memory control, delay spaced training with Puromycin (25 mg per L)\2+3\All blocks')
		path_save = Path(r'E:\Results (paper)\2024 07_Long-term memory control, delay spaced training with Puromycin (25 mg per L)\2+3\All blocks')

		cond0 = 'controlSpacedNoPuromycin'
		cond1 = 'controlSpacedPuromycin'
		cond2 = 'delaySpacedNoPuromycin'
		cond3 = 'delaySpacedPuromycin'

		us_latency_delay = -1  # s

		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : 'lightblue', name : 'Control Spaced No Puromycin', id_in_path : 'controlSpacedNoPuromycin'},
				  cond1 : {color :  'pink', name : 'Control Spaced Puromycin', id_in_path : 'controlSpacedPuromycin'},
				  cond2 : {color : 'blue', name : 'Delay Spaced No Puromycin', id_in_path : 'delaySpacedNoPuromycin'},
				  cond3 : {color : 'red', name : 'Delay Spaced Puromycin', id_in_path : 'delaySpacedPuromycin'}
				  }

		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 64 + 70 + 50
		min_number_us_trials = 59 + 52 + 35

		# To estimate maximum duration of the experiment.
		expected_number_cs = 64 + 70 + 50
		expected_number_us = 59 + 52 + 35
		# time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		# catch_trials_training = [25, 39, 53, 59] + [25 + 64, 39 + 64, 53 + 64, 59 + 64]

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
								[*range(165,175)],
								[*range(175,185)]
								]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [[*range(18,28)],
								[*range(28,37)],
								[*range(37,46)],
								[*range(46,55)],
								[*range(55,64)],

								[*range(82,92)],
								[*range(92,101)],
								[*range(101,110)],
								[*range(110,119)],

								[*range(119,129)],
								[*range(129,138)],
								[*range(138,147)],
								[*range(147,156)],
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

			'Test 4',
			'Test 5',
			'Test 6',
			'Retrain 6',
			'Retrain 7',
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
			'Retrain 5',

			'Retrain 6',
			'Retrain 7',
		]



		trials_cs_blocks_phases = [
			
			np.arange(5,15), np.arange(15,65),
			
			np.arange(65, 95), np.arange(95, 115), np.arange(115, 135),

			np.arange(135, 165), np.arange(165, 185)
		]

		trials_us_blocks_phases = [
			
			np.arange(18,64),
			
			np.arange(82, 101), np.arange(101, 119),

			np.arange(119, 156)
		]



		names_cs_blocks_phases = [
			'Pre',
			'Train',
			'Test 1',
			'Retrain 1',
			'Retrain 2',
			'Test 2',
			'Retrain 3'
			]

		names_us_blocks_phases = [
			'Train',
			'Retrain 1',
			'Retrain 2',
			'Retrain 3'
			]




	case 'delayDanionella':

	#! June 2024 basic delay experiment

		#* Path where the raw data is.
		path_home = Path(r'F:\Pilot studies\2024 06_Delay with Danionella')
		path_save = Path(r'F:\Pilot studies\2024 06_Delay with Danionella\Results')


		cond0 = 'control'
		cond1 = 'delay'


		min_trace_interval_stable_numb_trials = 10
		max_trace_interval_stable_numb_trials = 10

		us_latency_trace_min = 0.5 # s
		us_latency_trace_max = 3  # s

		#! Careful with the number of catch trials!
		# number_us_trials_increasing_trace = 26

		# us_latency_conditioning_list = cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)/1000


		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
				  cond1 : {color :  [216, 31, 98], name : 'Delay CC', id_in_path : 'delay', us_latency : [9]*46}}

#! Exception
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 90
		min_number_us_trials = 65

		# To estimate maximum duration of the experiment.
		expected_number_cs = 90
		expected_number_us = 65
		time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		# catch_trials_training = [25, 39, 53, 59]

		# catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
		# list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



		# To plot pooled trials.

		# 9*10 trials
		trials_cs_blocks_10 = [[*range(1,11)],
								[*range(11,21)],
								[*range(21,31)],
								[*range(31,41)],
								[*range(41,11)],
								[*range(51,61)],
								[*range(61,71)],
								[*range(71,81)],
								[*range(81,91)],
								]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [[*range(5,15)],
								[*range(15,24)],
								[*range(24,33)],
								[*range(33,42)],
								[*range(42,51)]]

		names_cs_blocks_10 = [
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Post-conditioning 1',
			'Post-conditioning 2',
			'Post-conditioning 3']

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5']



		trials_cs_blocks_phases = [np.arange(1,11), np.arange(11,61), np.arange(61,91)]

		trials_us_blocks_phases = [np.arange(5,51)]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
			'Post']

		names_us_blocks_phases = [
			'Train']




	case 'delayMK801':

	#! November 2022 basic delay and delay experiment

		#* Path where the raw data is.
		# path_home = Path(r'I:\Joaquim\data')
		path_home = Path(r'D:\2024 06_Delay with MK 801 (10 vs 100 μM)')
		path_save = Path(r'E:\Results (paper)\2024 06_Delay with MK 801 (10 vs 100 μM)')

		cond0 = 'controlHighMK801'
		cond1 = 'delayHighMK801'
		
		cond2 = 'controlLowMK801'
		cond3 = 'delayLowMK801'
		
		cond4 = 'controlNoMK801'
		cond5 = 'delayNoMK801'

		min_delay_interval_stable_numb_trials = 10
		max_delay_interval_stable_numb_trials = 10

		# us_latency_delay = 3  # s

		#! Careful with the number of catch trials!
		number_us_trials_increasing_delay = 26


		#TODO add more stuff to this dict.
		condition_dict = {
				cond0 : {color :  sns.color_palette('colorblind')[0], name : 'Control High MK801', id_in_path : 'controlHighMK801'},

				  cond1 : {color :  sns.color_palette('colorblind')[4], name : 'Delay High MK801', id_in_path : 'delayHighMK801', us_latency : [9]*46},

				  cond2 : {color :  'blue', name : 'Control Low MK801', id_in_path : 'controlLowMK801'},

   	              cond3 : {color : 'magenta', name : 'Delay Low MK801', id_in_path : 'delayLowMK801', us_latency : [9]*46},
				
				  cond4 : {color :  sns.color_palette('colorblind')[-1], name : 'Control No MK801', id_in_path : 'controlNoMK801'},

   	              cond5 : {color :  sns.color_palette('colorblind')[-4], name : 'Delay No MK801', id_in_path : 'delayNoMK801', us_latency : [9]*46},
}





#! Exception
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 94
		min_number_us_trials = 78

		# To estimate maximum duration of the experiment.
		expected_number_cs = 94
		expected_number_us = 78
		time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		catch_trials_training = [25, 39, 53, 59]

		catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
		# list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



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
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Post-conditioning 1',
			'Post-conditioning 2',
			'Post-conditioning 3']

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5']



		trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

		trials_us_blocks_phases = [np.arange(15,64)]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
			'Post']

		names_us_blocks_phases = [
			'Train']




	case '2-P single plane':

		#* Path where the raw data is.

		path_home = Path(r'E:\2024 10_Delay 2-P single plane\Behavior')
		path_save = Path(r'E:\2024 10_Delay 2-P single plane\Behavior\Results')


		cond0 = 'control'
		cond1 = 'delay'
		# cond2 = 'trace'

		# min_trace_interval_stable_numb_trials = 10
		# max_trace_interval_stable_numb_trials = 10

		# us_latency_trace_min = 0.5 # s
		# us_latency_trace_max = 3  # s

		#! Careful with the number of catch trials!
		# number_us_trials_increasing_trace = 26

		# us_latency_conditioning_list = cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)/1000

		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
				  cond1 : {color :  [216, 31, 98], name : 'Delay CC', id_in_path : 'delay', us_latency : [9]*57},
				#   cond2 : {color : [226, 166, 14], name : 'Trace CC', id_in_path : 'trace',
	    #    [128, 139, 164]

	    #    us_latency :
		#    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}
		}

#! Exception
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 84
		min_number_us_trials = 69

		# To estimate maximum duration of the experiment.
		expected_number_cs = 84
		expected_number_us = 69
		time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		catch_trials_training = [12, 26, 34]

		# catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
		# list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



		# To plot pooled trials.

		# 9*10 trials
		trials_cs_blocks_10 = [[*range(5,15)],
						 		[*range(15,25)],
								[*range(25,35)],
								[*range(35,45)],
								[*range(45,55)],
								[*range(55,65)],
								[*range(65,75)],
								[*range(75,85)]
								]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [[*range(18,28)],
								[*range(28,37)],
								[*range(37,46)],
								[*range(46,55)],
								# [*range(55,64)]
								]
		
		names_cs_blocks_10 = [
			'Pre-train 1',
			# 'Pre-train 2',
			# 'Pre-train 3',
			'Train 1',
			'Train 2',
			'Train 3',
			'Train 4',
			# 'Train 5',
			'Test 1',
			'Test 2',
			'Test 3'
			]

		names_us_blocks_10 = [
			'Train 1',
			'Train 2',
			'Train 3',
			'Train 4',
			# 'Train 5'
		]



		trials_cs_blocks_phases = [[*range(5,15)], [*range(15,55)], [*range(55,85)]]

		trials_us_blocks_phases = [[*range(18,55)]]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
			'Test'
		]

		names_us_blocks_phases = [
			'Train'
		]





	case '2-P multiple planes top':

		#* Path where the raw data is.

		path_home = Path(r'E:\2024 03_Delay 2-P 15 planes top part\Behavior')
		path_save = Path(r'E:\2024 03_Delay 2-P 15 planes top part\Behavior\Results')

		cond0 = 'control'
		cond1 = 'delay'
		# cond2 = 'trace'

		# min_trace_interval_stable_numb_trials = 10
		# max_trace_interval_stable_numb_trials = 10

		# us_latency_trace_min = 0.5 # s
		# us_latency_trace_max = 3  # s

		#! Careful with the number of catch trials!
		# number_us_trials_increasing_trace = 26

		# us_latency_conditioning_list = cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)/1000

		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
				  cond1 : {color :  [216, 31, 98], name : 'Delay CC', id_in_path : 'delay', us_latency : [9]*41},
				#   cond2 : {color : [226, 166, 14], name : 'Trace CC', id_in_path : 'trace',
	    #    [128, 139, 164]

	    #    us_latency :
		#    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}
		}

#! Exception
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 70
		min_number_us_trials = 71

		# To estimate maximum duration of the experiment.
		expected_number_cs = 70
		expected_number_us = 71
		time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		# catch_trials_training = [25, 39, 53, 59]

		# catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
		# list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



		# To plot pooled trials.

		# 9*10 trials
		trials_cs_blocks_10 = [[*range(5,15)],
						 		[*range(15,25)],
								[*range(25,35)],
								[*range(35,45)],
								[*range(45,55)],
								[*range(55,65)],
								[*range(65,75)]]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [[*range(22,32)],
								[*range(32,42)],
								[*range(42,52)],
								[*range(52,62)]]

		names_cs_blocks_10 = [
			'Pre-conditioning 1',
			'Pre-conditioning 2',
			'Pre-conditioning 3',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			# 'Conditioning 5',
			# 'Post-conditioning 1',
			# 'Post-conditioning 2',
			# 'Post-conditioning 3'
			]

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			# 'Conditioning 5'
		]



		trials_cs_blocks_phases = [[*range(5,35)], [*range(35,75)]]

		trials_us_blocks_phases = [[*range(22,62)]]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
			# 'Post'
		]

		names_us_blocks_phases = [
			'Train'
		]




	case '2-P multiple planes bottom':

		#* Path where the raw data is.

		path_home = Path(r'E:\2024 10_Delay 2-P multiple planes bottom\Behavior')
		path_save = Path(r'E:\2024 10_Delay 2-P multiple planes bottom\Behavior\Results')

		cond0 = 'control'
		cond1 = 'delay'
		# cond2 = 'trace'

		# min_trace_interval_stable_numb_trials = 10
		# max_trace_interval_stable_numb_trials = 10

		# us_latency_trace_min = 0.5 # s
		# us_latency_trace_max = 3  # s

		#! Careful with the number of catch trials!
		# number_us_trials_increasing_trace = 26

		# us_latency_conditioning_list = cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)/1000

		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
				  cond1 : {color :  [216, 31, 98], name : 'Delay CC', id_in_path : 'delay', us_latency : [9]*41},
				#   cond2 : {color : [226, 166, 14], name : 'Trace CC', id_in_path : 'trace',
	    #    [128, 139, 164]

	    #    us_latency :
		#    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}
		}

#! Exception
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 70
		min_number_us_trials = 71

		# To estimate maximum duration of the experiment.
		expected_number_cs = 70
		expected_number_us = 71
		time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		# catch_trials_training = [25, 39, 53, 59]

		# catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
		# list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



		# To plot pooled trials.

		# 9*10 trials
		trials_cs_blocks_10 = [[*range(5,15)],
						 		[*range(15,25)],
								[*range(25,35)],
								[*range(35,45)],
								[*range(45,55)],
								[*range(55,65)],
								[*range(65,75)]]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [[*range(22,32)],
								[*range(32,42)],
								[*range(42,52)],
								[*range(52,62)]]

		names_cs_blocks_10 = [
			'Pre-conditioning 1',
			'Pre-conditioning 2',
			'Pre-conditioning 3',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			# 'Conditioning 5',
			# 'Post-conditioning 1',
			# 'Post-conditioning 2',
			# 'Post-conditioning 3'
			]

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			# 'Conditioning 5'
		]



		trials_cs_blocks_phases = [[*range(5,35)], [*range(35,75)]]

		trials_us_blocks_phases = [[*range(22,62)]]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
			# 'Post'
		]

		names_us_blocks_phases = [
			'Train'
		]



	case '2-P multiple planes zoom in':

		#* Path where the raw data is.

		path_home = Path(r'E:\2024 09_Delay 2-P 4 planes JC neurons\Behavior')
		path_save = Path(r'E:\2024 09_Delay 2-P 4 planes JC neurons\Behavior\Results')

		cond0 = 'control'
		cond1 = 'delay'
		# cond2 = 'trace'

		# min_trace_interval_stable_numb_trials = 10
		# max_trace_interval_stable_numb_trials = 10

		# us_latency_trace_min = 0.5 # s
		# us_latency_trace_max = 3  # s

		#! Careful with the number of catch trials!
		# number_us_trials_increasing_trace = 26

		# us_latency_conditioning_list = cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)/1000

		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
				  cond1 : {color :  [216, 31, 98], name : 'Delay CC', id_in_path : 'delay', us_latency : [9]*57},
				#   cond2 : {color : [226, 166, 14], name : 'Trace CC', id_in_path : 'trace',
	    #    [128, 139, 164]

	    #    us_latency :
		#    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}
		}

#! Exception
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 84
		min_number_us_trials = 69

		# To estimate maximum duration of the experiment.
		expected_number_cs = 84
		expected_number_us = 69
		time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		catch_trials_training = [12, 26, 34]

		# catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
		# list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



		# To plot pooled trials.

		# 9*10 trials
		trials_cs_blocks_10 = [[*range(5,15)],
						 		[*range(15,25)],
								[*range(25,35)],
								[*range(35,45)],
								[*range(45,55)],
								[*range(55,65)],
								[*range(65,75)],
								[*range(75,85)]
								]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [[*range(18,28)],
								[*range(28,37)],
								[*range(37,46)],
								[*range(46,55)],
								# [*range(55,64)]
								]
		
		names_cs_blocks_10 = [
			'Pre-train 1',
			# 'Pre-train 2',
			# 'Pre-train 3',
			'Train 1',
			'Train 2',
			'Train 3',
			'Train 4',
			# 'Train 5',
			'Test 1',
			'Test 2',
			'Test 3'
			]

		names_us_blocks_10 = [
			'Train 1',
			'Train 2',
			'Train 3',
			'Train 4',
			# 'Train 5'
		]



		trials_cs_blocks_phases = [[*range(5,15)], [*range(15,55)], [*range(55,85)]]

		trials_us_blocks_phases = [[*range(18,55)]]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
			'Test'
		]

		names_us_blocks_phases = [
			'Train'
		]



	case '2-P multiple planes ca8':

		#* Path where the raw data is.

		path_home = Path(r'E:\2024 10_Delay 2-P multiplane imaging ca8\Behavior')
		path_save = Path(r'E:\2024 10_Delay 2-P multiplane imaging ca8\Behavior\Results')

		cond0 = 'control'
		cond1 = 'delay'
		# cond2 = 'trace'

		# min_trace_interval_stable_numb_trials = 10
		# max_trace_interval_stable_numb_trials = 10

		# us_latency_trace_min = 0.5 # s
		# us_latency_trace_max = 3  # s

		#! Careful with the number of catch trials!
		# number_us_trials_increasing_trace = 26

		# us_latency_conditioning_list = cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)/1000

		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
				  cond1 : {color :  [216, 31, 98], name : 'Delay CC', id_in_path : 'delay', us_latency : [9]*41},
				#   cond2 : {color : [226, 166, 14], name : 'Trace CC', id_in_path : 'trace',
	    #    [128, 139, 164]

	    #    us_latency :
		#    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}
		}

#! Exception
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 70
		min_number_us_trials = 71

		# To estimate maximum duration of the experiment.
		expected_number_cs = 70
		expected_number_us = 71
		time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		# catch_trials_training = [25, 39, 53, 59]

		# catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
		# list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



		# To plot pooled trials.

		# 9*10 trials
		trials_cs_blocks_10 = [[*range(5,15)],
						 		[*range(15,25)],
								[*range(25,35)],
								[*range(35,45)],
								[*range(45,55)],
								[*range(55,65)],
								[*range(65,75)]]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [[*range(22,32)],
								[*range(32,42)],
								[*range(42,52)],
								[*range(52,62)]]

		names_cs_blocks_10 = [
			'Pre-conditioning 1',
			'Pre-conditioning 2',
			'Pre-conditioning 3',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			# 'Conditioning 5',
			# 'Post-conditioning 1',
			# 'Post-conditioning 2',
			# 'Post-conditioning 3'
			]

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			# 'Conditioning 5'
		]



		trials_cs_blocks_phases = [[*range(5,35)], [*range(35,75)]]

		trials_us_blocks_phases = [[*range(22,62)]]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
			# 'Post'
		]

		names_us_blocks_phases = [
			'Train'
		]





	case '2p':

	#! November 2022 basic delay and trace experiment

		#* Path where the raw data is.

		path_home = Path(r'D:\2024 02_Delay 2p\Behavior')
		path_save = Path(r'D:\2024 02_Delay 2p\Behavior\Results')

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
		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', id_in_path : 'control'},
				  cond1 : {color :  [216, 31, 98], name : 'Delay CC', id_in_path : 'delay', us_latency : [9]*46},
				  cond2 : {color : [226, 166, 14], name : 'Trace CC', id_in_path : 'trace',
	    #    [128, 139, 164]

	       us_latency :
		   cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}}

#! Exception
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 94
		min_number_us_trials = 78

		# To estimate maximum duration of the experiment.
		expected_number_cs = 94
		expected_number_us = 78
		time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		catch_trials_training = [25, 39, 53, 59]

		catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
		# list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



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
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Post-conditioning 1',
			'Post-conditioning 2',
			'Post-conditioning 3']

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5'
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

	case 'original':

	#! November 2022 basic delay and trace experiment

		#* Path where the raw data is.
		# path_home = Path(r'I:\Joaquim\data')
		# path_home = Path(r'D:\2024 06_Delay with Pkj cells ablated')
		# path_save = Path(r'E:\Results (paper)\2024 06_Delay with Pkj cells ablated')
		# path_home = Path(r'E:\2024 05_Delay c-Fos')
		# path_save = Path(r'E:\2024 05_Delay c-Fos\Results')
		path_home = Path(r'D:\2022 11_Basic delay and (increasing) trace CC paradigm\Raw data')
		path_save = Path(r'E:\Results (paper)\2022 11_Basic delay and (increasing) trace CC paradigm')

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
		condition_dict = {cond0 : {color : sns.color_palette('colorblind')[0], name : 'Control', id_in_path : 'control'},
				  cond1 : {color :  sns.color_palette('colorblind')[4], name : 'Delay CC', id_in_path : 'delay', us_latency : [9]*46},
				  cond2 : {color : sns.color_palette('colorblind')[1], name : 'Trace CC', id_in_path : 'trace',
	    #    [128, 139, 164]

	       us_latency :
		   cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace_max, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace_max] * max_trace_interval_stable_numb_trials)}}

#! Exception
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 94
		min_number_us_trials = 78

		# To estimate maximum duration of the experiment.
		expected_number_cs = 94
		expected_number_us = 78
		time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		catch_trials_training = [25, 39, 53, 59]

		catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]
		# list(np.arange(5,15)) + catch_trials_training + list(np.arange(65,95))



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
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Post-conditioning 1',
			'Post-conditioning 2',
			'Post-conditioning 3']

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5']



		trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

		trials_us_blocks_phases = [np.arange(18,64)]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
			'Post']

		names_us_blocks_phases = [
			'Train']



	case 'fixedVsIncreasingTrace':

	#! End February-March 2023 comparison between 3-s fixed trace and 3-s increasing trace

		#* Path where the raw data is.
		path_home = Path(r'D:\2023 02-03_Fixed vs increasing trace (3 s)\Raw data')
		path_save = Path(r'E:\Results (paper)\2023 02-03_Fixed vs increasing trace (3 s)')

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
		condition_dict = {cond0 : {color : sns.color_palette('colorblind')[0], name : 'Control', id_in_path : 'control'},
				  cond1 : {color :  sns.color_palette('colorblind')[1], name : 'Trace CC fixed', id_in_path : 'fixedTrace', us_latency : [9]*46},
				  cond2 : {color : sns.color_palette('colorblind')[2], name : 'Trace CC increasing', id_in_path : 'increasingTrace', us_latency : 
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

		catch_trials_training = [25, 39, 53, 59]

		catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]


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
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Post-conditioning 1',
			'Post-conditioning 2',
			'Post-conditioning 3']

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5'
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
		path_home = Path(r'D:\2023 03_3-s vs 10-s fixed trace\Raw data')
		path_save = Path(r'E:\Results (paper)\2023 03_3-s vs 10-s fixed trace')

		
		cond0 = 'control'
		cond1 = '3sFixedTrace'
		cond2 = '10sFixedTrace'


		#* Colors for data in the plots.

		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : sns.color_palette('colorblind')[0], name : 'Control', id_in_path : 'control'},
				  cond1 : {color : sns.color_palette('colorblind')[1], name : 'Trace 3 s fixed', id_in_path : '3sFixedTrace', us_latency : [13]*46},
				  cond2 : {color : sns.color_palette('colorblind')[8], name : 'Trace 10 s fixed', id_in_path : '10sFixedTrace', us_latency : [20]*46}
				  }

		cr_window = 13  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 94
		min_number_us_trials = 78

		# To estimate maximum duration of the experiment.
		expected_number_cs = 94
		expected_number_us = 78
		time_aft_last_trial = 1 # min

		catch_trials_training = [25, 39, 53, 59]

		catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]



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
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Post-conditioning 1',
			'Post-conditioning 2',
			'Post-conditioning 3']

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5'
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


	case '3sTraceFrom2DiffExp':

	#! End February-March 2023 comparison between 3-s fixed trace and 3-s increasing trace

		#* Path where the raw data is.
		path_home = Path(r'D:\\')
		path_save = Path(r'E:\Results (paper)\2023 02-03_Fixed 3-s trace and unpaired control (pool of 2 different experiments)')

		cond0 = 'control'
		cond1 = 'trace'

		us_latency_trace_max = 3  # s


		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : sns.color_palette('colorblind')[0], name : 'Control', id_in_path : 'control'},
							cond1 : {color :  sns.color_palette('colorblind')[1], name : 'Trace CC fixed', id_in_path : 'fixedTrace', us_latency : [9]*46}}


		cr_window = 13  # s


		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 94
		min_number_us_trials = 78

		# To estimate maximum duration of the experiment.
		expected_number_cs = 94
		expected_number_us = 78
		time_aft_last_trial = 1 # min

		catch_trials_training = [25, 39, 53, 59]

		catch_trials_plot = [12,13,14] + catch_trials_training + [65,66,67]


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
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Post-conditioning 1',
			'Post-conditioning 2',
			'Post-conditioning 3']

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5'
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
		path_home = Path(r'D:\2023 04_Fixed 3-s trace with MK 801 (100 μM)\Raw data')
		path_save = Path(r'E:\Results (paper)\2023 04_Fixed 3-s trace with MK 801 (100 μM)')


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
		condition_dict = {cond0 : {color : sns.color_palette('colorblind')[0], name : 'Control No MK801', id_in_path : 'controlNoMK801'},

				  cond1 : {color : sns.color_palette('colorblind')[1], name : 'Trace No MK801', id_in_path : 'traceNoMK801',
	       us_latency : [13]*46
		#    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
		   },

				  cond2 : {color : sns.color_palette('colorblind')[-1], name : 'Control MK801', id_in_path : 'controlMK801'},

   	              cond3 : {color : sns.color_palette('colorblind')[-2], name : 'Trace MK801', id_in_path : 'traceMK801',
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
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Post-conditioning 1',
			'Post-conditioning 2',
			'Post-conditioning 3']

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5'
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
		path_home = Path(r'D:\2023 05 & 08_Fixed 3-s trace with Puromycin (5 mg per L)\Raw data')
		path_save = Path(r'E:\Results (paper)\2023 05 & 08_Fixed 3-s trace with Puromycin (5 mg per L)')

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
		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control No Puromycin', id_in_path : 'controlNoPuromycin'},

				  cond1 : {color : [246, 190, 0], name : 'Trace No Puromycin', id_in_path : 'traceNoPuromycin',
	       us_latency : [13]*46
		#    cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
		   },

				  cond2 : {color :  [0, 0, 139], name : 'Control Puromycin', id_in_path : 'controlPuromycin'},

   	              cond3 : {color : [245, 130, 48], name : 'Trace Puromycin', id_in_path : 'tracePuromycin',
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
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Post-conditioning 1',
			'Post-conditioning 2',
			'Post-conditioning 3']

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5'
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
		# path_home = Path(r'I:\Joaquim\data')
		
		path_home = Path(r'C:\Users\joaqc\Desktop\PR')
		#(r'D:\2023 08_Fully reinforced fixed 3-s trace vs partially reinforced\3\Raw data')
		path_save = Path(r'C:\Users\joaqc\Desktop\PR\results')
		# (r'E:\Results (paper)\2023 08_Fully reinforced fixed 3-s trace vs partially reinforced\3')

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
		cr_window = 10  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 94
		min_number_us_trials = 78

		# To estimate maximum duration of the experiment.
		expected_number_cs = 94
		expected_number_us = 78
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
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Post-conditioning 1',
			'Post-conditioning 2',
			'Post-conditioning 3']

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5'
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


	case 'longTermSpacedVsMassed-1':

	#! August 2023 trace with massed or spaced schedule

		#* Path where the raw data is.
		# path_home = Path(r'I:\Joaquim\data')
		# path_home = Path(r'D:\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\5\Block 1\Raw data')
		# path_save = Path(r'E:\Results (paper)\2023 08-09_Long-term memory spaced vs massed fixed 3-s trace\5\Block 1')
		path_home = Path(r'D:\2023 10_Long-term memory spaced vs massed delay\Block 1\Raw data')
		path_save = Path(r'E:\Results (paper)\2023 10_Long-term memory spaced vs massed delay\Block 1')

		cond0 = 'controlMassed'
		cond1 = 'controlSpaced'
		cond2 = 'delayMassed'
		cond3 = 'delaySpaced'

		us_latency_delay = -1  # s


		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control Massed', id_in_path : 'controlMassed'},
				  cond1 : {color :  [245, 130, 48], name : 'Control Spaced', id_in_path : 'controlSpaced'},
				  cond2 : {color : [0, 0, 139], name : 'Delay Massed', id_in_path : 'delayMassed'},
				  cond3 : {color : [246, 190, 0], name : 'Delay Spaced', id_in_path : 'delaySpaced'}
				  }

		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 64
		min_number_us_trials = 59

		# To estimate maximum duration of the experiment.
		expected_number_cs = 64
		expected_number_us = 59
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


		trials_us_blocks_10 = [[*range(18,28)],
								[*range(28,37)],
								[*range(37,46)],
								[*range(46,55)],
								[*range(55,64)]]
		
								# [[*range(13,22)],
								# [*range(23,32)],
								# [*range(32,41)],
								# [*range(41,50)],
								# [*range(50,59)]]

		names_cs_blocks_10 = [
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
		]

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5'
		]



		trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65)]

		trials_us_blocks_phases = [np.arange(18,64)]
		# [np.arange(15,65)]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
		]

		names_us_blocks_phases = [
			'Train'
		]

	case 'longTermSpacedVsMassed-2':

	#! August 2023 delay with massed or spaced schedule

		#* Path where the raw data is.
		# path_home = Path(r'I:\Joaquim\data')
		# path_home = Path(r'D:\2023 08-09_Long-term memory spaced vs massed fixed 3-s delay\5\Block 2\Raw data')
		# path_save = Path(r'E:\Results (paper)\2023 08-09_Long-term memory spaced vs massed fixed 3-s delay\5\Block 2')
		path_home = Path(r'D:\2023 10_Long-term memory spaced vs massed delay\Block 2\Raw data')
		path_save = Path(r'E:\Results (paper)\2023 10_Long-term memory spaced vs massed delay\Block 2')


		cond0 = 'controlMassed'
		cond1 = 'controlSpaced'
		cond2 = 'delayMassed'
		cond3 = 'delaySpaced'

		us_latency_delay = -1  # s


		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control Massed', id_in_path : 'controlMassed'},
				  cond1 : {color :  [245, 130, 48], name : 'Control Spaced', id_in_path : 'controlSpaced'},
				  cond2 : {color : [0, 0, 139], name : 'delay Massed', id_in_path : 'delayMassed'},
				  cond3 : {color : [246, 190, 0], name : 'delay Spaced', id_in_path : 'delaySpaced'}
				  }

		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 60
		min_number_us_trials = 44

		# To estimate maximum duration of the experiment.
		expected_number_cs = 60
		expected_number_us = 44
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
								# [*range(61,71)],
								# [*range(71,81)],
								# [*range(81,91)],
								# [*range(91,101)],
								# [*range(101,111)]
								]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [
								[*range(17,27)],
						 		[*range(27,36)],
								[*range(36,45)],
								]

		names_cs_blocks_10 = [
						'Test 1',
						'Test 2',
						'Test 3',
						'Retrain 1',
						'Retrain 2',
						'Retrain 3',
						# 'Retrain 4',
						# 'Retrain 5',
						# 'Post-retrain 1',
						# 'Post-retrain 2',
						# 'Post-retrain 3'
						]

		names_us_blocks_10 = [
						'Retrain 1',
						'Retrain 2',
						'Retrain 3',
						# 'Retrain 4',
						# 'Retrain 5'
						]



		trials_cs_blocks_phases = [np.arange(1,31), np.arange(31,61),]
							#  np.arange(81,111)]

		trials_us_blocks_phases = [np.arange(16,45)]

		names_cs_blocks_phases = ['Test',
							'Retrain',]
							# 'Post-retrain']

		names_us_blocks_phases = ['Retrain']

	case 'longTermSpacedVsMassed-all':

	#! August 2023 delay with massed or spaced schedule

		#* Path where the raw data is.
		# path_home = Path(r'I:\Joaquim\data')
		path_home = Path(r'D:\2023 10_Long-term memory spaced vs massed delay\All blocks\Raw data')
		path_save = Path(r'E:\Results (paper)\2023 10_Long-term memory spaced vs massed delay\All blocks')

		cond0 = 'controlMassed'
		cond1 = 'controlSpaced'
		cond2 = 'delayMassed'
		cond3 = 'delaySpaced'

		us_latency_delay = -1  # s


		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control Massed', id_in_path : 'controlMassed'},
				  cond1 : {color :  [245, 130, 48], name : 'Control Spaced', id_in_path : 'controlSpaced'},
				  cond2 : {color : [0, 0, 139], name : 'delay Massed', id_in_path : 'delayMassed'},
				  cond3 : {color : [246, 190, 0], name : 'delay Spaced', id_in_path : 'delaySpaced'}
				  }

		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 64 + 60
		min_number_us_trials = 59 + 44

		# To estimate maximum duration of the experiment.
		expected_number_cs = 64 + 60
		expected_number_us = 59 + 44
		# time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		# catch_trials_training = [25, 39, 53, 59] + [25 + 64, 39 + 64, 53 + 64, 59 + 64]

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
								# [*range(125,135)],
								# [*range(135,145)],
								# [*range(145,155)],
								# [*range(155,165)],
								# [*range(165,175)]
								]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [[*range(18,28)],
								[*range(28,37)],
								[*range(37,46)],
								[*range(46,55)],
								[*range(55,64)],

								[*range(81,91)],
						 		[*range(91,100)],
								[*range(100,109)]
								]

		names_cs_blocks_10 = [
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Test 1',
			'Test 2',
			'Test 3',
			'Retrain 1',
			'Retrain 2',
			'Retrain 3',
			# 'Retrain 4',
			# 'Retrain 5',
			# 'Post-retrain 1',
			# 'Post-retrain 2',
			# 'Post-retrain 3'
		]

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Retrain 1',
			'Retrain 2',
			'Retrain 3',
			# 'Retrain 4',
			# 'Retrain 5'
		]



		trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65, 95), np.arange(95, 125)]

		trials_us_blocks_phases = [np.arange(18,64), np.arange(81, 109)]



		names_cs_blocks_phases = [
			'Pre',
			'Train',
			'Test',
			'Retrain',
			# 'Post-retrain'
			]

		names_us_blocks_phases = [
			'Train',
			'Retrain']




	case 'longTermDelaySpacedPuromycin-1':

	#! August 2023 trace with massed or spaced schedule

		#* Path where the raw data is.
		path_home = Path(r'D:\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\2\Block 1\Raw data')		
		path_save = Path(r'E:\Results (paper)\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\2\Block 1')
		# path_home = Path(r'F:\Pilot studies\2024 01_delay with Puromycin (10 mg per L)\Block 1\Raw data')
		# path_save = Path(r'F:\Pilot studies\2024 01_delay with Puromycin (10 mg per L)\Block 1')


		cond0 = 'controlSpacedNoPuromycin'
		cond1 = 'controlSpacedPuromycin'
		cond2 = 'delaySpacedNoPuromycin'
		cond3 = 'delaySpacedPuromycin'

		us_latency_delay = -1  # s


		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : 'darkblue', name : 'Control Spaced No Puromycin', id_in_path : 'controlSpacedNoPuromycin'},
				  cond1 : {color :  'red', name : 'Control Spaced Puromycin', id_in_path : 'controlSpacedPuromycin'},
				  cond2 : {color : 'blue', name : 'Delay Spaced No Puromycin', id_in_path : 'delaySpacedNoPuromycin'},
				  cond3 : {color : 'pink', name : 'Delay Spaced Puromycin', id_in_path : 'delaySpacedPuromycin'}
				  }
				  
		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 64
		min_number_us_trials = 59

		# To estimate maximum duration of the experiment.
		expected_number_cs = 64
		expected_number_us = 59
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


		trials_us_blocks_10 = [[*range(18,28)],
								[*range(28,37)],
								[*range(37,46)],
								[*range(46,55)],
								[*range(55,64)]]
		
								# [[*range(13,22)],
								# [*range(23,32)],
								# [*range(32,41)],
								# [*range(41,50)],
								# [*range(50,59)]]

		names_cs_blocks_10 = [
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
		]

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5'
		]



		trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65)]

		trials_us_blocks_phases = [np.arange(18,64)]
		# [np.arange(15,65)]

		names_cs_blocks_phases = [
			'Pre',
			'Train',
		]

		names_us_blocks_phases = [
			'Train'
		]

	case 'longTermDelaySpacedPuromycin-2':

	#! August 2023 delay with massed or spaced schedule

		#* Path where the raw data is.
		# path_home = Path(r'D:\2023 10_Long-term memory spaced vs massed delay\Block 2\Raw data')
		# path_save = Path(r'E:\Results (paper)\2023 10_Long-term memory spaced vs massed delay\Block 2')

		path_home = Path(r'D:\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\2\Block 2\Raw data')
		path_save = Path(r'E:\Results (paper)\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\2\Block 2')
		# path_home = Path(r'F:\Pilot studies\2024 01_delay with Puromycin (10 mg per L)\Block 2\Raw data')
		# path_save = Path(r'F:\Pilot studies\2024 01_delay with Puromycin (10 mg per L)\Block 2')

		cond0 = 'controlSpacedNoPuromycin'
		cond1 = 'controlSpacedPuromycin'
		cond2 = 'delaySpacedNoPuromycin'
		cond3 = 'delaySpacedPuromycin'

		us_latency_delay = -1  # s


		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : 'darkblue', name : 'Control Spaced No Puromycin', id_in_path : 'controlSpacedNoPuromycin'},
				  cond1 : {color :  'red', name : 'Control Spaced Puromycin', id_in_path : 'controlSpacedPuromycin'},
				  cond2 : {color : 'blue', name : 'Delay Spaced No Puromycin', id_in_path : 'delaySpacedNoPuromycin'},
				  cond3 : {color : 'pink', name : 'Delay Spaced Puromycin', id_in_path : 'delaySpacedPuromycin'}
				  }

		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 60
		min_number_us_trials = 44

		# To estimate maximum duration of the experiment.
		expected_number_cs = 60
		expected_number_us = 44
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
								# [*range(61,71)],
								# [*range(71,81)],
								# [*range(81,91)],
								# [*range(91,101)],
								# [*range(101,111)]
								]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [
								[*range(17,27)],
						 		[*range(27,36)],
								[*range(36,45)],
								]

		names_cs_blocks_10 = [
						'Test 1',
						'Test 2',
						'Test 3',
						'Retrain 1',
						'Retrain 2',
						'Retrain 3',
						# 'Retrain 4',
						# 'Retrain 5',
						# 'Post-retrain 1',
						# 'Post-retrain 2',
						# 'Post-retrain 3'
						]

		names_us_blocks_10 = [
						'Retrain 1',
						'Retrain 2',
						'Retrain 3',
						# 'Retrain 4',
						# 'Retrain 5'
						]



		trials_cs_blocks_phases = [np.arange(1,31), np.arange(31,61),]
							#  np.arange(81,111)]

		trials_us_blocks_phases = [np.arange(16,45)]

		names_cs_blocks_phases = ['Test',
							'Retrain',]
							# 'Post-retrain']

		names_us_blocks_phases = ['Retrain']

	case 'longTermDelaySpacedPuromycin-all':

	#! August 2023 delay with massed or spaced schedule

		#* Path where the raw data is.
		# path_home = Path(r'D:\2023 10_Long-term memory spaced vs massed delay\All blocks\Raw data')
		# path_save = Path(r'E:\Results (paper)\2023 10_Long-term memory spaced vs massed delay\All blocks')

		path_home = Path(r'D:\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\2\All blocks\Raw data')
		path_save = Path(r'E:\Results (paper)\2023 11_Long-term memory control, delay spaced training with Puromycin (5 mg per L)\2\All blocks')
		# path_home = Path(r'F:\Pilot studies\2024 01_delay with Puromycin (10 mg per L)\All blocks\Raw data')
		# path_save = Path(r'F:\Pilot studies\2024 01_delay with Puromycin (10 mg per L)\All blocks')

		cond0 = 'controlSpacedNoPuromycin'
		cond1 = 'controlSpacedPuromycin'
		cond2 = 'delaySpacedNoPuromycin'
		cond3 = 'delaySpacedPuromycin'

		us_latency_delay = -1  # s


		#TODO add more stuff to this dict.
		condition_dict = {cond0 : {color : sns.color_palette('colorblind')[0], name : 'Control Spaced No Puromycin', id_in_path : 'controlSpacedNoPuromycin'},
				  cond1 : {color :  'red', name : 'Control Spaced Puromycin', id_in_path : 'controlSpacedPuromycin'},
				  cond2 : {color : sns.color_palette('colorblind')[4], name : 'Delay Spaced No Puromycin', id_in_path : 'delaySpacedNoPuromycin'},
				  cond3 : {color : 'pink', name : 'Delay Spaced Puromycin', id_in_path : 'delaySpacedPuromycin'}
				  }


		cr_window = 9  # s

		# Minimum number of trials to not discard an experiment.
		min_number_cs_trials = 64 + 60
		min_number_us_trials = 59 + 44

		# To estimate maximum duration of the experiment.
		expected_number_cs = 64 + 60
		expected_number_us = 59 + 44
		# time_aft_last_trial = 1 # min


#TODO add this to the other experiments
		# catch_trials_training = [25, 39, 53, 59] + [25 + 64, 39 + 64, 53 + 64, 59 + 64]

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
								# [*range(125,135)],
								# [*range(135,145)],
								# [*range(145,155)],
								# [*range(155,165)],
								# [*range(165,175)]
								]

		# Starts at 18 because the first US is discarded in the analysis.
		trials_us_blocks_10 = [[*range(18,28)],
								[*range(28,37)],
								[*range(37,46)],
								[*range(46,55)],
								[*range(55,64)],

								[*range(81,91)],
						 		[*range(91,100)],
								[*range(100,109)]
								]

		names_cs_blocks_10 = [
			'Pre-conditioning',
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Test 1',
			'Test 2',
			'Test 3',
			'Retrain 1',
			'Retrain 2',
			'Retrain 3',
			# 'Retrain 4',
			# 'Retrain 5',
			# 'Post-retrain 1',
			# 'Post-retrain 2',
			# 'Post-retrain 3'
		]

		names_us_blocks_10 = [
			'Conditioning 1',
			'Conditioning 2',
			'Conditioning 3',
			'Conditioning 4',
			'Conditioning 5',
			'Retrain 1',
			'Retrain 2',
			'Retrain 3',
			# 'Retrain 4',
			# 'Retrain 5'
		]



		trials_cs_blocks_phases = [np.arange(5,15), np.arange(15,65), np.arange(65, 95), np.arange(95, 125)]

		trials_us_blocks_phases = [np.arange(18,64), np.arange(81, 109)]



		names_cs_blocks_phases = [
			'Pre',
			'Train',
			'Test',
			'Retrain',
			# 'Post-retrain'
			]

		names_us_blocks_phases = [
			'Train',
			'Retrain']



path_lost_frames, path_summary_exp, path_summary_beh, path_processed_data, path_cropped_exp_with_bout_detection, path_tail_angle_fig_cs, path_tail_angle_fig_us, path_raw_vigor_fig_cs, path_raw_vigor_fig_us, path_scaled_vigor_fig_cs, path_scaled_vigor_fig_us, path_suppression_ratio_fig_cs, path_suppression_ratio_fig_us, path_pooled_vigor_fig, path_analysis_protocols, path_orig_pkl, path_all_fish, path_pooled = f.create_folders(path_save)




exp_types = [*condition_dict.keys()]
exp_types_names = [condition_dict[k][id_in_path] for k in [*condition_dict.keys()]]
exp_types_order = [cond.lower() for cond in condition_dict.keys()]



color_palette = [condition_dict[x][color] for x in condition_dict.keys()]




#!!!!!!!!!!!!!!!! uncomment
# for c in condition_dict.keys():
# 	condition_dict[c][color][:3] = [i / 255 for i in condition_dict[c][color][:3]]



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