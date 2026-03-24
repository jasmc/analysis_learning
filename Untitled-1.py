import pandas as pd


A = pd.read_csv(r"C:\Users\joaqc\Desktop\20221115_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfmp tail tracking.txt", sep=' ')
A = A.tail(210000)
A.to_csv(r"C:\Users\joaqc\Desktop\20000101_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfmp tail tracking.txt", sep=' ', index=False)


B = pd.read_csv(r"C:\Users\joaqc\Desktop\20221115_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfcam.txt", sep=' ')
B = B.tail(210000)
B.to_csv(r"C:\Users\joaqc\Desktop\20000101_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfcam.txt", sep=' ', index=False)


C = pd.read_csv(r"C:\Users\joaqc\Desktop\20221115_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfstim control.txt", sep=' ')
C.to_csv(r"C:\Users\joaqc\Desktop\20000101_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfstim control.txt", sep=' ', index=False)


					


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
					condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', name_in_path : 'control'},
							  cond1 : {color : [245, 130, 48], name : 'Trace 3 s fixed', name_in_path : '3sFixedTrace', us_latency_after_cs_onset : [13]*46},
							  cond2 : {color : [139,69,19], name : 'Trace 10 s fixed', name_in_path : '10sFixedTrace', us_latency_after_cs_onset : [20]*46}
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



					trials_cs_blocks_blocks = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

					trials_us_blocks_blocks = [np.arange(15,65)]

					names_blocks_cs = [
						'Pre',
						'Train',
						'Post'
					]

					names_blocks_us = [
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
					condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control No MK801', name_in_path : 'controlNoMK801'},

							  cond1 : {color : [245, 130, 48], name : 'Trace No MK801', name_in_path : 'traceNoMK801',
					   us_latency_after_cs_onset : [13]*46
					#	cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
					   },

							  cond2 : {color :  [0, 0, 139], name : 'Control MK801', name_in_path : 'controlMK801'},

			   				  cond3 : {color : [246, 190, 0], name : 'Trace MK801', name_in_path : 'traceMK801',
					   us_latency_after_cs_onset : [13]*46
					#	cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
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



					trials_cs_blocks_blocks = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

					trials_us_blocks_blocks = [np.arange(15,65)]

					names_blocks_cs = [
						'Pre',
						'Train',
						'Post'
					]

					names_blocks_us = [
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
					condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control No Puromycin', name_in_path : 'controlNoPuromycin'},

							  cond1 : {color : [245, 130, 48], name : 'Trace No Puromycin', name_in_path : 'traceNoPuromycin',
					   us_latency_after_cs_onset : [13]*46
					#	cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
					   },

							  cond2 : {color :  [0, 0, 139], name : 'Control Puromycin', name_in_path : 'controlPuromycin'},

			   				  cond3 : {color : [246, 190, 0], name : 'Trace Puromycin', name_in_path : 'tracePuromycin',
					   us_latency_after_cs_onset : [13]*46
					#	cs_duration + np.array([us_latency_trace_min] * (min_trace_interval_stable_numb_trials - 1) + list(np.linspace(us_latency_trace_min, us_latency_trace, num=number_us_trials_increasing_trace + 1, endpoint=False, dtype='float')) + [us_latency_trace] * max_trace_interval_stable_numb_trials)
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



					trials_cs_blocks_blocks = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

					trials_us_blocks_blocks = [np.arange(15,65)]

					names_blocks_cs = [
						'Pre',
						'Train',
						'Post'
					]

					names_blocks_us = [
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
					condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', name_in_path : 'control'},
							  cond1 : {color :  [216, 31, 98], name : 'Trace Partially Reinforced', name_in_path : 'tracePartiallyReinforced'},
							  cond2 : {color : [226, 166, 14], name : 'Trace Fully Reinforced', name_in_path : 'traceFullyReinforced'}}

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
					catch_trials_train = [25, 39, 53, 59]

					catch_trials_plot = list(np.arange(5,15)) + catch_trials_train + list(np.arange(65,95))

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



					trials_cs_blocks_blocks = [np.arange(5,15), np.arange(15,65), np.arange(65,95)]

					trials_us_blocks_blocks = [np.arange(18,65)]

					names_blocks_cs = [
						'Pre',
						'Train',
						'Post'
					]

					names_blocks_us = [
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
					condition_dict = {cond0 : {color : [58, 129, 195], name : 'Control', name_in_path : 'control'},
							  cond1 : {color :  [216, 31, 98], name : 'Trace Spaced', name_in_path : 'traceSpaced'},
							#   cond2 : {color : [226, 166, 14], name : 'Trace Massed', name_in_path : 'traceMassed'}
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
					# catch_trials_train = [25, 39, 53, 59]

					# catch_trials_plot = list(np.arange(5,15)) + catch_trials_train + list(np.arange(65,95))

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



					trials_cs_blocks_blocks = [np.arange(5,15), np.arange(15,65), np.arange(65,70)]

					trials_us_blocks_blocks = [np.arange(15,65)]

					names_blocks_cs = [
						'Pre',
						'Train',
						'Post'
					]

					names_blocks_us = [
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
					condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control Massed', name_in_path : 'controlMassed'},
							  cond1 : {color :  [245, 130, 48], name : 'Control Spaced', name_in_path : 'controlSpaced'},
							  cond2 : {color : [0, 0, 139], name : 'Trace Massed', name_in_path : 'traceMassed'},
							  cond3 : {color : [246, 190, 0], name : 'Trace Spaced', name_in_path : 'traceSpaced'}
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
					# catch_trials_train = [25, 39, 53, 59]

					# catch_trials_plot = list(np.arange(5,15)) + catch_trials_train + list(np.arange(65,95))

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



					trials_cs_blocks_blocks = [np.arange(5,15), np.arange(15,65)]

					trials_us_blocks_blocks = [np.arange(15,65)]

					names_blocks_cs = [
						'Pre',
						'Train',
					]

					names_blocks_us = [
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
					condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control Massed', name_in_path : 'controlMassed'},
							  cond1 : {color :  [245, 130, 48], name : 'Control Spaced', name_in_path : 'controlSpaced'},
							  cond2 : {color : [0, 0, 139], name : 'Trace Massed', name_in_path : 'traceMassed'},
							  cond3 : {color : [246, 190, 0], name : 'Trace Spaced', name_in_path : 'traceSpaced'}
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
					# catch_trials_train = [25, 39, 53, 59]

					# catch_trials_plot = list(np.arange(5,15)) + catch_trials_train + list(np.arange(65,95))

					# To plot pooled trials.

					# 9*10 trials
					trials_cs_blocks_10 = [[*range(64,74)]]

					# Starts at 18 because the first US is discarded in the analysis.
					trials_us_blocks_10 = []
					names_cs_blocks_10 = ['Test']

					names_us_blocks_10 = []



					trials_cs_blocks_blocks = [np.arange(64,74)]

					trials_us_blocks_blocks = []

					names_blocks_cs = ['Test']

					names_blocks_us = []

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
					condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control Massed', name_in_path : 'controlMassed'},
							  cond1 : {color :  [245, 130, 48], name : 'Control Spaced', name_in_path : 'controlSpaced'},
							  cond2 : {color : [0, 0, 139], name : 'Trace Massed', name_in_path : 'traceMassed'},
							  cond3 : {color : [246, 190, 0], name : 'Trace Spaced', name_in_path : 'traceSpaced'}
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
					catch_trials_train = [25, 39, 53, 59]

					# catch_trials_plot = list(np.arange(5,15)) + catch_trials_train + list(np.arange(65,95))

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



					trials_cs_blocks_blocks = [np.arange(1,31), np.arange(31,81), np.arange(81,111)]

					trials_us_blocks_blocks = [np.arange(14,60)]

					names_blocks_cs = ['Test',
										'Retrain',
										'Post-retrain']

					names_blocks_us = ['Retrain']



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
					condition_dict = {cond0 : {color : [173, 216, 230], name : 'Control Massed', name_in_path : 'controlMassed'},
							  cond1 : {color :  [245, 130, 48], name : 'Control Spaced', name_in_path : 'controlSpaced'},
							  cond2 : {color : [0, 0, 139], name : 'Trace Massed', name_in_path : 'traceMassed'},
							  cond3 : {color : [246, 190, 0], name : 'Trace Spaced', name_in_path : 'traceSpaced'}
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
					catch_trials_train = [25, 39, 53, 59] + [25 + 64, 39 + 64, 53 + 64, 59 + 64]

					# catch_trials_plot = list(np.arange(5,15)) + catch_trials_train + list(np.arange(65,95))

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



					trials_cs_blocks_blocks = [np.arange(5,15), np.arange(15,65), np.arange(65, 95), np.arange(95, 145), np.arange(145, 175)]

					trials_us_blocks_blocks = [np.arange(15,65), np.arange(79, 117)]



					names_blocks_cs = [
						'Pre',
						'Train',
						'Test',
						'Retrain',
						'Post-retrain']

					names_blocks_us = [
						'Train',
						'Retrain']

				# case _:


	