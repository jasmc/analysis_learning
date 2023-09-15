from my_general_variables import *
from pathlib import Path


map_folder_to_experiment = {r'2022 11_Basic delay and (increasing) trace CC paradigm' : 'Delay_increasingTrace',
							
}

experiments_info = {'Delay_increasingTrace':
										{'path_home' : Path(r'D:\2022 11_Basic delay and (increasing) trace CC paradigm\Raw data'),
										'path_save' : Path(r'E:\Results (paper)\2022 11_Basic delay and (increasing) trace CC paradigm'),
 
										# 'expected_number_us' : 78,
#! may move this to somewhere else (like to 'parts'...)
										# 'catch_trials_train' : [25, 39, 53, 59],


										'parts' : {'trials' :
					 									{'cs' : {'elements' : range(1,95),
																'names_elements' : None},
														'us' : {'elements' : range(1,79),
																'names_elements' : None}},
													
								#!!!!! 
								WORK HERE
													catch_trials_plot : list(np.arange(5,15)) + catch_trials_train + list(np.arange(65,95)),
													'blocks' :
														{'cs' : {'elements' : (range(5,15),
																					range(15,25),
																					range(25,35),
																					range(35,45),
																					range(45,55),
																					range(55,65),
																					range(65,75),
																					range(75,85),
																					range(85,95)),
																'names_elements' : ('Pre-train',
																					'Train 1',
																					'Train 2',
																					'Train 3',
																					'Train 4',
																					'Train 5',
																					'Post-train 1',
																					'Post-train 2',
																					'Post-train 3')},
														# Starts at 18 because the first US is discarded in the analysis.,
														'us' : {'elements' : (range(18,28),
																					range(28,37),
																					range(37,46),
																					range(46,55),
																					range(55,64)),
																'names_elements' : ('Train 1',
																					'Train 2',
																					'Train 3',
																					'Train 4',
																					'Train 5')}},
													'phases' : 
														{'cs' : {'elements' : (range(5,15),
																		   			range(15,65),
																					range(65,95)),
																'names_elements' : ('Pre-train',
																				  	'Train',
																				  	'Test')},
														# Starts at 18 because the first US is discarded in the analysis.,
														'us' : {'elements' : (range(15,65)),
																'names_elements' : ('Train',)}}},

										'conditions' : {'control' :
					  											{'name' : 'Control',
																'name_in_path' : 'control',
																'color' : 'blue',
																'cr_window' : (0.5, cs_duration),

																'us_latency_after_cs_onset' : None,
																'number_reinforced_trials' : None,
																'min_us_latency_trace' : None},
												  		
														# 'delay' : Condition('Delay').set_color(magenta).set_name_in_path('control').get_us_latency(us_latency_after_cs_onset='stable', number_reinforced_trials=46, min_us_latency_trace=3),
														
														# 'trace' : Condition('Trace').set_color(yellow).set_name_in_path('control').get_us_latency(us_latency_after_cs_onset=9, number_reinforced_trials=46, min_us_latency_trace=0.5, max_us_latency_trace=3, min_trace_interval_stable_numb_trials=10, max_trace_interval_stable_numb_trials=10)}
														}
											}
											}
