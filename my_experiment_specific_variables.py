from my_general_variables import *
from pathlib import Path


map_folder_to_experiment = {
	r'2000 01_Test' : 'Test'
	# r'2022 11_Basic delay and (increasing) trace CC paradigm' : 'Delay_increasingTrace',
							
}

experiments_info = {
	# 'Delay_increasingTrace':
	'Test' : 
					
										{'path_home' : Path(r'D:\2022 11_Basic delay and (increasing) trace CC paradigm\Raw data'),
										'path_save' : Path(r'E:\Results (paper)\2022 11_Basic delay and (increasing) trace CC paradigm'),
										'parts' : {'trials' :
					 									{'CS' : {'elements' : (1,95),
				#! is names of the elements necessary???
																'names_elements' : None},
														'US' : {'elements' : (1,79),
																'names_elements' : None}},
													# 'blocks' :
													# 	{'CS' : {'elements' : ((5,15),
													# 								(15,25),
													# 								(25,35),
													# 								(35,45),
													# 								(45,55),
													# 								(55,65),
													# 								(65,75),
													# 								(75,85),
													# 								(85,95)),
													# 			'names_elements' : ('Pre-train',
													# 								'Train 1',
													# 								'Train 2',
													# 								'Train 3',
													# 								'Train 4',
													# 								'Train 5',
													# 								'Test 1',
													# 								'Test 2',
													# 								'Test 3')},
													# 	# Starts at 18 because the first US is discarded in the analysis.,
													# 	'US' : {'elements' : ((18,28),
													# 								(28,37),
													# 								(37,46),
													# 								(46,55),
													# 								(55,64)),
													# 			'names_elements' : ('Train 1',
													# 								'Train 2',
													# 								'Train 3',
													# 								'Train 4',
													# 								'Train 5')}},
													'phases' : 
														{'CS' : {'elements' : ((5,15),
																				(15,65),
																				(65,95)),
																'names_elements' : ('Pre-train',
																				  	'Train',
																				  	'Test')},
														# Starts at 18 because the first US is discarded in the analysis.,
														'US' : {'elements' : ((15,65),),
																'names_elements' : ('Train',)}},
													'catch_trials' : 
														{'CS' : {'elements' : tuple((5,15)) + (25, 39, 53, 59) + tuple((65,95)),
					   											'names_elements' : ('Pre-train',
									   												'Train',
																				   	'Test')}},
										'conditions' : {'control' :
					  											{'name' : 'Control',
																'name_in_path' : 'control',
																'color' : 'blue',
																'cr_window' : (0.5, cs_duration),

																'us_latency_after_cs_onset' : None,
																'number_reinforced_trials' : None,
																'min_us_latency_trace' : None},
														
														'delay' :
					  											{'name' : 'Delay',
																'name_in_path' : 'delay',
																'color' : 'magenta',
																'cr_window' : (0.5, cs_duration),

																'us_latency_after_cs_onset' : 'stable',
																'number_reinforced_trials' : 46,
																'min_us_latency_trace' : '3'},
														
														'trace' :
					  											{'name' : 'Trace',
																'name_in_path' : 'trace',
																'color' : 'yellow',
																'cr_window' : (0.5, cs_duration),

																'us_latency_after_cs_onset' : 'stable',
																'number_reinforced_trials' : 46,
																'min_us_latency_trace' : 3}}}}}


														# 'delay' : Condition('Delay').set_color(magenta).set_name_in_path('control').get_us_latency(us_latency_after_cs_onset='stable', number_reinforced_trials=46, min_us_latency_trace=3),
														
														# 'trace' : Condition('Trace').set_color(yellow).set_name_in_path('control').get_us_latency(us_latency_after_cs_onset=9, number_reinforced_trials=46, min_us_latency_trace=0.5, max_us_latency_trace=3, min_trace_interval_stable_numb_trials=10, max_trace_interval_stable_numb_trials=10)}