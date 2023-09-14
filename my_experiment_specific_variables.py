from my_general_variables import *
from pathlib import Path


experiments_dictionary = {
		'Delay_increasingTrace' : {
		
		# 'name' : 'Delay_increasingTrace',

		#* Path where the raw data is.
		'path_home' : Path(r'D:\2022 11_Basic delay and (increasing) trace CC paradigm\Raw data'),
		'path_save' : Path(r'E:\Results (paper)\2022 11_Basic delay and (increasing) trace CC paradigm'),

		'expected_number_cs' : 94,
		'expected_number_us' : 78,

		'catch_trials_train' : [25, 39, 53, 59],
		# catch_trials_plot : list(np.arange(5,15)) + catch_trials_train + list(np.arange(65,95)),


		'trials_blocks_cs' : (range(5,15),
							range(15,25),
							range(25,35),
							range(35,45),
							range(45,55),
							range(55,65),
							range(65,75),
							range(75,85),
							range(85,95)),

		# Starts at 18 because the first US is discarded in the analysis.,
		'trials_blocks_us' : (range(18,28),
							range(28,37),
							range(37,46),
							range(46,55),
							range(55,64)),

		'names_blocks_cs' : ('Pre-train',
							'Train 1',
							'Train 2',
							'Train 3',
							'Train 4',
							'Train 5',
							'Post-train 1',
							'Post-train 2',
							'Post-train 3'),

		'names_blocks_us' : ('Train 1',
							'Train 2',
							'Train 3',
							'Train 4',
							'Train 5'),

		'trials_phases_cs' : (range(5,15),
					   			range(15,65),
								range(65,95)),

		'trials_phases_us' : (range(15,65)),

		'names_phases_cs' : ('Pre-train',
							  'Train',
							  'Test'),

		'names_phases_us' : ('Train',)
		}
}