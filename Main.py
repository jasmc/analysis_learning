from my_general_variables import *
from my_experiment_specific_variables import *
import my_functions as f


experiments = ['original'
	# 'longTermSpacedVsMassed-2',
			   # 'traceMK801'
				# 'fixedVsIncreasingTrace'
				# '10sTrace'
				# 'tracePuromycin'
				# 'tracePartialReinforcement'
				# 'traceSpaced'
				]




# script (part of the analysis)


for exp in experiments:

	experiment = Experiment(exp)


	

experiment.cr_window
