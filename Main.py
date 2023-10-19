
# TODO remove
from scipy import interpolate

#TODO import my_general_variables as g
from my_general_variables import *

import my_functions as f
from my_classes import *

# TODO remove
from importlib import reload
reload(f)


test_fish = r"C:\Users\joaqc\Desktop\2000 01_Test\Raw data\20000101_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfmp tail tracking.txt"
# test_fish = r"C:\Users\joaqc\Desktop\20221115_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfmp tail tracking.txt"

data_path = test_fish
camera_path = data_path.replace('mp tail tracking', 'cam')
protocol_path = data_path.replace('mp tail tracking', 'stim control')

fig_camera_name = r'C:\Users\joaqc\Desktop\2000 01_Test\Lost frames\test_fish' + '_camera.png'




#* Open the file with information about the time of each frame.
camera = f.read_camera(camera_path)


#* Estimate the true framerate.
predicted_framerate, reference_frame_id, reference_frame_time, Lost_frames = f.framerate_and_reference_frame(camera, name, fig_camera_name)

#TODO
# if Lost_frames:
# 	return None

#* Open tail tracking data.
data = f.read_tail_tracking_data(data_path)

#TODO
# tracking_errors

#* Add information about the time of each frame to data.
data = f.merge_camera_with_data(data, camera)




#* Interpolate data to the expected framerate
data = f.interpolate_data(data, expected_framerate, predicted_framerate)

	data_ = data.copy()

	#* Interpolate tail tracking data_ to the expected framerate.
	data_[time_experiment_f] -= data_[time_experiment_f].iat[0]

	data_[time_experiment_f] *= expected_framerate/predicted_framerate

	interp_function = interpolate.interp1d(data_[time_experiment_f], data_.drop(columns=time_experiment_f), kind='slinear', axis=0, assume_sorted=True, bounds_error=False, fill_value="extrapolate")
this is not fully correct
	data = pd.DataFrame(np.arange(data_[time_experiment_f].iat[0], data_[time_experiment_f].iat[-1]), columns=[time_experiment_f])

	data[data_.drop(columns=time_experiment_f).columns] = interp_function(data[time_experiment_f])








#* Open the stim log.
protocol = f.read_protocol(protocol_path)


data = f.highlight_stim_in_data(data, protocol)




data.dtypes






import numpy as np

np.arange(1.1,10.3)
ZERO THE TIME COLS



del camera, protocol





data.dtypes











		#* Filter tail tracking data.
		data = f.filter_data(data, space_bcf_window, time_bcf_window)



		#* Segment bouts
		data = f.vigor_for_bout_detection(data, chosen_tail_point, time_min_window, time_max_window)
		data = f.identify_bouts(data, bout_detection_thr_1, min_bout_duration, min_interbout_time, bout_detection_thr_2)


		f.plot_behavior_overview(data, stem_fish_path_orig, fig_behavior_name)










data.dtypes

data.columns


fix dtypes
	set some to categorical data




dtypes_ditc = { frame_id : 'int64'
	abs_time : 'int64',			   
				}









# f.read_camera((camera_path))
# a = repr("C:\Users\joaqc\Desktop\2000 01_Test\Raw data\20000101_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfcam.txt")


# repr("C:\Users\joaqc")
# from pathlib import Path
# Path(r"C:\Users\joaqc\Desktop\2000 01_Test\Raw data\20000101_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfcam.txt").exists()

fish = Fish.from_path(test_fish)
fish.preprocess(False)


# f.read_camera(camera_path)




# fish.experiment
# experiments = ['original'
# 	# 'longTermSpacedVsMassed-2',
# 			   # 'traceMK801'
# 				# 'fixedVsIncreasingTrace'
# 				# '10sTrace'
# 				# 'tracePuromycin'
# 				# 'tracePartialReinforcement'
# 				# 'traceSpaced'
# 				]




# # script (part of the analysis)


# for exp in experiments:

# 	experiment = Experiment(exp)


	

# experiment.cr_window
