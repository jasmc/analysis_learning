
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
fig_behavior_name = r'C:\Users\joaqc\Desktop\2000 01_Test\Behavior\test_fish' + '_behavior.png'

fish_name = r'20000101_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfmp tail tracking'


#* Open the file with information about the time of each frame.
camera = f.read_camera(camera_path)

#* Estimate the true framerate.
predicted_framerate, reference_frame_id, reference_frame_time, Lost_frames = f.framerate_and_reference_frame(camera, name, fig_camera_name)

#!!!!!!!!!!!! Very important to use the reference times!!!!!!! because of missing frames and abs_time not being 'exact'
#! need to correct the absolute time ... to the time at which things are acquired

#TODO
# if Lost_frames:
# 	return None

#* Open tail tracking data.
data = f.read_tail_tracking_data(data_path)

#TODO
# tracking_errors

#* Add information about the time of each frame to data.
data = f.merge_camera_with_data(data, camera)

del camera

#* Interpolate data to the expected framerate
data = f.interpolate_data(data, predicted_framerate)

#* Open the stim log.
protocol = f.read_protocol(protocol_path)


#!confirm
data = f.highlight_stim_in_data(data, protocol)

del protocol

# TODO clean up also coordinates of the tail
#* Filter tail tracking data.
#! Only filtering the angle data so far.
data = f.filter_data(data)

	# #TODO might want to test Adrien's way of filtering data (from Megabouts)

	# data[angle_cols] = data[angle_cols].cumsum(axis=1)

	# from sklearn.decomposition import PCA
	# from scipy.signal import savgol_filter

	# pca = PCA(n_components=4)
	# low_D = pca.fit_transform(data[angle_cols])
	# data[angle_cols] = pca.inverse_transform(low_D)

	# data[angle_cols] = savgol_filter(data[angle_cols], window_length=11, polyorder=2, deriv=0, delta=1.0, axis=0, mode='interp', cval=0.0)

f.plot_behavior_overview(data, fish_name, fig_behavior_name)

#* Identify tail bouts.
data = f.identify_bouts(data)





	#* Identify blocks of trials.
	data = f.identify_blocks_trials(data, blocks_dict)




	if f.lost_stim(len(data[data[cs_beg]!=0]), len(data[data[us_beg]!=0]), experiment.expected_number_cs, experiment.expected_number_us, experiment.protocol_info_path, stem_fish_path_orig, 2):

		print(data[data[cs_beg]!=0])
		print(data[data[us_beg]!=0])
		continue






data[tail_angle].plot()





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
