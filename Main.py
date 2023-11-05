
# TODO remove
from scipy import interpolate

#TODO import my_general_variables as g
from my_general_variables import *

import my_functions as f
from my_classes import *

# TODO remove
from importlib import reload
reload(f)


import my_experiment_specific_variables as exp_var



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
predicted_framerate, reference_frame_id, Lost_frames = f.framerate_and_reference_frame(camera, name, fig_camera_name)

#TODO
# if Lost_frames:
# 	return None

camera = camera.drop(columns=ela_time)

#* Discard frames that will not be used (in camera and hence further down).
# The calculated interframe interval before the reference frame is variable. Discard what happens up to then (also achieved by using how='inner' in merge_camera_with_data).
camera = camera[camera[frame_id] >= reference_frame_id]

#* Open tail tracking data.
data = f.read_tail_tracking_data(data_path)

#TODO
# tracking_errors

#* Add information about the time of each frame to data.
data = f.merge_camera_with_data(data, camera)

del camera

#* Fix abs_time so that the time of each frame becomes closer to the time at which the frames were acquired by the camera and not when they were caught by the computer.
# The delay between acquiring and catching the frame is unknown and therefore disregarded.
data[abs_time] = np.linspace(data[abs_time].iat[0], data[abs_time].iat[0] + len(data) * (1000 / predicted_framerate), len(data))

#* Interpolate data to the expected framerate.
data = f.interpolate_data(data, predicted_framerate)

#* Open the stim log.
protocol = f.read_protocol(protocol_path)


#* Identify the stimuli, trials and blocks of the experiment.
#TODO replace by exp_var.experiments_info[Experiment.name]
data = f.highlight_stim_trials_blocks_in_data(data, protocol, blocks_info = exp_var.experiments_info['Test']['parts']['blocks'][cs])

del protocol



#TODO
	# if f.lost_stim(len(data[data[cs_beg]!=0]), len(data[data[us_beg]!=0]), experiment.expected_number_cs, experiment.expected_number_us, experiment.protocol_info_path, stem_fish_path_orig, 2):

	# 	print(data[data[cs_beg]!=0])
	# 	print(data[data[us_beg]!=0])
	# 	continue




# TODO clean up also coordinates of the tail
#* Filter tail tracking data.
#! Only filtering the angle data so far.
# TODO might want to test Adrien's way of filtering data (from Megabouts)
data = f.filter_data(data)


f.plot_behavior_overview(data, fish_name, fig_behavior_name)

#* Identify tail bouts.
data = f.identify_bouts(data)







#TODO implement this??? Include in one of the functions above????
	#* time_bef_frame and time_aft_frame are for expected_framerate (700 FPS).
	data = f.extract_data_around_stimuli(data, protocol, time_bef_frame, time_aft_frame, time_bcf_window, time_max_window, time_min_window)


	data.iloc[:,0] = data.iloc[:,0] - data.iat[0,0]


	f.plot_cropped_experiment(data, expected_framerate, bout_detection_thr_1, bout_detection_thr_2, downsampling_step, stem_fish_path_orig, fig_cropped_exp_with_bout_detection_name)


	data.drop(columns=vigor_bout_detection, inplace=True)



#TODO not sure if should save this as well
	#* Calculate tail vigor.
	data[vigor_raw] = data.iloc[:,1:2+chosen_tail_point].diff().abs().sum(axis=1) * (expected_framerate / 1000) # deg/ms



#! check if this works with HDF

	strain, day, fish_number, age, experiment, condition, rig_name, protocol_number, rig_cs_color = self.fish_info()


	data.attrs = {'Strain' : strain,
				'Day' : day,
				'Fish no.' : fish_number,
				'Age (dpf)' : age,
				'Experiment' : experiment,
				'Condition' : condition,
				'Rig name' : rig_name,
				'Protocol number' : protocol_number,
				'CS color' : rig_cs_color}




#! HDF5   !!!!!!!!!
#!  organize hierarchically in experiment/condition and then all the corresponding fish
#! save all fish in the same HDF5
pd.set_option('io.hdf.default_format','table')

#! try gzip


#!!!!!!!!!!!!!!!!!!!!! UPDATE PREPOCESS METHOD WITH THIS. BUT FIRST CHECK WHAT IS ALREADY THERE



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
