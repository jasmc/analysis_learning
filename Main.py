
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


store_raw = r"C:\Users\joaqc\Desktop\2000 01_Test\all_data_raw.h5"

# test_fish = r"C:\Users\joaqc\Desktop\2000 01_Test\Raw data\20000101_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfmp tail tracking.txt"

test_fish = r"C:\Users\joaqc\Desktop\2000 01_Test\Raw data\20221125_12_delay_black-8_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_9dpfcam.txt"



fish = Fish('Test', fish_raw_path=test_fish)


print(fish.__dict__)





all_data_raw = AllData(store_raw)


if not all_data_raw.fish_is_in_store(fish):
	
	fish.preprocess()
	
	all_data_raw.add_fish_raw_data(fish)

print('done')









# #* Retrieve metadata
# with pd.HDFStore(store_raw, complevel=4, complib="zlib") as store:
# 	print(store.get_storer(fish._key()).attrs['Fish info'])




# fish.__dict__.items()

# all_data_raw.__dict__

# #! save in fixed format 
# # , Raw_data=True
# [(key, value) for key, value in fish.__dict__.items() if key != 'data']
# fish.fish_info()



# fish.name
# # test_fish = r"C:\Users\joaqc\Desktop\20221115_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfm
# # 265404p tail tracking.txt"






# data_path = test_fish
# camera_path = data_path.replace('mp tail tracking', 'cam')
# protocol_path = data_path.replace('mp tail tracking', 'stim control')

# fig_camera_name = r'C:\Users\joaqc\Desktop\2000 01_Test\Lost frames\test_fish' + '_camera.png'
# fig_behavior_name = r'C:\Users\joaqc\Desktop\2000 01_Test\Behavior\test_fish' + '_behavior.png'

# fish_name = r'20000101_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpf'







# #* Open the file with information about the time of each frame.
# camera = f.read_camera(camera_path)

# #* Estimate the true framerate.
# predicted_framerate, reference_frame_id, Lost_frames = f.framerate_and_reference_frame(camera, name, fig_camera_name)

# #TODO
# # if Lost_frames:
# # 	return None

# camera = camera.drop(columns=ela_time)

# #* Discard frames that will not be used (in camera and hence further down).
# # The calculated interframe interval before the reference frame is variable. Discard what happens up to then (also achieved by using how='inner' in merge_camera_with_data).
# camera = camera[camera[frame_id] >= reference_frame_id]

# #* Open tail tracking data.
# data = f.read_tail_tracking_data(data_path)

# #TODO
# # tracking_errors

# #* Add information about the time of each frame to data.
# data = f.merge_camera_with_data(data, camera)

# del camera

# #* Fix abs_time so that the time of each frame becomes closer to the time at which the frames were acquired by the camera and not when they were caught by the computer.
# # The delay between acquiring and catching the frame is unknown and therefore disregarded.
# data[abs_time] = np.linspace(data[abs_time].iat[0], data[abs_time].iat[0] + len(data) * (1000 / predicted_framerate), len(data))

# #* Interpolate data to the expected framerate.
# data = f.interpolate_data(data, predicted_framerate)

# #* Open the stim log.
# protocol = f.read_protocol(protocol_path)


# #* Identify the stimuli, trials and blocks of the experiment.
# #TODO replace by exp_var.experiments_info[Experiment.name]
# data = f.identify_trials(data, protocol)
# # , blocks_info = exp_var.experiments_info['Test']['parts']['phases']
# del protocol




# #TODO
# 	# if f.lost_stim(len(data[data[cs_beg]!=0]), len(data[data[us_beg]!=0]), experiment.expected_number_cs, experiment.expected_number_us, experiment.protocol_info_path, stem_fish_path_orig, 2):

# 	# 	print(data[data[cs_beg]!=0])
# 	# 	print(data[data[us_beg]!=0])
# 	# 	continue




# # TODO clean up also coordinates of the tail
# #* Filter tail tracking data.
# #! Only filtering the angle data so far.
# # TODO might want to test Adrien's way of filtering data (from Megabouts)
# data = f.filter_data(data)



# store = pd.HDFStore(r"C:\Users\joaqc\Desktop\2000 01_Test\all_data.h5", complevel=4, complib="zlib")
# store.append(fish_name, data, data_columns=[cs, us], expectedrows=len(data))
# store.get_storer(fish_name).attrs['Fish info'] = self.fish_info()
# store.close()








# 	f.plot_behavior_overview(data, fish_name, fig_behavior_name)


# 	#* Identify tail bouts.
# 	data = f.identify_bouts(data)







# #TODO create a function to calculate time_trial_f and insert it in the dataset.
# 	# #Todo move to function this part
# 	# data[[cs_time_trial_f, us_time_trial_f]] = np.nan

# 	# extra_time_window = np.max([time_bcf_window, time_max_window, time_min_window])


# #TODO implement this??? Include in one of the functions above????
# 	#* time_bef_frame and time_aft_frame are for expected_framerate (700 FPS).
# 	data = f.extract_data_around_stimuli(data, protocol, time_bef_frame, time_aft_frame, time_bcf_window, time_max_window, time_min_window)


# 	# data.iloc[:,0] = data.iloc[:,0] - data.iat[0,0]


# 	f.plot_cropped_experiment(data, expected_framerate, bout_detection_thr_1, bout_detection_thr_2, downsampling_step, stem_fish_path_orig, fig_cropped_exp_with_bout_detection_name)


# 	# data.drop(columns=vigor_bout_detection, inplace=True)



# # #TODO not sure if should save this as well
# # 	#* Calculate tail vigor.
# # 	data[vigor_raw] = data.iloc[:,1:2+chosen_tail_point].diff().abs().sum(axis=1) * (expected_framerate / 1000) # deg/ms












# #!!!!!!!!!!

# Oftentimes when appending large amounts of data to a store, it is useful to turn off index creation for each append, then recreate at the end.
# df_1 = pd.DataFrame(np.random.randn(10, 2), columns=list("AB"))

# df_2 = pd.DataFrame(np.random.randn(10, 2), columns=list("AB"))

# st = pd.HDFStore("appends.h5", mode="w")

# st.append("df", df_1, data_columns=["B"], index=False)

# st.append("df", df_2, data_columns=["B"], index=False)

# st.get_storer("df").table
# Out[540]: 
# /df/table (Table(20,)) ''
#   description := {
#   "index": Int64Col(shape=(), dflt=0, pos=0),
#   "values_block_0": Float64Col(shape=(1,), dflt=0.0, pos=1),
#   "B": Float64Col(shape=(), dflt=0.0, pos=2)}
#   byteorder := 'little'
#   chunkshape := (2730,)
# Then create the index when finished appending.

# st.create_table_index("df", columns=["B"], optlevel=9, kind="full")

# st.get_storer("df").table
# Out[542]: 
# /df/table (Table(20,)) ''
#   description := {
#   "index": Int64Col(shape=(), dflt=0, pos=0),
#   "values_block_0": Float64Col(shape=(1,), dflt=0.0, pos=1),
#   "B": Float64Col(shape=(), dflt=0.0, pos=2)}
#   byteorder := 'little'
#   chunkshape := (2730,)
#   autoindex := True
#   colindexes := {
# 	"B": Index(9, fullshuffle, zlib(1)).is_csi=True}

# st.close()





# #!!!!!!!!!!!!!
# You can pass chunksize=<int> to append, specifying the write chunksize (default is 50000). This will significantly lower your memory usage on writing.


# #!!!!!!!!!!
# You can pass expectedrows=<int> to the first append, to set the TOTAL number of rows that PyTables will expect. This will optimize read/write performance.






# #!!!!!!!!!!!!!!!!!!!!! UPDATE PREPOCESS METHOD WITH THIS. BUT FIRST CHECK WHAT IS ALREADY THERE





# #! Use HDFStore.append() to concat all the fish a single experiment

# # f.read_camera((camera_path))
# # a = repr("C:\Users\joaqc\Desktop\2000 01_Test\Raw data\20000101_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfcam.txt")


# # repr("C:\Users\joaqc")
# # from pathlib import Path
# # Path(r"C:\Users\joaqc\Desktop\2000 01_Test\Raw data\20000101_01_delay_orange-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_6dpfcam.txt").exists()

# fish = Fish.from_path(test_fish)
# fish.preprocess(False)


# # f.read_camera(camera_path)




# # fish.experiment
# # experiments = ['original'
# # 	# 'longTermSpacedVsMassed-2',
# # 			   # 'traceMK801'
# # 				# 'fixedVsIncreasingTrace'
# # 				# '10sTrace'
# # 				# 'tracePuromycin'
# # 				# 'tracePartialReinforcement'
# # 				# 'traceSpaced'
# # 				]




# # # script (part of the analysis)


# # for exp in experiments:

# # 	experiment = Experiment(exp)


	

# # experiment.cr_window
