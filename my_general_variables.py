from os import name
from typing import Final
import numpy as np
import math
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)


# plt.style.use('plot_style.mplstyle')
# plt.style.use(['seaborn-v0_8-talk'])
# plt.rcParams["svg.fonttype"] = 'none'
# plt.rcParams['font.family'] = 'nunito'
# , 'plot_style'


#* Acronyms
'''
	cs. conditined stimulus or conditioned stumili
	us. unconditined stimulus or unconditioned stumili

	cr. conditioned response

	bef. before
	aft. after

	beg. beginning

	min. [at the beginning] minimum
	max. maximum

	thr. threshold

	bcf. boxcar filter


	ms. miliseconds
	s. seconds
	min. [at the end] minutes

	id. identification

	abs. absolute

	stim. stimulus or stimuli

	ifi. interframe interval

'''


#! create a dictionary for the dtypes of each col.




#* Parameters for the analysis


number_tail_points: Final = 16

# To define the whole extent of trials, i.e., regions in time before and after a stimulus.
time_bef_ms: Final = -45000 # ms
time_aft_ms: Final = 45000 # ms
time_bef: Final = -45 # ms
time_aft: Final = 45 # ms



time_aft_last_trial: Final = 1 # min



baseline_window = 15 # s

# To crop data, before pooling.
t_crop_data_bef_s = -baseline_window #s
t_crop_data_aft_s = baseline_window #s

# To check whether any frame was lost.
number_frames_discard_beg: Final = 60*700 # around 60 s at 700 FPS (number of frames)
buffer_size: Final = 700 # frames
max_interval_between_frames: Final = 0.005 # ms
lag_thr: Final = 50 # ms

# To check for tracking errors.
single_point_tracking_error_thr: Final = 2 * 180/np.pi # deg

# Parameteres for filtering raw data.
#! Need to add here the units.
#? If you change this, you will have to change the rest of the code.
# All below in number of frames at expected_framerate (700 FPS)
space_bcf_window: Final = 3 # number segments of the tail
time_bcf_window: Final = 10
time_max_window: Final = 20
time_min_window: Final = 400
#center_window = True
filtering_window: Final = 350 # frames

# Parameters for bout detection.
bout_detection_thr_1: Final = 4 # deg/ms   # Changed from 1 to 4 on 17 Jan 2023.
min_interbout_time: Final = 10 # frames
min_bout_duration: Final = 40 # frames
bout_detection_thr_2: Final = 1 # deg/ms

# Interpolate data to this framerate.
expected_framerate: Final = 700 # FPS
time_experiment_f = 'Time (frame) [{} FPS]'.format(expected_framerate)

# After interpolating, time_bef and time_aft can be used in number of frames.
time_bef_frame: Final = int(np.ceil(time_bef_ms * expected_framerate/1000)) # frames
time_aft_frame: Final = int(np.ceil(time_aft_ms * expected_framerate/1000)) # frames



#! CHANGE THIS TO A MORE CAUDAL POINT
# Point of the tail chosen as the last one to be considered in the analysis, using a tracking with 15 points on the tail.
chosen_tail_point: Final = number_tail_points - space_bcf_window // 2




# Width of the bins used to group bout_beg and bout_end data, in s.
binning_window: Final = 0.5 # s

binning_window_long: Final = 1 # s

time_bins_short = list(np.arange(-binning_window, t_crop_data_bef_s-binning_window, -binning_window))[::-1] + list(np.arange(0, t_crop_data_aft_s+binning_window, binning_window))

time_bins_long = list(np.arange(-binning_window_long, time_bef-binning_window_long, -binning_window_long))[::-1] + list(np.arange(0, time_aft+binning_window_long, binning_window_long))

bin_or_window_name = str(binning_window) + '-s window'


# Estimated duration of stimuli, for pooling data.
cs_duration: Final = 10 # s
us_duration: Final = 0.1 # s


us_struggle_window = 15 # s


#* Parameters for clipping scaled vigor
clip_low = -0.05
clip_high = 1.05
vigor_no_bout = -0.1



#* Parameters for plotting data

# Donwsampling step to plot the whole experimental data, using plot_cropped_experiment function.
downsampling_step: Final = 5



mean_bef_onset = 'Mean '+str(baseline_window)+' s before'


#* Variables containing strings

stim_type: Final = 'Type'
beg: Final = 'beg (ms)'
end: Final = 'end (ms)'

abs_time = 'AbsoluteTime'
ela_time = 'ElapsedTime'


cs: Final = 'CS'
us: Final = 'US'

cs_beg: Final = 'CS beg'
cs_end: Final = 'CS end'
us_beg: Final = 'US beg'
us_end: Final = 'US end'

vigor_bout_detection: Final = 'Vigour for bout detection (deg/ms)'
vigor_raw: Final = 'Vigour (deg/ms)'
vigor_digested: Final = 'Scaled vigour (AU)'
bout: Final = 'Bout'
bout_beg: Final = 'Bout beg'
bout_end: Final = 'Bout end'

bout_bef: Final = bout + ' before'
bout_aft: Final = bout + ' after'
bout_diff: Final = bout + ' division'

# bout_beg_end: Final = ' Bout beg and end'
vigor_mean: Final = vigor_raw + ' mean'
vigor_sem: Final = vigor_raw + ' sem'
# normalized_vigor = 'Normalized vigor'
bout_mean: Final = bout + ' mean'
# bout_sem = bout + ' sem'
# bout_beg_sum: Final = bout_beg + ' sum'
# bout_end_sum: Final = ('Bout end') + ' sum'
bout_beg_mean: Final = bout_beg + ' mean'
# bout_beg_sem = bout_beg + ' sem'
bout_end_mean: Final = bout_end + ' mean'
# bout_end_sem = bout_end + ' sem'
count: Final = 'Count'
trial: Final = 'Trial'



# experiment: Final = 'Exp.'


number_trial: Final = 'Trial number'
type_trial_csus: Final = 'Trial type'
# alignment: Final = 'Alignment'
#! Change this name!
time_trial: Final = 'Trial time (frame) [700 FPS]'
time_trial_s: Final = 'Trial time (s)'
block_name: Final = 'Block name'




color: Final = 'color'

blue: Final = [x/255 for x in [58, 129, 195]]
magenta: Final = [x/255 for x in [216, 31, 98]]
yellow: Final = [x/255 for x in [216, 31, 98]]



name: Final = 'name'


control: Final = 'control'
delay: Final = 'delay'
trace: Final = 'trace'
# delayNoOpt: Final = 'delaynoopt'

# cond1: Final = 'fixedtrace'
# cond2: Final = 'increasingtrace'


trials: Final = 'trials'
blocks: Final = 'blocks'
phases: Final = 'phases'


number_elements: Final = 'trials in each block'
names_trials_blocks_phases: Final = 'names of blocks'

name_in_path: Final = 'name in original path'

number_cols_or_rows: Final = 'number of cols or rows'

horizontal_fig: Final = 'horizontal'
fig_size: Final = 'figure size'

us_latency_after_cs_onset: Final = 'US latency'


folder_name: Final = 'folder name'


# , bout_beg, bout_end]
# metrics: Final = [bout_mean, bout_beg_mean, bout_end_mean]

# x_label: Final = 'x label'
x_label: Final = '\nt (s)'




fish: Final = 'Fish'

division: Final = 'division'

cols_position: Final = ['Decrease', 'Increase', 'No change']


sr = 'Suppression ratio'

segments_analysis: Final = [mean_bef_onset, '', sr]


function_add_data: Final = 'function to add data to plot'
y_label: Final = 'y label'
y_lim: Final = 'y limit'
y_ticks: Final = 'y ticks'


camera_value: Final = 'CameraValue'
photodiode_value: Final = 'PhotodiodeValue'
galvo_value: Final = 'GalvoValue'
arduino_value: Final = 'ArduinoValue'

frame_id: Final = 'Frame number'


x_name: Final = 'X '
y_name: Final = 'Y '
angle_name: Final = 'Angle (deg) '

tail_angle: Final = angle_name + str(chosen_tail_point - 1)  # It's -1 because of 0-indexing in Python.




cols_to_use_orig = ['FrameID'] + ['x' + str(i) for i in range(number_tail_points)] + ['y' + str(i) for i in range(number_tail_points)] + ['angle' + str(i) for i in range(number_tail_points - 1)]

# x_cols = []
# y_cols = []
# angle_cols = []

x_cols = [x_name + str(i) for i in range(number_tail_points)]
y_cols = [y_name + str(i) for i in range(number_tail_points)]
angle_cols = [angle_name + str(i) for i in range(number_tail_points - 1)]
data_cols = x_cols + y_cols + angle_cols

# for i in range(number_tail_points):

# 	cols_to_use_orig.append('x' + str(i))
# 	cols_to_use_orig.append('y' + str(i))

# 	x_cols.append(x_name + str(i))
# 	y_cols.append(y_name + str(i))
# 	angle_cols.append(angle_name + str(i))

# 	if i < 15:
# 		cols_to_use_orig.append('angle' + str(i))


# 	data_cols.append(x_name + str(i))
# 	data_cols.append(y_name + str(i))

# for i in range(number_tail_points-1):
# 	data_cols.append(angle_name + str(i))






cols_stim = [cs_beg, cs_end, us_beg, us_end]
			#  , type_trial_csus, number_trial, block_name]

cols_bout = [bout_beg, bout_end, bout]

# cols_stats = [vigor_mean, vigor_sem, bout_mean, bout_sem, bout_beg_mean, bout_beg_sem, bout_end_mean, bout_end_sem]
cols_stats = [vigor_mean, bout_mean, bout_beg_mean, bout_end_mean]


# cols_ordered = [[time_trial], cols_stim, cols[1:], [vigor_raw], cols_bout]
# cols_ordered = [i for j in cols_ordered for i in j]





cs_color: Final = [i/255 for i in [13, 129, 54]] # green
us_color: Final = [i/255 for i in [112, 46, 120]] # purple
# [130, 55, 140]





metrics_dict = {}
metrics_single_trials_dict = {}


# metrics_single_fish = [tail_angle]
# [vigor]
# [bout_beg, bout_end]
# [bout]



# if tail_angle in metrics_single_fish:
metrics_single_trials_dict[tail_angle] = {}
metrics_single_trials_dict[tail_angle][folder_name] = '2.0. single fish angle'
metrics_single_trials_dict[tail_angle][y_label] = 'Tail end angle (deg)'
metrics_single_trials_dict[tail_angle][y_lim] = (-45,45)
metrics_single_trials_dict[tail_angle][y_ticks] = np.arange(0,-45,45)

# if vigor in metrics_single_fish:
metrics_single_trials_dict[vigor_raw] = {}
metrics_single_trials_dict[vigor_raw][folder_name] = '2.1. single fish vigor'
metrics_single_trials_dict[vigor_raw][y_label] = 'Tail vigor (deg/ms)'
metrics_single_trials_dict[vigor_raw][y_lim] = (0,10)
metrics_single_trials_dict[vigor_raw][y_ticks] = []


# if bout in metrics_single_fish:
metrics_single_trials_dict[bout] = {}
metrics_single_trials_dict[bout][folder_name] = '2.2. single fish bouts'
metrics_single_trials_dict[bout][y_label] = 'Tail movement'
metrics_single_trials_dict[bout][y_lim] = (None,None)
metrics_single_trials_dict[bout][y_ticks] = []

# if bout_beg in metrics_single_fish:
metrics_single_trials_dict[bout_beg] = {}
metrics_single_trials_dict[bout_beg][folder_name] = '2.3. single fish beg bouts'
metrics_single_trials_dict[bout_beg][y_label] = 'Start of tail movement'
metrics_single_trials_dict[bout_beg][y_lim] = (None,None)
metrics_single_trials_dict[bout_beg][y_ticks] = []

# if bout_end in metrics_single_fish:
metrics_single_trials_dict[bout_end] = {}
metrics_single_trials_dict[bout_end][folder_name] = '2.4. single fish end bouts'
metrics_single_trials_dict[bout_end][y_label] = 'End of tail movement'
metrics_single_trials_dict[bout_end][y_lim] = (None,None)
metrics_single_trials_dict[bout_end][y_ticks] = []













	
metrics = [tail_angle, vigor_raw, bout, bout_beg, bout_end]
# [bout]
# bout_beg, bout_end]

#TODO REMOVE THE YLIMS

if vigor_raw in metrics:
	metrics_dict[vigor_raw] = {}
	metrics_dict[vigor_raw][folder_name] = '3.1. pooled fish vigor'
	metrics_dict[vigor_raw][y_label] = 'Tail movement vigor (deg/ms)'
	metrics_dict[vigor_raw][y_lim] = (-1,25)
	metrics_dict[vigor_raw][y_ticks] = (np.arange(0,21,20))

if bout in metrics:
	metrics_dict[bout] = {}
	metrics_dict[bout][folder_name] = '3.2. pooled fish bouts'
	metrics_dict[bout][y_label] = 'Proportion of trials with tail movement'
	metrics_dict[bout][y_lim] = (0,0.95)
	metrics_dict[bout][y_ticks] =  []

if bout_beg in metrics:
	metrics_dict[bout_beg] = {}
	metrics_dict[bout_beg][folder_name] = '3.3. pooled fish beg bouts'
	metrics_dict[bout_beg][y_label] = 'Proportion of trials start tail movement'
	metrics_dict[bout_beg][y_lim] = (-0.0003,0.002)
	metrics_dict[bout_beg][y_ticks] = []

if bout_end in metrics:
	metrics_dict[bout_end] = {}
	metrics_dict[bout_end][folder_name] = '3.4. pooled fish end bouts'
	metrics_dict[bout_end][y_label] = 'Proportion of trials end tail movement'
	metrics_dict[bout_end][y_lim] = (-0.0003,0.002)
	metrics_dict[bout_end][y_ticks] = []

