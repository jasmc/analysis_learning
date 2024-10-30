import numpy as np


#* Parameters
# %%
#region Parameters
light_percentage_increase_thr = 5
average_light_derivative_thr = 10
top_bottom_frame_slice = 50  # number pixels
front_back_frame_slice = 50  # number pixels


# step_size = 0.001  # mm
# number_repetitions_the_plane = 20
images_bin_size = 30
number_repetitions_the_plane_consecutively_stable = 40
# step_between_repetitions_of_the_plane = 1

median_filter_kernel = 3
gaussian_filter_sigma = 1

# kernel_size = 3
# ddepth = cv2.CV_16S

total_motion_thr = 0.5


#! debug
nrows = None
# 100000000

number_rows_read = None


galvo_value_height_threshold = 0.5
galvo_value_distance_threshold = 100
galvo_value_width_threshold = 20


xy_movement_allowed = 0.15  # fraction of the real image


number_imaged_planes = 15
number_reps_plane_consective = 2
relevant_cs = np.concatenate([range(5,35), range(45,75)])

motion_thr_from_trial_average = 5

correlation_map_sigma = 2
voxel_bin_size = 5

