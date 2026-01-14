"""
Analysis Parameters Configuration

This module defines all tunable parameters for the 2-photon imaging and behavioral analysis pipeline.
Parameters are organized by processing stage and can be adjusted without modifying core analysis scripts.

Parameter Categories:
1. Data Loading & Preprocessing
2. Motion Correction
3. Anatomical Stack Alignment
4. Activity Response Calculation
5. Voxel/Pixel Analysis
6. ROI Detection
7. Behavioral Analysis
8. Visualization

Usage:
    import my_parameters as p
    max_shift = p.xy_movement_allowed
    
Conventions:
- Spatial measurements in pixels unless otherwise noted
- Time measurements in milliseconds
- Thresholds are inclusive (>= or <=)
- Frame rates in Hz

Note: When changing parameters, document the rationale and date.
Critical parameters affecting motion correction and response detection
should be validated on test datasets before batch processing.
"""

# TODO: Define parameter values
# Example structure:
# # Motion Correction Parameters
# xy_movement_allowed = 0.1  # Maximum shift as fraction of image size
# total_motion_thr = 50.0    # Threshold for high-motion trial exclusion (pixels)
# median_filter_kernel = (1, 3, 3)  # Kernel size for spatial smoothing (time, y, x)

# # Response Analysis Parameters
# voxel_bin_size = 2  # Spatial binning factor for voxel analysis
# min_intensity_threshold = 100  # Minimum baseline fluorescence
# frame_rate = 30  # Imaging frame rate (Hz)

# # Anatomical Alignment Parameters
# low_high = 10  # Search range for plane identification (planes)
# image_crop_ = 20  # Pixels to crop from edges for template matching
# image_crop_template_matching = 40  # Crop for correlation calculation

# # Behavioral Parameters
# light_percentage_increase_thr = 0.15  # CS detection threshold

#* Parameters
# %%
#region Parameters
light_percentage_increase_thr = 5
light_percentage_decrease_PMT = 10
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


# # number_imaged_planes = 15
# # number_reps_plane_consective = 2
# # relevant_cs = nconcatenate([range(5,35), range(45,75)])

# number_imaged_planes = 4
# number_reps_plane_consective = 2
# relevant_cs = [range(5,35),
#               range(45,75)]
#             # [range(5,13),
#             #   range(15,23), range(25,33), range(35,43), range(45,53),
#             #   range(55,63), range(67,75), range(77,85)]

# index_list = [np.concatenate([[i+2*x*number_imaged_planes, i+2*x*number_imaged_planes+1] for x in range(len(relevant_cs))]) for i in range(0, number_reps_plane_consective * number_imaged_planes, 2)]

# relevant_cs = np.concatenate(relevant_cs)

motion_thr_from_trial_average = 5

correlation_map_sigma = 3
voxel_bin_size = 5


time_bef_cs_onset = 45000  # ms
time_aft_cs_onset = 35000  # ms


image_crop = 5  # number of pixels to crop around the images
image_crop_template_matching = int(image_crop*1.5)

motion_thr_within_trial = 5  # threshold for motion within trial (number of pixels)
motion_thr_across_trials = 20 # threshold for motion across trials (number of pixels)

image_crop_ = image_crop + motion_thr_within_trial

number_iterations_within_trial = 3


min_intensity_threshold = 1e-3

low_high = 10



border_size = 2*voxel_bin_size
correlation_thr = 0.3
median_thr = 5

softthresh=100

step = 2 # Process trials in pairs