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

from dataclasses import dataclass

# region Parameters
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
motion_thr_across_trials = 20  # threshold for motion across trials (number of pixels)

image_crop_ = image_crop + motion_thr_within_trial

number_iterations_within_trial = 3


min_intensity_threshold = 1e-3

low_high = 10



border_size = 2*voxel_bin_size
correlation_thr = 0.3
median_thr = 5

softthresh = 100

step = 2  # Process trials in pairs
# endregion


@dataclass(frozen=True)
class ParameterSet:
    light_percentage_increase_thr: float = light_percentage_increase_thr
    light_percentage_decrease_PMT: float = light_percentage_decrease_PMT
    average_light_derivative_thr: float = average_light_derivative_thr
    top_bottom_frame_slice: int = top_bottom_frame_slice
    front_back_frame_slice: int = front_back_frame_slice
    images_bin_size: int = images_bin_size
    number_repetitions_the_plane_consecutively_stable: int = number_repetitions_the_plane_consecutively_stable
    median_filter_kernel: int = median_filter_kernel
    gaussian_filter_sigma: int = gaussian_filter_sigma
    total_motion_thr: float = total_motion_thr
    nrows: int | None = nrows
    number_rows_read: int | None = number_rows_read
    galvo_value_height_threshold: float = galvo_value_height_threshold
    galvo_value_distance_threshold: float = galvo_value_distance_threshold
    galvo_value_width_threshold: int = galvo_value_width_threshold
    xy_movement_allowed: float = xy_movement_allowed
    motion_thr_from_trial_average: float = motion_thr_from_trial_average
    correlation_map_sigma: int = correlation_map_sigma
    voxel_bin_size: int = voxel_bin_size
    time_bef_cs_onset: int = time_bef_cs_onset
    time_aft_cs_onset: int = time_aft_cs_onset
    image_crop: int = image_crop
    image_crop_template_matching: int = image_crop_template_matching
    motion_thr_within_trial: int = motion_thr_within_trial
    motion_thr_across_trials: int = motion_thr_across_trials
    image_crop_: int = image_crop_
    number_iterations_within_trial: int = number_iterations_within_trial
    min_intensity_threshold: float = min_intensity_threshold
    low_high: int = low_high
    border_size: int = border_size
    correlation_thr: float = correlation_thr
    median_thr: int = median_thr
    softthresh: int = softthresh
    step: int = step


PARAMETERS = ParameterSet()


def get_parameters() -> ParameterSet:
    return PARAMETERS
