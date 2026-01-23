"""
Analysis Parameters Configuration
=================================

Defines tunable parameters for the imaging pipeline.
"""

from dataclasses import dataclass

# region Parameters

# Data Loading
nrows = None  # Read all rows

# Preprocessing / Timing
time_bef_cs_onset = 45000  # ms
time_aft_cs_onset = 35000  # ms

# Image Filtering
median_filter_kernel = 3

# Motion Correction / Cropping
image_crop = 5
motion_thr_within_trial = 5
image_crop_ = image_crop + motion_thr_within_trial
image_crop_template_matching = int(image_crop * 1.5)
low_high = 10  # Plane search range

# endregion

@dataclass(frozen=True)
class ParameterSet:
    nrows: int | None = nrows
    time_bef_cs_onset: int = time_bef_cs_onset
    time_aft_cs_onset: int = time_aft_cs_onset
    median_filter_kernel: int = median_filter_kernel
    image_crop: int = image_crop
    image_crop_: int = image_crop_
    image_crop_template_matching: int = image_crop_template_matching
    low_high: int = low_high

PARAMETERS = ParameterSet()

def get_parameters() -> ParameterSet:
    return PARAMETERS
