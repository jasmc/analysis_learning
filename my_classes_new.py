"""
Core Data Structures
====================

Defines the core data classes used across the pipeline: Trial, Plane, and Data.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr

# region Classes

@dataclass
class Trial:
    """Represents a single trial within a plane."""
    trial_number: int
    protocol: pd.DataFrame
    behavior: pd.DataFrame
    images: xr.DataArray
    mask_good_frames: np.ndarray = None
    template_image: np.ndarray = None
    position_anatomical_stack: int = None
    shift_correction: np.ndarray = None
    pre_cs_mean: xr.DataArray = None
    cs_us_mean: xr.DataArray = None
    post_us_mean: xr.DataArray = None
    cs_us_vs_pre: np.ndarray = None
    post_us_vs_pre: np.ndarray = None
    anatomy: np.ndarray = None

    def get_stim_index(self, cs_us_col: str):
        """Get the indices of the stimulus onset and offset in the image time series."""
        cs_beg, cs_end = self.protocol.loc[self.protocol[cs_us_col] != 0, 'Time (ms)'].values[[0, -1]]

        a = self.images.coords['Time (ms)'].values < cs_beg
        cs_beg_index = np.where(np.diff(a))[0][0]

        b = self.images.coords['Time (ms)'].values > cs_end
        cs_end_index = np.where(np.diff(b))[0][0]

        return np.array([cs_beg_index, cs_end_index])


@dataclass
class Plane:
    """Represents an imaging plane containing multiple trials."""
    trials: list[Trial]
    template_image_position_anatomical_stack: int = None

    def get_reference_position(self):
        return round(np.median([trial.position_anatomical_stack for trial in self.trials]))

    def get_all_images(self):
        return np.concatenate([trial.images.values for trial in self.trials])


@dataclass
class Data:
    """Represents the entire dataset for a fish."""
    planes: list[Plane]
    anatomical_stack: xr.DataArray

    def get_planes(self, plane_numbers: list[int]):
        return [self.planes[i] for i in plane_numbers]

    def get_trials(self, plane_numbers: list[int] | str, trial_numbers: list[int]):
        if plane_numbers == 'all':
            plane_numbers = range(len(self.planes))
        return [self.planes[i].trials[j] for i in plane_numbers for j in trial_numbers]

# endregion
