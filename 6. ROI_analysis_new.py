"""
6. ROI Analysis - Correlation-based ROI Detection and Analysis

This script performs correlation-based ROI detection and stimulus-response analysis
on 2-photon calcium imaging data exported to HDF5 format.

Workflow:
1. Load imaging data from HDF5 file (output of script 5.Save_data_as_HDF5.py)
2. Concatenate all trials for a single plane with timing information
3. Calculate pixel-wise correlation maps for ROI seeding
4. Detect ROIs using correlation-based region growing
5. Extract ROI traces and correlate with CS/US/learning regressors
6. Visualize and cluster ROI responses

Key Parameters:
- gausswidth: Gaussian smoothing width for correlation map (default: 2)
- deTrendScl: Detrending filter window size (default: 55)
- corrthresh: Correlation threshold for ROI growth (default: 0.4)
- stopthresh: Minimum correlation to stop ROI search (default: 0.1)
- maximumroisize: Maximum pixels per ROI (default: 200)

Output:
- ROI masks and traces per plane
- Correlation values with CS/US/learning regressors
- Clustering and visualization of responsive cells
"""

import pickle
# %% Imports
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster.hierarchy import (dendrogram, fcluster, linkage,
                                     optimal_leaf_ordering)
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from tqdm import tqdm

import plotting_style_new as plotting_style
from my_general_variables import *
from my_paths_new import fish_name as default_fish_name
from my_paths_new import path_home as default_path_home

# region Configuration
PANDAS_OPTIONS = {
	"mode.copy_on_write": True,
	"compute.use_numba": True,
	"compute.use_numexpr": True,
	"compute.use_bottleneck": True,
}
# endregion


def configure_environment() -> None:
	for option, value in PANDAS_OPTIONS.items():
		pd.set_option(option, value)
	plotting_style.set_plot_style(use_constrained_layout=False)


configure_environment()

# %% Data classes for ROI analysis results


# region Paths
PATH_HOME_OVERRIDE = None
FISH_NAME_OVERRIDE = None
# endregion

path_home = PATH_HOME_OVERRIDE or default_path_home
fish_name = FISH_NAME_OVERRIDE or default_fish_name



@dataclass
class PlaneTimings:
	"""Stimulus timing information for a single trial.
	
	All times are relative to the trial's first image time (times - times[0]).
	cs_start_time / cs_end_time : conditioned stimulus window (ms).
	us_start_time / us_end_time : unconditioned stimulus window (ms) or None if absent.
	"""
	cs_start_time: float = 0.0
	cs_end_time: float = 0.0
	us_start_time: Optional[float] = None
	us_end_time: Optional[float] = None


@dataclass
class PlaneData:
    """Data for a single plane and trial from HDF5."""
    imagedata: np.ndarray = field(default_factory=lambda: np.array([]))
    badframes: np.ndarray = field(default_factory=lambda: np.array([]))
    goodframes: np.ndarray = field(default_factory=lambda: np.array([]))
    pooledata: np.ndarray = field(default_factory=lambda: np.array([]))
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    behaviortimes: np.ndarray = field(default_factory=lambda: np.array([]))
    tailtrace: np.ndarray = field(default_factory=lambda: np.array([]))
    timings: PlaneTimings = field(default_factory=PlaneTimings)


@dataclass
class AllData:
    """Concatenated data for all trials in a single plane."""
    maskgoodframes: np.ndarray = field(default_factory=lambda: np.array([]))
    imagedata: np.ndarray = field(default_factory=lambda: np.array([]))
    badframes: np.ndarray = field(default_factory=lambda: np.array([]))
    goodframes: np.ndarray = field(default_factory=lambda: np.array([]))
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    compressedtimes: np.ndarray = field(default_factory=lambda: np.array([]))
    behaviortimes: np.ndarray = field(default_factory=lambda: np.array([]))
    tailtrace: np.ndarray = field(default_factory=lambda: np.array([]))
    csflaghires: np.ndarray = field(default_factory=lambda: np.array([]))
    usflaghires: np.ndarray = field(default_factory=lambda: np.array([]))
    csflaglores: np.ndarray = field(default_factory=lambda: np.array([]))
    usflaglores: np.ndarray = field(default_factory=lambda: np.array([]))
    trialstarttime: np.ndarray = field(default_factory=lambda: np.array([]))
    trialendtime: np.ndarray = field(default_factory=lambda: np.array([]))
    trialhasus: np.ndarray = field(default_factory=lambda: np.array([]))
    csRegressor: np.ndarray = field(default_factory=lambda: np.array([]))
    csMask: np.ndarray = field(default_factory=lambda: np.array([]))
    learnRegressor: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ROIParams:
    """Parameters for ROI detection."""
    deTrendScl: int = 55
    maximumroisize: int = 200
    corrthresh: float = 0.4
    stopthresh: float = 0.1
    maxperplane: int = 1000
    showfigs: bool = False


@dataclass
class ROIData:
    """Results from ROI detection."""
    allnums: np.ndarray = field(default_factory=lambda: np.array([]))
    corrmapval: np.ndarray = field(default_factory=lambda: np.array([]))
    avcorrval: np.ndarray = field(default_factory=lambda: np.array([]))
    mytraces: np.ndarray = field(default_factory=lambda: np.array([]))
    mytracesraw: np.ndarray = field(default_factory=lambda: np.array([]))
    allrois: np.ndarray = field(default_factory=lambda: np.array([]))


# %% Functions

def _resolve_trial_group(h5_file: h5py.File, plane_number: int, trial_number: int) -> tuple[h5py.Group, str]:
    """
    Resolve the HDF5 group for a given plane/trial, supporting both:
    - /planes/plane_{N}/trial_{M}
    - /planes/item_{N}/trials/item_{M}
    """
    planes_group = h5_file.get('planes')
    if planes_group is None:
        raise KeyError("'planes' group not found in HDF5 file")

    plane_key_candidates = [f'plane_{plane_number}', f'item_{plane_number}']
    trial_key_candidates = [f'trial_{trial_number}', f'item_{trial_number}']

    for plane_key in plane_key_candidates:
        if plane_key not in planes_group:
            continue
        plane_group = planes_group[plane_key]

        for trial_key in trial_key_candidates:
            if trial_key in plane_group:
                return plane_group[trial_key], f'/planes/{plane_key}/{trial_key}'

        if 'trials' in plane_group:
            trials_group = plane_group['trials']
            for trial_key in trial_key_candidates:
                if trial_key in trials_group:
                    return trials_group[trial_key], f'/planes/{plane_key}/trials/{trial_key}'

    raise KeyError(f"Trial {trial_number} not found for plane {plane_number} in HDF5 file")


def _read_group_dataset(h5_group: h5py.Group, name: str) -> Optional[np.ndarray]:
    """
    Read a dataset that may be stored directly or inside a group (xarray/DataFrame style).
    Returns None if the dataset is not found.
    """
    if name in h5_group:
        obj = h5_group[name]
        if isinstance(obj, h5py.Dataset):
            return obj[:]
        if isinstance(obj, h5py.Group):
            for key in ('data', 'values'):
                if key in obj:
                    return obj[key][:]

    fallback_candidates = [f'{name}_data_fallback', f'{name}_values_fallback']
    for fallback in fallback_candidates:
        if fallback in h5_group:
            return h5_group[fallback][:]

    return None


def _read_xarray_coords(xr_group: h5py.Group) -> dict[str, np.ndarray]:
    coords = {}
    coords_group = xr_group.get('coords')
    if coords_group is None:
        return coords
    for key in coords_group.keys():
        coords[key] = coords_group[key][:]
    return coords


def _find_coord(coords: dict[str, np.ndarray], candidates: list[str]) -> Optional[np.ndarray]:
    for name in candidates:
        if name in coords:
            return coords[name]
    return None


def get_plane_data(filename: Path, plane_number: int, trial_number: int) -> PlaneData:
	# Read one plane/trial from HDF5 and return structured PlaneData.
	# Expected HDF5 structure (from script 5.Save_data_as_HDF5.py):
	#   /planes/item_{N}/trials/item_{M}/protocol/values   [events, stim_cols]
	#   /planes/item_{N}/trials/item_{M}/behavior/values   [time, features]
	#   /planes/item_{N}/trials/item_{M}/images/data       [time, Y, X]
	#   /planes/item_{N}/trials/item_{M}/images/coords/Time (ms)
	#   /planes/item_{N}/trials/item_{M}/images/coords/mask good frames
	# Times are typically in ms; we convert to relative time by subtracting first timepoint.
    """
    Read data for a single plane and trial from HDF5 file.
    
    Parameters
    ----------
    filename : Path
        Path to the HDF5 file
    plane_number : int
        Plane index (0-based)
    trial_number : int
        Trial index (0-based)
    
    Returns
    -------
    PlaneData
        Dataclass containing imaging data, frames info, behavior, and timings
    """
    planedata = PlaneData()
    with h5py.File(filename, 'r') as f:
        trial_group, datastem = _resolve_trial_group(f, plane_number, trial_number)

        images_group = trial_group.get('images')
        imagedata_raw = _read_group_dataset(trial_group, 'images')
        if imagedata_raw is None:
            raise KeyError(f"'images' data not found in {datastem}")

        coords = {}
        if isinstance(images_group, h5py.Group):
            coords = _read_xarray_coords(images_group)

        # Script 5 saves as [time, Y, X], we need [Y, X, time] for processing
        if imagedata_raw.ndim == 3:
            # Transpose from [T, Y, X] to [Y, X, T]
            imagedata = np.transpose(imagedata_raw, (1, 2, 0))
        else:
            imagedata = imagedata_raw
        planedata.imagedata = imagedata.astype(np.float32)
        
        n_frames = imagedata.shape[-1]
        
        # Read good/bad frame masks
        mask_data = _find_coord(
            coords,
            ['mask good frames', 'mask_good_frames', 'mask_good_frames_bool']
        )
        if mask_data is None:
            mask_data = _read_group_dataset(trial_group, 'mask_good_frames')
        
        if mask_data is None:
            # Fallback: assume all frames are good
            planedata.goodframes = np.arange(n_frames)
            planedata.badframes = np.array([], dtype=int)
        else:
            # Handle different mask formats
            if mask_data.dtype.kind in ['S', 'U', 'O']:  # String types
                mask_str = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in mask_data])
                mask_str = np.char.upper(mask_str)
                planedata.goodframes = np.where(mask_str == 'TRUE')[0]
                planedata.badframes = np.where(mask_str == 'FALSE')[0]
            elif mask_data.dtype == bool:
                planedata.goodframes = np.where(mask_data)[0]
                planedata.badframes = np.where(~mask_data)[0]
            else:
                # Numeric: non-zero means good
                planedata.goodframes = np.where(mask_data != 0)[0]
                planedata.badframes = np.where(mask_data == 0)[0]

        # Create pooled data (spatial binning) for quick QC
        # Purpose: build representative traces without full image stack
        if imagedata.ndim >= 2:
            # Average across first spatial dimension (Y) to get [X, T] then pool
            flatdata = np.mean(imagedata.astype(np.float32), axis=0)
            n_spatial = flatdata.shape[0] if flatdata.ndim > 1 else 1
            n_pools = min(30, max(1, n_spatial // 20))
            n_time = flatdata.shape[-1] if flatdata.ndim > 1 else len(flatdata)
            pooledata = np.zeros((n_pools, n_time), dtype=np.float32)
            
            for n in range(n_pools):
                start_idx = n * 20
                end_idx = min(start_idx + 20, n_spatial)
                if flatdata.ndim > 1:
                    pooledata[n, :] = np.mean(flatdata[start_idx:end_idx, :], axis=0)
                else:
                    pooledata[n, :] = flatdata
            
            # Replace bad frames with median of good frames
            if len(planedata.goodframes) > 0 and pooledata.shape[1] > 0:
                good_idx = planedata.goodframes[planedata.goodframes < pooledata.shape[1]]
                if len(good_idx) > 0:
                    repval = np.median(pooledata[:, good_idx], axis=1)
                    for bad_idx in planedata.badframes:
                        if bad_idx < pooledata.shape[1]:
                            pooledata[:, bad_idx] = repval
            planedata.pooledata = pooledata
        
        # Read protocol/stimulus timing - expected columns:
        # [Time (ms), CS beg, CS end, US beg, US end]
        protocol = _read_group_dataset(trial_group, 'protocol')
        if protocol is not None:
            # Protocol may be [n_events, n_cols] or [n_cols, n_events]
            # We need time in first row/column
            if protocol.ndim == 2:
                if protocol.shape[0] > protocol.shape[1]:
                    # [n_events, n_cols] format - transpose to [n_cols, n_events]
                    protocol = protocol.T
                
                protocoltimes = protocol[0, :]
                
                # Extract CS timing (columns 1 and 2 typically mark CS start/end events)
                cs_start_idx = np.where(protocol[1, :] > 0)[0] if protocol.shape[0] > 1 else []
                cs_end_idx = np.where(protocol[2, :] > 0)[0] if protocol.shape[0] > 2 else []
                cs_start_time = protocoltimes[cs_start_idx[0]] if len(cs_start_idx) > 0 else 0
                cs_end_time = protocoltimes[cs_end_idx[0]] if len(cs_end_idx) > 0 else 0
                
                # Extract US timing (columns 3 and 4 typically mark US start/end events)
                us_start_idx = np.where(protocol[3, :] > 0)[0] if protocol.shape[0] > 3 else []
                us_end_idx = np.where(protocol[4, :] > 0)[0] if protocol.shape[0] > 4 else []
                us_start_time = protocoltimes[us_start_idx[0]] if len(us_start_idx) > 0 else None
                us_end_time = protocoltimes[us_end_idx[0]] if len(us_end_idx) > 0 else None
                
                # Times will be made relative after we read image times
                planedata.timings = PlaneTimings(
                    cs_start_time=cs_start_time,
                    cs_end_time=cs_end_time,
                    us_start_time=us_start_time,
                    us_end_time=us_end_time
                )
            else:
                # 1D protocol - no timing info available
                planedata.timings = PlaneTimings()
        else:
            planedata.timings = PlaneTimings()
        
        # Read or construct time array from image coords
        times = _find_coord(coords, ['Time (ms)', 'time_ms', 'time', 'Time'])
        if times is None:
            for key in ('times', 'time_ms', 'images_time'):
                if key in trial_group:
                    times = trial_group[key][:]
                    break
        
        if times is None:
            # Construct times assuming ~30 Hz imaging
            frame_interval = 33.0  # ms, approximately 30 Hz
            times = np.arange(n_frames) * frame_interval
        
        # Convert times to relative (starting from 0)
        time_offset = times[0]
        planedata.times = times - time_offset
        
        # Adjust stimulus timings to be relative to first frame
        planedata.timings = PlaneTimings(
            cs_start_time=planedata.timings.cs_start_time - time_offset,
            cs_end_time=planedata.timings.cs_end_time - time_offset,
            us_start_time=planedata.timings.us_start_time - time_offset if planedata.timings.us_start_time is not None else None,
            us_end_time=planedata.timings.us_end_time - time_offset if planedata.timings.us_end_time is not None else None
        )
        
        # Read behavior data - expected shape [time, features]
        # Features typically include: time, cumulative tail segments, etc.
        behaviordata = _read_group_dataset(trial_group, 'behavior')
        if behaviordata is not None:
            
            # Behavior may be [n_timepoints, n_features] or transposed
            if behaviordata.ndim == 2:
                # Ensure shape is [n_timepoints, n_features]
                if behaviordata.shape[0] < behaviordata.shape[1]:
                    behaviordata = behaviordata.T
                
                # First column is typically time
                behaviortimes = behaviordata[:, 0]
                
                # Remaining columns are behavior features (cumulative tail segments)
                if behaviordata.shape[1] > 1:
                    allbehavior = np.cumsum(behaviordata[:, 1:], axis=1)

                    # Extract tail trace - typically column 15 (index 14) or last column
                    tail_col = min(14, allbehavior.shape[1] - 1)
                    tailtrace = allbehavior[:, tail_col]
                else:
                    tailtrace = np.zeros(len(behaviortimes))
                
                planedata.behaviortimes = behaviortimes - time_offset
                planedata.tailtrace = tailtrace
            else:
                planedata.behaviortimes = np.array([])
                planedata.tailtrace = np.array([])
        else:
            planedata.behaviortimes = np.array([])
            planedata.tailtrace = np.array([])
    
    return planedata


def _create_stimulus_flag(times: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
	# Helper: make a binary vector marking stimulus frames in 'times'.
	# Inclusive at start, exclusive at end: [start_time, end_time)
	# Note: this helper expects numeric start_time and end_time. Callers should avoid passing None.
    """
    Create a binary stimulus flag array.
    
    Parameters
    ----------
    times : np.ndarray
        Time points
    start_time : float
        Stimulus start time
    end_time : float
        Stimulus end time
    
    Returns
    -------
    np.ndarray
        Binary flag array (1 during stimulus, 0 otherwise)
    """
    flag = np.zeros(len(times), dtype=int)
    mask = (times >= start_time) & (times < end_time)
    flag[mask] = 1
    return flag


def _create_us_flag_lores(times: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
    """
    Create a US flag for low-res (imaging) timing using MATLAB's indexing logic:
    flag(times(1:end-1) >= start) = 1
    flag(times(1:end-1) >= end) + 1 = 0
    """
    flag = np.zeros(len(times), dtype=int)
    if len(times) < 2:
        return flag

    start_idx = np.where(times[:-1] >= start_time)[0]
    if len(start_idx) > 0:
        flag[start_idx] = 1

    end_idx = np.where(times[:-1] >= end_time)[0]
    if len(end_idx) > 0:
        flag[end_idx + 1] = 0

    return flag


def concat_single_plane_file(
	whichfolder: Path,
	whichfile: str,
	whichplane: int,
	savedata: bool = True,
	overwrite: bool = False,
	numbertrials: int = 16
) -> AllData:
	# Concatenate trials for a plane into one AllData object.
	# Key points:
	# - savefolder/plane{plane:05d}.pkl is used to cache concatenated results
	# - compressedtimes tracks continuous time across concatenated trials
	# - offsets are applied to good/bad frame indices when concatenating frames
	# - behavior arrays are truncated per-trial to match imaging trial duration
    """
    Concatenate all trials for a single plane into one dataset with timing flags.
    
    Parameters
    ----------
    whichfolder : Path
        Folder containing the HDF5 file
    whichfile : str
        Name of the HDF5 file
    whichplane : int
        Plane index to process
    savedata : bool
        Whether to save the concatenated data
    overwrite : bool
        Whether to overwrite existing saved data
    numbertrials : int
        Number of trials to concatenate
    
    Returns
    -------
    AllData
        Dataclass containing concatenated data with timing flags
    """
    filename = whichfolder / whichfile
    savefolder = whichfolder / 'savefolder'
    savefolder.mkdir(exist_ok=True)
    
    savefilename = f'plane{whichplane:05d}.pkl'
    fullsavepath = savefolder / savefilename
    
    # Check for existing saved data
    if fullsavepath.exists() and not overwrite:
        print(f'Loading existing data from {fullsavepath}')
        with open(fullsavepath, 'rb') as f:
            return pickle.load(f)
    
    alldata = AllData()
    trialhasus = np.zeros(numbertrials, dtype=int)
    trialstarttime = []
    trialendtime = []
    
    # Process first trial (trial 0)
    whichtrial = 0
    thisdata = get_plane_data(filename, whichplane, whichtrial)
    
    imagedata = thisdata.imagedata
    badframes = thisdata.badframes.copy()
    goodframes = thisdata.goodframes.copy()
    times = thisdata.times.copy()
    
    compressedtimes = times - times[0]
    interframetime = np.mean(np.diff(times)) if len(times) > 1 else 33.0  # Default ~30Hz
    nextrialtime = times[-1] + interframetime
    
    behaviortimes = thisdata.behaviortimes.copy()
    tailtrace = thisdata.tailtrace.copy()
    
    trialstarttime.append(0)
    trialendtime.append(nextrialtime)
    
    # Truncate behavior to trial duration
    if len(behaviortimes) > 0:
        lastbehaviorframe = np.where(behaviortimes < nextrialtime)[0]
        if len(lastbehaviorframe) > 0:
            last_idx = lastbehaviorframe[-1]
            behaviortimes = behaviortimes[:last_idx]
            tailtrace = tailtrace[:last_idx]
    
    behinterframetime = np.mean(np.diff(behaviortimes)) if len(behaviortimes) > 1 else 1.43  # ~700Hz
    nexbehtrialtime = behaviortimes[-1] + behinterframetime if len(behaviortimes) > 0 else 0
    
    compressedbehaviortimes = behaviortimes - behaviortimes[0] if len(behaviortimes) > 0 else behaviortimes
    
    # Create CS/US flags using MATLAB-aligned logic
    cs_end = thisdata.timings.cs_end_time
    csflaghires = _create_stimulus_flag(behaviortimes, thisdata.timings.cs_start_time, cs_end)
    csflaglores = _create_stimulus_flag(times, thisdata.timings.cs_start_time, cs_end)
    
    if thisdata.timings.us_start_time is not None:
        trialhasus[whichtrial] = 1
        us_end = thisdata.timings.us_end_time if thisdata.timings.us_end_time is not None else nextrialtime
        usflaghires = _create_stimulus_flag(behaviortimes, thisdata.timings.us_start_time, us_end)
        usflaglores = _create_us_flag_lores(times, thisdata.timings.us_start_time, us_end)
    else:
        usflaghires = np.zeros(len(behaviortimes), dtype=int)
        usflaglores = np.zeros(len(times), dtype=int)
    
    # Process remaining trials
    for whichtrial in tqdm(range(1, numbertrials), desc=f'Concatenating plane {whichplane}'):
        thisdata = get_plane_data(filename, whichplane, whichtrial)
        
        # Concatenate bad/good frames with offset
        offset = imagedata.shape[-1]
        badframes = np.concatenate([badframes, thisdata.badframes + offset])
        goodframes = np.concatenate([goodframes, thisdata.goodframes + offset])
        
        # Concatenate image data
        imagedata = np.concatenate([imagedata, thisdata.imagedata], axis=-1)
        
        thistimes = thisdata.times
        thisbehaviortimes = thisdata.behaviortimes
        thistailtrace = thisdata.tailtrace
        nextrialtime_local = thistimes[-1] + interframetime
        
        # Truncate behavior
        if len(thisbehaviortimes) > 0:
            lastbehaviorframe = np.where(thisbehaviortimes < nextrialtime_local)[0]
            if len(lastbehaviorframe) > 0:
                last_idx = lastbehaviorframe[-1]
                thisbehaviortimes = thisbehaviortimes[:last_idx]
                thistailtrace = thistailtrace[:last_idx]
        
        times = np.concatenate([times, thistimes])
        
        nextrialtime = compressedtimes[-1] + interframetime
        trialstarttime.append(nextrialtime)
        compressedtimes = np.concatenate([compressedtimes, thistimes - thistimes[0] + nextrialtime])
        
        nextrialtime = compressedtimes[-1] + interframetime
        trialendtime.append(nextrialtime)
        
        behaviortimes = np.concatenate([behaviortimes, thisbehaviortimes])
        tailtrace = np.concatenate([tailtrace, thistailtrace])
        
        if len(thisbehaviortimes) > 0:
            compressedbehaviortimes = np.concatenate([
                compressedbehaviortimes, 
                thisbehaviortimes - thisbehaviortimes[0] + nexbehtrialtime
            ])
            nexbehtrialtime = compressedbehaviortimes[-1] + behinterframetime
        
        # CS/US flags for this trial using helper function
        cs_end = thisdata.timings.cs_end_time
        newcsflaghires = _create_stimulus_flag(thisbehaviortimes, thisdata.timings.cs_start_time, cs_end)
        newcsflaglores = _create_stimulus_flag(thistimes, thisdata.timings.cs_start_time, cs_end)
        
        if thisdata.timings.us_start_time is not None:
            trialhasus[whichtrial] = 1
            us_end = thisdata.timings.us_end_time if thisdata.timings.us_end_time is not None else nextrialtime_local
            newusflaghires = _create_stimulus_flag(thisbehaviortimes, thisdata.timings.us_start_time, us_end)
            newusflaglores = _create_us_flag_lores(thistimes, thisdata.timings.us_start_time, us_end)
        else:
            newusflaghires = np.zeros(len(thisbehaviortimes), dtype=int)
            newusflaglores = np.zeros(len(thistimes), dtype=int)
        
        csflaghires = np.concatenate([csflaghires, newcsflaghires])
        usflaghires = np.concatenate([usflaghires, newusflaghires])
        csflaglores = np.concatenate([csflaglores, newcsflaglores])
        usflaglores = np.concatenate([usflaglores, newusflaglores])
    
    # Create mask for good frames
    maskgoodframes = np.zeros(imagedata.shape[-1], dtype=int)
    maskgoodframes[goodframes.astype(int)] = 1
    
    # Populate AllData
    alldata.maskgoodframes = maskgoodframes
    alldata.imagedata = imagedata
    alldata.badframes = badframes
    alldata.goodframes = goodframes
    alldata.times = times
    alldata.compressedtimes = compressedtimes
    alldata.behaviortimes = behaviortimes
    alldata.tailtrace = tailtrace
    alldata.csflaghires = csflaghires
    alldata.usflaghires = usflaghires
    alldata.csflaglores = csflaglores
    alldata.usflaglores = usflaglores
    alldata.trialstarttime = np.array(trialstarttime)
    alldata.trialendtime = np.array(trialendtime)
    alldata.trialhasus = trialhasus
    
    # Construct CS, US, and learning regressors
    csRegressor = np.zeros(len(alldata.times), dtype=int)
    csMask = np.zeros(len(alldata.times), dtype=int)
    learnRegressor = np.zeros(len(alldata.times), dtype=int)
    
    for n in range(numbertrials):
        # Find CS start in this trial
        cs_mask = (alldata.compressedtimes > alldata.trialstarttime[n]) & (alldata.csflaglores > 0)
        startcs_idx = np.where(cs_mask)[0]
        
        if len(startcs_idx) > 0:
            startcs = startcs_idx[0]
            
            if alldata.trialhasus[n] == 1:
                # Find US onset
                us_mask = (alldata.compressedtimes > alldata.trialstarttime[n]) & (alldata.usflaglores > 0)
                endcs_idx = np.where(us_mask)[0]
                
                if len(endcs_idx) > 0:
                    endcs = endcs_idx[0]
                    csRegressor[startcs:endcs] = 1
                    
                    # Mark post-US period
                    endflag_idx = np.where(alldata.compressedtimes < alldata.trialendtime[n])[0]
                    if len(endflag_idx) > 0:
                        endflag = endflag_idx[-1]
                        csMask[endcs:endflag + 1] = 1
                    
                    learnRegressor[startcs:endcs] = 1
            else:
                # No US - use full CS period
                cs_end_mask = (alldata.compressedtimes < alldata.trialendtime[n]) & (alldata.csflaglores > 0)
                endcs_idx = np.where(cs_end_mask)[0]
                
                if len(endcs_idx) > 0:
                    endcs = endcs_idx[-1]
                    csRegressor[startcs:endcs + 1] = 1
    
    alldata.csRegressor = csRegressor
    alldata.csMask = csMask
    alldata.learnRegressor = learnRegressor
    
    if savedata:
        print(f'Saving to {fullsavepath}')
        with open(fullsavepath, 'wb') as f:
            pickle.dump(alldata, f)
    
    return alldata


def calculate_correlation_map_single_plane(
	alldata: AllData,
	gausswidth: float = 2.0,
	deTrendScl: int = 55,
	howmanyframes: int = 0,
	myborder: int = 10
) -> np.ndarray:
	# Compute a correlation-based "seed" map for ROI detection.
	# Steps:
	# 1) select good frames and optionally limit number of frames used
	# 2) spatially smooth each frame (gaussian) -> FplaneData
	# 3) subtract mean image and apply temporal detrending filter
	# 4) compute per-pixel norms and derive corrmap = meannorm^2 / (filtered_indnorm^2)
	# 5) subtract global median to center baseline and zero-out image borders
	# Notes:
	# - indnorm: per-pixel norm of detrended raw signal
	# - meannorm: per-pixel norm of detrended, spatially filtered signal
	# - the ratio highlights pixels where smoothed (neural) signal power is concentrated
	# - a tiny epsilon avoids division by zero
    """
    Calculate pixel-wise correlation map for ROI seeding.
    
    Parameters
    ----------
    alldata : AllData
        Concatenated plane data
    gausswidth : float
        Gaussian smoothing sigma
    deTrendScl : int
        Detrending filter window size
    howmanyframes : int
        Number of frames to use (0 = all)
    myborder : int
        Border pixels to set to zero
    
    Returns
    -------
    np.ndarray
        Correlation map (h x w)
    """
    total_frames = alldata.imagedata.shape[2]
    if howmanyframes == 0 or howmanyframes > total_frames:
        howmanyframes = total_frames

    goodframe_indices = np.where(alldata.maskgoodframes[:howmanyframes])[0]
    
    h, w = alldata.imagedata.shape[:2]
    
    # Get good frames only
    planeData = alldata.imagedata[:, :, goodframe_indices].astype(np.float32)
    zzc = planeData.shape[2]
    
    print(f'Processing {zzc} frames for correlation map...')
    
    # Apply Gaussian filter to each frame
	# Applying spatial smoothing reduces pixel-level noise and emphasizes coherent structures.
    FplaneData = np.zeros_like(planeData)
    for n in tqdm(range(zzc), desc='Gaussian filtering'):
        FplaneData[:, :, n] = gaussian_filter(planeData[:, :, n], sigma=gausswidth, mode='nearest')
    
    # Compute mean images
    planeDataM = np.mean(planeData, axis=2)
    FplaneDataM = np.mean(FplaneData, axis=2)
    
    # Create detrending filter
    myfilter = np.ones(deTrendScl) * (-1 / deTrendScl)
    center_idx = int(np.ceil(deTrendScl / 2)) - 1
    myfilter[center_idx] += 1
    
    print('Subtracting mean...')
    # Subtract mean
    for n in range(planeData.shape[2]):
        planeData[:, :, n] -= planeDataM
        FplaneData[:, :, n] -= FplaneDataM
    
    print('Detrending...')
    # Apply detrending filter along time axis
    for row in tqdm(range(planeData.shape[0]), desc='Detrending rows'):
        for col in range(planeData.shape[1]):
            planeData[row, col, :] = np.convolve(planeData[row, col, :], myfilter, mode='same')
            FplaneData[row, col, :] = np.convolve(FplaneData[row, col, :], myfilter, mode='same')
    
    # Calculate correlation map
    indnorm = np.linalg.norm(planeData, axis=2)
    meannorm = np.linalg.norm(FplaneData, axis=2)
    meannormsq = np.power(meannorm, 2)
    findnorm = gaussian_filter(indnorm, sigma=gausswidth, mode='nearest')
    findnormsq = np.power(findnorm, 2)
    
    corrmap = meannormsq / (findnormsq + 1e-10)  # Avoid division by zero
    
    # Subtract baseline correlation
    corrmap = corrmap - np.median(corrmap)
    
    # Zero out borders
    corrmap[:myborder, :] = 0
    corrmap[:, :myborder] = 0
    corrmap[:, -myborder:] = 0
    corrmap[-myborder:, :] = 0
    
    return corrmap


def find_rois_single_plane(
	alldata: AllData,
	thisplanecorr: np.ndarray,
	roiparams: ROIParams
) -> ROIData:
	# ROI region-growing based on correlation seeds.
	# Algorithm summary:
	# - pick maximum in correlation map as seed
	# - iteratively dilate the ROI border and evaluate new pixels' correlation
	# - accept best half of candidate pixels by correlation, then threshold by corrthresh
	# - apply morphological closing to include adjacent pixels
	# - stop when ROI exceeds maximum size or no more qualified pixels
	# - mark pixels as assigned in allrois to avoid reuse
	# Important notes:
	# - traces are detrended before correlation comparisons
	# - allrois uses integer labels: 0 means unassigned, positive integers are ROI ids
    """
    Find ROIs using correlation-based region growing.
    
    Parameters
    ----------
    alldata : AllData
        Concatenated plane data
    thisplanecorr : np.ndarray
        Correlation map for seeding ROIs
    roiparams : ROIParams
        ROI detection parameters
    
    Returns
    -------
    ROIData
        Detected ROIs with traces and statistics
    """
    deTrendScl = roiparams.deTrendScl
    maximumroisize = roiparams.maximumroisize
    corrthresh = roiparams.corrthresh
    stopthresh = roiparams.stopthresh
    maxperplane = roiparams.maxperplane
    showfigs = roiparams.showfigs
    
    # Get good frames
    goodframe_indices = np.where(alldata.maskgoodframes)[0]
    planeData4ROIs = alldata.imagedata[:, :, goodframe_indices].astype(np.float64)
    
    h, w = thisplanecorr.shape
    timepfull = planeData4ROIs.shape[2]
    
    # Create detrending filter
    myfilter = np.ones(deTrendScl) * (-1 / deTrendScl)
    center_idx = int(np.ceil(deTrendScl / 2)) - 1
    myfilter[center_idx] += 1
    
    # Initialize ROI tracking
    allrois = np.zeros(h * w, dtype=int)  # linearized map; 0 means unassigned, >0 is ROI id
    thisroi = np.zeros((h, w), dtype=bool)

    # Find initial seed point
    maxindex = np.argmax(thisplanecorr)
    maxval = thisplanecorr.flat[maxindex]
    i, j = np.unravel_index(maxindex, thisplanecorr.shape)
    thisplanecorr[i, j] = 0
    thisroi[i, j] = True
    
    # Initialize traces
    thistrace = planeData4ROIs[i, j, :].copy()
    thistracef = np.convolve(thistrace, myfilter, mode='same')
    
    # Storage for results
    mytraces = []
    mytracesraw = []
    allnums = []
    corrmapval = []
    avcorrval = []
    
    numcells = 0
    dilation_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    close_kernel = dilation_kernel
    
    print(f'Starting ROI detection (stopthresh={stopthresh}, maxperplane={maxperplane})...')
    
    with tqdm(total=maxperplane, desc='Finding ROIs') as pbar:
        while np.max(thisplanecorr) > stopthresh and numcells < maxperplane:
            numcells += 1
            processing = True
            thisnum = 1
            thisavcorr = 0
            iter_count = 0
            
            while processing:
                iter_count += 1
                
                # Grow ROI points using dilation
                newroipoints = ndimage.binary_dilation(thisroi, structure=dilation_kernel)
                newroiindices = np.where(newroipoints & ~thisroi)
                
                # Filter out already-found pixels
                if len(newroiindices[0]) == 0:
                    processing = False
                    continue
                    
                new_linear = np.ravel_multi_index(newroiindices, (h, w))
                valid_mask = allrois[new_linear] == 0
                newroiindices = (newroiindices[0][valid_mask], newroiindices[1][valid_mask])
                
                if len(newroiindices[0]) == 0:
                    processing = False
                    continue
                
                i_arr, j_arr = newroiindices
                
                # Calculate correlations for new pixels
                alltracesf = np.zeros((timepfull, len(i_arr)))
                alltracestrue = np.zeros((timepfull, len(i_arr)))
                thiscorr = np.zeros(len(i_arr))
                
                for n, (ii, jj) in enumerate(zip(i_arr, j_arr)):
                    newtrace = planeData4ROIs[ii, jj, :]
                    newtracef = np.convolve(newtrace, myfilter, mode='same')
                    alltracesf[:, n] = newtracef
                    alltracestrue[:, n] = newtrace
                    # Compute correlation safely
                    if np.std(newtracef) > 0 and np.std(thistracef) > 0:
                        thiscorr[n] = np.corrcoef(newtracef, thistracef)[0, 1]
                    else:
                        thiscorr[n] = 0
                
                # Select best half of pixels
                num2inc = max(1, int(np.ceil(len(i_arr) / 2)))
                sortind = np.argsort(thiscorr)
                bestind = sortind[-num2inc:]
                
                thiscorr = thiscorr[bestind]
                alltracestrue = alltracestrue[:, bestind]
                i_arr = i_arr[bestind]
                j_arr = j_arr[bestind]
                
                # Apply correlation threshold
                if iter_count == 1:
                    sigpixels = np.where(thiscorr > 0)[0]
                else:
                    sigpixels = np.where(thiscorr > corrthresh)[0]
                
                if len(sigpixels) == 0:
                    processing = False
                    continue
                
                # Add significant pixels to ROI
                alltracestrue = alltracestrue[:, sigpixels]
                tracecorrs = thiscorr[sigpixels]
                i_arr = i_arr[sigpixels]
                j_arr = j_arr[sigpixels]
                
                thisroi[i_arr, j_arr] = True
                thisplanecorr[i_arr, j_arr] = 0
                
                # Update trace (weighted average)
                thistrace = (thistrace * thisnum + np.sum(alltracestrue, axis=1)) / (len(sigpixels) + thisnum)
                thistracef = np.convolve(thistrace, myfilter, mode='same')
                
                # Update average correlation
                thisavcorr = (thisavcorr * (thisnum - 1) + np.sum(tracecorrs)) / (len(sigpixels) + thisnum - 1)
                thisnum += len(sigpixels)
                
                # Apply morphological closing
                newroipoints = ndimage.binary_closing(thisroi, structure=close_kernel)
                close_indices = np.where(newroipoints & ~thisroi)
                
                if len(close_indices[0]) > 0:
                    close_linear = np.ravel_multi_index(close_indices, (h, w))
                    close_valid = allrois[close_linear] == 0
                    close_indices = (close_indices[0][close_valid], close_indices[1][close_valid])
                    
                    for ii, jj in zip(close_indices[0], close_indices[1]):
                        newtrace = planeData4ROIs[ii, jj, :]
                        newtracef = np.convolve(newtrace, myfilter, mode='same')
                        
                        if np.std(newtracef) > 0 and np.std(thistracef) > 0:
                            corr_val = np.corrcoef(newtracef, thistracef)[0, 1]
                        else:
                            corr_val = 0
                        
                        thisroi[ii, jj] = True
                        thisplanecorr[ii, jj] = 0
                        
                        thistrace = (thistrace * thisnum + newtrace) / (thisnum + 1)
                        thistracef = np.convolve(thistrace, myfilter, mode='same')
                        thisavcorr = (thisavcorr * (thisnum - 1) + corr_val) / thisnum
                        thisnum += 1
                
                if thisnum > maximumroisize:
                    processing = False
            
            # Store this ROI's trace
            if np.std(thistrace) > 0:
                mytraces.append(zscore(thistrace))
            else:
                mytraces.append(thistrace)
            mytracesraw.append(thistrace.copy())
            
            # Mark ROI pixels
            roi_indices = np.where(thisroi)
            if len(roi_indices[0]) > 0:
                roi_linear = np.ravel_multi_index(roi_indices, (h, w))
                allrois[roi_linear] = numcells
            
            # Store statistics
            allnums.append(thisnum)
            corrmapval.append(maxval)
            avcorrval.append(thisavcorr)
            
            # Reset for next ROI
            thisroi[:] = False
            
            maxindex = np.argmax(thisplanecorr)
            maxval = thisplanecorr.flat[maxindex]
            i, j = np.unravel_index(maxindex, thisplanecorr.shape)
            thisplanecorr[i, j] = 0
            thisroi[i, j] = True
            
            thistrace = planeData4ROIs[i, j, :].copy()
            thistracef = np.convolve(thistrace, myfilter, mode='same')
            
            pbar.update(1)
    
    # Create ROIData result
    roidata = ROIData(
        allnums=np.array(allnums),
        corrmapval=np.array(corrmapval),
        avcorrval=np.array(avcorrval),
        mytraces=np.column_stack(mytraces) if mytraces else np.array([]),
        mytracesraw=np.column_stack(mytracesraw) if mytracesraw else np.array([]),
        allrois=allrois.reshape(h, w)
    )
    
    return roidata


def analyze_roi_responses(
	alldata: AllData,
	roidata: ROIData,
	min_roi_size: int = 50
) -> dict:
	# Correlate ROI traces with CS/US/learning regressors.
	# Notes:
	# - roidata.mytraces is time x n_rois (z-scored when possible)
	# - good_roi_indices maps the filtered ROI list back to the original roidata.allnums indices:
	#     original_roi_id = good_roi_indices[filtered_index]
	# - CS correlation masks out the US period to avoid confounds between CS and US evoked responses
    """
    Analyze ROI responses by correlating with CS/US/learning regressors.
    
    Parameters
    ----------
    alldata : AllData
        Concatenated plane data with regressors
    roidata : ROIData
        Detected ROIs with traces
    min_roi_size : int
        Minimum ROI size to include in analysis
    
    Returns
    -------
    dict
        Dictionary containing correlation results and cell indices
    """
    # Filter ROIs by size
    good_roi_mask = roidata.allnums > min_roi_size
    
    if roidata.mytraces.size == 0:
        print('No ROI traces available!')
        return {}
    
    mytracestruegood = roidata.mytraces[:, good_roi_mask]
    
    print(f'Analyzing {np.sum(good_roi_mask)} ROIs with >{min_roi_size} pixels')
    
    if mytracestruegood.shape[1] == 0:
        print('No ROIs passed size threshold!')
        return {}
    
    # Get regressors for good frames only
    goodframe_indices = np.where(alldata.maskgoodframes)[0]
    csRegressor = alldata.csRegressor[goodframe_indices]
    csMask = alldata.csMask[goodframe_indices]
    learnRegressor = alldata.learnRegressor[goodframe_indices]
    
    # Mask out US period for CS correlation
    non_us_mask = csMask == 0
    csRegressor_masked = csRegressor[non_us_mask]
    learnRegressor_masked = learnRegressor[non_us_mask]
    
    # Calculate correlations
    n_rois = mytracestruegood.shape[1]
    ccorr = np.zeros(n_rois)
    ucorr = np.zeros(n_rois)
    lcorr = np.zeros(n_rois)
    
    for n in range(n_rois):
        trace = mytracestruegood[:, n]
        masked_trace = trace[non_us_mask]
        
        # CS correlation (masked) - with NaN check
        if np.std(masked_trace) > 0 and np.std(csRegressor_masked) > 0:
            corr_val = np.corrcoef(masked_trace, csRegressor_masked)[0, 1]
            ccorr[n] = corr_val if not np.isnan(corr_val) else 0
        
        # US correlation (full trace) - with NaN check
        if np.std(trace) > 0 and np.std(csMask) > 0:
            corr_val = np.corrcoef(trace, csMask)[0, 1]
            ucorr[n] = corr_val if not np.isnan(corr_val) else 0
        
        # Learning correlation (masked) - with NaN check
        if np.std(masked_trace) > 0 and np.std(learnRegressor_masked) > 0:
            corr_val = np.corrcoef(masked_trace, learnRegressor_masked)[0, 1]
            lcorr[n] = corr_val if not np.isnan(corr_val) else 0
    
    # Find responsive cells based on correlation thresholds
    cs_threshold = 0.15
    us_threshold = 0.15
    learn_threshold = 0.10
    
    cs_responsive = np.where(ccorr > cs_threshold)[0]
    us_responsive = np.where(ucorr > us_threshold)[0]
    learn_responsive = np.where(lcorr > learn_threshold)[0]
    
    # Get original ROI indices
    good_roi_indices = np.where(good_roi_mask)[0]
    
    results = {
        'ccorr': ccorr,
        'ucorr': ucorr,
        'lcorr': lcorr,
        'cs_responsive': cs_responsive,
        'us_responsive': us_responsive,
        'learn_responsive': learn_responsive,
        'good_roi_mask': good_roi_mask,
        'good_roi_indices': good_roi_indices,
        'mytracestruegood': mytracestruegood,
        'n_rois': n_rois
    }
    
    print(f'Found {len(cs_responsive)} CS-responsive, '
          f'{len(us_responsive)} US-responsive, '
          f'{len(learn_responsive)} learning-responsive ROIs')
    
    return results


def cluster_roi_responses(
	mytraces: np.ndarray,
	n_clusters: int = 5,
	method: str = 'ward'
) -> tuple:
	# Cluster traces for visualization and grouping.
	# - pdist(metric='correlation') computes distances based on 1 - Pearson_correlation
	#   so smaller distance => more similar traces. Values are capped via nan_to_num.
    """
    Cluster ROI traces using hierarchical clustering.
    
    Parameters
    ----------
    mytraces : np.ndarray
        ROI traces (time x n_rois)
    n_clusters : int
        Number of clusters to form
    method : str
        Linkage method ('ward', 'average', 'complete')
    
    Returns
    -------
    tuple
        (cluster_labels, linkage_matrix, ordered_indices)
    """
    if mytraces.size == 0 or mytraces.shape[1] < 2:
        print('Not enough ROIs for clustering')
        return np.array([]), np.array([]), np.array([])
    
    # Compute pairwise distances (1 - Pearson correlation). pdist(metric='correlation')
    # returns 0 for identical traces, larger values when traces are dissimilar.
    distances = pdist(mytraces.T, metric='correlation')
    
    # Handle NaN distances
    distances = np.nan_to_num(distances, nan=1.0)
    
    # Hierarchical clustering
    Z = linkage(distances, method=method)
    
    # Optimal leaf ordering for visualization
    try:
        Z_ordered = optimal_leaf_ordering(Z, distances)
    except Exception:
        Z_ordered = Z
    
    # Get cluster assignments
    cluster_labels = fcluster(Z_ordered, n_clusters, criterion='maxclust')
    
    # Get ordered indices from dendrogram
    dendro = dendrogram(Z_ordered, no_plot=True)
    ordered_indices = np.array(dendro['leaves'])
    
    return cluster_labels, Z_ordered, ordered_indices


def plot_roi_heatmap(
    mytraces: np.ndarray,
    times: np.ndarray,
    ordered_indices: np.ndarray,
    cluster_labels: np.ndarray,
    trial_boundaries: np.ndarray,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot clustered ROI activity heatmap.
    
    Parameters
    ----------
    mytraces : np.ndarray
        ROI traces (time x n_rois)
    times : np.ndarray
        Time points
    ordered_indices : np.ndarray
        Indices for ordered display
    cluster_labels : np.ndarray
        Cluster assignments
    trial_boundaries : np.ndarray
        Trial start times for vertical lines
    save_path : Path, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Reorder traces
    if len(ordered_indices) > 0:
        ordered_traces = mytraces[:, ordered_indices]
    else:
        ordered_traces = mytraces
    
    # Plot heatmap
    im = ax.imshow(
        ordered_traces.T,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-2, vmax=2,
        extent=[times[0], times[-1], 0, ordered_traces.shape[1]]
    )
    
    # Add trial boundaries
    for boundary in trial_boundaries:
        ax.axvline(boundary, color='white', linewidth=0.5, alpha=0.5)
    
    # Add cluster boundaries
    if len(cluster_labels) > 0 and len(ordered_indices) > 0:
        ordered_labels = cluster_labels[ordered_indices]
        cluster_changes = np.where(np.diff(ordered_labels) != 0)[0]
        for change in cluster_changes:
            ax.axhline(change + 0.5, color='black', linewidth=1)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ROI (ordered by cluster)')
    ax.set_title('ROI Activity Heatmap')
    
    plt.colorbar(im, ax=ax, label='z-score')
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f'Saved heatmap to {save_path}')
    
    return fig


def plot_correlation_scatter(
    results: dict,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot scatter plots of CS vs US and CS vs Learning correlations.
    
    Parameters
    ----------
    results : dict
        Output from analyze_roi_responses
    save_path : Path, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ccorr = results['ccorr']
    ucorr = results['ucorr']
    lcorr = results['lcorr']
    
    # CS vs US correlation
    axes[0].scatter(ccorr, ucorr, alpha=0.5, s=20)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].set_xlabel('CS correlation')
    axes[0].set_ylabel('US correlation')
    axes[0].set_title('CS vs US Responsiveness')
    axes[0].set_xlim([-0.5, 0.5])
    axes[0].set_ylim([-0.5, 0.5])
    
    # CS vs Learning correlation
    axes[1].scatter(ccorr, lcorr, alpha=0.5, s=20, color='green')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('CS correlation')
    axes[1].set_ylabel('Learning correlation')
    axes[1].set_title('CS vs Learning Responsiveness')
    axes[1].set_xlim([-0.5, 0.5])
    axes[1].set_ylim([-0.5, 0.5])
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f'Saved scatter plot to {save_path}')
    
    return fig


def plot_responsive_traces(
    mytraces: np.ndarray,
    responsive_indices: np.ndarray,
    regressor: np.ndarray,
    title: str = 'Responsive Cells',
    offset: float = 3.0,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
	# Plot multiple traces stacked vertically. Caller must ensure:
	# - mytraces shape: (time, n_rois)
	# - regressor length matches the time dimension (usually constructed using good-frame indices)
    """
    Plot traces of responsive cells with regressor overlay.
    
    Parameters
    ----------
    mytraces : np.ndarray
        ROI traces (time x n_rois)
    responsive_indices : np.ndarray
        Indices of responsive ROIs to plot
    regressor : np.ndarray
        Regressor to overlay (e.g., CS, US)
    title : str
        Plot title
    offset : float
        Vertical offset between traces
    ax : plt.Axes, optional
        Axes to plot on
    
    Returns
    -------
    plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    for n, idx in enumerate(responsive_indices):
        ax.plot(mytraces[:, idx] + offset * n, alpha=0.7)
    
    # Scale and plot regressor
    if len(responsive_indices) > 0:
        regressor_scaled = regressor * offset * 0.8
        ax.plot(regressor_scaled, 'k--', alpha=0.5, label='Regressor')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Activity (z-score + offset)')
    ax.set_title(f'{title} (n={len(responsive_indices)})')
    
    return ax


def plot_cell_type_traces(
    results: dict,
    alldata: AllData,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot traces for CS, US, and learning responsive cells.
    
    Parameters
    ----------
    results : dict
        Output from analyze_roi_responses
    alldata : AllData
        Concatenated plane data
    save_path : Path, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    mytraces = results['mytracestruegood']
    goodframe_indices = np.where(alldata.maskgoodframes)[0]
    csRegressor = alldata.csRegressor[goodframe_indices]
    csMask = alldata.csMask[goodframe_indices]
    
    # CS responsive cells
    plot_responsive_traces(
        mytraces, results['cs_responsive'], csRegressor,
        title='CS Responsive', ax=axes[0]
    )
    
    # Learning responsive cells
    plot_responsive_traces(
        mytraces, results['learn_responsive'], csRegressor,
        title='Learning Responsive', ax=axes[1]
    )
    
    # US responsive cells
    plot_responsive_traces(
        mytraces, results['us_responsive'], csMask,
        title='US Responsive', ax=axes[2]
    )
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f'Saved cell type traces to {save_path}')
    
    return fig


def create_roi_colormap(
    allrois: np.ndarray,
    roi_indices: np.ndarray,
    colors: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create RGB image showing ROI locations with colors.
    
    Parameters
    ----------
    allrois : np.ndarray
        ROI label map (h x w)
    roi_indices : np.ndarray
        ROI indices to highlight (1-based, matching allrois values)
    colors : np.ndarray, optional
        RGB colors for each ROI (n_rois x 3), values 0-1
    
    Returns
    -------
    np.ndarray
        RGB image (h x w x 3)
    """
    h, w = allrois.shape
    colimage = np.zeros((h, w, 3), dtype=np.float32)
    
    if colors is None:
        # Default: white for all ROIs
        colors = np.ones((len(roi_indices), 3))
    
    for n, roi_idx in enumerate(roi_indices):
        mask = allrois == roi_idx
        for c in range(3):
            colimage[:, :, c][mask] = colors[n % len(colors), c]
    
    return colimage


def plot_roi_map(
    roidata: ROIData,
    results: dict,
    cell_type: str = 'all',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot spatial map of ROIs colored by cell type.
    
    Parameters
    ----------
    roidata : ROIData
        ROI detection results
    results : dict
        Output from analyze_roi_responses
    cell_type : str
        'cs', 'us', 'learn', or 'all'
    save_path : Path, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    good_roi_indices = results['good_roi_indices']
    
    # Define colors for different cell types
    colors = {
        'cs': np.array([0, 1, 0]),      # Green
        'us': np.array([1, 0, 0]),      # Red
        'learn': np.array([0, 0, 1]),   # Blue
    }
    
    if cell_type == 'all':
        # Create composite image
        colimage = np.zeros((*roidata.allrois.shape, 3), dtype=np.float32)
        
        # CS cells in green
        for idx in results['cs_responsive']:
            roi_idx = good_roi_indices[idx] + 1  # +1 because allrois is 1-indexed
            mask = roidata.allrois == roi_idx
            colimage[:, :, 1][mask] = 1.0
        
        # US cells in red
        for idx in results['us_responsive']:
            roi_idx = good_roi_indices[idx] + 1
            mask = roidata.allrois == roi_idx
            colimage[:, :, 0][mask] = 1.0
        
        # Learning cells in blue
        for idx in results['learn_responsive']:
            roi_idx = good_roi_indices[idx] + 1
            mask = roidata.allrois == roi_idx
            colimage[:, :, 2][mask] = 1.0
        
        title = 'All Responsive Cells (G=CS, R=US, B=Learn)'
    else:
        responsive_map = {
            'cs': results['cs_responsive'],
            'us': results['us_responsive'],
            'learn': results['learn_responsive']
        }
        responsive_indices = responsive_map.get(cell_type, [])
        roi_indices = good_roi_indices[responsive_indices] + 1
        
        color_array = np.tile(colors[cell_type], (len(roi_indices), 1))
        colimage = create_roi_colormap(roidata.allrois, roi_indices, color_array)
        title = f'{cell_type.upper()} Responsive Cells'
    
    ax.imshow(colimage)
    ax.set_title(title)
    ax.axis('off')
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f'Saved ROI map to {save_path}')
    
    return fig


def plot_mean_response_by_type(
    results: dict,
    alldata: AllData,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot mean response traces for each cell type with regressor overlay.
    
    Parameters
    ----------
    results : dict
        Output from analyze_roi_responses
    alldata : AllData
        Concatenated plane data
    save_path : Path, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    mytraces = results['mytracestruegood']
    ccorr = results['ccorr']
    ucorr = results['ucorr']
    lcorr = results['lcorr']
    
    goodframe_indices = np.where(alldata.maskgoodframes)[0]
    csRegressor = alldata.csRegressor[goodframe_indices]
    csMask = alldata.csMask[goodframe_indices]
    
    # Learning cells: high learning corr, higher than CS corr
    learning_cells = np.where((lcorr > 0.1) & (lcorr - ccorr > 0.05))[0]
    non_us_mask = csMask == 0
    
    if len(learning_cells) > 0:
        masked_traces = mytraces[non_us_mask, :][:, learning_cells]
        mean_trace = np.mean(masked_traces, axis=1)
        axes[0].plot(csRegressor[non_us_mask] * 3, 'g--', label='CS Regressor')
        axes[0].plot(mean_trace, 'b-', label=f'Mean (n={len(learning_cells)})')
        axes[0].set_title('Learning Cells (lcorr > 0.1, lcorr - ccorr > 0.05)')
        axes[0].legend()
    
    # US cells
    us_cells = np.where(ucorr > 0.1)[0]
    if len(us_cells) > 0:
        mean_trace = np.mean(mytraces[:, us_cells], axis=1)
        axes[1].plot(csRegressor * 3, 'g--', label='CS Regressor')
        axes[1].plot(csMask * 3, 'r--', label='US Mask')
        axes[1].plot(mean_trace, 'b-', label=f'Mean (n={len(us_cells)})')
        axes[1].set_title('US Cells (ucorr > 0.1)')
        axes[1].legend()
    
    # CS cells (not learning)
    cs_cells = np.where((ccorr > 0.1) & (lcorr - ccorr < 0))[0]
    if len(cs_cells) > 0:
        masked_traces = mytraces[non_us_mask, :][:, cs_cells]
        mean_trace = np.mean(masked_traces, axis=1)
        axes[2].plot(csRegressor[non_us_mask] * 3, 'g--', label='CS Regressor')
        axes[2].plot(mean_trace, 'b-', label=f'Mean (n={len(cs_cells)})')
        axes[2].set_title('CS Cells (ccorr > 0.1, lcorr - ccorr < 0)')
        axes[2].legend()
    
    axes[2].set_xlabel('Frame')
    
    for ax in axes:
        ax.set_ylabel('Activity (z-score)')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f'Saved mean response plot to {save_path}')
    
    return fig


def run_full_roi_analysis(
	whichfolder: Path,
	whichfile: str,
	whichplane: int = 0,
	numbertrials: int = 16,
	gausswidth: float = 2.0,
	roiparams: Optional[ROIParams] = None,
	save_results: bool = True,
	plot_results: bool = True
) -> dict:
	# High-level wrapper that runs the full pipeline:
	# 1) concatenate trials -> AllData
	# 2) compute correlation map -> corrmap
	# 3) detect ROIs -> ROIData
	# 4) analyze responses -> results dict
	# 5) cluster and plot/save outputs if requested
	# All intermediate results are returned for programmatic inspection.
    """
    Run the complete ROI analysis pipeline for a single plane.
    
    Parameters
    ----------
    whichfolder : Path
        Folder containing the HDF5 file
    whichfile : str
        Name of the HDF5 file
    whichplane : int
        Plane index to analyze
    numbertrials : int
        Number of trials
    gausswidth : float
        Gaussian smoothing width for correlation map
    roiparams : ROIParams, optional
        ROI detection parameters
    save_results : bool
        Whether to save results to disk
    plot_results : bool
        Whether to generate plots
    
    Returns
    -------
    dict
        Dictionary containing all analysis results
    """
    if roiparams is None:
        roiparams = ROIParams()
    
    print(f'=== ROI Analysis for plane {whichplane} ===')
    
    # Step 1: Concatenate data
    print('\n1. Concatenating plane data...')
    alldata = concat_single_plane_file(
        whichfolder, whichfile, whichplane,
        savedata=save_results, numbertrials=numbertrials
    )
    
    # Step 2: Calculate correlation map
    print('\n2. Calculating correlation map...')
    corrmap = calculate_correlation_map_single_plane(
        alldata, gausswidth=gausswidth, deTrendScl=roiparams.deTrendScl
    )
    
    # Step 3: Find ROIs
    print('\n3. Finding ROIs...')
    roidata = find_rois_single_plane(alldata, corrmap.copy(), roiparams)
    
    # Step 4: Analyze responses
    print('\n4. Analyzing ROI responses...')
    results = analyze_roi_responses(alldata, roidata)
    
    # Step 5: Cluster ROIs
    print('\n5. Clustering ROIs...')
    if results and results['mytracestruegood'].shape[1] > 2:
        cluster_labels, Z, ordered_indices = cluster_roi_responses(
            results['mytracestruegood'], n_clusters=5
        )
        results['cluster_labels'] = cluster_labels
        results['linkage_matrix'] = Z
        results['ordered_indices'] = ordered_indices
    
    # Save ROI data
    if save_results:
        savefolder = whichfolder / 'savefolder'
        savefolder.mkdir(exist_ok=True)
        roi_savepath = savefolder / f'rois{whichplane:05d}.pkl'
        print(f'\nSaving ROI data to {roi_savepath}')
        with open(roi_savepath, 'wb') as f:
            pickle.dump({'roidata': roidata, 'results': results}, f)
    
    # Generate plots
    if plot_results and results:
        print('\n6. Generating plots...')
        savefolder = whichfolder / 'savefolder' / 'figures'
        savefolder.mkdir(exist_ok=True, parents=True)
        
        # Correlation scatter
        plot_correlation_scatter(
            results, save_path=savefolder / f'plane{whichplane:05d}_scatter.png'
        )
        
        # Cell type traces
        plot_cell_type_traces(
            results, alldata,
            save_path=savefolder / f'plane{whichplane:05d}_traces.png'
        )
        
        # ROI map
        plot_roi_map(
            roidata, results, cell_type='all',
            save_path=savefolder / f'plane{whichplane:05d}_roi_map.png'
        )
        
        # Mean responses
        plot_mean_response_by_type(
            results, alldata,
            save_path=savefolder / f'plane{whichplane:05d}_mean_responses.png'
        )
        
        # Heatmap
        if 'ordered_indices' in results:
            goodframe_indices = np.where(alldata.maskgoodframes)[0]
            times = alldata.compressedtimes[goodframe_indices]
            plot_roi_heatmap(
                results['mytracestruegood'], times,
                results['ordered_indices'], results['cluster_labels'],
                alldata.trialstarttime,
                save_path=savefolder / f'plane{whichplane:05d}_heatmap.png'
            )
        
        plt.close('all')
    
    return {
        'alldata': alldata,
        'roidata': roidata,
        'corrmap': corrmap,
        'results': results
    }


# %% Main execution block
if __name__ == '__main__':
	# Main guard: brief runtime instructions added
	# - edit hdf5_folder and fish_name at top to point to dataset
	# - run this script as a module to perform step-by-step debugging (the script includes interactive cells)
	# - large computations (ROI detection) can be commented out during quick tests
    # Example usage
    print(f'Processing fish: {fish_name}')
    print(f'Path: {path_home}')
    
    # Define paths
    hdf5_folder = Path(r"H:\2-P imaging\2024 09_Delay 2-P 4 planes JC neurons\20241007_03_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf")
    # path_home / 'Processed data'
    hdf5_file = f'{fish_name}_data.h5'
    
    # Check if file exists
    if not (hdf5_folder / hdf5_file).exists():
        print(f'HDF5 file not found: {hdf5_folder / hdf5_file}')
        print('Run script 5.Save_data_as_HDF5.py first.')
    else:
        print(f'Found HDF5 file: {hdf5_file}')
        print('Ready for ROI analysis.')
        
        # Uncomment to run full analysis:
        # results = run_full_roi_analysis(
        #     hdf5_folder, hdf5_file,
        #     whichplane=0,
        #     numbertrials=16,
        #     roiparams=ROIParams(
        #         deTrendScl=55,
        #         maximumroisize=200,
        #         corrthresh=0.4,
        #         stopthresh=0.1,
        #         maxperplane=1000
        #     ),
        #     save_results=True,
        #     plot_results=True
        # )

    # %% Step-by-step analysis for debugging
    # Parameters
    whichplane = 0
    numbertrials = 16
    gausswidth = 2.0
    roiparams = ROIParams(
        deTrendScl=55,
        maximumroisize=200,
        corrthresh=0.4,
        stopthresh=0.1,
        maxperplane=1000
    )

    # %% Step 1: Concatenate plane data
    print('\n=== Step 1: Concatenating plane data ===')
    alldata = concat_single_plane_file(
        hdf5_folder, hdf5_file, whichplane,
        savedata=True, overwrite=False, numbertrials=numbertrials
    )
    print(f'Image data shape: {alldata.imagedata.shape}')
    print(f'Number of good frames: {len(alldata.goodframes)}')
    print(f'Number of trials: {len(alldata.trialstarttime)}')

    # %% Step 2: Calculate correlation map
    print('\n=== Step 2: Calculating correlation map ===')
    corrmap = calculate_correlation_map_single_plane(
        alldata, gausswidth=gausswidth, deTrendScl=roiparams.deTrendScl
    )
    print(f'Correlation map shape: {corrmap.shape}')
    print(f'Correlation map range: [{corrmap.min():.3f}, {corrmap.max():.3f}]')

    # Plot correlation map
    plt.figure(figsize=(10, 8))
    plt.imshow(corrmap * 3, cmap='hot')
    plt.colorbar(label='Correlation')
    plt.title(f'Correlation Map - Plane {whichplane}')
    plt.show()

    # %% Step 3: Find ROIs
    print('\n=== Step 3: Finding ROIs ===')
    roidata = find_rois_single_plane(alldata, corrmap.copy(), roiparams)
    print(f'Number of ROIs found: {len(roidata.allnums)}')
    print(f'ROI sizes: min={roidata.allnums.min()}, max={roidata.allnums.max()}, mean={roidata.allnums.mean():.1f}')
    print(f'Traces shape: {roidata.mytraces.shape}')

    # Plot ROI map
    plt.figure(figsize=(10, 8))
    plt.imshow(roidata.allrois, cmap='nipy_spectral')
    plt.colorbar(label='ROI ID')
    plt.title(f'ROI Map - Plane {whichplane} ({len(roidata.allnums)} ROIs)')
    plt.show()

    # %% Step 4: Analyze responses
    print('\n=== Step 4: Analyzing ROI responses ===')
    results = analyze_roi_responses(alldata, roidata, min_roi_size=50)

    if results:
        print(f'Number of good ROIs: {results["n_rois"]}')
        print(f'CS responsive: {len(results["cs_responsive"])}')
        print(f'US responsive: {len(results["us_responsive"])}')
        print(f'Learning responsive: {len(results["learn_responsive"])}')

    # %% Step 5: Cluster ROIs
    print('\n=== Step 5: Clustering ROIs ===')
    if results and results['mytracestruegood'].shape[1] > 2:
        cluster_labels, Z, ordered_indices = cluster_roi_responses(
        results['mytracestruegood'], n_clusters=5
        )
        results['cluster_labels'] = cluster_labels
        results['linkage_matrix'] = Z
        results['ordered_indices'] = ordered_indices
        print(f'Cluster sizes: {np.bincount(cluster_labels)[1:]}')

    # %% Step 6: Generate plots
    print('\n=== Step 6: Generating plots ===')

    # Correlation scatter
    if results:
        fig_scatter = plot_correlation_scatter(results)
        plt.show()

    # %% Cell type traces
    if results:
        fig_traces = plot_cell_type_traces(results, alldata)
        plt.show()

    # %% ROI map with cell types
    if results:
        fig_roi_map = plot_roi_map(roidata, results, cell_type='all')
        plt.show()

    # %% Mean responses by cell type
    if results:
        fig_mean = plot_mean_response_by_type(results, alldata)
        plt.show()

    # %% Heatmap
    if results and 'ordered_indices' in results:
        goodframe_indices = np.where(alldata.maskgoodframes)[0]
        times = alldata.compressedtimes[goodframe_indices]
        fig_heatmap = plot_roi_heatmap(
        results['mytracestruegood'], times,
        results['ordered_indices'], results['cluster_labels'],
        alldata.trialstarttime
        )
        plt.show()


# %% Interactive analysis cells

# %% Load and concatenate data for a single plane
# whichplane = 0
# alldata = concat_single_plane_file(hdf5_folder, hdf5_file, whichplane, savedata=True)

# %% Calculate correlation map
# corrmap = calculate_correlation_map_single_plane(alldata, gausswidth=2.0)
# plt.figure(); plt.imshow(corrmap * 3, cmap='hot'); plt.colorbar(); plt.title('Correlation Map')

# %% Find ROIs
# roiparams = ROIParams(corrthresh=0.4, stopthresh=0.1, maxperplane=500)
# roidata = find_rois_single_plane(alldata, corrmap.copy(), roiparams)

# %% Analyze responses
# results = analyze_roi_responses(alldata, roidata)

# %% Plot results
# plot_correlation_scatter(results)
# plot_cell_type_traces(results, alldata)
# plot_roi_map(roidata, results, cell_type='all')
# plot_mean_response_by_type(results, alldata)
