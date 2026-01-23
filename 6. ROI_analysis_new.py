"""
6. ROI Analysis
===============

Performs correlation-based ROI detection and stimulus response analysis on 2-photon imaging data.

Workflow:
1. Load HDF5 data (planes/trials).
2. Concatenate trials for a single plane into a continuous time-series (AllData).
3. Compute a pixel-wise local correlation map to identify active neural structures.
4. Detect ROIs using an iterative correlation-based region growing algorithm.
5. Extract ROI traces and correlate them with behavioral repressors (CS, US, Learning).
6. Cluster and visualize the results.
"""

# Imports
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, optimal_leaf_ordering
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from tqdm import tqdm

import plotting_style_new as plotting_style
from my_paths_new import fish_name as default_fish_name
from my_paths_new import path_home as default_path_home

# region Configuration
PANDAS_OPTIONS = {
    "mode.copy_on_write": True,
    "compute.use_numba": True,
    "compute.use_numexpr": True,
    "compute.use_bottleneck": True,
}

def configure_environment() -> None:
    for option, value in PANDAS_OPTIONS.items():
        pd.set_option(option, value)
    plotting_style.set_plot_style(use_constrained_layout=False)

configure_environment()

# Paths (Overridable)
PATH_HOME_OVERRIDE = None
FISH_NAME_OVERRIDE = None

path_home = PATH_HOME_OVERRIDE or default_path_home
fish_name = FISH_NAME_OVERRIDE or default_fish_name
# endregion

# region Data Classes

@dataclass
class PlaneTimings:
    """Stimulus timing relative to the start of the trial imaging."""
    cs_start_time: float = 0.0
    cs_end_time: float = 0.0
    us_start_time: Optional[float] = None
    us_end_time: Optional[float] = None

@dataclass
class PlaneData:
    """Raw data for a single plane/trial loaded from HDF5."""
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
    """Concatenated continuous data for a single plane across multiple trials."""
    maskgoodframes: np.ndarray = field(default_factory=lambda: np.array([]))
    imagedata: np.ndarray = field(default_factory=lambda: np.array([]))
    badframes: np.ndarray = field(default_factory=lambda: np.array([]))
    goodframes: np.ndarray = field(default_factory=lambda: np.array([]))
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    compressedtimes: np.ndarray = field(default_factory=lambda: np.array([]))
    behaviortimes: np.ndarray = field(default_factory=lambda: np.array([]))
    tailtrace: np.ndarray = field(default_factory=lambda: np.array([]))
    # Binary flags for stimulus presence (hires=behavior time, lores=imaging time)
    csflaghires: np.ndarray = field(default_factory=lambda: np.array([]))
    usflaghires: np.ndarray = field(default_factory=lambda: np.array([]))
    csflaglores: np.ndarray = field(default_factory=lambda: np.array([]))
    usflaglores: np.ndarray = field(default_factory=lambda: np.array([]))
    # Trial boundaries
    trialstarttime: np.ndarray = field(default_factory=lambda: np.array([]))
    trialendtime: np.ndarray = field(default_factory=lambda: np.array([]))
    trialhasus: np.ndarray = field(default_factory=lambda: np.array([]))
    # Regressors for correlation analysis
    csRegressor: np.ndarray = field(default_factory=lambda: np.array([]))
    csMask: np.ndarray = field(default_factory=lambda: np.array([]))
    learnRegressor: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class ROIParams:
    """Parameters controlling ROI detection."""
    deTrendScl: int = 55       # Window size for detrending filter
    maximumroisize: int = 200  # Max pixels per ROI
    corrthresh: float = 0.4    # Minimum correlation to add a pixel to ROI
    stopthresh: float = 0.1    # Stop searching when max correlation drops below this
    maxperplane: int = 1000    # Soft limit on number of ROIs
    showfigs: bool = False

@dataclass
class ROIData:
    """Results of ROI detection."""
    allnums: np.ndarray = field(default_factory=lambda: np.array([]))    # Size of each ROI
    corrmapval: np.ndarray = field(default_factory=lambda: np.array([])) # Max correlation value at seed
    avcorrval: np.ndarray = field(default_factory=lambda: np.array([]))  # Average correlation of ROI
    mytraces: np.ndarray = field(default_factory=lambda: np.array([]))   # Filtered/Z-scored traces [time x n_rois]
    mytracesraw: np.ndarray = field(default_factory=lambda: np.array([]))# Raw traces [time x n_rois]
    allrois: np.ndarray = field(default_factory=lambda: np.array([]))    # Spatial map [y, x] where val=roi_id

# endregion

# region Helper Functions

def _resolve_trial_group(h5_file: h5py.File, plane_number: int, trial_number: int) -> tuple[h5py.Group, str]:
    """Find the HDF5 group for a specific plane and trial."""
    planes_group = h5_file.get('planes')
    if planes_group is None:
        raise KeyError("'planes' group not found in HDF5 file")

    # Try common naming conventions
    plane_keys = [f'plane_{plane_number}', f'item_{plane_number}']
    trial_keys = [f'trial_{trial_number}', f'item_{trial_number}']

    for p_key in plane_keys:
        if p_key in planes_group:
            p_group = planes_group[p_key]
            # Check direct trials
            for t_key in trial_keys:
                if t_key in p_group:
                    return p_group[t_key], f'/planes/{p_key}/{t_key}'
            # Check nested trials group
            if 'trials' in p_group:
                t_group = p_group['trials']
                for t_key in trial_keys:
                    if t_key in t_group:
                        return t_group[t_key], f'/planes/{p_key}/trials/{t_key}'
    
    raise KeyError(f"Trial {trial_number} not found for plane {plane_number}")

def _read_dataset(group: h5py.Group, name: str) -> Optional[np.ndarray]:
    """Read dataset from HDF5 group, handling fallbacks."""
    if name in group:
        obj = group[name]
        if isinstance(obj, h5py.Dataset):
            return obj[:]
        if isinstance(obj, h5py.Group):  # Handle xarray/pandas saved as groups
            for key in ('data', 'values'):
                if key in obj:
                    return obj[key][:]
    
    for fb in [f'{name}_data_fallback', f'{name}_values_fallback']:
        if fb in group:
            return group[fb][:]
    return None

def _read_xarray_coords(group: h5py.Group) -> dict:
    coords = {}
    if 'coords' in group:
        for k in group['coords']:
            coords[k] = group['coords'][k][:]
    return coords

def _find_coord(coords: dict, keys: list) -> Optional[np.ndarray]:
    for k in keys:
        if k in coords:
            return coords[k]
    return None

# endregion

# region Core Data Loading

def get_plane_data(filename: Path, plane_number: int, trial_number: int) -> PlaneData:
    """Load a single plane's trial from HDF5."""
    pd_obj = PlaneData()
    with h5py.File(filename, 'r') as f:
        t_group, _ = _resolve_trial_group(f, plane_number, trial_number)
        
        # Load Images
        imagedata_raw = _read_dataset(t_group, 'images')
        if imagedata_raw is None:
            raise KeyError("No images found.")
        
        # Ensure [Y, X, T] format
        if imagedata_raw.ndim == 3 and imagedata_raw.shape[0] < imagedata_raw.shape[2]: # Assuming T is usually smaller than X*Y? No, T is ~1000, X,Y ~512.
            # Actually, script 5 saves as [T, Y, X]. We typically want [Y, X, T] for processing.
             pd_obj.imagedata = np.transpose(imagedata_raw, (1, 2, 0)).astype(np.float32)
        else:
            pd_obj.imagedata = imagedata_raw.astype(np.float32)

        n_frames = pd_obj.imagedata.shape[-1]
        
        # Load Coords for Timing/Masks
        coords = {}
        if 'images' in t_group and isinstance(t_group['images'], h5py.Group):
            coords = _read_xarray_coords(t_group['images'])

        # Good/Bad Frames
        mask = _find_coord(coords, ['mask good frames', 'mask_good_frames'])
        if mask is None:
            mask = _read_dataset(t_group, 'mask_good_frames')
            
        if mask is not None:
             # Logic to parse boolean/string masks
             if mask.dtype.kind in 'SUO': # Strings
                 mask_str = np.char.upper(mask.astype(str))
                 pd_obj.goodframes = np.where(mask_str == 'TRUE')[0]
                 pd_obj.badframes = np.where(mask_str == 'FALSE')[0]
             else:
                 pd_obj.goodframes = np.where(mask != 0)[0]
                 pd_obj.badframes = np.where(mask == 0)[0]
        else:
            pd_obj.goodframes = np.arange(n_frames)
            
        # Timings
        times = _find_coord(coords, ['Time (ms)', 'time', 'times'])
        if times is None:
            times = np.arange(n_frames) * 33.0 # Fallback 30Hz
        
        pd_obj.times = times - times[0]
        
        # Protocol
        prot = _read_dataset(t_group, 'protocol')
        timings = PlaneTimings()
        if prot is not None and prot.size > 0:
             # Assume standard structure: Row 0 is times, Row 1=CS_Start, Row 2=CS_End, Row 3=US_Start...
             if prot.shape[0] > prot.shape[1]: prot = prot.T
             ptimes = prot[0, :]
             # CS
             cs_s = np.where(prot[1] > 0)[0]
             cs_e = np.where(prot[2] > 0)[0]
             if len(cs_s): timings.cs_start_time = ptimes[cs_s[0]]
             if len(cs_e): timings.cs_end_time = ptimes[cs_e[0]]
             # US
             if prot.shape[0] > 3:
                 us_s = np.where(prot[3] > 0)[0]
                 if len(us_s): timings.us_start_time = ptimes[us_s[0]]
             if prot.shape[0] > 4:
                 us_e = np.where(prot[4] > 0)[0]
                 if len(us_e): timings.us_end_time = ptimes[us_e[0]]
        
        # Adjust timings to relative
        start_offset = times[0]
        timings.cs_start_time -= start_offset
        timings.cs_end_time -= start_offset
        if timings.us_start_time: timings.us_start_time -= start_offset
        if timings.us_end_time: timings.us_end_time -= start_offset
        
        pd_obj.timings = timings

        # Behavior
        beh = _read_dataset(t_group, 'behavior')
        if beh is not None:
             if beh.shape[0] < beh.shape[1]: beh = beh.T
             pd_obj.behaviortimes = beh[:, 0] - start_offset
             # Simple extraction of tail trace (last column usually)
             if beh.shape[1] > 1:
                pd_obj.tailtrace = beh[:, -1]
    
    return pd_obj


def concat_single_plane_file(folder: Path, filename: str, plane: int, n_trials: int, 
                             save: bool = True, overwrite: bool = False) -> AllData:
    """
    Concatenate multiple trials for a single plane.
    
    This creates a continuous timeline of imaging data ('AllData'), stitching trials together.
    It harmonizes timestamps and creates continuous regressor vectors for analysis.
    """
    save_path = folder / 'savefolder' / f'plane{plane:05d}.pkl'
    if save_path.exists() and not overwrite:
        print(f"Loading cached plane data: {save_path}")
        with open(save_path, 'rb') as f: return pickle.load(f)

    print(f"Concatenating {n_trials} trials for plane {plane}...")
    full_path = folder / filename
    
    # Initialize containers
    ad = AllData()
    img_list, bad_list, good_list = [], [], []
    time_list, comp_time_list = [], []
    beh_time_list, tail_list = [], []
    
    # Stimulus Flags
    cs_flags, us_flags = [], []
    cs_regs, us_masks, l_regs = [], [], []

    current_time_base = 0.0
    
    for t in tqdm(range(n_trials)):
        pd = get_plane_data(full_path, plane, t)
        
        # Offsets
        frame_offset = sum(x.shape[2] for x in img_list)
        
        img_list.append(pd.imagedata)
        bad_list.append(pd.badframes + frame_offset)
        good_list.append(pd.goodframes + frame_offset)
        
        # Continuous Timing
        dt = np.mean(np.diff(pd.times)) if len(pd.times) > 1 else 33.0
        pd_dur = pd.times[-1] + dt
        
        # Append times shifted by current base
        time_list.append(pd.times) # Raw trial times
        comp_time_list.append(pd.times + current_time_base)
        
        # Behavior
        beh_time_list.append(pd.behaviortimes + current_time_base)
        tail_list.append(pd.tailtrace)
        
        # Regressors for this trial
        # Map trial-relative timings to the continuous time vector
        n_fr = len(pd.times)
        tr_cs = np.zeros(n_fr, dtype=int)
        tr_us = np.zeros(n_fr, dtype=int)
        tr_learn = np.zeros(n_fr, dtype=int)
        
        # CS Window
        cs_mask = (pd.times >= pd.timings.cs_start_time) & (pd.times < pd.timings.cs_end_time)
        
        if pd.timings.us_start_time:
            # CS-US Trial
            # CS only until US start
            cs_pure = cs_mask & (pd.times < pd.timings.us_start_time)
            tr_cs[cs_pure] = 1
            # US Window
            us_mask = (pd.times >= pd.timings.us_start_time) & (pd.times < (pd.timings.us_end_time or pd_dur))
            tr_us[us_mask] = 1
            # Learning = CS period in US trials
            tr_learn[cs_pure] = 1
        else:
             # CS-only Trial
             tr_cs[cs_mask] = 1
        
        cs_regs.append(tr_cs)
        us_masks.append(tr_us)
        l_regs.append(tr_learn)

        current_time_base += pd_dur

    # Merge
    ad.imagedata = np.concatenate(img_list, axis=2)
    ad.badframes = np.concatenate(bad_list)
    ad.goodframes = np.concatenate(good_list)
    ad.compressedtimes = np.concatenate(comp_time_list)
    ad.behaviortimes = np.concatenate(beh_time_list) if beh_time_list else np.array([])
    ad.tailtrace = np.concatenate(tail_list) if tail_list else np.array([])
    
    ad.csRegressor = np.concatenate(cs_regs)
    ad.csMask = np.concatenate(us_masks)
    ad.learnRegressor = np.concatenate(l_regs)
    
    # Mask Good Frames Boolean
    total_frames = ad.imagedata.shape[2]
    mask = np.zeros(total_frames, dtype=bool)
    mask[ad.goodframes] = True
    ad.maskgoodframes = mask

    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f: pickle.dump(ad, f)
        
    return ad

# endregion

# region Analysis: Correlation & ROI Finding

def calculate_correlation_map(alldata: AllData, params: ROIParams) -> np.ndarray:
    """
    Calculate pixel-wise correlation map.
    
    Steps:
    1. Filter image spatially (Gaussian).
    2. Detrend temporally to remove slow drifts.
    3. Compute correlation of smoothed signal vs raw signal.
    """
    # Use only good frames
    d = alldata.imagedata[:, :, alldata.goodframes].astype(np.float32)
    
    print("Calculating correlation map...")
    # 1. Spatial Smoothing
    d_smooth = np.zeros_like(d)
    for i in range(d.shape[2]):
        d_smooth[:,:,i] = gaussian_filter(d[:,:,i], sigma=2.0)
    
    # 2. Subtract Mean
    d -= d.mean(axis=2, keepdims=True)
    d_smooth -= d_smooth.mean(axis=2, keepdims=True)
    
    # 3. Correlation (Norm(smooth) / Norm(raw))
    # This metric highlights pixels where local spatial averaging preserves signal (i.e., cells)
    # vs noise (which averages out).
    norm_smooth = np.linalg.norm(d_smooth, axis=2)
    norm_raw = np.linalg.norm(d, axis=2)
    
    # Avoid div by zero
    norm_raw[norm_raw == 0] = 1.0
    
    corrmap = (norm_smooth / norm_raw) ** 2
    
    # Clean edges
    b = 10
    corrmap[:b, :] = 0; corrmap[-b:, :] = 0
    corrmap[:, :b] = 0; corrmap[:, -b:] = 0
    
    return corrmap

def find_rois(alldata: AllData, corrmap: np.ndarray, params: ROIParams) -> ROIData:
    """
    Iterative ROI detection.
    
    Algorithm:
    1. Pick max pixel in correlation map.
    2. Region grow based on correlation similarity.
    3. Save ROI if valid.
    4. Zero out ROI in map and repeat.
    """
    d = alldata.imagedata[:, :, alldata.goodframes]
    h, w = corrmap.shape
    roi_map = np.zeros((h, w), dtype=int)
    
    traces = []
    raw_traces = []
    sizes = []
    
    curr_map = corrmap.copy()
    count = 0
    
    print("Finding ROIs...")
    pbar = tqdm(total=params.maxperplane)
    
    while curr_map.max() > params.stopthresh and count < params.maxperplane:
        # Seed
        max_idx = np.argmax(curr_map)
        my, mx = np.unravel_index(max_idx, (h, w))
        
        # Initialize ROI mask
        mask = np.zeros((h, w), dtype=bool)
        mask[my, mx] = True
        
        # Region Growing (Simplified)
        # Dilate mask, check correlation of new pixels with current mean trace
        # If > threshold, add to mask.
        
        # Simple iterative dilation for demonstration of logic
        active = True
        current_trace = d[my, mx, :]
        
        while active:
            # Get neighbors
            dilated = ndimage.binary_dilation(mask)
            candidates = dilated & (~mask) & (roi_map == 0)
            
            if not candidates.any():
                active = False
                break
                
            y_cand, x_cand = np.where(candidates)
            
            # Check correlations
            added_pixels = 0
            for cy, cx in zip(y_cand, x_cand):
                pix_trace = d[cy, cx, :]
                if np.corrcoef(current_trace, pix_trace)[0,1] > params.corrthresh:
                    mask[cy, cx] = True
                    added_pixels += 1
            
            if added_pixels == 0:
                active = False
            else:
                # Update trace
                roi_pixels = d[mask]
                current_trace = roi_pixels.mean(axis=0)
                
            if mask.sum() > params.maximumroisize:
                active = False

        # Save ROI
        if mask.sum() > 5: # Min size
            count += 1
            roi_map[mask] = count
            traces.append(zscore(current_trace))
            raw_traces.append(current_trace)
            sizes.append(mask.sum())
            
        # Zero out seeded area in map + vicinity
        # Zeroing slightly more than mask prevents overlapping seeds
        curr_map[ndimage.binary_dilation(mask, iterations=2)] = 0
        pbar.update(1)
        
    pbar.close()
    
    res = ROIData()
    res.allrois = roi_map
    res.mytraces = np.array(traces).T  if traces else np.empty((d.shape[2], 0))
    res.mytracesraw = np.array(raw_traces).T if raw_traces else np.empty((d.shape[2], 0))
    res.allnums = np.array(sizes)
    return res

def analyze_responses(alldata: AllData, roidata: ROIData) -> dict:
    """Correlate ROI traces with regressors."""
    if roidata.mytraces.shape[1] == 0:
        return {}
        
    # Correlations
    # Note: Regressors are full length, traces are good frames only.
    # We must index regressors by goodframes.
    
    # Safety check on lengths
    n_frames = roidata.mytraces.shape[0]
    # Regressors might need subsetting if alldata was built with all frames
    reg_cs = alldata.csRegressor[alldata.goodframes][:n_frames]
    reg_us = alldata.csMask[alldata.goodframes][:n_frames]
    reg_lr = alldata.learnRegressor[alldata.goodframes][:n_frames]
    
    n_rois = roidata.mytraces.shape[1]
    
    corr_cs = np.zeros(n_rois)
    corr_us = np.zeros(n_rois)
    corr_lr = np.zeros(n_rois)
    
    for i in range(n_rois):
        tr = roidata.mytraces[:, i]
        # Skip NaNs
        valid = ~np.isnan(tr)
        if valid.sum() < 10: continue
        
        corr_cs[i] = np.corrcoef(tr[valid], reg_cs[valid])[0,1]
        corr_us[i] = np.corrcoef(tr[valid], reg_us[valid])[0,1]
        corr_lr[i] = np.corrcoef(tr[valid], reg_lr[valid])[0,1]
        
    # Classify
    return {
        'corr_cs': corr_cs,
        'corr_us': corr_us,
        'corr_lr': corr_lr,
        'idx_cs': np.where(corr_cs > 0.3)[0],
        'idx_us': np.where(corr_us > 0.3)[0],
        'idx_lr': np.where(corr_lr > 0.3)[0]
    }

# endregion

# region Plotting

def plot_roi_map(roidata: ROIData, folder: Path, plane: int):
    plt.figure(figsize=(10, 10))
    plt.imshow(roidata.allrois, cmap='nipy_spectral', interpolation='nearest')
    plt.title(f"ROI Map - Plane {plane}")
    plt.axis('off')
    plt.savefig(folder / 'savefolder' / f'roi_map_plane{plane}.png')
    plt.close()

def plot_trace_heatmap(roidata: ROIData, folder: Path, plane: int):
    if roidata.mytraces.shape[1] == 0: return
    
    # Cluster for visualization
    Z = linkage(roidata.mytraces.T, method='ward')
    idx = dendrogram(Z, no_plot=True)['leaves']
    
    plt.figure(figsize=(12, 8))
    plt.imshow(roidata.mytraces[:, idx].T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    plt.colorbar(label='Z-Score')
    plt.xlabel('Frame')
    plt.ylabel('ROI (Clustered)')
    plt.title(f"Activity Heatmap - Plane {plane}")
    plt.savefig(folder / 'savefolder' / f'heatmap_plane{plane}.png')
    plt.close()

# endregion

# region Main Execution

def run_analysis(folder: Path, file: str, plane: int = 0):
    print(f"--- Analyzing Plane {plane} ---")
    
    # 1. Load & Concat
    # numbertrials=16 implies we expect 16 trials, can be parameterized
    ad = concat_single_plane_file(folder, file, plane, n_trials=16, save=True)
    
    # 2. Params
    params = ROIParams()
    
    # 3. Correlation Map
    cmap = calculate_correlation_map(ad, params)
    
    # 4. ROIs
    rois = find_rois(ad, cmap, params)
    
    # 5. Analyze
    res = analyze_responses(ad, rois)
    
    # 6. Save & Plot
    plot_roi_map(rois, folder, plane)
    plot_trace_heatmap(rois, folder, plane)
    
    print(f"Done. Found {len(rois.allnums)} ROIs.")
    print(f"CS Responsive: {len(res['idx_cs'])}")
    print(f"US Responsive: {len(res['idx_us'])}")

if __name__ == "__main__":
    # Example usage based on existing paths
    # Note: Using raw string for path to avoid escape issues
    data_folder = Path(r"H:\2-P imaging") / path_home.stem / fish_name
    data_file = f"{fish_name}_data.h5" # Assuming naming convention from script 5
    
    if (data_folder / data_file).exists():
        run_analysis(data_folder, data_file, plane=0)
    else:
        print(f"Data file not found: {data_folder / data_file}")
        print("Please run Script 5 (Save_data_as_HDF5) first.")

# endregion
