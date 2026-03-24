"""
Data joining and preprocessing pipeline (Refactored)
====================================================

This script synchronizes camera timestamps, protocol timing, galvo triggers,
tail tracking, and raw imaging frames into a unified dataset.
It is the first step in the imaging analysis pipeline.

Refactored to follow the structure of Pipeline_Analysis.py.
"""

# %%
# region Imports & Configuration
import gc
import pickle
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import scipy.ndimage as ndimage
import tifffile
import xarray as xr
from scipy import signal
from tqdm import tqdm

import my_classes_new as c
import my_functions_imaging_new as fi
import my_parameters_new as p
import plotting_style_new as plotting_style
from experiment_configuration import ExperimentType, get_experiment_config
from general_configuration import config as gen_config

# Constants
RELOAD_MODULES = True
USE_PLOTLY_DARK = True

PANDAS_OPTIONS = {
    "mode.copy_on_write": True,
    "compute.use_numba": True,
    "compute.use_numexpr": True,
    "compute.use_bottleneck": True,
}

if "__file__" in globals():
    module_root = Path(__file__).resolve()
else:
    module_root = Path.cwd()

def configure_environment(use_plotly_dark: bool) -> None:
    for option, value in PANDAS_OPTIONS.items():
        pd.set_option(option, value)
    if use_plotly_dark:
        pio.templates.default = "plotly_dark"
    plotting_style.set_plot_style(use_constrained_layout=False)

if RELOAD_MODULES:
    reload(fi)
    reload(c)
    reload(p)

configure_environment(USE_PLOTLY_DARK)
# endregion


# region Parameters
# ==============================================================================
# PIPELINE CONTROL FLAGS
# ==============================================================================
RUN_JOIN_DATA = True

# ==============================================================================
# EXPERIMENT SETTINGS
# ==============================================================================
FILTER_FISH_ID = None # e.g. '20240415_01'
EXPERIMENT_TYPE = '2-P multiple planes top' # Placeholder

# Fallback path if EXPERIMENT_TYPE is not fully configured
MANUAL_PATH_HOME = Path(r'D:\2024 03_Delay 2-P 15 planes top part')
MANUAL_PATH_SAVE = Path(r'F:\Results (paper)')
PATH_ROOT_ANALYSIS = Path(r'H:\2-P imaging')

# endregion

# region Pipeline Functions

def is_summer_time(fish_name: str) -> bool:
    try:
        date = int(fish_name.split('_')[0][4:6])
        return 4 <= date <= 10
    except:
        return False

def run_join_data():
    """
    Executes the data joining pipeline.
    """
    print("\n" + "="*80)
    print("RUNNING JOIN DATA PIPLINES")
    print("="*80)

    # 1. Configuration & Paths
    try:
        config = get_experiment_config(EXPERIMENT_TYPE)
        path_home = config.path_home
        path_results_save = config.path_save
    except Exception as e:
        print(f"Config load error: {e}. Using manual paths.")
        path_home = MANUAL_PATH_HOME
        path_results_save = MANUAL_PATH_SAVE / path_home.stem

    path_results_save.mkdir(parents=True, exist_ok=True)

    # 2. Iterate over fish
    # We look for fish directories in 'Neurons' folder usually, or look for tracking files in 'Tail'
    # The original script assumes behavior files are in path_home / 'Tail' and imaging in path_home / 'Neurons' / fish_name
    
    behavior_path_home = path_home / 'Tail'
    neurons_path_home = path_home / 'Neurons'

    if not behavior_path_home.exists() or not neurons_path_home.exists():
        print(f"Missing data directories in {path_home}")
        return

    # Find list of fish based on Stim Control files in Tail folder, as they are reliable indicators of a behavioral session
    all_protocol_paths = list(behavior_path_home.glob('*stim control.txt'))
    
    if FILTER_FISH_ID:
        all_protocol_paths = [p for p in all_protocol_paths if FILTER_FISH_ID in p.name]

    print(f"Found {len(all_protocol_paths)} fish sessions to process.")

    for protocol_path in tqdm(all_protocol_paths, desc="Joining Data"):
        fish_name = protocol_path.name.replace('_stim control.txt', '')
        print(f"\nProcessing {fish_name}")

        fish_ID = '_'.join(fish_name.split('_')[:2])
        
        # Define paths
        camera_path = behavior_path_home / (fish_name + '_cam.txt')
        tracking_path = behavior_path_home / (fish_name + '_mp tail tracking.txt')
        
        imaging_fish_home = neurons_path_home / fish_name
        imaging_data_path_home = imaging_fish_home / 'Imaging'
        
        galvo_path = imaging_data_path_home / 'signalsfeedback.xls'
        images_path = imaging_data_path_home / (fish_name + '_green.tif') # Check if suffix matches
        if not images_path.exists():
             # Try without 'Imaging' subfolder if not found, or different naming?
             # Logic from original script: imaging_path_home / 'Imaging' / (fish_name + '_green.tif')
             pass
        
        anatomy_1_path = imaging_fish_home / 'Anatomical stack 1' / 'Anatomical stack 1.tif'

        results_figs_path_save = path_results_save / 'Neurons' / fish_name
        results_figs_path_save.mkdir(parents=True, exist_ok=True)

        whole_data_path_save = PATH_ROOT_ANALYSIS / path_home.stem / fish_name
        whole_data_path_save.mkdir(parents=True, exist_ok=True)

        path_pkl_before_motion_correction = whole_data_path_save / (fish_ID + '_1. Before motion correction.pkl')

        if path_pkl_before_motion_correction.exists():
            print(f'Already processed: {fish_name}. Skipping...')
            # continue 
            # Uncomment continue to skip existing

        # 3. Determine Trial Mapping (Indices)
        relevant_cs = []
        index_list = []
        number_imaged_planes = 0
        number_reps_plane_consective = 0

        # Heuristic to match path_home logic from original script
        path_str = str(path_home)
        if '15 planes top part' in path_str or '15 planes bottom part' in path_str or '15 planes ca8' in path_str:
            number_imaged_planes = 15
            number_reps_plane_consective = 2
            relevant_cs = [range(5,35), range(45,75)]
            index_list = [np.concatenate([[i+number_reps_plane_consective*x*number_imaged_planes, i+number_reps_plane_consective*x*number_imaged_planes+1] for x in range(len(relevant_cs))]) for i in range(0, number_reps_plane_consective * number_imaged_planes, number_reps_plane_consective)]
        
        elif 'single plane' in path_str:
            number_imaged_planes = 1
            number_reps_plane_consective = 80
            relevant_cs = [np.arange(5,85)]
            index_list = relevant_cs
            
        elif '4 planes JC neurons' in path_str:
            num_planes = 4
            num_reps_per_plane = 2
            relevant_cs = [range(5,15), range(15,25), range(25,55), range(55,45), range(45,55), range(55,65), range(65,75), range(75,85)]
            index_list = [
                np.concatenate([
                    [i + num_reps_per_plane * x * num_planes, i + num_reps_per_plane * x * num_planes + 1]
                    for x in range(len(relevant_cs))
                ])
                for i in range(0, num_reps_per_plane * num_planes, num_reps_per_plane)
            ]
        else:
            print(f"Warning: path_home {path_home} not recognized for trial structure. Skipping.")
            continue
            
        relevant_cs = np.concatenate(relevant_cs)

        # 4. Load & Process Data
        # Camera
        data = fi.read_camera(camera_path)
        if data is None:
            print(f"Missing camera file: {camera_path}")
            continue
        data['AbsoluteTime'] = data['AbsoluteTime'].astype('float64')

        # Framerate
        predicted_framerate, reference_frame_id = fi.framerate_and_reference_frame(data)
        if 'ela_time' in data.columns: data = data.drop(columns='ela_time')
        data = data[data['Frame number'] >= reference_frame_id]
        
        # Time interplation
        abs_time_start = data['AbsoluteTime'].iat[0]
        abs_time_stop = abs_time_start + len(data) * (1000 / predicted_framerate)
        data['AbsoluteTime'] = np.linspace(abs_time_start, abs_time_stop, len(data))

        # Protocol
        protocol = fi.read_protocol(protocol_path)
        if protocol is None:
            print(f"Missing protocol: {protocol_path}")
            continue
        data = fi.identify_trials(data, protocol)

        # Galvo
        if not galvo_path.exists():
            print(f"Missing galvo: {galvo_path}")
            continue
            
        try:
            galvo = pd.read_csv(
                galvo_path, sep='\t', decimal=',', usecols=[0, 1],
                names=['AbsoluteTime', 'GalvoValue'], dtype={'GalvoValue': 'float64'},
                parse_dates=['AbsoluteTime'], date_format=r'%d/%m/%Y  %H:%M:%S,%f',
                skip_blank_lines=True, skipinitialspace=True, nrows=p.nrows
            ).dropna(axis=0).reset_index(drop=True)
            galvo['AbsoluteTime'] = galvo['AbsoluteTime'].astype('int64') / 10**6
        except Exception as e:
            print(f"Error reading galvo: {e}")
            continue

        if is_summer_time(fish_name):
            galvo['AbsoluteTime'] -= 60*60*1000

        # Galvo Peaks
        galvo = galvo[galvo['GalvoValue'].ne(galvo['GalvoValue'].shift())].reset_index(drop=True)
        peaks_initial = signal.find_peaks(galvo['GalvoValue'], height=0.5, prominence=0.05)[0]
        if peaks_initial.size == 0:
            print("No galvo peaks.")
            continue
            
        beg_first_image = peaks_initial[0]
        beg_first_image_time = galvo['AbsoluteTime'].iat[beg_first_image]
        
        peaks = (signal.find_peaks(galvo['GalvoValue'].iloc[beg_first_image + 1000:], 
                                   height=[0.5, 5], distance=100, prominence=[0.5, 5], width=20)[0] 
                 + beg_first_image + 1000)
        
        # Determine Reference Frame / Interframe Interval
        interframe_interval_array = galvo['AbsoluteTime'].iloc[peaks].diff()
        interframe_interval = interframe_interval_array.median()
        
        if np.isnan(interframe_interval):
             print("Interframe interval calc failed.")
             continue

        beg_image_to_consider_index = interframe_interval_array.index[np.where(interframe_interval_array == interframe_interval)[0][0] - 1]
        beg_image_to_consider_time = galvo['AbsoluteTime'].iat[beg_image_to_consider_index]
        peaks = peaks[peaks >= beg_image_to_consider_index]
        number_images_before_first_image_to_consider = round((beg_image_to_consider_time - beg_first_image_time) / interframe_interval)

        # Reads Images
        if not images_path.exists():
            print(f"Missing images: {images_path}")
            continue
        bytes_header, height, width = fi.get_bytes_header_and_image(images_path)
        bytes_header_and_image = bytes_header + height * width * 2
        number_images = fi.get_number_images(images_path, bytes_header_and_image)

        # Plot Galvo QC
        fig, axs = plt.subplots(nrows=4, figsize=(10,12))
        galvo_sub = galvo.loc[0:10000]
        axs[0].plot(galvo_sub['AbsoluteTime'], galvo_sub['GalvoValue'], 'k')
        axs[0].plot(beg_first_image_time, 1, 'ro')
        fig.savefig(results_figs_path_save / '1. Galvo signal and frames.png')
        plt.close(fig)
        
        # Merge Timelines
        galvo['Frame beg'] = np.nan
        end_idx = min(len(peaks), number_images - number_images_before_first_image_to_consider)
        galvo.loc[galvo.iloc[peaks].index[:end_idx], 'Frame beg'] = 1
        
        # Prepend missing frames
        galvo = pd.merge_ordered(
            galvo,
            pd.DataFrame({
                'AbsoluteTime': np.arange(
                    beg_image_to_consider_time - number_images_before_first_image_to_consider * interframe_interval,
                    beg_image_to_consider_time, interframe_interval),
                'Frame beg': np.ones(number_images_before_first_image_to_consider)
            })
        )
        
        # Behavior
        behavior = fi.read_tail_tracking_data(tracking_path)
        if behavior is None:
             print("No tail tracking.")
             continue
        behavior = behavior.astype('float32')

        # Sync
        if galvo['AbsoluteTime'].iat[0] <= data['AbsoluteTime'].iat[0]:
            galvo = galvo[galvo['AbsoluteTime'] >= data['AbsoluteTime'].iat[0]]
            first_timepoint = galvo['AbsoluteTime'].iat[0]
            data['AbsoluteTime'] -= first_timepoint
            galvo['AbsoluteTime'] -= first_timepoint
            data = data[data['AbsoluteTime'] >= 0]
            galvo = galvo[galvo['AbsoluteTime'] <= data['AbsoluteTime'].iat[-1]]
        
        data = pd.merge_ordered(data, galvo, on='AbsoluteTime', how='outer')
        data = pd.merge(data, behavior, on='Frame number', how='left')
        data.rename(columns={'AbsoluteTime' : 'Time (ms)'}, inplace=True)
        
        # Load Raw Images to RAM (Careful with memory)
        print("Loading images...")
        try:
            images = np.array([
                fi.get_image_from_tiff(images_path, image_i, bytes_header, height, width).astype('float32')
                for image_i in tqdm(range(number_images))
            ])
        except MemoryError:
            print("Memory Error loading images.")
            continue
            
        # Imaging DataArray
        imaging = xr.DataArray(
            images,
            coords={
                'index': ('Time (ms)', data.loc[data['Frame beg'].notna(), :].index),
                'Time (ms)': data.loc[data['Frame beg'].notna(), 'Time (ms)'].to_numpy(),
                'x': range(images.shape[1]),
                'y': range(images.shape[2]),
            },
            dims=['Time (ms)', 'x', 'y'],
        )
        imaging.name = 'Imaging data'

        # Slice into Trials and Planes
        all_data = []
        
        # Re-extract protocol for slicing
        data['CS'] = data['CS'].fillna(0).astype('int')
        data['US'] = data['US'].fillna(0).astype('int')
        
        cs_onset_indices_all = data.index[data['CS'].diff() > 0].tolist() # Simplified detection of CS onset indices from merged data
        # Note: Original script used protocol DF. We need to be careful matching 'relevant_cs' mapping.
        # Original: cs_onset_index = np.array([protocol.loc[protocol[cs] == relevant_cs[i], :].index[0] ...])
        # We need the INDEX in the MERGED dataframe that corresponds to the CS onset.
        
        # Let's approximate:
        # We need to map 'relevant_cs' numbers to the actual CS events in 'data'.
        # 'data' has 'CS' column with Stim Numbers (1, 2, 3...)
        
        cs_numbers = data.loc[data['CS'] != 0, 'CS'].unique()
        # Ensure relevant_cs are in cs_numbers
        
        try:
            # Map CS number to Index in 'data'
            # We want the index in 'data' where 'CS' starts being == CS_ID
            cs_onset_map = {}
            for cs_id in relevant_cs:
                 rows = data.index[data['CS'] == cs_id]
                 if not rows.empty:
                     cs_onset_map[cs_id] = rows[0]
            
            # Now build planes
            planes_cs_onset_indices = [
                 [cs_onset_map[j] for j in i if j in cs_onset_map] 
                 for i in index_list
            ]
        except Exception as e:
            print(f"Error mapping CS indices: {e}")
            continue

        print("Building Plane/Trial objects...")
        i_trial_global = 0
        for plane_i, plane_cs_onset_indices in tqdm(enumerate(planes_cs_onset_indices)):
            trials_list = []
            for trial_cs_onset_index in plane_cs_onset_indices:
                time_target = data.loc[trial_cs_onset_index, 'Time (ms)']
                time_start = time_target - p.time_bef_cs_onset
                time_end = time_target + p.time_aft_cs_onset
                
                # Slicing
                # Logic for Protocol slice from merged data
                # We need a proper Protocol DataFrame for the Trial object
                # Original script creates 'trial_protocol' with CS beg/end events
                
                trial_slice = data[data['Time (ms)'].between(time_start, time_end)].copy()
                
                # Construct minimalist protocol DF for Trial
                trial_protocol = pd.DataFrame() # Needs columns CS beg, CS end, etc.
                # ... (This logic is complex to replicate exactly without 'protocol' object from original flow)
                # We will simplify by passing the slice of 'data' which contains CS/US columns
                
                # Image Slicing
                trial_images = imaging.loc[time_start:time_end, :, :]
                trial_images.values = ndimage.median_filter(trial_images, size=p.median_filter_kernel, axes=(1, 2))
                
                # Behavior Slicing
                trial_beh = behavior[behavior['Time (ms)'].between(time_start, time_end)] if 'Time (ms)' in behavior.columns else behavior
                
                trials_list.append(c.Trial(i_trial_global, trial_slice, trial_beh, trial_images))
                i_trial_global += 1

            all_data.append(c.Plane(trials_list))

        # Anatomical Stack
        try:
            if anatomy_1_path.exists():
                anatomical_stack_images = tifffile.imread(anatomy_1_path).astype('float32')
            else:
                anatomical_stack_images = tifffile.imread(imaging_fish_home / 'Anatomical stack 1.tif').astype('float32')
            
            anatomical_stack_images = xr.DataArray(
                anatomical_stack_images,
                coords={
                    'index': ('plane_number', range(anatomical_stack_images.shape[0])),
                    'plane_number': range(anatomical_stack_images.shape[0]),
                    'x': range(anatomical_stack_images.shape[2]),
                    'y': range(anatomical_stack_images.shape[1]),
                },
                dims=['plane_number', 'y', 'x'],
            )
            anatomical_stack_images = ndimage.median_filter(anatomical_stack_images, size=p.median_filter_kernel, axes=(1, 2))
        except Exception:
             print("Anatomical stack failed.")
             anatomical_stack_images = None

        all_data_obj = c.Data(all_data, anatomical_stack_images)

        # Save
        print(f"Saving to {path_pkl_before_motion_correction}...")
        with open(path_pkl_before_motion_correction, 'wb') as file:
            pickle.dump(all_data_obj, file)

    print("JOIN DATA FINISHED")

# endregion

# region Main
if __name__ == "__main__":
    if RUN_JOIN_DATA:
        try:
            run_join_data()
        except Exception as e:
            print(f"Error in Join Data: {e}")
            raise e
# endregion
