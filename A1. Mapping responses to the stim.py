"""
Mapping responses to the stim (Refactored)
==========================================

This script checks for responses (CS and US) in the imaging data.
It follows the structure of Pipeline_Analysis.py.

Usage:
    - Set the parameters in the 'Parameters' region.
    - Run the script.
"""
# %%
# region Imports & Configuration
import gc
import pickle
from copy import deepcopy
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import tifffile
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import my_classes_new as c
import my_functions_imaging_new as fi
import my_parameters_new as p
import plotting_style_new as plotting_style
from experiment_configuration import ExperimentType, get_experiment_config
from general_configuration import config as gen_config

# Set pandas options
pd.set_option("mode.copy_on_write", True)
pd.set_option("compute.use_numba", True)
pd.set_option("compute.use_numexpr", True)
pd.set_option("compute.use_bottleneck", True)

# Apply plotting style
pio.templates.default = "plotly_dark"
plotting_style.set_plot_style(use_constrained_layout=False)

if "__file__" in globals():
    module_root = Path(__file__).resolve()
else:
    module_root = Path.cwd()

# Reload custom modules if needed
reload(fi)
reload(c)
reload(p)
# endregion


# region Parameters
# ==============================================================================
# PIPELINE CONTROL FLAGS
# ==============================================================================
RUN_MAPPING = True

# ==============================================================================
# EXPERIMENT SETTINGS
# ==============================================================================
# Filter to process specific fish. Set to None to process all found fish.
FILTER_FISH_ID = '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf' 

# Define the experiment type to load configuration
# Note: Ensure this experiment type is implemented in experiment_configuration.py
# If NOT implemented, you may need to manually define path_home below.
EXPERIMENT_TYPE = '2-P multiple planes top' 

# Fallback path if EXPERIMENT_TYPE is not fully configured for this specific dataset
# This preserves the path from the original script
MANUAL_PATH_HOME = Path(r'D:\2024 03_Delay 2-P 15 planes top part')
MANUAL_PATH_SAVE = Path(r'F:\Results (paper)')

# ==============================================================================
# ANALYSIS PARAMETERS
# ==============================================================================
BORDER_SIZE = 2 * p.voxel_bin_size
CORRELATION_THR = 0.3
MEDIAN_THR = 5
SOFTTHRESH = 50
RANGE_COLOR_THR = 30

# Path to the base directory where H drive results are stored
PATH_ROOT_Analysis = Path(r'H:\2-P imaging')
# endregion


# region Pipeline Functions

def run_response_mapping():
    """
    Executes the response mapping analysis.
    
    1. Loads the analysis pickle file (Before motion correction or Analysis 1).
    2. Maps CS and US responses.
    3. Generates and saves visualization plots.
    """
    print("\n" + "="*80)
    print("RUNNING RESPONSE MAPPING")
    print("="*80)

    # 1. Configuration & Paths
    try:
        config = get_experiment_config(EXPERIMENT_TYPE)
        path_home = config.path_home
        path_results_save = config.path_save
    except Exception as e:
        print(f"Could not load config for {EXPERIMENT_TYPE}: {e}")
        print(f"Using manual paths.")
        path_home = MANUAL_PATH_HOME
        path_results_save = MANUAL_PATH_SAVE / path_home.stem

    if not path_home.exists():
        print(f"Warning: path_home {path_home} does not exist.")

    # Find fish folders
    # Assuming standard structure: path_home / 'Neurons' / fish_name or similar, 
    # but original script looked in path_home directly or hardcoded fish names.
    # We will try to find relevant files based on the file structure implied.
    
    # In the original script, fish_name was hardcoded. Here we try to iterate if FILTER_FISH_ID is set,
    # or find all subdirectories.
    
    # Check if we should search for fish directories
    potential_fish_dirs = []
    if (path_home / 'Neurons').exists():
         potential_fish_dirs = [x for x in (path_home / 'Neurons').iterdir() if x.is_dir()]
    elif (path_home / 'Imaging').exists():
         potential_fish_dirs = [x for x in (path_home / 'Imaging').iterdir() if x.is_dir()]
    
    # If explicit match requested
    if FILTER_FISH_ID:
        # If FILTER_FISH_ID looks like a full fish name, allow exact match
        potential_fish_dirs = [d for d in potential_fish_dirs if FILTER_FISH_ID in d.name]
        
        # If no dirs found, maybe FILTER_FISH_ID is the name itself and we just need to construct the path
        if not potential_fish_dirs:
             # Try constructing path assuming standard layout
             potential_fish_dirs = [path_home / 'Neurons' / FILTER_FISH_ID]

    if not potential_fish_dirs:
        print(f"No fish directories found for ID: {FILTER_FISH_ID} in {path_home}")
        return

    for fish_dir in tqdm(potential_fish_dirs, desc="Processing Fish"):
        fish_name = fish_dir.name
        print(f"Analyzing fish: {fish_name}")

        # Construct paths
        fish_id_short = '_'.join(fish_name.split('_')[:2])
        behavior_path_home = path_home / 'Tail'
        imaging_path_home = path_home / 'Neurons' / fish_name # refined assumption
        
        results_figs_path_save = path_results_save / 'Neurons' / fish_name
        results_figs_path_save.mkdir(parents=True, exist_ok=True)
        
        whole_data_path_save = PATH_ROOT_Analysis / path_home.stem / fish_name
        
        # Define input pickle path
        # Original script used '_analysis 1.pkl' or '_before motion correction.pkl'
        path_pkl_analysis_1 = whole_data_path_save / f"{fish_name}_analysis 1.pkl"
        path_pkl_bmc = whole_data_path_save / f"{fish_id_short}_before motion correction.pkl"

        file_to_load = None
        if path_pkl_analysis_1.exists():
            file_to_load = path_pkl_analysis_1
        elif path_pkl_bmc.exists():
            file_to_load = path_pkl_bmc
        
        if not file_to_load:
            print(f"Could not find analysis pickle file for {fish_name}")
            print(f"Checked: \n {path_pkl_analysis_1} \n {path_pkl_bmc}")
            continue

        print(f"Loading data from: {file_to_load}")
        try:
            with open(file_to_load, 'rb') as file:
                all_data = pickle.load(file)
        except Exception as e:
            print(f"Failed to load data: {e}")
            continue

        # 2. Main Analysis Logic
        
        # Determine delay vs trace vs control intervals
        interval_between_CS_onset_US_onset = 9 # default
        if 'delay' in fish_name:
            interval_between_CS_onset_US_onset = 9
        elif 'trace' in fish_name:
            interval_between_CS_onset_US_onset = 13
        elif 'control' in fish_name:
            interval_between_CS_onset_US_onset = 9

        # Example analysis: Mean image of a specific trial (e.g., trial 3 in plane 0)
        if len(all_data.planes) > 0 and len(all_data.planes[0].trials) > 3:
            mean_img = all_data.planes[0].trials[3].images.mean('Time (ms)')
            # fig, ax = plt.subplots()
            # ax.imshow(mean_img)
            # fig.savefig(results_figs_path_save / 'Mean_image_trial_3.png')
            # plt.close(fig)

        # Check for CA8 logic (black box)
        shape_ = all_data.planes[0].trials[0].images.shape[1:]
        if 'ca8' in str(path_home):
            x_black_box_beg = shape_[0] - 20
            x_black_box_end = shape_[0] - 5
            y_black_box_beg = shape_[1] - 20
            y_black_box_end = shape_[1] - 5
        else:
            x_black_box_beg = 330
            x_black_box_end = 345
            y_black_box_beg = 594
            y_black_box_end = 609

        ALL_DATA_COPY = deepcopy(all_data)

        # Split data into 3 periods and color code
        print("Processing planes and trials for color mapping...")
        for plane_i, plane in enumerate(all_data.planes):
            for trial_i, trial in enumerate(plane.trials):
                trial_images = trial.images.copy()

                # Anatomy / Template
                anatomy = trial.template_image.copy()
                if np.median(anatomy) > 0:
                    anatomy /= np.median(anatomy * RANGE_COLOR_THR)
                anatomy = np.clip(anatomy, 0, 1)
                trial.anatomy_channel = anatomy

                # CS Positive Response
                if hasattr(trial, 'CS_US_vs_pre'):
                    color_frame_original = np.clip(trial.CS_US_vs_pre, 0, 1)
                    trial.CS_positive_response = fi.add_colors_to_world(anatomy, color_frame_original)
                else:
                    # Fallback if attribute missing
                    trial.CS_positive_response = anatomy

                all_data.planes[plane_i].trials[trial_i] = trial

        # Plotting Composite
        if len(all_data.planes) > 0:
            plane = all_data.planes[0] # Use first plane for sizing
            
            # Sort planes by position if available
            plane_numbers = np.zeros((len(all_data.planes), len(plane.trials)))
            for p_i in range(len(all_data.planes)):
                for t_i in range(len(all_data.planes[p_i].trials)):
                     # Assuming 'position_anatomical_stack' exists
                     if hasattr(all_data.planes[p_i].trials[t_i], 'position_anatomical_stack'):
                        plane_numbers[p_i, t_i] = all_data.planes[p_i].trials[t_i].position_anatomical_stack
                     else:
                        plane_numbers[p_i, t_i] = p_i

            # Create big figure
            n_rows = len(all_data.planes)
            n_cols = len(plane.trials)
            
            print(f"Generating summary figure: {n_rows} rows x {n_cols} cols")
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1, n_rows * 2), squeeze=False)
            
            for p_i in range(n_rows):
                # Use sorted position if valid, else p_i
                position = int(plane_numbers[p_i, 0]) if p_i < plane_numbers.shape[0] else p_i
                if position >= n_rows: position = p_i

                for t_i in range(n_cols):
                    if t_i < len(all_data.planes[p_i].trials):
                        img = all_data.planes[p_i].trials[t_i].CS_positive_response
                        axs[position, t_i].imshow(img)
                    axs[position, t_i].set_xticks([])
                    axs[position, t_i].set_yticks([])

            fig.subplots_adjust(hspace=0.05, wspace=0.02)
            fig.tight_layout()
            
            save_path_fig = results_figs_path_save / 'CS_positive_response_summary.png'
            fig.savefig(save_path_fig, dpi=300)
            print(f"Saved summary figure to {save_path_fig}")
            plt.close(fig)

        # Rotated Tiff Saving Example
        # (Adapted from original script logic)
        # Note: This logic assumes specific trial structure (pre, train, test)
        # We will wrap it in a try/except or check
        
        try:
             # Example: taking one plane's trials and saving as multipage tiff
            plane_idx = 3 if len(all_data.planes) > 3 else 0
            trials = all_data.planes[plane_idx].trials
            
            tiff_images = []
            for trial in trials:
                # Rotate 90 deg
                img_rot = np.rot90(trial.CS_positive_response)
                # Convert to PIL for text
                pil_img = Image.fromarray((img_rot * 255).astype(np.uint8))
                tiff_images.append(pil_img)
            
            if tiff_images:
                tiff_save_path = results_figs_path_save / 'CS_positive_response_run_rotated.tiff'
                tiff_images[0].save(tiff_save_path, save_all=True, append_images=tiff_images[1:])
                print(f"Saved rotated multipage tiff to {tiff_save_path}")

        except Exception as e:
            print(f"Skipping tiff generation: {e}")

    print("RESPONSE MAPPING FINISHED")

# endregion


# region Main
if __name__ == "__main__":
    if RUN_MAPPING:
        try:
            run_response_mapping()
        except Exception as e:
            print(f"Error in Response Mapping: {e}")
            raise e
# endregion