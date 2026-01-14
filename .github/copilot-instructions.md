# Copilot Instructions for Analysis Learning Codebase

This repository contains a Python-based data analysis pipeline for 2-photon calcium imaging and behavioral data in zebrafish. The workflow is structured around sequential scripts and shared configuration modules.

## Project Architecture & Workflow

### 1. Sequential Pipeline
The analysis is driven by numbered scripts that must be executed in order. Do not try to combine them into a single execution unless requested.
- **Preprocessing:** `0.2_Prepocessing.py` (Behavioral data, tail tracking)
- **Data Joining:** `1.Join_all_data.py`
- **Motion Correction:** `2.Motion_correction.py` / `2.Motion_correction_Suite2p.py`
- **Analysis:** `3.x` scripts (Pixel/Voxel analysis), `4.Activity_maps.py`
- **Export:** `5.Save_data_as_HDF5.py`

### 2. Configuration Management
Configuration is decentralized across three key files. Always check these before modifying logic:
- **`my_paths.py`**: Defines the *current* experiment context (`path_home`, `fish_name`). This is the primary switch for changing datasets.
- **`my_parameters.py`**: Contains analysis thresholds (e.g., `light_percentage_increase_thr`, `total_motion_thr`) and filter settings.
- **`my_experiment_specific_variables_NEW.py`**: Defines the metadata structure for experiments (trials, blocks, phases, conditions).

### 3. Shared Modules
- **`my_classes.py`**: Core data structures (`Experiment`, `Condition`).
- **`my_functions_imaging.py`**: Heavy lifting for image processing and visualization.
- **`my_functions_behavior.py`**: Behavioral analysis logic.
- **`my_general_variables.py`**: Global constants imported via `*` (be aware of namespace pollution).

## Coding Conventions & Patterns

- **Module Reloading:** Scripts frequently use `importlib.reload()` (e.g., `reload(fi)`, `reload(c)`) to support interactive development. Preserve this pattern when modifying scripts.
- **Path Handling:** Paths are often constructed relative to `path_home` defined in `my_paths.py`. Use `pathlib.Path` for all file operations.
- **Naming:** Custom modules are consistently prefixed with `my_`.
- **Data Structures:**
  - **`experiments_info`** (in `my_experiment_specific_variables_NEW.py`): A nested dictionary defining the experiment structure.
  - **Output Directory Structure:** Defined in `my_functions_imaging.create_folders()`. Respect this hierarchy (e.g., `Processed data/`, `Lost frames/`).

## Key Dependencies
- **Imaging:** `suite2p`, `tifffile`, `opencv-python` (`cv2`), `scikit-image`
- **Data Analysis:** `numpy`, `pandas`, `xarray`, `h5py`
- **Visualization:** `matplotlib`, `plotly`

## Developer Tips
- **Context Switching:** To switch experiments, the user modifies `my_paths.py`. Do not hardcode paths in scripts; reference `my_paths.path_home`.
- **Global Variables:** Be cautious of variables imported from `my_general_variables.py`. If a variable seems undefined, check there first.
- **Interactive Execution:** The code is designed to be run interactively (e.g., in VS Code Interactive Window or Jupyter). Scripts often have `# %%` cell markers.
