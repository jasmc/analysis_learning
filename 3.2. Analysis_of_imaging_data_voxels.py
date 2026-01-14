"""
3.2 Voxel-Level Imaging Data Analysis

This script performs spatial downsampling (voxelization) and temporal response analysis on motion-corrected 2-photon calcium imaging data.

Workflow:
1. Loads motion-corrected data from pickle file (output of script 2.x)
2. Segments each trial into three temporal periods (Pre-CS, CS-US, Post-US) based on experimental protocol
3. Applies frame alignment using previously computed shift corrections
4. **Spatial Downsampling (Voxelization):**
   - Downsamples imaging data using block reduction (mean aggregation)
   - Block size determined by p.voxel_bin_size parameter
   - Upsamples back to original resolution for visualization and mapping
5. Computes mean fluorescence for each temporal period (excluding bad frames)
6. Calculates normalized activity responses:
   - ΔF/F₀ = (F - F₀) / (F₀ + softthresh)
   - Applied to CS-US vs Pre-CS and Post-US vs Pre-CS comparisons
   - Full time-series ΔF/F₀ also calculated for movie generation
7. Generates analysis outputs:
   - Time-series plots of ΔF/F₀ for all voxels (Fig 7)
   - Multi-page TIFF movies showing positive responses with CS timing indicators
   - Composite movies per plane showing all trials side-by-side

Key Differences from Pixel-Level Analysis (3.1):
- Spatial downsampling via block_reduce reduces computational load
- No gaussian smoothing applied (already averaged in voxels)
- Generates time-series plots and movies for quality control
- Creates composite multi-trial visualization movies

Parameters:
- p.voxel_bin_size: Spatial binning factor (e.g., 2 → 2x2 pixel blocks)
- softthresh: Baseline offset (100) to prevent division by near-zero values
- activity_threshold_viz: Threshold (0.7) for displaying activity in colored overlays
- indicator_color: Visual marker color ('lime') for CS timing in movies

Output Files:
- Pickle file with voxelized trial data including:
  - pre_cs_mean, cs_us_mean, post_us_mean: temporal period averages
  - cs_us_vs_pre, post_us_vs_pre: normalized response maps
  - deltaFOverF: full time-series normalized responses
  - anatomy: normalized template image
- PNG: Time-series plots (7. Negative responses to CS.png)
- Multi-page TIFFs: Individual trial activity movies with CS indicators
- Multi-page TIFFs: Composite movies per plane showing all trials

Movie Features:
- Green rectangle indicator shows CS presentation timing
- Side-by-side trial layout for cross-trial comparison
- Inferno colormap for positive activity visualization
"""

#* Imports

# %%
# region Imports

##   
# region Imports
import pickle
from copy import deepcopy
from importlib import reload
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import tifffile
from PIL import Image
from skimage.measure import block_reduce
from tqdm import tqdm

#* Load custom functions and classes
import my_classes as c
# import my_experiment_specific_variables as spec_var
import my_functions_imaging as fi
import my_parameters as p
from my_general_variables import *
from PIL import Image, ImageDraw
import numpy as np

# Save all data in a single pickle file.
# Anatomical stack images and imaging data are median filtered.


#* Imports


# endregion

reload(fi)
reload(c)
reload(p)

#* Settings
##    Settings
# region Settings

# %matplotlib ipympl

pio.templates.default = "plotly_dark"

pd.set_option("mode.copy_on_write", True)
pd.set_option("compute.use_numba", True)
pd.set_option("compute.use_numexpr", True)
pd.set_option("compute.use_bottleneck", True)
#endregion



#* Paths
##   
# region Paths
path_home = Path(r'E:\2024 09_Delay 2-P 4 planes JC neurons')
# Path(r'D:\2024 10_Delay 2-P 15 planes bottom part')
# Path(r'D:\2024 03_Delay 2-P 15 planes top part')
# Path(r'E:\2024 10_Delay 2-P single plane')
# Path(r'E:\2024 10_Delay 2-P 15 planes ca8 neurons')
# Path(r'E:\2024 09_Delay 2-P zoom in multiplane imaging')

path_results_save = Path(r'F:\Results (paper)') / path_home.stem

# fish_list = [f for f in (path_home / 'Imaging').iterdir() if fi.is_dir()]
# fish_names_list = [fi.stem for f in fish_list]

fish_name = r'20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241017_01_delay_2p-4_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# r'20241016_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240417_01_delay_2p-4_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
# '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'
# '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'
# '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf'



fish_ID = '_'.join(fish_name.split('_')[:2])


behavior_path_home = path_home / 'Tail'
imaging_path_home = path_home / 'Neurons' / fish_name

behavior_path_save = path_results_save / 'Tail'
results_figs_path_save = path_results_save / 'Neurons' / fish_name

whole_data_path_save = Path(r'H:\2-P imaging') / path_home.stem / fish_name


path_pkl_after_motion_correction = whole_data_path_save / (fish_ID + '_2. After motion correction.pkl')

path_pkl_responses = whole_data_path_save / (fish_ID + '_3. Responses.pkl')

			# # Create the directory if it does not exist
			# (results_path / 'ROI traces').mkdir(parents=True, exist_ok=True)
			# (results_path / 'Videos with deltaF over F').mkdir(parents=True, exist_ok=True)
			# (results_path / 'Anatomy with ROIs').mkdir(parents=True, exist_ok=True)
			# os.makedirs(whole_data_path_save, exist_ok=True)




border_size = 2*p.voxel_bin_size
correlation_thr = 0.3
median_thr = 5

softthresh=100

step = 2 # Process trials in pairs


if 'delay' in fish_name:
	interval_between_cs_onset_us_onset = 9  # s
elif 'trace' in fish_name:
	interval_between_cs_onset_us_onset = 13  # s
elif 'control' in fish_name:
	interval_between_cs_onset_us_onset = 9  # s
else:
	interval_between_cs_onset_us_onset = None





#* Load the data before motion correction.
# region Load the data before motion correction
with open(path_pkl_after_motion_correction, 'rb') as file:
	all_data = pickle.load(file)



print('Analyzing fish: ', fish_name)


# endregion


shape_ = all_data.planes[0].trials[0].images.shape[1:]

# x_black_box_beg, x_black_box_end, y_black_box_beg, y_black_box_end = all_data.black_box

# x_black_box_beg = shape_[0] - 20
# x_black_box_end = shape_[0] - 5
# y_black_box_beg = shape_[1] - 20
# y_black_box_end = shape_[1] - 5

# ALL_DATA = deepcopy(all_data)
# all_data = deepcopy(ALL_DATA)




#* Split data into 3 periods: before cs, from cs onset to us onset and after us onset.
for plane_i, plane in tqdm(enumerate(all_data.planes)):
	for trial_i, trial in enumerate(plane.trials):

		protocol = trial.protocol

		cs_beg_time = protocol.loc[protocol['CS beg']!=0, 'Time (ms)'].values[0]
		cs_end_time = protocol.loc[protocol['CS end']!=0, 'Time (ms)'].values[0]
				
		if (us_beg_time := protocol.loc[protocol['US beg']!=0, 'Time (ms)']).empty:
			us_beg_time = cs_beg_time + interval_between_cs_onset_us_onset*1000
			us_end_time = us_beg_time + 100
		else:
			us_beg_time = us_beg_time.values[0]
			us_end_time = protocol.loc[protocol['US end']!=0, 'Time (ms)'].values[0]

		trial_images = trial.images.copy().fillna(0)

		trial_images['mask before cs'] = trial_images['Time (ms)'] < cs_beg_time
		trial_images['mask cs-us'] = (trial_images['Time (ms)'] > cs_beg_time) & (trial_images['Time (ms)'] < us_beg_time)
		trial_images['mask after us'] = trial_images['Time (ms)'] > us_beg_time

		trial_images.values = fi.align_frames(trial_images.to_numpy(), np.stack([trial_images['shift correction in X'].to_numpy(), trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))

		trial_images_downsampled = block_reduce(trial_images, block_size=(1, p.voxel_bin_size, p.voxel_bin_size), func=np.mean, cval=0)
		# Upsample the downsampled data back to original resolution for visualization or mapping
		# Note: Subsequent calculations (means) are still based on the downsampled data unless trial_images.values is reassigned.
		upsampled_values = np.repeat(np.repeat(trial_images_downsampled, p.voxel_bin_size, axis=1), p.voxel_bin_size, axis=2)

		target_shape = trial_images.values.shape[1:]

		upsampled_values = upsampled_values[:, :target_shape[0], :target_shape[1]]
		trial_images.values = upsampled_values

		trial.images = trial_images.copy()
	
		# plt.imshow(np.mean(trial_images, axis=0), interpolation='none')

		trial.pre_cs_mean = trial_images.sel({'Time (ms)':(trial_images['mask before cs']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')
		trial.cs_us_mean = trial_images.sel({'Time (ms)':(trial_images['mask cs-us']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')
		trial.post_us_mean = trial_images.sel({'Time (ms)':(trial_images['mask after us']) & (trial_images['mask good frames'])}).mean(dim='Time (ms)')


		pre_cs_data = trial.pre_cs_mean.values

		trial.cs_us_vs_pre = (trial.cs_us_mean.values - pre_cs_data) / (pre_cs_data + softthresh)
		trial.post_us_vs_pre = (trial.post_us_mean.values - pre_cs_data) / (pre_cs_data + softthresh)
		
		#* Calculate deltaF/F0 for each trial		
		trial.deltaFOverF = (trial.images - pre_cs_data) / (pre_cs_data + softthresh)


		trial.anatomy = fi.normalize_image(all_data.planes[plane_i].trials[trial_i].template_image, (0.01,0.99)) / 10


		all_data.planes[plane_i].trials[trial_i] = trial


	# 	break
	# break



#endregion



#* Plot the deltaF/F0 for each trial
fig, axs = plt.subplots(len(all_data.planes), len(plane.trials), figsize=(15, 32), squeeze=False, sharex=True, sharey=False)
row_min_y = 0
row_max_y = 1

for plane_i, plane in tqdm(enumerate(all_data.planes)):


	# First pass to determine y-limits for the row
	for trial_i, trial in enumerate(plane.trials):
		protocol = trial.protocol

		cs_beg_time = protocol.loc[protocol['CS beg']!=0, 'Time (ms)'].values[0]
		cs_end_time = protocol.loc[protocol['CS end']!=0, 'Time (ms)'].values[0]
				
		if (us_beg_time := protocol.loc[protocol['US beg']!=0, 'Time (ms)']).empty:
			us_beg_time = cs_beg_time + interval_between_cs_onset_us_onset*1000
			us_end_time = us_beg_time + 100
		else:
			us_beg_time = us_beg_time.values[0]
			us_end_time = protocol.loc[protocol['US end']!=0, 'Time (ms)'].values[0]


		pre_cs_data = trial.pre_cs_mean.values

		trial_deltaFOverF = (trial.images - pre_cs_data) / (pre_cs_data + softthresh)

# zscore

		for x, y in zip(range(trial_deltaFOverF.shape[1]), range(trial_deltaFOverF.shape[2])):
			axs[plane_i, trial_i].plot(trial_deltaFOverF.values[:,x,y], 'k')

		# Find the index closest to cs_beg_time
		time_coords = trial_deltaFOverF['Time (ms)'].values
		cs_beg_index = np.abs(time_coords - cs_beg_time).argmin()
		cs_end_index = np.abs(time_coords - cs_end_time).argmin()

		# Add the vertical line
		axs[plane_i, trial_i].axvline(cs_beg_index, color='green', linestyle='--', label='CS onset')
		axs[plane_i, trial_i].axvline(cs_end_index, color='green', linestyle='--', label='CS offset')

		current_min = np.nanmin(trial_deltaFOverF.values)
		current_max = np.nanmax(trial_deltaFOverF.values)
		if not np.isinf(current_min):
			row_min_y = min(row_min_y, current_min)
		if not np.isinf(current_max):
			row_max_y = max(row_max_y, current_max)


	# Add some padding to the limits
	y_padding = (row_max_y - row_min_y) * 0.05 
	row_ylim = (row_min_y - y_padding, row_max_y + y_padding)
	if np.isinf(row_ylim[0]) or np.isinf(row_ylim[1]): # Fallback if limits are infinite
		row_ylim = (0, 1) # Or some other sensible default

	for trial_i, trial in enumerate(plane.trials):
		
		axs[plane_i, trial_i].set_ylim(row_ylim)
		
	# 	break
	# break

	# all_trial_images = [trial.images for trial in plane.trials]
fig.subplots_adjust(hspace=0.01, wspace=0)
fig.suptitle(f'{fish_name}/nNegative responses to CS', fontsize=10, y=0.91)
fig.savefig(results_figs_path_save / '7. Negative responses to CS.png', dpi=600, facecolor='white', bbox_inches='tight')


# trial_deltaFOverF.max(dim='Time (ms)')




#* Make a movie of the deltaF/F0 for each trial

##* CS positive response

trial_deltaFOverF_trials_list = [[] for _ in range(len(all_data.planes))]

for plane_i, plane in enumerate(all_data.planes):
	for trial_i, trial in enumerate(plane.trials):

		response = trial.deltaFOverF.values

		response = np.where(response > 0, response, 0)
		response = [fi.normalize_image(response[i,:,:], quantiles=(0.3, 1)) for i in range(response.shape[0])]
		# response_thr = np.quantile(response_frame, 0.99)
		
		trial_positive_deltaFOverF = [fi.add_colors_to_world_improved_2(trial.anatomy, resp, colormap='inferno', activity_threshold=0.3, alpha=1) for resp in response]


		# axs[plane_i,trial_i].axis('off')
		# plt.imshow(fi.add_colors_to_world_improved_2(anatomy/5, color_frame_original, colormap='inferno', activity_threshold=0.5, alpha=1), interpolation='none')

		break
	break

tifffile = []
for resp in trial_positive_deltaFOverF:
	tifffile.append(Image.fromarray(resp))

tiff_path = results_figs_path_save / 'CS_positive_response_multipage.tiff'
tifffile[0].save(tiff_path, save_all=True, append_images=tifffile[1:])


plt.imshow(trial_positive_deltaFOverF[100], interpolation='none')



# 		trial_deltaFOverF_trials_list[plane_i].append(trial.deltaFOverF)
	
# trial_deltaFOverF_trials = [xr.concat(trial_deltaFOverF_trials_list[plane_i], dim='Time (ms)') for plane_i in range(len(all_data.planes))]


for plane_i, plane in enumerate(all_data.planes):

len(trial_deltaFOverF_trials)




#* Save the images as a multipage tiff
#region Save the images as a multipage tiff
tifffile = []
for trial_i, trial in enumerate(all_data.planes[2].trials):
	tifffile.append(Image.fromarray(trial.cs_positive_response))

# Save tifffile as a multipage tiff
tiff_path = imaging_path_ / 'cs_positive_response_multipage new 1.tiff'
tifffile[0].save(tiff_path, save_all=True, append_images=tifffile[1:])



# Load the TIFF file
tiff_path = imaging_path_ / 'cs_positive_response_multipage new 1.tiff'


# Label a specific frame in the TIFF stack
frame_index = 0  # Change this to the index of the frame you want to label
labeled_frame = tifffile[1]

# Display the labeled frame
plt.imshow(labeled_frame, cmap='gray')
plt.title(f'Labeled Frame {1}')
plt.colorbar()
plt.show()







#* Make a movie of the deltaF/F0 for each trial
# Define rectangle coordinates (top-right corner)
width, height = all_data.planes[0].trials[0].images.values.shape[1:]
square_size = 10 # Adaptive size, min 10px
margin = 3 # Small margin from the edge
rect_coords = [
	(width - square_size - margin, margin), # Top-left corner of rectangle
	(height - margin, square_size + margin)  # Bottom-right corner of rectangle
]


# Make sure ImageDraw is imported

for plane_i, plane in tqdm(enumerate(all_data.planes), desc="Generating Trial Movies"):
	for trial_i, trial in enumerate(plane.trials):


		# --- Get CS indices for this trial ---
		protocol = trial.protocol
		cs_beg_time = protocol.loc[protocol['CS beg']!=0, 'Time (ms)'].values[0]
		cs_end_time = protocol.loc[protocol['CS end']!=0, 'Time (ms)'].values[0]

		# Handle cases where US might be missing in protocol (using interval)
		if (us_beg_time_arr := protocol.loc[protocol['US beg']!=0, 'Time (ms)']).empty:
			us_beg_time = cs_beg_time + interval_between_cs_onset_us_onset*1000
		else:
			us_beg_time = us_beg_time_arr.values[0]

		# Use the deltaFOverF calculated earlier
		if not hasattr(trial, 'deltaFOverF'):
			print(f"Warning: deltaFOverF not found for Plane {plane_i}, Trial {trial_i}. Skipping movie generation.")
			continue

		time_coords = trial.deltaFOverF['Time (ms)'].values
		cs_beg_index = np.abs(time_coords - cs_beg_time).argmin()
		cs_end_index = np.abs(time_coords - cs_end_time).argmin()
		# --- ---

		# --- Calculate positive deltaF/F response frames for visualization ---
		response = trial.deltaFOverF.values
		response = np.where(response > 0, response, 0)
		response = [fi.normalize_image(response[i,:,:], quantiles=(0, 1)) for i in range(response.shape[0])]
		# response_thr = np.quantile(response_frame, 0.99)
		trial_positive_deltaFOverF = [fi.add_colors_to_world_improved_2(trial.anatomy, resp, colormap='inferno', activity_threshold=0.7, alpha=1) for resp in response]

		# --- ---
		# plt.imshow(trial_positive_deltaFOverF[100], interpolation='none')

		# --- Generate and save TIFF for this trial with indicator ---
		trial_tifffile_pil = []
		for frame_idx, frame_np in enumerate(trial_positive_deltaFOverF):
			# Ensure frame is uint8 for PIL
			# if frame_np.dtype != np.uint8:
			# 	 frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
			img = Image.fromarray(frame_np)
			# Ensure image is RGB for drawing color rectangle
			if img.mode != 'RGB':
				img = img.convert('RGB')
			draw = ImageDraw.Draw(img)

			# Draw green rectangle if within CS period
			if cs_beg_index <= frame_idx <= cs_end_index:

				try:
					draw.rectangle(rect_coords, fill="lime", width=3, ) # Green rectangle, 3px thick
				except Exception as e:
					print(f"Error drawing rectangle on frame {frame_idx} for P{plane_i} T{trial_i}: {e}")


			trial_tifffile_pil.append(img)

		# Save the list of PIL images as a multipage TIFF
		if trial_tifffile_pil: # Check if list is not empty
			tiff_path = results_figs_path_save / f'Trial_Movies_DeltaFOverF Plane_{plane_i}_Trial_{trial_i}_CS_positive_response_indicated.tiff'
			try:
				trial_tifffile_pil[0].save(
					tiff_path,
					save_all=True,
					append_images=trial_tifffile_pil[1:],
					duration=100, # Duration per frame in ms (adjust as needed, e.g., 1000 / frame_rate)
					loop=0       # Loop indefinitely (0) or play once (None or 1)
				)
			except Exception as e:
				print(f"Error saving TIFF {tiff_path}: {e}")
		# --- ---
	# break








#* Create one movie per plane, showing all trials side-by-side

# --- Configuration ---
activity_threshold_viz = 0.7 # Threshold for showing color overlay in visualization
indicator_color = "lime"
indicator_width = 2
padding_color = (0, 0, 0) # Black padding
padding_width = 2 # Width of padding between trial images

# --- Pre-calculate all individual trial frames and timings ---
n_planes = len(all_data.planes)
n_trials_per_plane = [len(plane.trials) for plane in all_data.planes]
all_trial_frames_data = [[[] for _ in range(n_trials_per_plane[p])] for p in range(n_planes)]
all_trial_cs_indices = [[() for _ in range(n_trials_per_plane[p])] for p in range(n_planes)]
max_frames_per_plane = [0] * n_planes
frame_height, frame_width = 0, 0 # Initialize frame dimensions

print("Preprocessing trial data for composite movies...")
for plane_i, plane in tqdm(enumerate(all_data.planes), total=n_planes, desc="Preprocessing Planes"):
	current_max_frames = 0
	for trial_i, trial in enumerate(plane.trials):

		# --- Get CS indices ---
		protocol = trial.protocol
		cs_beg_time = protocol.loc[protocol['CS beg']!=0, 'Time (ms)'].values[0]
		cs_end_time = protocol.loc[protocol['CS end']!=0, 'Time (ms)'].values[0]

		time_coords = trial.deltaFOverF['Time (ms)'].values
		cs_beg_index = np.abs(time_coords - cs_beg_time).argmin()
		cs_end_index = np.abs(time_coords - cs_end_time).argmin()
		all_trial_cs_indices[plane_i][trial_i] = (cs_beg_index, cs_end_index)

		# --- Calculate positive deltaF/F response frames ---
		response = trial.deltaFOverF.values
		response = np.where(response > 0, response, 0) # Keep only positive
		# Normalize only if there's positive activity to avoid division by zero or normalizing zeros
		response_normalized = [(fi.normalize_image(response[i,:,:], quantiles=(0, 1)) * 255).astype(np.uint8) if np.any(response[i,:,:] > 0) else np.zeros_like(response[i,:,:], dtype=np.uint8) for i in range(response.shape[0])]

		# Generate colored frames using the normalized response
		trial_frames_np = [fi.add_colors_to_world_improved_2(trial.anatomy, resp_norm, colormap='inferno', activity_threshold=int(activity_threshold_viz * 255), alpha=1) for resp_norm in response_normalized]


		all_trial_frames_data[plane_i][trial_i] = trial_frames_np
		current_max_frames = max(current_max_frames, len(trial_frames_np))

		# Store frame dimensions (assuming they are consistent across trials and planes)
		if frame_height == 0 and len(trial_frames_np) > 0:
			# Ensure anatomy is loaded if needed for dimensions
			if hasattr(trial, 'anatomy') and trial.anatomy is not None:
				frame_height, frame_width = trial.anatomy.shape[:2] # Get H, W from anatomy
			elif len(trial_frames_np) > 0:
				frame_height, frame_width, _ = trial_frames_np[0].shape # Get H, W, C from generated frame
			else:
				print(f"Warning: Could not determine frame dimensions for Plane {plane_i}, Trial {trial_i}.")
				# Set default or skip? For now, let it potentially fail later if still 0.

	max_frames_per_plane[plane_i] = current_max_frames

	# break




# Check if frame dimensions were determined
if frame_height == 0 or frame_width == 0:
	raise ValueError("Could not determine frame dimensions. Check input data and preprocessing steps.")


# --- Define indicator rectangle coordinates (relative to individual trial frame) ---
square_size = max(10, min(frame_width // 10, frame_height // 10)) # Adaptive size
margin = 3
rect_coords = [
	(frame_width - square_size - margin, margin), # Top-left
	(frame_width - margin, square_size + margin)  # Bottom-right
]

# --- Create and Save Composite Movie for Each Plane ---
print(f"\nGenerating {n_planes} composite movies...")
for plane_i in range(n_planes):
	n_cols = n_trials_per_plane[plane_i]
	max_frames = max_frames_per_plane[plane_i]

	if n_cols == 0 or max_frames == 0:
		print(f"Skipping Plane {plane_i}: No trials or frames found.")
		continue

	print(f"Processing Plane {plane_i} ({n_cols} trials, {max_frames} frames)...")

	# Calculate full image dimensions for this plane's movie
	composite_width = n_cols * frame_width + (n_cols - 1) * padding_width
	composite_height = frame_height

	composite_frames_pil_list = []

	for frame_idx in tqdm(range(max_frames), desc=f"  Generating Frames P{plane_i}", leave=False):
		# Create a new blank composite frame (black background)
		composite_img = Image.new('RGB', (composite_width, composite_height), padding_color)
		draw_composite = ImageDraw.Draw(composite_img) # Draw padding lines if needed

		for trial_i in range(n_cols):
			trial_frames = all_trial_frames_data[plane_i][trial_i]
			cs_beg_idx, cs_end_idx = all_trial_cs_indices[plane_i][trial_i]

			# Get the correct frame, use last frame if index is out of bounds, or black if no frames
			if not trial_frames:
				# Use a black frame if this trial had no data
				current_frame_np = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
			else:
				safe_frame_idx = min(frame_idx, len(trial_frames) - 1)
				current_frame_np = trial_frames[safe_frame_idx]

			# Ensure frame is uint8 RGB
			if current_frame_np.dtype != np.uint8:
				current_frame_np = current_frame_np.astype(np.uint8)
			if current_frame_np.shape != (frame_height, frame_width, 3):
				# Handle potential grayscale or other inconsistencies if necessary
				# This might happen if add_colors... fails or anatomy is grayscale
				print(f"Warning: Unexpected frame shape {current_frame_np.shape} for P{plane_i} T{trial_i} F{frame_idx}. Attempting conversion.")
				if len(current_frame_np.shape) == 2: # Grayscale
					current_frame_np = cv2.cvtColor(current_frame_np, cv2.COLOR_GRAY2RGB)
				elif current_frame_np.shape[2] == 4: # RGBA
					current_frame_np = cv2.cvtColor(current_frame_np, cv2.COLOR_RGBA2RGB)
				# Resize if necessary (shouldn't be needed if preprocessing is consistent)
				if current_frame_np.shape[:2] != (frame_height, frame_width):
					current_frame_np = cv2.resize(current_frame_np, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)


			# Convert to PIL Image
			img = Image.fromarray(current_frame_np)

			# Draw indicator if within CS period for this specific trial
			if cs_beg_idx != -1 and cs_beg_idx <= frame_idx <= cs_end_idx: # Check for valid indices
				draw_individual = ImageDraw.Draw(img)
				try:
					draw_individual.rectangle(rect_coords, outline=indicator_color, width=indicator_width)
				except Exception as e:
					print(f"Error drawing rectangle on P{plane_i} T{trial_i} F{frame_idx}: {e}")

			# Calculate position to paste this trial's image onto the composite frame
			paste_x = trial_i * (frame_width + padding_width)
			paste_y = 0 # Single row layout
			composite_img.paste(img, (paste_x, paste_y))

			# Draw vertical padding line (optional)
			if trial_i < n_cols - 1:
				line_x = paste_x + frame_width + padding_width // 2
				# draw_composite.line([(line_x, 0), (line_x, composite_height)], fill=padding_color, width=padding_width)


		composite_frames_pil_list.append(composite_img)

	# --- Save the composite movie for this plane using tifffile ---
	if composite_frames_pil_list:
		movie_save_path = results_figs_path_save / f'{fish_ID}_Plane_{plane_i}_Composite_Activity_Movie.tiff'
		print(f"  Saving composite movie for Plane {plane_i} to: {movie_save_path}")

		# Convert PIL images back to a NumPy stack
		np_frames = []
		for img_pil in composite_frames_pil_list:
			frame_np = np.array(img_pil) # Should already be RGB uint8
			np_frames.append(frame_np)

		# Stack frames along the time axis (axis=0)
		movie_stack = np.stack(np_frames, axis=0) # Shape: (frames, height, width, channels)

		# Calculate frame duration in seconds for tifffile metadata
		frame_rate = p.frame_rate if hasattr(p, 'frame_rate') and p.frame_rate > 0 else 10 # Default to 10 fps
		duration_sec = 1.0 / frame_rate

		try:
			# Save using tifffile, enabling BigTIFF for large files
			tifffile.imwrite(
				movie_save_path,
				movie_stack,
				# imagej=True, # Make it ImageJ compatible (adds metadata)
				resolution=(1.0, 1.0), # Placeholder resolution
				metadata={'spacing': duration_sec, 'unit': 'sec', 'axes': 'TYXC'}, # Time, Y, X, Channels
				bigtiff=False # Crucial for files potentially > 4GB
			)
			print(f"  Successfully saved movie for Plane {plane_i}.")
		except Exception as e:
			print(f"Error saving TIFF for Plane {plane_i} ({movie_save_path}): {e}")
	else:
		print(f"  No frames generated for Plane {plane_i}, skipping save.")


	# break



print("\nComposite movie generation complete.")


# ... rest of the code ...


# 		np.stack([plane_positive_deltaFOverF_frames, trial_positive_deltaFOverF_frames], axis=1) # Append frames to the plane list

# plt.imshow(trial_positive_deltaFOverF_frames[0], interpolation='none')
# 		# break
# 	# break





plt.imshow(movie_stack[0,:,:])






















# Save the list of PIL images as a multipage TIFF
if trial_tifffile_pil: # Check if list is not empty
	# Ensure results directory exists
	tiff_path = results_figs_path_save / f'Plane_{plane_i}_Trial_{trial_i}_CS_positive_indicated.tiff'
	try:
		trial_tifffile_pil[0].save(
			tiff_path,
			save_all=True,
			append_images=trial_tifffile_pil[1:],
			duration=int(1000 / p.frame_rate) if hasattr(p, 'frame_rate') else 100, # Use frame rate if available
			loop=0       # Loop indefinitely
		)
	except Exception as e:
		print(f"Error saving TIFF {tiff_path}: {e}")
		# --- ---
		# --- ---
	# break












for plane_i, plane in enumerate(all_data.planes):
	for trial_i, trial in enumerate(plane.trials):

		pre_cs_data = trial.pre_cs_mean.values

		trial.cs_us_vs_pre = (trial.cs_us_mean.values - pre_cs_data) / (pre_cs_data + softthresh)
		trial.post_us_vs_pre = (trial.post_us_mean.values - pre_cs_data) / (pre_cs_data + softthresh)
		# trial.cs_us_vs_pre = gaussian_filter(((trial.cs_us_mean.values - pre_cs_data) / (pre_cs_data + softthresh)), sigma=2)
		# trial.post_us_vs_pre = gaussian_filter((trial.post_us_mean.values - pre_cs_data / (pre_cs_data + softthresh)), sigma=2)

		# anatomy = fi.normalize_image(fi.calculate_anatomy(trial_images.values.astype(np.float64), border_size), (0.01,0.99))

		# anatomy = anatomies[trial_i//2]
		# trial.anatomy = fi.scale_slide(all_data.planes[plane_i].trials[trial_i].template_image, p.min_intensity_threshold) / 5
		trial.anatomy = fi.normalize_image(all_data.planes[plane_i].trials[trial_i].template_image, (0.01,0.99)) / 10

		#* Give some colors to the world.
		# color_frame_original = fi.scale_slide(trial.cs_us_vs_pre, p.min_intensity_threshold)
		# color_frame_original.max()

		# plt.imshow(fi.add_colors_to_world(anatomy/5, color_frame_original), interpolation='none')
		# plt.imshow(fi.add_colors_to_world_improved(anatomy, color_frame_original, activity_scaling=2.0, anatomy_brightness=0.5), interpolation='none')


with open(path_pkl_responses, 'wb') as file:
	pickle.dump(all_data, file)