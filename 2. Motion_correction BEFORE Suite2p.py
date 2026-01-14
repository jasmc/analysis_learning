"""
2. Motion Correction (Custom Implementation)

This script performs rigid and non-rigid motion correction on 2-photon calcium imaging data
using a custom correlation-based registration algorithm.

Workflow:
1. Loads raw imaging data from pickle file (output of script 1.Join_all_data.py)
2. Computes reference template for each plane:
   - Uses median or mean of high-contrast frames
   - Applies gaussian smoothing for robust template
3. Within-trial motion correction:
   - Computes phase correlation or cross-correlation between frames and template
   - Calculates rigid (x,y) shifts for frame alignment
   - Optionally applies non-rigid corrections for tissue deformation
4. Between-trial alignment:
   - Aligns trial templates to a common reference
   - Ensures spatial consistency across experimental sessions
5. Quality control:
   - Identifies bad frames based on correlation thresholds
   - Flags frames with excessive motion (> maxshift)
   - Calculates total motion per trial for exclusion criteria
6. Generates motion-corrected template images per trial
7. Saves corrected data and shift vectors to pickle file

Key Parameters (from my_parameters.py):
- xy_movement_allowed: Maximum allowed shift as fraction of frame size
- total_motion_thr: Threshold for flagging high-motion trials
- median_filter_kernel: Spatial smoothing for templates
- correlation_threshold: Minimum correlation for frame acceptance

Motion Correction Features:
- Phase correlation for sub-pixel accuracy
- Iterative template refinement
- Bidirectional scan correction (optional)
- ROI-based registration to exclude artifacts

Output Structure:
- Trial objects with attributes:
  - template_image: motion-corrected reference
  - shift_correction: [x, y] displacement per frame
  - mask_good_frames: boolean mask excluding bad frames
  - correlation_scores: frame-to-template similarity

Output File:
- Pickle file: {fish_ID}_2. After motion correction.pkl

Note: For Suite2p-based registration, use 2.Motion_correction_Suite2p.py instead.
This custom implementation provides more control over registration parameters
and is optimized for multiplane imaging with large FOVs.
"""

# Save all data in a single pickle file.
# Anatomical stack images and imaging data are median filtered.


#* Imports

##   
# region Imports
import os
import pickle
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import xarray as xr
from suite2p.registration import register
from tqdm import tqdm

refImg = register.pick_initial_reference(ops)

#* Load custom functions and classes
import my_classes as c
# import my_experiment_specific_variables as spec_var
import my_functions_imaging as fi
import my_parameters as p
from my_general_variables import *
from my_paths import fish_name, path_home

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
# path_home = Path(r'D:\2024 10_Delay 2-P 15 planes ca8 neurons')
# Path(r'D:\2024 09_Delay 2-P 4 planes JC neurons')
# Path(r'D:\2024 10_Delay 2-P 15 planes bottom part')
# Path(r'D:\2024 03_Delay 2-P 15 planes top part')
# Path(r'D:\2024 10_Delay 2-P single plane')
# Path(r'D:\2024 09_Delay 2-P zoom in multiplane imaging')

path_results_save = Path(r'F:\Results (paper)') / path_home.stem

# fish_list = [f for f in (path_home / 'Imaging').iterdir() if fi.is_dir()]
# fish_names_list = [fi.stem for f in fish_list]

fish_name = r'20241015_03_delay_2p-9_mitfaminusminus,ca8e1bgcamp6s_6dpf'
# '20241002_02_delay_2p-1_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf'
# r'20241022_01_delay_2p-10_mitfaminusminus,ca8e1bgcamp6s_5dpf'
# '20241015_01_delay_2p-7_mitfaminusminus,ca8e1bgcamp6s_6dpf'
# '20241008_03_delay_2p-6_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241007_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# r'20241013_03_control_2p-3_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20241013_01_control_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240930_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241014_01_trace_2p-4_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241014_03_trace_2p-6_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241010_01_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241008_02_delay_2p-5_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241008_01_delay_2p-4_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241007_03_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'

# '20241017_01_delay_2p-4_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# r'20241016_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240417_01_delay_2p-4_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240416_01_delay_2p-3_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf'


# '20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'  not fluroescent
# '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# # '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf'


# '20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'

# '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'


# '20241013_01_control_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241013_02_control_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241009_03_delay_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'

# '20241024_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
# '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240415_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'

# 20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf

# '20241015_03_delay_2p-9_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf'


# '20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'



fish_ID = '_'.join(fish_name.split('_')[:2])


behavior_path_home = path_home / 'Tail'
imaging_path_home = path_home / 'Neurons' / fish_name

behavior_path_save = path_results_save / 'Tail'
results_figs_path_save = path_results_save / 'Neurons' / fish_name

whole_data_path_save = Path(r'H:\2-P imaging') / path_home.stem / fish_name
os.makedirs(whole_data_path_save, exist_ok=True)

path_pkl_before_motion_correction = whole_data_path_save / (fish_ID + '_1. Before motion correction.pkl')

path_pkl_after_motion_correction = whole_data_path_save / (fish_ID + '_2. After motion correction.pkl')






# for fish_i, fish_name in enumerate(fish_names_list):

# 	try:

# imaging_path_ = imaging_path / 'Imaging'

if path_pkl_after_motion_correction.exists():

	print('Already preprocessed: ', fish_name)
	print(path_pkl_after_motion_correction)
	
	# continue

print('Analyzing fish: ', fish_name)









			# #!!!!!!
			# trial = all_data.planes[0].trials[0]


			# path_ = path_home / fish_name / 'test.h5'
			# os.makedirs(os.path.dirname(path_), exist_ok=True)

			# path_2 = path_home / fish_name / 'test2.h5'
			# os.makedirs(os.path.dirname(path_2), exist_ok=True)


			# xr.DataArray(trial.behavior,
			# 			coords={'index': ('Time (ms)', trial.behavior.index), 'Time (ms)': trial.behavior['Time (ms)'].to_numpy()}, dims=['Time (ms)', 'cols'])



			# ds = xr.Dataset(dict(images=trial.images, behavior=xr.DataArray(trial.behavior,
			# 			coords={'index': ('Time (ms)', trial.behavior.index), 'Time (ms)': trial.behavior['Time (ms)'].to_numpy()}, dims=['Time (ms)', 'cols'])))



			# trial.images.to_netcdf(path_2, mode='a', group='images', format='NETCDF4', engine='netcdf4', encoding={'Imaging data': {'zlib': True, 'complevel': 4, 'chunksizes': (trial.images.shape[0],np.ceil(trial.images.shape[1]/8),np.ceil(trial.images.shape[2]/8))}})

			# trial.images.dims

			# a = trial.behavior.to_xarray().set_coords('Time (ms)')
			# a.name = 'Tracking data'



			# imaging = xr.DataArray(images, coords={'index': ('Time (ms)', data.loc[data['Frame beg'].notna(), :].index), 'Time (ms)': data.loc[data['Frame beg'].notna(), abs_time].to_numpy(), 'x': range(images.shape[1]), 'y': range(images.shape[2])}, dims=['Time (ms)', 'x', 'y'])



			# # .set_dims('Time (ms)')
			# trial.images

			# #!
			# with pd.HDFStore(path_2, complevel=4, complib='zlib') as store:
			# 	store.append('behavior', trial.behavior, expectedrows=len(trial.behavior), append=False, chunksize=len(trial.behavior))

			# 	#! store.select(fish.dataset_key(), where=where_to_query)
			# # pd.read_hdf(self._path, key=fish.dataset_key(), mode='r', complevel=self._compression_level, complib=self._compression_library)



			# # Save the data as an HDF5 file
			# h5_path = path_home / fish_name / (fish_name + '_before_motion_correction.h5')
			# os.makedirs(os.path.dirname(h5_path), exist_ok=True)

			# with h5py.File(h5_path, 'w') as h5_file:
			# 	# Save anatomical stack images with compression
			# 	h5_file.create_dataset('anatomical_stack_images', data=anatomical_stack_images, compression="gzip", compression_opts=4)
				
			# 	planes_group =  h5_file.create_group(f'planes')

			# 	# Save planes data with compression
			# 	for plane_i, plane in enumerate(all_data.planes):
			# 		plane_group = planes_group.create_group(f'plane_{plane_i}')
					
			# 		# Save trials data with compression
			# 		for trial_i, trial in enumerate(plane.trials):
			# 			trial_group = plane_group.create_group(f'trial_{trial_i}')
			# 			trial_group.create_dataset('protocol', data=trial.protocol, compression="gzip", compression_opts=4)
			# 			trial_group.create_dataset('behavior', data=trial.behavior, compression="gzip", compression_opts=4, chunks=(trial.behavior.shape[0],trial.behavior.shape[1]))
			# 			trial_group.create_dataset('images', data=trial.images, compression="gzip", compression_opts=4, chunks=(trial.images.shape[0],np.ceil(trial.images.shape[1]/8),np.ceil(trial.images.shape[2]/8)))


# # Load the HDF5 file

# with h5py.File(h5_path, 'r') as h5_file:

# 	# Load the planes data
# 	planes_group = h5_file['planes']
# 	planes = []
# 	for plane_name in planes_group:
# 		plane_group = planes_group[plane_name]
# 		trials = []
# 		for t_i, trial_name in enumerate(plane_group):
# 			trial_group = plane_group[trial_name]
# 			trial = c.Trial(
# 				trial_number=t_i,
# 				protocol=trial_group['protocol'][()],
# 				behavior=trial_group['behavior'][()],
# 				images=trial_group['images'][()]
# 			)
# 			# if 'mask_good_frames' in trial_group:
# 			# 	trial.mask_good_frames = trial_group['mask_good_frames']
# 			# if 'template_image' in trial_group:
# 			# 	trial.template_image = trial_group['template_image']
# 			# if 'position_anatomical_stack' in trial_group:
# 			# 	trial.position_anatomical_stack = trial_group['position_anatomical_stack']
# 			trials.append(trial)
# 		planes.append(c.Plane(trials=trials))

# 	all_data = c.Data(planes=planes, anatomical_stack=h5_file['anatomical_stack_images'][()])

# # all_data.__dict__.keys()

# # anatomical_stack_images = all_data.anatomical_stack


#* Load the data before motion correction.
# region Load the data before motion correction
with open(path_pkl_before_motion_correction, 'rb') as file:
	all_data = pickle.load(file)




shape_ = all_data.planes[0].trials[0].images.shape[1:]

# x_black_box_beg, x_black_box_end, y_black_box_beg, y_black_box_end = all_data.black_box


if ('ca8' in str(path_home)) | ('4' in str(path_home)):
	
	x_black_box_beg = shape_[0] - 20
	x_black_box_end = shape_[0] - 5
	y_black_box_beg = shape_[1] - 20
	y_black_box_end = shape_[1] - 5
	
elif 'single' in str(path_home):
		
		
	x_black_box_beg = shape_[0] - 10
	x_black_box_end = shape_[0] - 5
	y_black_box_beg = shape_[1] - 10
	y_black_box_end = shape_[1] - 5

else:
	x_black_box_beg = 330
	x_black_box_end = 345
	y_black_box_beg = 594
	y_black_box_end = 609


# x_black_box_beg = 330
# x_black_box_end = 345
# y_black_box_beg = 594
# y_black_box_end = 609



# region Check where dark mask is.

##* Subtract the background from the images.

###* Anatomical stack images.
anatomical_stack_images = all_data.anatomical_stack

anatomical_stack_images = xr.DataArray(anatomical_stack_images, coords={'index': ('plane_number', range(anatomical_stack_images.shape[0])), 'plane_number': range(anatomical_stack_images.shape[0]), 'x': range(anatomical_stack_images.shape[2]), 'y': range(anatomical_stack_images.shape[1])}, dims=['plane_number', 'y', 'x'])

#! The dark mask ("eye mask"), if present, contains more than 0.1% of the pixels in each frame.
####* Consider that the background is 10% of each frame. Subtract it to the images and clip the values to 0.
# The background will be always in more than 10% of the pixels in each frame.
#? or take the mean of each frame and subtract it from the frame?
anatomical_stack_images = anatomical_stack_images - anatomical_stack_images.quantile(0.01, dim=('y', 'x'))
anatomical_stack_images = anatomical_stack_images.clip(0, None)
anatomical_stack_images = anatomical_stack_images.drop_vars('quantile')

plt.imshow(np.mean(anatomical_stack_images, axis=0))
plt.colorbar(shrink=0.5)
plt.savefig(results_figs_path_save / '4. Mean of anatomical stack without background.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# # Create a histogram of the values in all_data.anatomical_stack
# plt.figure()
# plt.hist(anatomical_stack_images.to_numpy().ravel(), bins=500, color='blue', alpha=0.7, range=(0, 100))
# plt.title('Histogram of Anatomical Stack Values')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')
# plt.show()


# plt.imshow(all_data.anatomical_stack[100,-30:,-30:], vmin=0, vmax=500)
# plt.colorbar(shrink=0.5)

###* Imaging images.
for plane_i, plane in enumerate(all_data.planes):
	for trial_i, trial in enumerate(plane.trials):

		# trial_images = trial.images.copy()
		# plt.imshow(np.mean(trial_images, axis=0), vmin=0, vmax=None, interpolation='None')
		# plt.colorbar(shrink=0.5)

		# try:
		trial.images = trial.images - trial.images.quantile(0.01, dim=('y', 'x'))
		trial.images = trial.images.clip(0, None)
		trial.images.drop_vars('quantile')
		all_data.planes[plane_i].trials[trial_i].images = trial.images
		# except:
		# 	pass
	# 	break
	# break
# trial.__dict__.keys()

#! WHEN TO RELY ON THE EYE MASK

# #* Check the black box (mask)
# plt.figure()
# plt.imshow(all_data.planes[0].trials[0].images[100,:,:], vmin=0, vmax=500)
# plt.colorbar(shrink=0.5)

# plt.figure()
# plt.imshow(all_data.planes[0].trials[0].images[100][x_black_box_beg:x_black_box_end, y_black_box_beg: y_black_box_end], vmin=0, vmax=None)
# plt.colorbar(shrink=0.5)

# all_data.planes[0].trials[0].images[100].shape

#endregion





#ToDo this should take the pixel spacing into account!!!
#* Correct motion within and across trials.
# region Correct motion


_, y_dim, x_dim = np.array(anatomical_stack_images.shape)

x_dim = int(x_dim * p.xy_movement_allowed/2)
y_dim = int(y_dim * p.xy_movement_allowed/2)







# ALL_DATA = deepcopy(all_data)
# all_data = deepcopy(ALL_DATA)



for plane_i, plane in tqdm(enumerate(all_data.planes)):

	print('Plane: ', plane_i)

	motions = list(np.zeros(len(plane.trials), dtype='int32'))
	template_images = np.zeros((len(plane.trials), plane.trials[0].images.shape[1], plane.trials[0].images.shape[2]))

	#* Motion correction within trial.
	for trial_i, trial in tqdm(enumerate(plane.trials)):

		# break

		print('Trial: ', trial_i)


		# Crop the images
		# Work with numpy array for efficiency and copy to avoid modifying original data
		trial_images_ = trial.images[:, p.image_crop:-p.image_crop, p.image_crop:-p.image_crop].copy().values

		##* Initial template generation
		# Discard bad frames due to gating of the PMT or other major artifacts
		initial_good_frames_mask = fi.get_good_images_indices(trial_images_, p.light_percentage_increase_thr)
		if not np.any(initial_good_frames_mask):
			print(f"  Warning: No initial good frames found based on PMT gating for Plane {plane_i}, Trial {trial_i}. Using mean image.")
			template_image_ = np.mean(trial_images_, axis=0)
			if np.isnan(template_image_).all(): # Handle case where all frames might be bad/nan
				print(f"  Error: Mean image is all NaN for Plane {plane_i}, Trial {trial_i}. Skipping trial motion correction.")
				# Store placeholders and skip to next trial
				motions.append(np.zeros((trial.images.shape[0], 2), dtype=int))
				template_images[trial_i] = np.zeros_like(trial.images[0].values)
				# plane_numbers[plane_i, trial_i] = -1 # Indicate failure
				continue # Skip to next trial
		else:
			template_image_ = fi.get_template_image(trial_images_[initial_good_frames_mask])

		# Check if initial template is valid
		if template_image_ is None or np.isnan(template_image_).all():
			print(f"  Error: Initial template image is invalid/NaN for Plane {plane_i}, Trial {trial_i}. Using mean image as fallback.")
			template_image_ = np.mean(trial_images_, axis=0)
			if np.isnan(template_image_).all():
				print(f"  Error: Fallback mean image is also NaN for Plane {plane_i}, Trial {trial_i}. Skipping trial motion correction.")
				motions.append(np.zeros((trial.images.shape[0], 2), dtype=int))
				template_images[trial_i] = np.zeros_like(trial.images[0].values)
				# plane_numbers[plane_i, trial_i] = -1
				continue

		best_template_image_ = template_image_.copy()
		# max_sharpness = fi.calculate_sharpness(best_template_image_)
		# print(f"  Initial Template Sharpness: {max_sharpness:.4f}")

		# Initialize motion_ corresponding to the best template found so far
		motion_ = np.zeros((trial_images_.shape[0], 2), dtype=int) # Start with zero motion assumption

		##* Iterative refinement loop
		for counter in tqdm(range(p.number_iterations_within_trial), desc="  Iterative Refinement", leave=False):

			#* Measure the motion of each frame relative to the *current* template image.
			current_motion = fi.measure_motion(trial_images_, template_image_, normalization=None)

			#* Get the total motion (Euclidean distance).
			total_motion = fi.get_total_motion(current_motion)

			#* Identify frames with acceptable motion (below threshold)
			mask_good_motion = total_motion <= p.motion_thr_within_trial

			# Ensure there are enough good frames to create a reliable new template
			min_good_frames = max(5, int(0.1 * trial_images_.shape[0])) # Require at least 5 frames or 10%
			if np.sum(mask_good_motion) < min_good_frames:
				print(f"  Iteration {counter+1}: Not enough frames ({np.sum(mask_good_motion)}) with motion <= {p.motion_thr_within_trial}. Stopping refinement.")
				# Keep the motion calculated in this iteration as the last estimate
				motion_ = current_motion
				break # Exit refinement loop

			#* Align the frames deemed good based on motion threshold, using the *current* motion estimate
			try:
				# Pass the motion corresponding only to the good frames
				aligned_good_frames = fi.align_frames(trial_images_[mask_good_motion], current_motion[mask_good_motion])
			except Exception as e:
				print(f"  Error during frame alignment in iteration {counter+1}: {e}. Stopping refinement.")
				motion_ = current_motion # Keep last motion estimate
				break


			#* Create a candidate template from the newly aligned good frames
			candidate_template_image_ = fi.get_template_image(aligned_good_frames)

			# Check if candidate template is valid
			if candidate_template_image_ is None or np.isnan(candidate_template_image_).all():
				print(f"  Warning: Candidate template image is invalid/NaN in iteration {counter+1}. Keeping previous template.")
				# Keep the previous template and motion, stop refinement
				break

			#* Calculate sharpness of the candidate template
			# current_sharpness = fi.calculate_sharpness(candidate_template_image_)
			# print(f"  Iteration {counter+1}: Candidate Sharpness: {current_sharpness:.4f}") # Optional: print sharpness each iteration

			#* Check for improvement (only update if significantly better to avoid noise)
			# sharpness_improvement_threshold = 1e-4 # Avoid updates for negligible changes
			# if current_sharpness > max_sharpness + sharpness_improvement_threshold:
				# print(f"  Iteration {counter+1}: Sharpness improved ({max_sharpness:.4f} -> {current_sharpness:.4f}). Updating template.")
			best_template_image_ = candidate_template_image_.copy()
			template_image_ = candidate_template_image_.copy() # Use the better template for the next iteration's motion estimation
			# max_sharpness = current_sharpness
			motion_ = current_motion # Store the motion corresponding to this new best template
			# else:
			# 	print(f"  Iteration {counter+1}: Sharpness did not improve sufficiently ({current_sharpness:.4f} <= {max_sharpness:.4f}). Stopping refinement.")
			# 	# Keep the 'motion_' from the iteration that produced the current 'best_template_image_'
			# 	# Exit the loop as further iterations with the same template are unlikely to help
			# 	break

		# After the loop, 'motion_' holds the shifts relative to the 'best_template_image_'
		# Recalculate total motion based on the final 'motion_' estimate for reporting/masking purposes
		total_motion = fi.get_total_motion(motion_)
		template_image_ = best_template_image_ # Ensure the final template is the best one found

		# print(f"  Final Template Sharpness: {max_sharpness:.4f}")

		# axs_comp[plane_i,trial_i].imshow(template_image_, interpolation='None',vmin=np.quantile(template_image_, 0.05) , vmax=np.quantile(template_image_, 0.95))



		#!!!!!!!!!! Discard frames where the motion exceeded the threshold and frames where there was large light changes.
		#* Motion correction across trials.

		mask_good_frames_motion = total_motion <= p.motion_thr_within_trial
		mask_good_frames_no_PMT = fi.get_good_images_indices(trial.images, p.light_percentage_decrease_PMT)
		mask_good_frames = mask_good_frames_motion & mask_good_frames_no_PMT

		template_image_ = fi.get_template_image(fi.align_frames(trial.images[mask_good_frames], motion_))


		#* Frames to ignore due to too much motion (or gating of the PMT, which causes a huge "motion").
		# trial_images = trial.images.values
		# Mask with True where the frames are bad (due to gating of the PMT or motion).
		# mask_good_frames = (~fi.get_good_images_indices_1(aligned_frames)) | (np.where(fi.get_total_motion(motions[trial_i]) > p.motion_thr_from_trial_average, True, False))

		all_data.planes[plane_i].trials[trial_i].images = all_data.planes[plane_i].trials[trial_i].images.assign_coords({'mask good frames' : ('Time (ms)', mask_good_frames)})
		#! all_data.planes[plane_i].trials[trial_i].mask_good_frames = mask_good_frames


		# mask_good_frames = mask_good_frames_motion | mask_good_frames_no_PMT
		motions[trial_i] = motion_
		template_images[trial_i] = template_image_


	#* Motion correction across trials of the same plane.
	# Normalize each template image using the provided function
	# np.stack is used for clarity in creating an array from a list of arrays
	templates = np.stack([fi.normalize_image(image) for image in template_images])
		
	motion_ = fi.measure_motion(templates[1: , p.image_crop_:-p.image_crop_, p.image_crop_:-p.image_crop_], templates[0, p.image_crop_:-p.image_crop_, p.image_crop_:-p.image_crop_], normalization='phase')

	# Calculate total motion for the across-trial shifts
	# motion_[k] represents the shift needed to align template k+1 with template 0.
	total_motion_across_trials = fi.get_total_motion(motion_)

	# Apply across-trial correction, checking the threshold for each trial relative to trial 0
	for trial_i in range(len(plane.trials)):

		# Trial 0 is the reference, its motion array doesn't need across-trial correction added.
		if trial_i > 0:
			# Check if the calculated motion to align this trial's template (trial_i)
			# with the first trial's template (trial 0) exceeds the threshold.
			# The relevant shift is stored in motion_[trial_i-1].
			if total_motion_across_trials[trial_i-1] <= p.motion_thr_across_trials:
				# If motion is acceptable, add the across-trial shift
				# to the existing within-trial shifts for this trial.
				motions[trial_i] = motions[trial_i] + motion_[trial_i-1]
			else:
				# If motion is excessive, print a warning and do not apply the across-trial correction.
				# motions[trial_i] will retain only the within-trial shifts.
				print(f"  Warning: Excessive across-trial motion detected for Plane {plane_i}, Trial {trial_i} relative to Trial 0 ({total_motion_across_trials[trial_i-1]:.2f} > {p.motion_thr_across_trials}). Skipping across-trial correction for this trial.")

		# Apply the final calculated shifts (within + valid across-trial) to the images
		# and update other relevant data for each trial.

		# Update the trial data with the final motion correction shifts
		# Ensure motions are stored as float32 as in the original code following this block
		all_data.planes[plane_i].trials[trial_i].images = all_data.planes[plane_i].trials[trial_i].images.assign_coords({'shift correction in X' : ('Time (ms)', motions[trial_i][:,0].astype('float32'))})
		all_data.planes[plane_i].trials[trial_i].images = all_data.planes[plane_i].trials[trial_i].images.assign_coords({'shift correction in Y' : ('Time (ms)', motions[trial_i][:,1].astype('float32'))})

		all_data.planes[plane_i].trials[trial_i].template_image = templates[trial_i,:,:]


#endregion


anatomical_stack_images = np.stack([fi.normalize_image(plane, (0.05, 0.95)) for plane in anatomical_stack_images])

#* Identify the plane number of the trial.
plane_numbers = np.zeros((len(all_data.planes), len(plane.trials)), dtype='int32')

for plane_i, plane in enumerate(all_data.planes):
	# if plane_i == 1:
	# 	break
	for trial_i, trial in enumerate(plane.trials):
		
		template_image = all_data.planes[plane_i].trials[trial_i].template_image
		
		if trial_i == 0:
			reference_plane_number_low = 3
			reference_plane_number_high = anatomical_stack_images.shape[0]-3

		elif trial_i % 2 == 0:
			reference_plane_number = np.median((plane_numbers[plane_i, 0], plane_numbers[plane_i, 1]))
			reference_plane_number_low = reference_plane_number - p.low_high
			reference_plane_number_high = reference_plane_number + p.low_high

		reference_plane_number_low = int(np.clip(reference_plane_number_low, 3, anatomical_stack_images.shape[0] - 2*p.low_high))
		reference_plane_number_high = int(np.clip(reference_plane_number_high, 2*p.low_high, anatomical_stack_images.shape[0]-3))

		plane_numbers[plane_i, trial_i] = reference_plane_number_low + fi.find_plane_in_anatomical_stack(anatomical_stack_images[reference_plane_number_low:reference_plane_number_high,  p.image_crop_:- p.image_crop_,  p.image_crop_:- p.image_crop_], template_image[p.image_crop_template_matching:-p.image_crop_template_matching, p.image_crop_template_matching:-p.image_crop_template_matching])[0]

		all_data.planes[plane_i].trials[trial_i].position_anatomical_stack = plane_numbers[plane_i, trial_i]

		# print(reference_plane_number_low, reference_plane_number_high)


		# if trial_i == 1:
		# 	break

print(plane_numbers)




#* Save the data.
# %%
# region Save the data




all_data = c.Data(all_data.planes, anatomical_stack_images)



with open(path_pkl_after_motion_correction, 'wb') as file:
	pickle.dump(all_data, file)









# planes_ = np.round(np.median(plane_numbers, axis=1)).astype('int')
# plane_position_stack = np.argsort(planes_)

# for plane_i in range(len(all_data.planes)):
	
# 	all_data.planes[plane_i].template_image_position_anatomical_stack = int(plane_position_stack[plane_i])



# fig, axs = plt.subplots(len(all_data.planes), len(plane.trials)+1, figsize=(10, 50), squeeze=False)

# for plane_i in range(len(all_data.planes)):
# 	for trial_i in range(len(plane.trials)):

# 		anatomical_plane = anatomical_stack_images[planes_[plane_i],:,:]
		
# 		axs[plane_i,trial_i+1].imshow(all_data.planes[plane_i].trials[trial_i].template_image, interpolation='None')
# 		axs[plane_i,trial_i+1].set_xticks([])
# 		axs[plane_i,trial_i+1].set_yticks([])
# 		# axs.title(f'Plane {plane_i}, Trial {trial_i}, Anat. Pos. {plane_numbers[plane_i, trial_i]}')
# 	axs[plane_i,0].imshow(anatomical_plane, interpolation='None', vmin=np.quantile(anatomical_plane, 0.05), vmax=np.quantile(anatomical_plane, 0.99))
# 	axs[plane_i,0].set_xticks([])
# 	axs[plane_i,0].set_yticks([])

# 	# break

# fig.set_size_inches(10, 20)
# fig.subplots_adjust(hspace=0, wspace=0)

# fig.savefig(results_figs_path_save / '5. Template images.png')








# fig, axs = plt.subplots(len(all_data.planes), len(plane.trials)+1, figsize=(10, 50), squeeze=False)

# for plane_i in range(len(all_data.planes)):
# 	for trial_i in range(len(plane.trials)):

# 		anatomical_plane = anatomical_stack_images[planes_[plane_i],:,:]
		
# 		axs[plane_position_stack[plane_i],trial_i+1].imshow(all_data.planes[plane_i].trials[trial_i].template_image, interpolation='None')
# 		axs[plane_position_stack[plane_i],trial_i+1].set_xticks([])
# 		axs[plane_position_stack[plane_i],trial_i+1].set_yticks([])
# 		# axs.title(f'Plane {plane_i}, Trial {trial_i}, Anat. Pos. {plane_numbers[plane_i, trial_i]}')
# 	axs[plane_position_stack[plane_i],0].imshow(anatomical_plane, interpolation='None', vmin=np.quantile(anatomical_plane, 0.05), vmax=np.quantile(anatomical_plane, 0.99))
# 	axs[plane_position_stack[plane_i],0].set_xticks([])
# 	axs[plane_position_stack[plane_i],0].set_yticks([])

# 	break

# fig.set_size_inches(10, 20)
# fig.subplots_adjust(hspace=0, wspace=0)

# # fig.savefig(results_figs_path_save / '5. Template images.png')




# fig, axs = plt.subplots(len(all_data.planes), 1, figsize=(10, 50), squeeze=False)
# for plane_i in range(len(all_data.planes)):
# 	axs[plane_i,0].imshow(anatomical_stack_images[planes_[plane_i], :, :], interpolation='None')

# #* Save the coordinates of the black box.
# # all_data.black_box = [x_black_box_beg, x_black_box_end, y_black_box_beg, y_black_box_end]




exec(open('3.1.Analysis_of_imaging_data_pixels.py').read())

print('END')






# 				# fig, axs = plt.subplots(15, 2, figsize=(10, 50))

# 				# for i in range(30):
# 				# 	if i<=14:
# 				# 		im = axs[i,0].imshow(C[i*2], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
# 				# 		axs[i,0].axis('off')

# 				# 	if i>14 and i<=29:
# 				# 		axs[i-15,1].imshow(C[(i-15)*2+1], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
# 				# 		axs[i-15,1].axis('off')

# 				# fig.tight_layout()
# 				# # fig.suptitle('Templates Before Correction', fontsize=16)
# 				# fig.savefig(r'H:\My Drive\PhD\Lab meetings\templates before.png', dpi=300, bbox_inches='tight')



# 				# fig, axs = plt.subplots(15, 2, figsize=(10, 50))

# 				# for i in range(30):
# 				# 	if i<=14:
# 				# 		axs[i,0].imshow(D[i*2], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
# 				# 		axs[i,0].axis('off')

# 				# 	if i>14 and i<=29:
# 				# 		axs[i-15,1].imshow(D[(i-15)*2+1], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
# 				# 		axs[i-15,1].axis('off')

# 				# fig.tight_layout()
# 				# # fig.suptitle('Templates After Correction', fontsize=16)
# 				# fig.savefig(r'H:\My Drive\PhD\Lab meetings\templates after .png', dpi=300, bbox_inches='tight')

# #  endregion

# #* Load the data.
# path_pkl = path_home / 'Imaging' / fish_name / (fish_name + '_2.pkl')
# # path_pkl = r"E:\2024 03_Delay 2-P multiple planes\20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf\20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf.pkl"

# with open(path_pkl, 'rb') as file:
# 	all_data = pickle.load(file)



# # Create a new HDF5 file
# h5_file = h5py.File(path_home / 'Imaging' / fish_name / (fish_name + '.h5'), 'w')

# # Create a group for the planes data
# planes_group = h5_file.create_group('planes')

# # Loop through each plane in all_data
# for plane_i, plane in enumerate(all_data.planes):
# 	# Create a group for the current plane
# 	plane_group = planes_group.create_group(f'plane_{plane_i}')
	
# 	# Loop through each trial in the plane
# 	for trial_i, trial in enumerate(plane.trials):
# 		# Create a group for the current trial
# 		trial_group = plane_group.create_group(f'trial_{trial_i}')
		
# 		trial_group.create_dataset('trial_number', data=trial.trial_number)
# 		trial_group.create_dataset('protocol', data=trial.protocol)
# 		trial_group.create_dataset('behavior', data=trial.behavior)
# 		trial_group.create_dataset('images', data=trial.images)
# 		try:
# 			trial_group.create_dataset('mask_good_frames', data=trial.mask_good_frames)
# 			trial_group.create_dataset('template_image', data=trial.template_image)
# 			trial_group.create_dataset('position_anatomical_stack', data=trial.position_anatomical_stack)
# 		except:
# 			pass

# # Close the HDF5 file
# h5_file.close()
