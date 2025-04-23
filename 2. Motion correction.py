#* Imports

# %%
# region Imports
import os
import pickle
from importlib import reload
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import scipy.ndimage as ndimage
import seaborn as sns
import xarray as xr
from tqdm import tqdm

#* Load custom functions and classes
import my_classes as c
import my_functions_imaging as fi
import my_parameters as p
from my_general_variables import *

reload(fi)
reload(c)
reload(p)
# endregion

#* Settings
# %% Settings
# region Settings
pio.templates.default = "plotly_dark"
pd.set_option("mode.copy_on_write", True)
# pd.set_option("compute.use_numba", True)
pd.set_option("compute.use_numexpr", True)
pd.set_option("compute.use_bottleneck", True)
#endregion


#* Paths
# %%
# region Paths


path_home = Path(r'C:\Users\joaqc\Desktop\WIP')
# Path(r'E:\2024 03_Delay 2-P 15 planes top part')
# Path(r'E:\2024 09_Delay 2-P 4 planes JC neurons')
# Path(r'E:\2024 10_Delay 2-P single plane')
# Path(r'E:\2024 10_Delay 2-P 15 planes ca8 neurons')
# Path(r'E:\2024 09_Delay 2-P zoom in multiplane imaging')

# fish_list = [f for f in (path_home / 'Imaging').iterdir() if f.is_dir()]
# fish_names_list = [f.stem for f in fish_list]

fish_name = r'20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'

# '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'


# '20241013_01_control_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241013_02_control_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241009_03_delay_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'

# '20241024_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
# '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240415_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'



# 20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf


# '20241015_03_delay_2p-9_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf'




# '20240926_03_trace_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'

behavior_path = path_home / fish_name / 'Behavior'
imaging_path = path_home / fish_name / 'Imaging'

path_pkl_before_motion_correction = path_home / fish_name / (fish_name + '_before motion correction' + '.pkl')

path_pkl_after_motion_correction = path_home / fish_name / (fish_name + '_after motion correction' + '.pkl')




		# path_home = Path(r'E:\2024 09_Delay 2-P 4 planes JC neurons')
		# # Path(r'E:\2024 10_Delay 2-P single plane')
		# # Path(r'E:\2024 03_Delay 2-P 15 planes top part')
		# # Path(r'E:\2024 10_Delay 2-P 15 planes ca8 neurons')
		# # Path(r'E:\2024 09_Delay 2-P zoom in multiplane imaging')

		# # fish_list = [f for f in (path_home / 'Imaging').iterdir() if f.is_dir()]
		# # fish_names_list = [f.stem for f in fish_list]

		# fish_name = r'20241013_01_control_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
		# # '20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
		# # '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf'

		# # '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
		# # '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
		# # '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
		# #! 
		# #! 20240415_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf
		# #! 20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf

		# # r'20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
		# # '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
		# # '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
		# # '20240415_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
		# # '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
		# # '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'




		# # '20241015_03_delay_2p-9_mitfaMinusMinus,ca8E1BGCaMP6s_6dpf'


		# # '20240926_03_trace_2p-9_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
		# # '20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
		# # r'20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'
		# # '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'


		# imaging_path = path_home / 'Imaging'


		# # for fish_i, fish_name in enumerate(fish_names_list):

		# # 	try:

		# imaging_path_ = imaging_path / fish_name / 'Imaging'


		# path_pkl_before_motion_correction = path_home / fish_name / (fish_name + '_before motion correction' + '.pkl')


		# # h5_path = path_home / fish_name / (fish_name + '_before_motion_correction.h5')

		# # h5_path = imaging_path_ / (fish_name + '_before_motion_correction.h5')

		# # h5_path = r"E:\2024 03_Delay 2-P 15 planes top part\Imaging\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf\Imaging\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf_before_motion_correction.h5"

#endregion



# x_black_box_beg = 330
# x_black_box_end = 345
# y_black_box_beg = 594
# y_black_box_end = 609



image_crop = 5  # number of pixels to crop around the images
image_crop_template_matching = int(image_crop*1.5)

motion_thr_within_trial = 5  # threshold for motion within trial (number of pixels)

image_crop_ = image_crop + motion_thr_within_trial


number_iterations_within_trial = 3






#* Load the data before motion correction.
# %%
# region Load the data before motion correction
with open(path_pkl_before_motion_correction, 'rb') as file:
	all_data = pickle.load(file)





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





# %%
# region Check where dark mask is.

##* Subtract the background from the images.

###* Anatomical stack images.
anatomical_stack_images = all_data.anatomical_stack

anatomical_stack_images = xr.DataArray(anatomical_stack_images, coords={'index': ('plane_number', range(anatomical_stack_images.shape[0])), 'plane_number': range(anatomical_stack_images.shape[0]), 'x': range(anatomical_stack_images.shape[2]), 'y': range(anatomical_stack_images.shape[1])}, dims=['plane_number', 'y', 'x'])


#!
#! The dark mask ("eye mask"), if present, contains more than 0.1% of the pixels in each frame.
####* Consider that the background is 10% of each frame. Subtract it to the images and clip the values to 0.
# The background will be always in more than 10% of the pixels in each frame.
#? or take the mean of each frame and subtract it from the frame?
anatomical_stack_images = anatomical_stack_images - anatomical_stack_images.quantile(0.01, dim=('y', 'x'))
anatomical_stack_images = anatomical_stack_images.clip(0, None)
anatomical_stack_images.drop_vars('quantile')

# plt.imshow(np.mean(anatomical_stack, axis=0))
# plt.colorbar(shrink=0.5)
# plt.savefig(path_home / 'anatomical_stack_mean.png', dpi=300, bbox_inches='tight')


# # Create a histogram of the values in all_data.anatomical_stack
# plt.figure()
# plt.hist(anatomical_stack.to_numpy().ravel(), bins=500, color='blue', alpha=0.7, range=(0, 500))
# plt.title('Histogram of Anatomical Stack Values')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')
# plt.show()


# plt.imshow(all_data.anatomical_stack[100,-30:,-30:], vmin=0, vmax=500)
# plt.colorbar(shrink=0.5)

###* Imaging images.
for plane_i, plane in enumerate(all_data.planes):
	for trial_i, trial in enumerate(plane.trials):
		# trial_images = trial.images
		# plt.imshow(np.mean(trial_images, axis=0), vmin=0, vmax=None)
		# trial.images.values = ndimage.median_filter(trial.images.to_numpy(), size=p.median_filter_kernel, axes=(1,2))

		try:
			trial.images = trial.images - trial.images.quantile(0.01, dim=('y', 'x'))
			trial.images = trial.images.clip(0, None)
			# plt.imshow(np.mean(trial.images, axis=0), vmin=0, vmax=500)

			trial.images.drop_vars('quantile')

			all_data.planes[plane_i].trials[trial_i].images = trial.images
		except:
			pass
	# 	break
	# break


# trial.__dict__.keys()

#! WHEN TO RELY ON THE EYE MASK



#* Check the black box (mask)
plt.figure()
plt.imshow(all_data.planes[0].trials[0].images[100,:,:], vmin=0, vmax=500)
plt.colorbar(shrink=0.5)

plt.figure()
plt.imshow(all_data.planes[0].trials[0].images[100][x_black_box_beg:x_black_box_end, y_black_box_beg: y_black_box_end], vmin=0, vmax=None)
plt.colorbar(shrink=0.5)

# all_data.planes[0].trials[0].images[100].shape

#endregion




#* Correct motion within and across trials.
# %%
# region Correct motion



#ToDo this should take the pixel spacing into account!!!

_, y_dim, x_dim = np.array(anatomical_stack_images.shape)

x_dim = int(x_dim * p.xy_movement_allowed/2)
y_dim = int(y_dim * p.xy_movement_allowed/2)




plane_numbers = np.zeros((len(all_data.planes), len(plane.trials)), dtype='int32')

for plane_i, plane in tqdm(enumerate(all_data.planes)):

	# break

	# if plane_i == 0:
	# 	continue

	print('Plane: ', plane_i)

	motions = list(np.zeros(len(plane.trials), dtype='int32'))
	template_images = np.zeros((len(plane.trials), plane.trials[0].images.shape[1], plane.trials[0].images.shape[2]))

	#* Motion correction within trial.
	for trial_i, trial in tqdm(enumerate(plane.trials)):

		print('Trial: ', trial_i)

		#? NEED TO CROP THE IMAGES
		#! Crop the images
		trial_images_ = trial.images[:, image_crop:-image_crop, image_crop:-image_crop].copy()

		
		##* Discard bad frames due to motion, gating of the PMT or plane change when making a template image for the trial.
		# plane.trials[trial_i].images.values, plane.trials[trial_i].template_image, plane.trials[trial_i].position_anatomical_stack 
		# motions[trial_i], template_images[trial_i], plane_numbers[trial_i] = f.correct_motion_within_trial(trial, anatomical_stack_images, x_dim, y_dim, 3)

		template_image_ = fi.get_template_image(trial_images_[fi.get_good_images_indices_1(trial_images_)])

		
		# ##* Subtract the background from the images.
		# for image_i in range(trial_images_.shape[0]):
		# 	trial_images_[image_i] -= np.mean(trial_images_[image_i, y_black_box_beg:y_black_box_end, x_black_box_beg:x_black_box_end])

		# (ndimage.median_filter(frames, size=p.median_filter_kernel, axes=(1,2)), axis=0)

		# ##* Clip the values of trial_images_.
		# trial_images_ = np.clip(trial_images_, 0, None)


		for counter in tqdm(range(number_iterations_within_trial)):
			
			#* Measure the motion of each frame using phase cross-correlation.
			#! check phase_cross_correlation parameters
			motion_ = fi.measure_motion(trial_images_, template_image_, normalization=None)

			#* Get the total motion.
			total_motion = fi.get_total_motion(motion_)
			# Use half of the frames to get the template image.
			# motion_thr = int(np.quantile(total_motion, 0.2))

			if counter < number_iterations_within_trial-1:

				#* Align the frames to their average.
				# aligned_frames = trial_images_.copy()

				# mask_frames_shift = total_motion <= motion_thr_within_trial
				# aligned_frames[mask_frames_shift] = f.align_frames(trial_images_[mask_frames_shift], motion_)

				# aligned_frames = f.align_frames(trial_images_[total_motion <= motion_thr_within_trial], motion_)


				#* Motion correction relative to trials average.
				template_image_ = fi.get_template_image(fi.align_frames(trial_images_[total_motion <= motion_thr_within_trial], motion_))
	
			# break
	# 	break


		#!!!!!!!!!! Discard frames where the motion exceeded the threshold and frames where there was large light changes.
		#* Motion correction relative to trials average.

		mask_good_frames_motion = total_motion <= motion_thr_within_trial
		mask_good_frames_no_PMT = fi.get_good_images_indices_2(trial.images)
		mask_good_frames = mask_good_frames_motion & mask_good_frames_no_PMT

		template_image_ = fi.get_template_image(fi.align_frames(trial.images[mask_good_frames], motion_))


		# plt.figure(figsize=(10, 6))
		# plt.imshow(template_image_, vmin=0, vmax=500)
		# plt.colorbar(shrink=0.5)
		# plt.title('Anatomy')
		# plt.show()


		#* Identify the plane number of the trial.
		plane_number_, _ = fi.find_plane_in_anatomical_stack(anatomical_stack_images[:, image_crop:-image_crop, image_crop:-image_crop], template_image_[image_crop_template_matching:-image_crop_template_matching, image_crop_template_matching:-image_crop_template_matching])

		print(plane_number_)



		#* Frames to ignore due to too much motion (or gating of the PMT, which causes a huge "motion").
		# trial_images = trial.images.values
		# Mask with True where the frames are bad (due to gating of the PMT or motion).
		# mask_good_frames = (~f.get_good_images_indices_1(aligned_frames)) | (np.where(f.get_total_motion(motions[trial_i]) > p.motion_thr_from_trial_average, True, False))

		all_data.planes[plane_i].trials[trial_i].images = all_data.planes[plane_i].trials[trial_i].images.assign_coords({'mask good frames' : ('Time (ms)', mask_good_frames)})
		#! all_data.planes[plane_i].trials[trial_i].mask_good_frames = mask_good_frames


		# all_data.planes[plane_i].trials[trial_i].mask_good_frames = 
		# all_data.planes[plane_i].trials[trial_i].mask_good_frames = 
		# all_data.planes[plane_i].trials[trial_i].mask_good_frames = 

		# mask_good_frames = mask_good_frames_motion | mask_good_frames_no_PMT
		motions[trial_i] = motion_
		template_images[trial_i] = template_image_
		plane_numbers[plane_i, trial_i] = plane_number_

		all_data.planes[plane_i].trials[trial_i].template_image = template_images[trial_i]
		all_data.planes[plane_i].trials[trial_i].position_anatomical_stack = plane_numbers[plane_i, trial_i]
#!!!!!!!!!!!

		# if trial_i == 4:

		# break

		# template_image_[motion_thr:-motion_thr, motion_thr:-motion_thr]
		# a = f.get_template_image(f.get_maximum_number_good_last_images(trial.images.values))

		# fig, axs = plt.subplots(1, 2)
		# axs[0].imshow(template_images[trial_i], vmin=0, vmax=500)
		# axs[1].imshow(anatomical_stack_images[plane_numbers[plane_i, trial_i]], vmin=0, vmax=500)
		# axs[0].set_title('Template plane from\naverage of good frames')
		# axs[1].set_title('Anatomical stack plane number ' + str(plane_numbers[plane_i, trial_i]))
		# fig.set_size_inches((20, 10))
		# # fig.colorbar(axs[0].imshow(template_images[trial_i]), ax=axs[0], shrink=0.5)
		# # fig.colorbar(ax=axs[1], shrink=0.5)
		# # fig.show()


		# fig, axs = plt.subplots(1, 2)
		# fig.suptitle('Motion of each frame')
		# axs[0].plot(f.get_total_motion(motions[trial_i]), 'k.')
		# axs[1].scatter(motions[trial_i][:,0]-0.01+0.02*np.random.rand(motions[trial_i][:,0].shape[0]),motions[trial_i][:,1]-0.01+0.02*np.random.rand(motions[trial_i][:,1].shape[0]),s=0.5)
		# # fig.show()



		# plt.imshow(ndimage.median_filter(np.mean(f.align_frames(trial.images.to_numpy(), motions[trial_i], f.get_total_motion(motions[trial_i])), axis=0), size=p.median_filter_kernel), vmin=0)
		# plt.colorbar(shrink=0.5)
		
		
		# plt.imshow(ndimage.median_filter(np.mean(trial.images.to_numpy(),axis=0), size=p.median_filter_kernel), vmin=0)
		# plt.colorbar(shrink=0.5)





	#* Motion correction across trials of the same plane.
	motion_ = fi.measure_motion(template_images[1: , image_crop_:-image_crop_, image_crop_:-image_crop_], template_images[0, image_crop_:-image_crop_, image_crop_:-image_crop_], normalization=None)

	for trial_i in range(len(plane.trials)):
		
		if trial_i > 0:

			motions[trial_i] += motion_[trial_i-1]

		# plt.imshow(ndimage.median_filter(np.mean(f.align_frames(trial.images[mask_good_frames], motions[trial_i]), axis=0), size=p.median_filter_kernel))
		# plt.colorbar(shrink=0.5)
		# plt.show()
		# plt.imshow(f.get_template_image(f.align_frames(trial.images[mask_good_frames], motions[trial_i])))

		all_data.planes[plane_i].trials[trial_i].images = all_data.planes[plane_i].trials[trial_i].images.assign_coords({'shift correction in X' : ('Time (ms)', motions[trial_i][:,0].astype('float32'))})
		all_data.planes[plane_i].trials[trial_i].images = all_data.planes[plane_i].trials[trial_i].images.assign_coords({'shift correction in Y' : ('Time (ms)', motions[trial_i][:,1].astype('float32'))})

		#! all_data.planes[plane_i].trials[trial_i].shift_correction = motions[trial_i].astype('float32')

		# break
	
	
	# print('Plane:', plane_i)



	# break

print(plane_numbers)


#endregion


# trial.__dict__.keys()

# for plane in all_data.planes:
# 	for trial in plane.trials:

# 		template_image_ = f.get_template_image(f.align_frames(trial.images[trial.mask_good_frames], trial.shift_correction))

# 		plane_number_, _ = f.find_plane_in_anatomical_stack(anatomical_stack_images[:, image_crop:-image_crop, image_crop:-image_crop], template_image_[image_crop_template_matching:-image_crop_template_matching, image_crop_template_matching:-image_crop_template_matching])



plane_position_stack = np.argsort(plane_numbers[:,0])

for plane_i in range(len(all_data.planes)):
	
	all_data.planes[plane_i].template_image_position_anatomical_stack = int(plane_position_stack[plane_i])



# all_data.planes[1].trials[0].template_image

fig, axs = plt.subplots(len(all_data.planes), len(plane.trials), figsize=(10, 50), squeeze=False)

for plane_i in range(len(all_data.planes)):
	for trial_i in range(len(plane.trials)):

		axs[plane_i,trial_i].imshow(all_data.planes[plane_i].trials[trial_i].template_image, vmin=0, vmax=500)
		axs[plane_i,trial_i].set_xticks([])
		axs[plane_i,trial_i].set_yticks([])
	
	# break

fig.set_size_inches(15, 35)
fig.subplots_adjust(hspace=0, wspace=0)

fig.savefig(path_home / fish_name / 'Template images.png')



#* Save the data.
# %%
# region Save the data




all_data = c.Data(all_data.planes, anatomical_stack_images)


#* Save the coordinates of the black box.
all_data.black_box = [x_black_box_beg, x_black_box_end, y_black_box_beg, y_black_box_end]


path_pkl_after_motion_correction = path_home / fish_name / (fish_name + '_after motion correction' + '.pkl')

with open(path_pkl_after_motion_correction, 'wb') as file:
	pickle.dump(all_data, file)


print('END')


# DATA = all_data

# endregion










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
