#* Imports

# %%
# region Imports
import os
import pickle
from importlib import reload
from pathlib import Path

import xarray as xr
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import scipy.ndimage as ndimage
import seaborn as sns
from tqdm import tqdm

#* Load custom functions and classes
import my_classes as c
import my_functions as f
import my_parameters as p
from my_general_variables import *

reload(f)
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
path_home = Path(r'E:\2024 03_Delay 2-P 15 planes top part')
# Path(r'E:\2024 09_Delay 2-P zoom in multiplane imaging')
# Path(r'E:\2024 10_Delay 2-P multiplane imaging ca8')

# fish_list = [f for f in (path_home / 'Imaging').iterdir() if f.is_dir()]
# fish_names_list = [f.stem for f in fish_list]

fish_name = r'20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'


imaging_path = path_home / 'Imaging'


# for fish_i, fish_name in enumerate(fish_names_list):

# 	try:

imaging_path_ = imaging_path / fish_name / 'Imaging'


path_pkl_before_motion_correction = path_home / fish_name / (fish_name + '_before motion correction' + '.pkl')


# h5_path = path_home / fish_name / (fish_name + '_before_motion_correction.h5')

# h5_path = imaging_path_ / (fish_name + '_before_motion_correction.h5')

# h5_path = r"E:\2024 03_Delay 2-P 15 planes top part\Imaging\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf\Imaging\20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf_before_motion_correction.h5"

#endregion



x_black_box_beg = 330
x_black_box_end = 345
y_black_box_beg = 594
y_black_box_end = 609


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
			# 			coords={'index': ('time', trial.behavior.index), 'time': trial.behavior['Time (ms)'].to_numpy()}, dims=['time', 'cols'])



			# ds = xr.Dataset(dict(images=trial.images, behavior=xr.DataArray(trial.behavior,
			# 			coords={'index': ('time', trial.behavior.index), 'time': trial.behavior['Time (ms)'].to_numpy()}, dims=['time', 'cols'])))



			# trial.images.to_netcdf(path_2, mode='a', group='images', format='NETCDF4', engine='netcdf4', encoding={'Imaging data': {'zlib': True, 'complevel': 4, 'chunksizes': (trial.images.shape[0],np.ceil(trial.images.shape[1]/8),np.ceil(trial.images.shape[2]/8))}})

			# trial.images.dims

			# a = trial.behavior.to_xarray().set_coords('Time (ms)')
			# a.name = 'Tracking data'



			# imaging = xr.DataArray(images, coords={'index': ('time', data.loc[data['Frame beg'].notna(), :].index), 'time': data.loc[data['Frame beg'].notna(), abs_time].to_numpy(), 'x': range(images.shape[1]), 'y': range(images.shape[2])}, dims=['time', 'x', 'y'])



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
# 			# if 'mask_bad_frames' in trial_group:
# 			# 	trial.mask_bad_frames = trial_group['mask_bad_frames']
# 			# if 'template_image' in trial_group:
# 			# 	trial.template_image = trial_group['template_image']
# 			# if 'position_anatomical_stack' in trial_group:
# 			# 	trial.position_anatomical_stack = trial_group['position_anatomical_stack']
# 			trials.append(trial)
# 		planes.append(c.Plane(trials=trials))

# 	all_data = c.Data(planes=planes, anatomical_stack=h5_file['anatomical_stack_images'][()])

# # all_data.__dict__.keys()

# # anatomical_stack_images = all_data.anatomical_stack






# %%
# region Check where dark mask is.

##* Subtract the background from the images.

###* Anatomical stack images.
anatomical_stack = all_data.anatomical_stack

anatomical_stack = xr.DataArray(anatomical_stack, coords={'index': ('plane_number', range(anatomical_stack.shape[0])), 'plane_number': range(anatomical_stack.shape[0]), 'x': range(anatomical_stack.shape[2]), 'y': range(anatomical_stack.shape[1])}, dims=['plane_number', 'y', 'x'])


#!
#! The dark mask ("eye mask"), if present, contains more than 0.1% of the pixels in each frame.
####* Consider that the background is 10% of each frame. Subtract it to the images and clip the values to 0.
# The background will be always in more than 10% of the pixels in each frame.
#? or take the mean of each frame and subtract it from the frame?
anatomical_stack = anatomical_stack - anatomical_stack.quantile(0.01, dim=('y', 'x'))
anatomical_stack = anatomical_stack.clip(0, None)

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
		trial.images.values = ndimage.median_filter(trial.images.to_numpy(), size=p.median_filter_kernel, axes=(1,2))
		trial.images = trial.images - trial.images.quantile(0.01, dim=('y', 'x'))
		trial.images = trial.images.clip(0, None)
		# plt.imshow(np.mean(trial.images, axis=0), vmin=0, vmax=500)
	# 	break
	# break


# trial.__dict__.keys()

#! WHEN TO RELY ON THE EYE MASK



# #* Check the black box (mask)
# plt.figure()
# plt.imshow(all_data.planes[0].trials[0].images[100,:,:], vmin=0, vmax=500)
# plt.colorbar(shrink=0.5)

# plt.figure()
# plt.imshow(all_data.planes[0].trials[0].images[100][y_black_box_beg:y_black_box_end, x_black_box_beg:x_black_box_end], vmin=0, vmax=500)
# plt.colorbar(shrink=0.5)



#endregion




#* Correct motion within and across trials.
# %%
# region Correct motion



#ToDo this should take the pixel spacing into account!!!

_, y_dim, x_dim = np.array(anatomical_stack.shape)

x_dim = int(x_dim * p.xy_movement_allowed/2)
y_dim = int(y_dim * p.xy_movement_allowed/2)





reload(f)
reload(c)
reload(p)

#! no final, ha valores negativos de pixels

for plane_i, plane in tqdm(enumerate(all_data.planes)):

	# break

	print('Plane: ', plane_i)

	motions = list(np.zeros(len(plane.trials), dtype='int32'))
	template_images = np.zeros((len(plane.trials), plane.trials[0].images.shape[1], plane.trials[0].images.shape[2]))
	plane_numbers = np.zeros(len(plane.trials), dtype='int32')

	#* Motion correction within trial.
	for trial_i, trial in tqdm(enumerate(plane.trials)):

		print('Trial: ', trial_i)

		#? NEED TO CROP THE IMAGES
		#! Crop the images
		trial_images_ = trial.images[:, image_crop:-image_crop, image_crop:-image_crop].copy()

		
		##* Discard bad frames due to motion, gating of the PMT or plane change when making a template image for the trial.
		# plane.trials[trial_i].images.values, plane.trials[trial_i].template_image, plane.trials[trial_i].position_anatomical_stack 
		# motions[trial_i], template_images[trial_i], plane_numbers[trial_i] = f.correct_motion_within_trial(trial, anatomical_stack_images, x_dim, y_dim, 3)

		template_image_ = f.get_template_image(f.get_maximum_number_good_last_images(trial_images_))

		
		# ##* Subtract the background from the images.
		# for image_i in range(trial_images_.shape[0]):
		# 	trial_images_[image_i] -= np.mean(trial_images_[image_i, y_black_box_beg:y_black_box_end, x_black_box_beg:x_black_box_end])

		# (ndimage.median_filter(frames, size=p.median_filter_kernel, axes=(1,2)), axis=0)

		# ##* Clip the values of trial_images_.
		# trial_images_ = np.clip(trial_images_, 0, None)


		for counter in tqdm(range(number_iterations_within_trial)):
			
			#* Measure the motion of each frame using phase cross-correlation.
			#! check phase_cross_correlation parameters
			motion_ = f.measure_motion(trial_images_, template_image_, normalization=None)

			#* Get the total motion.
			total_motion = f.get_total_motion(motion_)
			# Use half of the frames to get the template image.
			# motion_thr = int(np.quantile(total_motion, 0.2))

			if counter < number_iterations_within_trial-1:

				#* Align the frames to their average.
				# aligned_frames = trial_images_.copy()

				# mask_frames_shift = total_motion <= motion_thr_within_trial
				# aligned_frames[mask_frames_shift] = f.align_frames(trial_images_[mask_frames_shift], motion_)

				# aligned_frames = f.align_frames(trial_images_[total_motion <= motion_thr_within_trial], motion_)


				#* Motion correction relative to trials average.
				template_image_ = f.get_template_image(f.align_frames(trial_images_[total_motion <= motion_thr_within_trial], motion_))
	
			# break
	# 	break


		#!!!!!!!!!! Discard frames where the motion exceeded the threshold and frames where there was large light changes.
		#* Motion correction relative to trials average.

		mask_bad_frames_motion = total_motion <= motion_thr_within_trial
		mask_bad_frames_no_PMT = f.get_good_images_indices_2(trial.images)
		mask_bad_frames = mask_bad_frames_motion & mask_bad_frames_no_PMT
		images_helper = f.align_frames(trial.images[mask_bad_frames], motion_)
		
		template_image_ = f.get_template_image(images_helper)

		# fig, axs = plt.subplots(1, 2)
		# axs[0].imshow(ndimage.median_filter(np.mean(aligned_frames, axis=0), size=p.median_filter_kernel))
		# axs[1].imshow(np.mean(ndimage.median_filter(aligned_frames, size=p.median_filter_kernel, axes=(1,2)), axis=0))
		# fig.show()

		#* Identify the plane number of the trial.
		plane_number_, _ = f.find_plane_in_anatomical_stack(anatomical_stack[:, image_crop:-image_crop, image_crop:-image_crop], template_image_[image_crop_template_matching:-image_crop_template_matching, image_crop_template_matching:-image_crop_template_matching])

		print(plane_number_)



		#* Frames to ignore due to too much motion (or gating of the PMT, which causes a huge "motion").
		# trial_images = trial.images.values
		# Mask with True where the frames are bad (due to gating of the PMT or motion).
		# mask_bad_frames = (~f.get_good_images_indices_1(aligned_frames)) | (np.where(f.get_total_motion(motions[trial_i]) > p.motion_thr_from_trial_average, True, False))
		all_data.planes[plane_i].trials[trial_i].mask_bad_frames = mask_bad_frames
		# all_data.planes[plane_i].trials[trial_i].mask_bad_frames = 
		# all_data.planes[plane_i].trials[trial_i].mask_bad_frames = 
		# all_data.planes[plane_i].trials[trial_i].mask_bad_frames = 


		# mask_bad_frames = mask_bad_frames_motion | mask_bad_frames_no_PMT
		motions[trial_i] = motion_
		template_images[trial_i] = template_image_
		plane_numbers[trial_i] = plane_number_

#!!!!!!!!!!!

		# if trial_i == 4:

		# break



		# template_image_[motion_thr:-motion_thr, motion_thr:-motion_thr]
		# a = f.get_template_image(f.get_maximum_number_good_last_images(trial.images.values))



		fig, axs = plt.subplots(1, 2)
		axs[0].imshow(template_images[trial_i], vmin=0, vmax=500)
		axs[1].imshow(anatomical_stack[plane_numbers[trial_i]], vmin=0, vmax=500)
		axs[0].set_title('Template plane from\naverage of good frames')
		axs[1].set_title('Anatomical stack plane number ' + str(plane_numbers[trial_i]))
		fig.set_size_inches((20, 10))
		# fig.colorbar(axs[0].imshow(template_images[trial_i]), ax=axs[0], shrink=0.5)
		# fig.colorbar(ax=axs[1], shrink=0.5)
		# fig.show()


		fig, axs = plt.subplots(1, 2)
		fig.suptitle('Motion of each frame')
		axs[0].plot(f.get_total_motion(motions[trial_i]), 'k.')
		axs[1].scatter(motions[trial_i][:,0]-0.01+0.02*np.random.rand(motions[trial_i][:,0].shape[0]),motions[trial_i][:,1]-0.01+0.02*np.random.rand(motions[trial_i][:,1].shape[0]),s=0.5)
		# fig.show()



		# plt.imshow(ndimage.median_filter(np.mean(f.align_frames(trial.images.to_numpy(), motions[trial_i], f.get_total_motion(motions[trial_i])), axis=0), size=p.median_filter_kernel), vmin=0)
		# plt.colorbar(shrink=0.5)
		
		
		# plt.imshow(ndimage.median_filter(np.mean(trial.images.to_numpy(),axis=0), size=p.median_filter_kernel), vmin=0)
		# plt.colorbar(shrink=0.5)





	#* Motion correction across trials of the same plane.
	motion_ = f.measure_motion(template_images[1: , image_crop_:-image_crop_, image_crop_:-image_crop_], template_images[0, image_crop_:-image_crop_, image_crop_:-image_crop_], normalization=None)

	for m_i, m in enumerate(motions):
		
		if m_i > 0:

			motions[m_i] += motion_[m_i-1]

		plt.imshow(ndimage.median_filter(np.mean(f.align_frames(trial.images, motions[trial_i]), axis=0), size=p.median_filter_kernel))
		plt.colorbar(shrink=0.5)
		plt.show()

	break

	# for trial_i, trial in enumerate(plane.trials):
		
	# 	if trial_i > 0:
	# 		#* Measure motion of each frame using phase cross-correlation.
			
	# 		# trial_images_ = trial.images[:, image_crop_:-image_crop_, image_crop_:-image_crop_].copy()

	# 		motion_ = f.measure_motion(np.expand_dims(template_images[0, image_crop_:-image_crop_, image_crop_:-image_crop_], axis=0), np.expand_dims(template_images[trial_i, image_crop_:-image_crop_, image_crop_:-image_crop_], axis=0), normalization=None)
	# 		# np.expand_dims(template_images[0, image_c<rop_:-image_crop_, image_crop_:-image_crop_], axis=0).shape
	# 		# np.expand_dims(template_images[trial_i, image_crop_:-image_crop_, image_crop_:-image_crop_], axis=0).shape
	# 		# motion = f.measure_motion(np.expand_dims(template_images[trial_i][image_crop:-image_crop, image_crop:-image_crop], axis=0), template_images[0][image_crop:-image_crop, image_crop:-image_crop], normalization=None)[0]

	# 		motions[1:] += motion_

		# #* Measure motion of each frame using phase cross-correlation.
		# total_motion = f.get_total_motion(motions[trial_i])
		# # Use half of the frames to get the template image.
		# motion_thr = np.median(total_motion)

		
		
		#* Align the frames to their average.
		# aligned_frames = f.align_frames(trial.images.to_numpy(), motions[trial_i], total_motion, None)

		# template_image = f.get_template_image(aligned_frames[np.where(total_motion <= motion_thr)[0]])

		# motion = np.array((np.ceil(motions[trial_i].max(axis=0))), dtype='int32')

#!!!!!!!!!!!!!!! FIX THIS TO REMOVE PADDED VALUES AROUND THE TEMPLATE IMAGE
		#* Identify the plane number of the trial.
#!!!!!!!!!!!!!!!!plane_number_, _ = f.find_plane_in_anatomical_stack(anatomical_stack[:, image_crop:-image_crop, image_crop:-image_crop], template_image_[image_crop_template_matching:-image_crop_template_matching, image_crop_template_matching:-image_crop_template_matching])

		all_data.planes[plane_i].trials[trial_i].images.values = aligned_frames
		all_data.planes[plane_i].trials[trial_i].template_image = template_image
		all_data.planes[plane_i].trials[trial_i].position_anatomical_stack = 1
		#! plane_number

		motions[trial_i] = motion_
		template_images[trial_i] = template_image_
		plane_numbers[trial_i] = plane_number_
SAVE THE NEW TEMPLATE
SAVE THE SHIFT
THE PLANE NUMBER IN THE ANATOMICAL STACK

SHIFT THE ORIGINAL IMAGES
	# 	break
	# break

	convert motion to np.array

		# break
	# break
	print('Plane:', plane_i, plane_numbers)
#endregion



#* Save the data.
# %%
# region Save the data
path_pkl = path_home / 'Imaging' / fish_name / (fish_name + '_2' + '.pkl')

all_data = c.Data(all_data.planes, anatomical_stack_images)

with open(path_pkl, 'wb') as file:
	pickle.dump(all_data, file)
# endregion


#* Plot the position in the anatomical stack.
# region Position in the anatomical stack
try:
	A = []
	B = []

	C = []
	D = []

	for i in range(len(all_data.planes)):

		for j in range(2):

			A.append(all_data.planes[i].trials[j].position_anatomical_stack)

			C.append(all_data.planes[i].trials[j].template_image)

		for l in range(2,4):

			B.append(all_data.planes[i].trials[l].position_anatomical_stack)

			D.append(all_data.planes[i].trials[l].template_image)


	A = np.array(A)
	B = np.array(B)

	C = np.array(C)
	D = np.array(D)


	sns.set_style('whitegrid')


	path_ = path_home / fish_name


	plt.xlabel('Trial before or after initial train')
	plt.ylabel('Plane number in anatomical stack')
	plt.plot(A, 'blue')
	plt.plot(B, 'red')
	plt.legend(['Before initial train', 'After initial train'])
	plt.savefig(path_ / ('Where in the anatomical stack' + '.png'), dpi=300, bbox_inches='tight')


	plt.xlabel('Trial before or after initial train')
	plt.ylabel('Difference between planes imaged\n before and after initial train (μm)')
	plt.plot(A-B, 'k')
	plt.ylim(-10, 10)
	plt.savefig(path_ / ('Difference when revisiting planes' + '.png'), dpi=300, bbox_inches='tight')


	sns.set_style('white')
	
except:
	pass



				# fig, axs = plt.subplots(15, 2, figsize=(10, 50))

				# for i in range(30):
				# 	if i<=14:
				# 		im = axs[i,0].imshow(C[i*2], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
				# 		axs[i,0].axis('off')

				# 	if i>14 and i<=29:
				# 		axs[i-15,1].imshow(C[(i-15)*2+1], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
				# 		axs[i-15,1].axis('off')

				# fig.tight_layout()
				# # fig.suptitle('Templates Before Correction', fontsize=16)
				# fig.savefig(r'H:\My Drive\PhD\Lab meetings\templates before.png', dpi=300, bbox_inches='tight')



				# fig, axs = plt.subplots(15, 2, figsize=(10, 50))

				# for i in range(30):
				# 	if i<=14:
				# 		axs[i,0].imshow(D[i*2], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
				# 		axs[i,0].axis('off')

				# 	if i>14 and i<=29:
				# 		axs[i-15,1].imshow(D[(i-15)*2+1], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
				# 		axs[i-15,1].axis('off')

				# fig.tight_layout()
				# # fig.suptitle('Templates After Correction', fontsize=16)
				# fig.savefig(r'H:\My Drive\PhD\Lab meetings\templates after .png', dpi=300, bbox_inches='tight')

#  endregion

#* Load the data.
path_pkl = path_home / 'Imaging' / fish_name / (fish_name + '_2.pkl')
# path_pkl = r"E:\2024 03_Delay 2-P multiple planes\20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf\20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf.pkl"

with open(path_pkl, 'rb') as file:
	all_data = pickle.load(file)



# Create a new HDF5 file
h5_file = h5py.File(path_home / 'Imaging' / fish_name / (fish_name + '.h5'), 'w')

# Create a group for the planes data
planes_group = h5_file.create_group('planes')

# Loop through each plane in all_data
for plane_i, plane in enumerate(all_data.planes):
	# Create a group for the current plane
	plane_group = planes_group.create_group(f'plane_{plane_i}')
	
	# Loop through each trial in the plane
	for trial_i, trial in enumerate(plane.trials):
		# Create a group for the current trial
		trial_group = plane_group.create_group(f'trial_{trial_i}')
		
		trial_group.create_dataset('trial_number', data=trial.trial_number)
		trial_group.create_dataset('protocol', data=trial.protocol)
		trial_group.create_dataset('behavior', data=trial.behavior)
		trial_group.create_dataset('images', data=trial.images)
		try:
			trial_group.create_dataset('mask_bad_frames', data=trial.mask_bad_frames)
			trial_group.create_dataset('template_image', data=trial.template_image)
			trial_group.create_dataset('position_anatomical_stack', data=trial.position_anatomical_stack)
		except:
			pass

# Close the HDF5 file
h5_file.close()

print('END')
