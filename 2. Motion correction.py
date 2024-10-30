#* Imports

# %%
# region Imports
import pickle
from dataclasses import dataclass
from importlib import reload
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import scipy.ndimage as ndimage
import seaborn as sns
import tifffile
import xarray as xr
from scipy import signal
from scipy.stats import pearsonr, zscore
# from skimage import morphology
# from skimage.measure import block_reduce
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
pd.set_option("compute.use_numba", True)
pd.set_option("compute.use_numexpr", True)
pd.set_option("compute.use_bottleneck", True)
#endregion


#* Paths
# %%
# region Paths
path_home = Path(r'E:\2024 03_Delay 2-P multiple planes')
# Path(r'E:\2024 09_Delay 2-P zoom in multiplane imaging')
# Path(r'E:\2024 10_Delay 2-P multiplane imaging ca8')


fish_name = r'20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'

# path_pkl = path_home / 'Imaging' / fish_name / (fish_name + '_before motion correction' + '.pkl')

imaging_path = path_home / 'Imaging'


# for fish_i, fish_name in enumerate(fish_names_list):

# 	try:

imaging_path_ = imaging_path / fish_name / 'Imaging'

h5_path = imaging_path_ / (fish_name + '_before_motion_correction.h5')


#endregion





x_black_box_beg = 330
x_black_box_end = 345
y_black_box_beg = 594
y_black_box_end = 609


image_crop = 5  # number of pixels to crop around the images


motion_thr_within_trial = 5  # threshold for motion within trial (number of pixels)




#* Load the data before motion correction.
# %%
# region Load the data before motion correction
# with open(path_pkl, 'rb') as file:
# 	all_data = pickle.load(file)

# Load the HDF5 file

with h5py.File(h5_path, 'r') as h5_file:

	# Load the planes data
	planes_group = h5_file['planes']
	planes = []
	for plane_name in planes_group:
		plane_group = planes_group[plane_name]
		trials = []
		for trial_name in plane_group:
			trial_group = plane_group[trial_name]
			trial = c.Trial(
				trial_number=trial_group['trial_number'][()],
				protocol=trial_group['protocol'][()],
				behavior=trial_group['behavior'][()],
				images=trial_group['images'][()]
			)
			# if 'mask_bad_frames' in trial_group:
			# 	trial.mask_bad_frames = trial_group['mask_bad_frames'][()]
			# if 'template_image' in trial_group:
			# 	trial.template_image = trial_group['template_image'][()]
			# if 'position_anatomical_stack' in trial_group:
			# 	trial.position_anatomical_stack = trial_group['position_anatomical_stack'][()]
			trials.append(trial)
		planes.append(c.Plane(trials=trials))

	all_data = c.Data(planes=planes, anatomical_stack=h5_file['anatomical_stack'][()])

# all_data.__dict__.keys()

anatomical_stack_images = all_data.anatomical_stack



# #* Check the black box (mask)
# plt.figure()
# plt.imshow(all_data.planes[0].trials[0].images[100,:,:], vmin=0, vmax=500)
# plt.colorbar(shrink=0.5)

# plt.figure()
# plt.imshow(all_data.planes[0].trials[0].images[100][y_black_box_beg:y_black_box_end, x_black_box_beg:x_black_box_end], vmin=0, vmax=500)
# plt.colorbar(shrink=0.5)


##* Subtract the background from the images.
for image_i in range(anatomical_stack_images.shape[0]):
	anatomical_stack_images[image_i] -= np.mean(anatomical_stack_images[image_i, y_black_box_beg:y_black_box_end, x_black_box_beg:x_black_box_end])

##* Clip the values of anatomical_stack_images.
anatomical_stack_images = np.clip(anatomical_stack_images, 0, None)
#endregion




#* Correct motion within and across trials.
# %%
# region Correct motion



#ToDo this should take the pixel spacing into account!!!

_, y_dim, x_dim = np.array(anatomical_stack_images.shape)

x_dim = int(x_dim * p.xy_movement_allowed/2)
y_dim = int(y_dim * p.xy_movement_allowed/2)






#! no final, ha valores negativos de pixels

for plane_i, plane in tqdm(enumerate(all_data.planes)):

	# break

	print('Plane: ', plane_i)

	motions = [_ for _ in range(len(plane.trials))]
	template_images = np.zeros((len(plane.trials), plane.trials[0].images.shape[1], plane.trials[0].images.shape[2]))
	plane_numbers = np.zeros(len(plane.trials), dtype='int32')

	#* Motion correction within trial.
	for trial_i, trial in tqdm(enumerate(plane.trials)):

		print('Trial: ', trial_i)

		trial_images_ = trial.images.copy().to_numpy()

		# break
		
		##* Discard bad frames due to motion, gating of the PMT or plane change when making a template image for the trial.
		# plane.trials[trial_i].images.values, plane.trials[trial_i].template_image, plane.trials[trial_i].position_anatomical_stack 
		# motions[trial_i], template_images[trial_i], plane_numbers[trial_i] = f.correct_motion_within_trial(trial, anatomical_stack_images, x_dim, y_dim, 3)
		template_image_ = f.get_template_image(f.get_maximum_number_good_last_images(trial_images_))
	



#!!!!!!!!!!!!! MAKE THIS A FUNCTION
		##* Subtract the background from the images.
		for image_i in range(trial_images_.shape[0]):
			trial_images_[image_i] -= np.mean(trial_images_[image_i, y_black_box_beg:y_black_box_end, x_black_box_beg:x_black_box_end])

(ndimage.median_filter(frames, size=p.median_filter_kernel, axes=(1,2)), axis=0)






		##* Clip the values of trial_images_.
		trial_images_ = np.clip(trial_images_, 0, None)

		#? NEED TO CROP THE IMAGES


		for _ in tqdm(range(3)):

			# break
					
			#* Measure the motion of each frame using phase cross-correlation.
			#! check phase_cross_correlation parameters
			motion_ = f.measure_motion(trial_images_[:, image_crop:-image_crop, image_crop:-image_crop], template_image_[image_crop:-image_crop, image_crop:-image_crop], normalization=None)

			#* Get the total motion.
			total_motion = f.get_total_motion(motion_)
			# Use half of the frames to get the template image.
			# motion_thr = int(np.quantile(total_motion, 0.2))

			#* Align the frames to their average.
			aligned_frames = f.align_frames(trial_images_, motion_, total_motion, total_motion_thr=motion_thr_within_trial)
			
			#* Motion correction relative to trials average.
			template_image_ = f.get_template_image(aligned_frames[np.where(total_motion <= motion_thr_within_trial)[0]])
			
			# plt.imshow(template_image_, vmin=0, vmax=500)



		# fig, axs = plt.subplots(1, 2)
		# axs[0].imshow(ndimage.median_filter(np.mean(aligned_frames, axis=0), size=p.median_filter_kernel))
		# axs[1].imshow(np.mean(ndimage.median_filter(aligned_frames, size=p.median_filter_kernel, axes=(1,2)), axis=0))
		# fig.show()

		#* Identify the plane number of the trial.
		plane_number_, _ = f.find_plane_in_anatomical_stack(anatomical_stack_images, template_image_.astype('float32'), x_dim, y_dim)


		print(plane_number_)

		if trial_i == 4:

			break


	break



		# template_image_[motion_thr:-motion_thr, motion_thr:-motion_thr]
		# a = f.get_template_image(f.get_maximum_number_good_last_images(trial.images.values))

		fig, axs = plt.subplots(1, 2)

		axs[0].imshow(template_images[trial_i], vmin=0)
		axs[1].imshow(anatomical_stack_images[plane_numbers[trial_i]], vmin=0)
		axs[0].set_title('Template plane from\naverage of good frames')
		axs[1].set_title('Anatomical stack plane number ' + str(plane_numbers[trial_i]))
		# fig.figure(figsize=(10, 6))
		fig.colorbar(axs[0].imshow(template_images[trial_i]), ax=axs[0], shrink=0.5)
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


		#* Frames to ignore due to too much motion (or gating of the PMT, which causes a huge "motion").
		# trial_images = trial.images.values

		# Mask with True where the frames are bad (due to gating of the PMT or motion).
		mask_bad_frames = (~f.get_good_images_indices(aligned_frames)) | (np.where(f.get_total_motion(motions[trial_i]) > p.motion_thr_from_trial_average, True, False))

		all_data.planes[plane_i].trials[trial_i].mask_bad_frames = mask_bad_frames

	# 	break
	# break


	#* Motion correction across trials of the same plane.
	for trial_i, trial in enumerate(plane.trials):
			
		if trial_i > 0:
			#* Measure motion of each frame using phase cross-correlation.
			motion = f.measure_motion(np.expand_dims(template_images[trial_i][5:-5, 5:-5], axis=0), template_images[0][5:-5, 5:-5], normalization=None)[0]

			motions[trial_i] += motion

		#* Measure motion of each frame using phase cross-correlation.
		total_motion = f.get_total_motion(motions[trial_i])
		# Use half of the frames to get the template image.
		motion_thr = np.median(total_motion)

		
		
		#* Align the frames to their average.
		aligned_frames = f.align_frames(trial.images.to_numpy(), motions[trial_i], total_motion, None)

		template_image = f.get_template_image(aligned_frames[np.where(total_motion <= motion_thr)[0]])

		motion = np.array((np.ceil(motions[trial_i].max(axis=0))), dtype='int32')

#!!!!!!!!!!!!!!! FIX THIS TO REMOVE PADDED VALUES AROUND THE TEMPLATE IMAGE
		#* Identify the plane number of the trial.
		#! plane_number, _ = f.find_plane_in_anatomical_stack(anatomical_stack_images, template_image.astype('float32')[motion[1]:-motion[1], motion[0]:-motion[0]], None, x_dim, y_dim)


		plt.imshow(ndimage.median_filter(np.mean(aligned_frames, axis=0), size=p.median_filter_kernel))
		plt.colorbar(shrink=0.5)
		plt.show()

		all_data.planes[plane_i].trials[trial_i].images.values = aligned_frames
		all_data.planes[plane_i].trials[trial_i].template_image = template_image
		all_data.planes[plane_i].trials[trial_i].position_anatomical_stack = 1
		#! plane_number

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
