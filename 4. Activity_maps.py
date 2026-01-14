"""
4. Activity Maps Visualization

This script generates publication-ready visualizations of neural activity responses from processed imaging data.

Workflow:
1. Loads response data from pickle file (output of script 3.x)
2. Generates activity maps for each plane and trial:
   - Positive CS responses: neurons showing increased activity during CS presentation
   - Negative CS responses: neurons showing decreased activity during CS presentation (optional)
   - Positive US responses: neurons responding to unconditioned stimulus (optional)
3. Creates composite visualizations using color overlays on anatomical templates
4. Exports multi-page TIFF files for external analysis tools
5. Generates summary figures showing all planes and trials

Visualization Features:
- Color-coded activity overlays (inferno colormap by default)
- Anatomical template integration for spatial context
- Configurable activity thresholds for display
- Trial grouping by experimental phase (Pre-Train, Train, Test)
- Multi-format outputs (PNG for figures, TIFF for detailed analysis)

Key Parameters:
- color_list: Trial phase color coding for labels
- phase_list: Experimental phase labels per trial
- trial_numbers_list: Trial numbering scheme per plane
- activity_threshold: Minimum normalized response for visualization (typically 0.3-0.4)
- anatomy_brightness: Scaling factor for anatomical template visibility

Input Structure:
- Requires processed data with trial attributes:
  - cs_us_vs_pre: normalized CS response maps
  - post_us_vs_pre: normalized US response maps
  - anatomy: anatomical template image

Output Files:
- PNG: Summary figures with all planes/trials (e.g., "6.1. Positive responses to CS.png")
- Multi-page TIFF: Individual plane responses for ROI analysis
- Rotated and labeled frames for presentation

Note: This script can be called automatically from 3.x scripts or run standalone.
Trial-specific visualization parameters should be configured based on experimental design.
"""



#* Imports

# %%
# region Imports
import pickle
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import tifffile
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
from tqdm import tqdm

#* Load custom functions and classes
import my_classes as c
# import my_experiment_specific_variables as spec_var
import my_functions_imaging as fi
import my_parameters as p
from my_general_variables import *
from my_paths import fish_name, path_home

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
# path_home = Path(r'D:\2024 09_Delay 2-P 4 planes JC neurons')
# Path(r'D:\2024 10_Delay 2-P 15 planes ca8 neurons')

path_results_save = Path(r'F:\Results (paper)') / path_home.stem

# fish_name = r'20241007_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241008_02_delay_2p-5_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241008_03_delay_2p-6_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241008_01_delay_2p-4_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'


color_list = ['cyan']*2 + ['red']*8 + ['yellow']*6
phase_list = ['Pre-Train']*2 + ['Train']*8 + ['Test']*6
helper = np.array([0,1, 0,1, 10,11, 20,21, 30,31, 0,1, 12,13, 22,23])+1
trial_numbers_list = np.array([helper, helper+2, helper+4, helper+6])

if 'top' in str(path_home):
	color_list = ['white']*2 + ['red']*2
	phase_list = ['Pre-Train']*2 + ['Train']*2
	helper = np.array([0,1, 10,11])+1
	trial_numbers_list = np.array([helper+i*2 for i in range(15)])


# '20241013_01_control_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241010_01_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241008_02_delay_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241008_03_delay_2p-6_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'



# '20241022_01_delay_2p-10_mitfaminusminus,ca8e1bgcamp6s_5dpf'
# '20241015_01_delay_2p-7_mitfaminusminus,ca8e1bgcamp6s_6dpf'
# '20241008_03_delay_2p-6_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241007_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'

# '20241007_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241013_01_control_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240930_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241014_03_trace_2p-6_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241010_01_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241008_01_delay_2p-4_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241007_03_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20241017_01_delay_2p-4_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241016_02_delay_2p-2_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240417_01_delay_2p-4_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240919_03_control_2p-1_mitfaminusminus,elavl3h2bgcamp6s_5dpf'
# '20240911_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240927_02_control_2p-5_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240910_02_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20240415_02_delay_2p-2_mitfaMinusMinus,elavl3BGCaMP6f_5dpf'
# '20240415_01_delay_2p-1_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240926_03_trace_2p-9_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20240920_03_trace_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6s_6dpf'
# '20240416_01_delay_2p-3_mitfaminusminus,elavl3h2bgcamp6f_6dpf'



fish_ID = '_'.join(fish_name.split('_')[:2])


behavior_path_home = path_home / 'Tail'
imaging_path_home = path_home / 'Neurons' / fish_name

behavior_path_save = path_results_save / 'Tail'
results_figs_path_save = path_results_save / 'Neurons' / fish_name

whole_data_path_save = Path(r'H:\2-P imaging') / path_home.stem / fish_name


# path_pkl_after_motion_correction = whole_data_path_save / (fish_ID + '_2. After motion correction.pkl')


# path_pkl_responses = whole_data_path_save / (fish_ID + '_3. Responses.pkl')
path_pkl_responses = whole_data_path_save / (fish_ID + '_3. Responses_Suite2p.pkl')

			# # Create the directory if it does not exist
			# (results_path / 'ROI traces').mkdir(parents=True, exist_ok=True)
			# (results_path / 'Videos with deltaF over F').mkdir(parents=True, exist_ok=True)
			# (results_path / 'Anatomy with ROIs').mkdir(parents=True, exist_ok=True)
			# os.makedirs(whole_data_path_save, exist_ok=True)






if 'delay' in fish_name:
	interval_between_cs_onset_us_onset = 9  # s
elif 'trace' in fish_name:
	interval_between_cs_onset_us_onset = 13  # s
elif 'control' in fish_name:
	interval_between_cs_onset_us_onset = 9  # s
else:
	interval_between_cs_onset_us_onset = None


all_data.planes[2].trials[3].protocol



#* Load the data before motion correction.
# region Load the data before motion correction
with open(path_pkl_responses, 'rb') as file:
	all_data = pickle.load(file)



print('Analyzing fish: ', fish_name)


# endregion


num_planes = len(all_data.planes)
num_trials_per_plane = len(all_data.planes[0].trials)

shape_ = all_data.planes[0].trials[0].images.shape[1:]
image_dim_1 = shape_[0]
image_dim_2 = shape_[1]

#region Maps per single trials
##* CS positive response








for plane_i, plane in tqdm(enumerate(all_data.planes)):
	for trial_i, trial in enumerate(plane.trials):

		# if trial_i != 2:
	# 	# 	continue
	# 	break
	# break

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



						# # 		#* Align the frames
						# # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! correct for single plane
						# # 		trial_images.values = fi.align_frames(trial_images.to_numpy(), np.stack([trial_images['shift correction in X'].to_numpy(), trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))


						# # 		break
						# # 	break


						# # trial.__dict__.keys()




						# # A = trial_images/trial.pre_cs_mean


						# # plt.imshow(fi.add_colors_to_world_improved_2(trial.anatomy/5, A[95,:,:].to_numpy(), colormap='inferno', activity_threshold=0.2, alpha=1), interpolation='none')

						# # # Create a list to store frames
						# # frames = []

						# # # Generate frames and add to the list
						# # for i in range(A.shape[0]):
						# # 	frame = fi.add_colors_to_world_improved_2(trial.anatomy / 5,
						# # 											  fi.normalize_image(A[i, :, :], quantiles=(0.5, 1)), 
						# # 											  colormap='inferno', activity_threshold=0.2, alpha=0.5)
						# # 	frames.append((frame * 255).astype(np.uint8))  # Convert to 8-bit format

						# # # Save the frames as a multipage TIFF
						# # tifffile.imwrite(r"C:\Users\joaquim\Desktop\activity_movie.tiff", frames, photometric='rgb')

						# # trial.__dict__.keys()


						# response_frames_per_plane = []  # List to store response frames for all planes

						# for plane_i, plane in enumerate(all_data.planes):
						# 	plane_frames = []  # List to store frames for the current plane
						# 	for trial_i, trial in enumerate(plane.trials):
						# 		# response = np.where(trial.cs_us_vs_pre > 0, trial.cs_us_vs_pre, 0)

						# 		# response = block_reduce(trial.cs_us_vs_pre, block_size=(p.voxel_bin_size, p.voxel_bin_size), func=np.mean, cval=0)
						# 		# plane_frames.append(response)
						# 		plane_frames.append(trial.cs_us_vs_pre)

						# 		# response_frame = fi.normalize_image(response, quantiles=(0, 1))
						# 		# plane_frames.append(response_frame)  # Add response_frame to the current plane's list

						# 	response_frames_per_plane.append(np.array(plane_frames))  # Add the current plane's frames to the main list

						# 	# break

						# 		# plt.imshow(response_frame, interpolation='none')

						# 		# break

						# 		# image = fi.add_colors_to_world_improved_2(trial.anatomy*2.5, response_frame, colormap='inferno', activity_threshold=0.4, alpha=1)

						# 		# print(trial.trial_number)
						# 	# break



						# response_frames_per_plane = np.array(response_frames_per_plane)




						# for plane_i, plane in enumerate(all_data.planes):
						# 	plane_frames = []  # List to store frames for the current plane
						# 	for trial_i, trial in enumerate(plane.trials):
						# 		# response = np.where(trial.cs_us_vs_pre > 0, trial.cs_us_vs_pre, 0)
						# 		# response_frame = fi.normalize_image(response, quantiles=(0, 1))
						# 		# image = fi.add_colors_to_world_improved_2(trial.anatomy*2.5, response_frame, colormap='inferno', activity_threshold=0.4, alpha=1)

						# 		# image = fi.normalize_image(response_frames_per_plane[plane_i,trial_i,:,:], quantiles=(0, 1))
						# 		image = response_frames_per_plane[plane_i,trial_i,:,:]
						# 		# image = (image * 255).astype(np.uint8)

								
						# 		# Rotate the image by 90 degrees to the left
						# 		image = np.rot90(image, k=1)

						# 		# # Add plane and trial number as text in the top-right corner
						# 		# plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
						# 		# plt.imshow(image, interpolation='none')
						# 		# plt.axis('off')
								
						# 		# plt.text(10, 10, f'Plane {plane_i+1}', color='darkgray', fontsize=20, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

						# 		# plt.text(image.shape[1] - 10, 10, f'{phase_list[trial_i]} {trial_numbers_list[plane_i][trial_i]}', color=color_list[trial_i], fontsize=20, ha='right', va='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

						# 		# plt.tight_layout(pad=0)
								
						# 		# # Render the image with the label
						# 		# plt_canvas = plt.gca().figure.canvas
						# 		# plt_canvas.draw()
								
						# 		# plt_canvas = FigureCanvas(plt.gcf())
						# 		# plt_canvas.draw()
						# 		# labeled_image = np.frombuffer(plt_canvas.buffer_rgba(), dtype=np.uint8).reshape(plt_canvas.get_width_height()[::-1] + (4,))
						# 		# # plt.show()
						# 		# plt.close()

						# 		plane_frames.append(image)  # Add labeled frame to the list



						# 	tiff_path_1 = Path(r"C:\Users\joaquim\Desktop") / f'{fish_name}_plane_{plane_i}_forMeasure.tiff'
						# 	tifffile.imwrite(tiff_path_1, np.array(plane_frames).astype(np.float32))

						# 		# print(trial.trial_number)
						# 	# break


						# 	# # Save the frames as a multipage TIFF for the current plane
						# 	# tiff_path = Path(r"C:\Users\joaquim\Desktop") / f'{fish_name}_plane_{plane_i}_positive_responses FOR ROI.tiff'
						# 	# tifffile.imwrite(tiff_path, plane_frames, photometric='rgb')






						# # import matplotlib.pyplot as plt
						# # import numpy as np
						# # import pandas as pd

						# # # Create the full X range
						# # x_vals = list(range(1,81))

						# # # Define normalized vigor full list (length 80)
						# # normalized_vigor = [
						# #     0.96818185, 0.82162558, 1.04754199, 0.81429924, 1.4116402, 0.92850147, 1.28186846, 0.87243636,
						# #     1.04699759, 0.76536788, 1.05207526, 0.8438421, 1.00827338, 1.03175947, 0.67962354, 0.91921722,
						# #     0.74633776, 0.76595058, 0.76682449, 0.92884683, 0.83953546, 0.80015024, 0.85795741, 0.89144142,
						# #     0.86608158, 0.85989069, 0.85999002, 0.66742527, 0.77793336, 0.92570658, 0.74650336, 0.81889618,
						# #     0.81210486, 0.81736349, 0.76812621, 0.82313555, 0.7307879, 0.86480148, 0.86040372, 0.80863186,
						# #     0.88226972, 1.02518478, 0.79287218, 0.97154286, 0.89952258, 0.892217, 0.88360497, 0.83900208,
						# #     0.94697894, 0.85920551, 0.94681429, 0.83125647, 0.79226415, 0.79833738, 0.93320453, 0.87190693,
						# #     0.84497395, 0.94659986, 0.9395544, 0.95693298, 0.90386718, 0.84557724, 0.95043089, 1.02985856,
						# #     0.86861104, 0.99763037, 0.95744782, 0.95152197, 0.90897697, 0.98695234, 0.96045253, 0.98875049,
						# #     1.01066576, 0.97929391, 1.023214, 1.02731923, 1.01134371, 0.90000822, 1.05893978, 1.02277828
						# # ]

						# # # Create DataFrame with NaNs
						# # df = pd.DataFrame({
						# #     'X': x_vals,
						# #     'normalized_vigor': normalized_vigor,
						# #     'A': np.nan,
						# #     'B': np.nan,
						# #     'C': np.nan,
						# #     'D': np.nan,
						# #     'E': np.nan,
						# #     'F': np.nan
						# # })

						# # # Insert known A–F values at correct X positions
						# # known_values = {
						# #     5: [-0.025, 0.186, 0.003, 0.055, -0.051, 0.015],
						# #     6: [-0.033, 0.118, -0.008, 0.128, -0.013, 0.038],
						# #     15: [0.003, 0.092, 0.006, 0.023, 0.006, 0.167],
						# #     16: [0.055, 0.132, -0.002, 0.023, 0.017, 0.185],
						# #     25: [0.089, 0.069, 0.01, 0.008, 0.178, -0.03],
						# #     26: [0.08, 0.036, 0.028, 0.017, 0.065, -0.023],
						# #     35: [0.068, 0.043, 0.03, 0.013, 0.115, 0.003],
						# #     36: [0.069, 0.04, 0.034, 0.036, 0.132, -0.011],
						# #     45: [0.125, 0.128, 0.083, 0.014, 0.083, 0.006],
						# #     46: [0.17, 0.14, 0.073, 0.033, 0.008, 0.041],
						# #     55: [0.093, 0.052, 0.041, -0.006, 0.046, 0.009],
						# #     56: [0.083, 0.064, 0.061, -0.003, 0.034, 0.013],
						# #     67: [0.008, 0.024, -0.008, 0.032, -0.007, 0.031],
						# #     68: [0.018, 0.039, 0.022, 0.002, 0.001, 0.055],
						# #     77: [0.024, 0.177, -0.001, 0.000824, -0.003, 0.009],
						# #     78: [0.016, 0.21, -0.000175, -0.000119, 0.012, -0.002]
						# # }

						# # # Fill known values
						# # for x, values in known_values.items():
						# # 	if x in x_vals:  # Ensure the key exists in x_vals
						# # 		idx = df.index[df['X'] == x][0]
						# # 		df.loc[idx, ['A', 'B', 'C', 'D', 'E', 'F']] = values

						# # # Plotting
						# # fig, ax1 = plt.subplots(figsize=(12, 6))

						# # # Remove rows with NaN values
						# # df_cleaned = df.dropna()

						# # # Primary Y-axis: A–F
						# # for col in ['A', 'C', 'D', 'E']:
						# # 	ax1.plot(df_cleaned['X'], df_cleaned[col], label=col, marker='o')
						# # ax1.set_xlabel('X Axis')
						# # ax1.set_ylabel('A–F values')
						# # ax1.grid(True)

						# # # Combined legend
						# # lines1, labels1 = ax1.get_legend_handles_labels()
						# # ax1.legend(lines1, labels1, loc='upper left')

						# # plt.title('A–F Time Series (X = 1 to 80)')
						# # plt.tight_layout()
						# # plt.show()









						# # Build one 5D stack for all planes and save a single multipage TIFF
						# # Axes order: P (planes), C (channels: 0=response, 1=anatomy), T (trials), Y, X
						# response_pos_planes = []
						# response_neg_planes = []
						# anatomy_planes = []

						# for plane in all_data.planes:
						# 	response_pos_frames = []
						# 	response_neg_frames = []
						# 	anatomy_frames = []

						# 	for trial in [plane.trials[-1]]:
						# 		resp_pos = np.where(trial.cs_us_vs_pre > 0, trial.cs_us_vs_pre, 0)
						# 		resp_pos_norm = fi.normalize_image(resp_pos, quantiles=(0, 1))

						# 		resp_neg = np.where(trial.cs_us_vs_pre < 0, trial.cs_us_vs_pre, 0)
						# 		resp_neg_norm = fi.normalize_image(-resp_neg, quantiles=(0, 1))

						# 		# Rotate by 90 degrees left
						# 		response_pos_frames.append(np.rot90(resp_pos_norm, k=1))
						# 		response_neg_frames.append(np.rot90(resp_neg_norm, k=1))
						# 		anatomy_frames.append(np.rot90(trial.anatomy, k=1))

						# 	response_pos_planes.append(np.asarray(response_pos_frames, dtype=np.float32))   # (T, Y, X)
						# 	response_neg_planes.append(np.asarray(response_neg_frames, dtype=np.float32))   # (T, Y, X)
						# 	anatomy_planes.append(np.asarray(anatomy_frames, dtype=np.float32))     # (T, Y, X)

						# # responses = np.stack(response_planes, axis=0).astype(np.float32)   # (P, T, Y, X)
						# # anatomies = np.stack(anatomy_planes, axis=0).astype(np.float32)    # (P, T, Y, X)

						# # combined_all = np.stack([responses, anatomies], axis=1)            # (P, C=2, T, Y, X)

						# tiff_path = Path(r"C:\Users\joaquim\Desktop") / f"{fish_name}_all_planes_anatomy.tiff"
						# tifffile.imwrite(tiff_path, anatomy_planes)


						# tiff_path = Path(r"C:\Users\joaquim\Desktop") / f"{fish_name}_all_planes_positiveResponses.tiff"
						# tifffile.imwrite(tiff_path, response_pos_planes)

						# tiff_path = Path(r"C:\Users\joaquim\Desktop") / f"{fish_name}_all_planes_negativeResponses.tiff"
						# tifffile.imwrite(tiff_path, response_neg_planes)




						# __________________




						# for plane_i, plane in enumerate(all_data.planes):
						# 	plane_frames = []  # List to store frames for the current plane
						# 	for trial_i, trial in enumerate(plane.trials):
						# 		response = np.where(trial.cs_us_vs_pre > 0, trial.cs_us_vs_pre, 0)
						# 		response_frame = fi.normalize_image(response, quantiles=(0, 1))
						# 		image = fi.add_colors_to_world_improved_2(trial.anatomy*5, response_frame, colormap='inferno', activity_threshold=0.4, alpha=1)

						# 		# Rotate the image by 90 degrees to the left
						# 		image = np.rot90(image, k=1)

						# 		# Add plane and trial number as text in the top-right corner
						# 		plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
						# 		plt.imshow(image, interpolation='none')
								
						# 		plt.axis('off')
								
						# 		plt.text(10, 10, f'{phase_list[trial_i]}, trial {trial_numbers_list[plane_i][trial_i]}', color=color_list[trial_i], fontsize=30, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

						# 		plt.text(image.shape[1] - 10, 10, f'Plane {plane_i+1}', color='darkgray', fontsize=15, ha='right', va='top', bbox=dict(facecolor='black', alpha=0.9, edgecolor='none'))

						# 		plt.tight_layout(pad=0)
								
						# 		# Render the image with the label
						# 		plt_canvas = plt.gca().figure.canvas
						# 		plt_canvas.draw()
								
						# 		plt_canvas = FigureCanvas(plt.gcf())
						# 		plt_canvas.draw()
						# 		labeled_image = np.frombuffer(plt_canvas.buffer_rgba(), dtype=np.uint8).reshape(plt_canvas.get_width_height()[::-1] + (4,))
						# 		plt.close()

						# 		# Save each frame as a PNG with 300 dpi
						# 		png_path = Path(r"C:\Users\joaquim\Desktop") / f'{fish_name}_plane_{plane_i}_trial_{trial_i}_positive_response  new.png'
						# 		# plt.imsave(png_path, labeled_image, dpi=600)

						# 		plane_frames.append(labeled_image)  # Add labeled frame to the list

						# 	# Save the frames as a multipage TIFF for the current plane
						# 	tiff_path = Path(r"C:\Users\joaquim\Desktop") / f'{fish_name}_plane_{plane_i}_positive_responses  new.tiff'
						# 	tifffile.imwrite(tiff_path, plane_frames, photometric='rgb')

						# plt.imshow(labeled_image)




						# #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


plt.imshow(np.sum(all_data.anatomical_stack, axis=0))

all_data.planes[0].trials[0].__dict__.keys()



fig, axs = plt.subplots(num_planes, num_trials_per_plane, squeeze=False, facecolor='w')
for plane_i, plane in enumerate(all_data.planes):
	plane_frames = []  # List to store frames for the current plane
	for trial_i, trial in enumerate(plane.trials):
		response = np.where(trial.cs_us_vs_pre > 0, trial.cs_us_vs_pre, 0)
		response_frame = fi.normalize_image(response, quantiles=(0, 1))
		image = fi.add_colors_to_world_improved_2(trial.anatomy*2.5, response_frame, colormap='inferno', activity_threshold=0.4, alpha=1)
		axs[plane_i, trial_i].imshow(image, interpolation='none')
		axs[plane_i, trial_i].axis('off')

		# Append the frame to the list for saving as TIFF
		plane_frames.append(image)  # Convert to 8-bit format

	# Save the frames as a multipage TIFF for the current plane
	tiff_path = Path(r"C:\Users\joaquim\Desktop") / f'{fish_ID}_plane_{plane_i}_positive_responses.tiff'
	tifffile.imwrite(tiff_path, plane_frames, photometric='rgb')

fig.subplots_adjust(hspace=0, wspace=0.05)

fig.set_size_inches(len(all_data.planes[0].trials) *image_dim_2/100,len(all_data.planes) * image_dim_1/100)

fig.suptitle(f'{fish_name} Positive responses to CS', fontsize=10, y=0.95)

fig.savefig(results_figs_path_save / '6.1. Positive responses to CS_Suite2p.png', dpi=600, facecolor='white', bbox_inches='tight')





							# # ##* CS negative response
							# # fig, axs = plt.subplots(num_planes, num_trials_per_plane, figsize=(15 * image_dim_1/614, 7 * image_dim_2/350), squeeze=False)
							# # for plane_i, plane in enumerate(all_data.planes):
							# # 	for trial_i, trial in enumerate(plane.trials):
							# # 		response = np.where(trial.cs_us_vs_pre < 0, trial.cs_us_vs_pre, 0)
							# # 		response_frame = fi.normalize_image(-response, quantiles=(0, 1))
							# # 		# response_thr = np.quantile(response_frame, 0.99)
							# # 		axs[plane_i,trial_i].imshow(fi.add_colors_to_world_improved_2(trial.anatomy * 2.5, response_frame, colormap='inferno', activity_threshold=0.5, alpha=1), interpolation='none')
							# # 		axs[plane_i,trial_i].axis('off')
							# # fig.subplots_adjust(hspace=0.01, wspace=0)
							# # fig.tight_layout()
							# # fig.suptitle(f'{fish_name}/nNegative responses to CS', fontsize=10, y=0.95)
							# # fig.savefig(results_figs_path_save / '7.1. Negative responses to CS.png', dpi=600, facecolor='white', bbox_inches='tight')

							# # ##* US positive response
							# fig, axs = plt.subplots(num_planes, num_trials_per_plane, figsize=(15 * image_dim_1/614, 7 * image_dim_2/350), squeeze=False)
							# for plane_i, plane in enumerate(all_data.planes):
							# 	for trial_i, trial in enumerate(plane.trials):
							# 		response = np.where(trial.post_us_vs_pre > 0, trial.post_us_vs_pre, 0)
							# 		response_frame = fi.normalize_image(response, quantiles=(0, 1))
							# 		axs[plane_i,trial_i].imshow(fi.add_colors_to_world_improved_2(trial.anatomy * 2.5, response_frame, colormap='inferno', activity_threshold=0.3, alpha=1), interpolation='none')
							# 		axs[plane_i,trial_i].axis('off')
							# fig.subplots_adjust(hspace=0.01, wspace=0)
							# fig.tight_layout()
							# fig.suptitle(f'{fish_name}/nPositive responses to US', fontsize=10, y=0.95)
							# fig.savefig(results_figs_path_save / '8. Positive responses to US.png', dpi=600, facecolor='white', bbox_inches='tight')

							# # ##* US negative response
							# # fig, axs = plt.subplots(num_planes, num_trials_per_plane, figsize=(15 * image_dim_1/614, 7 * image_dim_2/350), squeeze=False)
							# # for plane_i, plane in enumerate(all_data.planes):
							# # 	for trial_i, trial in enumerate(plane.trials):
							# # 		response = np.where(trial.post_us_vs_pre < 0, trial.cs_us_vs_pre, 0)
							# # 		response_frame = fi.normalize_image(-response, quantiles=(0, 1))
							# # 		axs[plane_i,trial_i].imshow(fi.add_colors_to_world_improved_2(trial.anatomy * 2.5, response_frame, colormap='inferno', activity_threshold=0, alpha=1), interpolation='none')
							# # 		axs[plane_i,trial_i].axis('off')
							# # fig.subplots_adjust(hspace=0.01, wspace=0)
							# # fig.tight_layout()
							# # fig.suptitle(f'{fish_name}/nNegative responses to US', fontsize=10, y=0.95)
							# # fig.savefig(results_figs_path_save / '9. Negative responses to US.png', dpi=600, facecolor='white', bbox_inches='tight')
							# # #endregion



							# #region Maps per pair of trials of the same plane
							# #* Create anatomical scaffold and calculate response for every pair of trials of a plane
							# for plane_i, plane in tqdm(enumerate(all_data.planes)):

							# 	plane_anatomies = np.stack([all_data.planes[plane_i].trials[i].template_image for i in range(len(all_data.planes[plane_i].trials))])
							# 	#* Anatomy of a pair of trials of the same plane

							# 	plane_anatomies = [np.mean(np.stack(plane_anatomies[start_index : start_index + p.step], axis=0), axis=0) for start_index in range(0, num_trials_per_plane, p.step)]

							# 	plane_anatomies = [fi.normalize_image(anatomy, (0.01,0.99)) / 5 for anatomy in plane_anatomies]


							# 	plane_cs_us_vs_pre = np.stack([all_data.planes[plane_i].trials[i].cs_us_vs_pre for i in range(len(all_data.planes[plane_i].trials))])

							# 	plane_cs_us_vs_pre = [np.mean(np.stack(plane_cs_us_vs_pre[start_index : start_index + p.step], axis=0), axis=0) for start_index in range(0, num_trials_per_plane, p.step)]


							# 	plane.anatomies = plane_anatomies
							# 	plane.cs_us_vs_pre = plane_cs_us_vs_pre

							# 	all_data.planes[plane_i] = plane

							# ##* CS positive response
							# num_pairs_trials = len(all_data.planes[0].trials) // 2

							# # phase_list = ['Pre-Train'] + ['Train']
							# # color_list = ['white', 'red']

							# phase_list = phase_list[::2]
							# color_list = color_list[::2]


							# # fig, axs = plt.subplots(num_planes, num_pairs_trials, figsize=(9 * image_dim_1 / 614, 7 * image_dim_2 / 350), squeeze=False)
							# for plane_i, plane in enumerate(all_data.planes):
							# 	plane_frames = []  # List to store frames for the current plane

							# 	for trial_pair_i in range(num_pairs_trials):
							# 		anatomy = plane.anatomies[trial_pair_i]
							# 		cs_us_vs_pre = plane.cs_us_vs_pre[trial_pair_i]

							# 		response = np.where(cs_us_vs_pre > 0, cs_us_vs_pre, 0)
							# 		response_frame = fi.normalize_image(response, quantiles=(0, 1))
							# 		image = fi.add_colors_to_world_improved_2(anatomy, response_frame, colormap='inferno', activity_threshold=0.3, alpha=1)

							# 		# Rotate the image by 90 degrees to the left
							# 		image = np.rot90(image, k=1)

							# 		# Add plane and trial pair number as text in the top-right corner
							# 		plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
							# 		plt.imshow(image, interpolation='none')
							# 		plt.axis('off')

							# 		plt.text(10, 10, f'Plane {plane_i+1}', color='darkgray', fontsize=18, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

							# 		trials_numbers = trial_numbers_list[plane_i,trial_pair_i*2:trial_pair_i*2+2]

							# 		plt.text(image.shape[1] - 10, 10, f'{phase_list[trial_pair_i]} ({trials_numbers[0]},{trials_numbers[1]})', color=color_list[trial_pair_i], fontsize=18, ha='right', va='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

							# 		plt.tight_layout(pad=0)

							# 		# Render the image with the label
							# 		plt_canvas = plt.gca().figure.canvas
							# 		plt_canvas.draw()

							# 		plt_canvas = FigureCanvas(plt.gcf())
							# 		plt_canvas.draw()
							# 		labeled_image = np.frombuffer(plt_canvas.buffer_rgba(), dtype=np.uint8).reshape(plt_canvas.get_width_height()[::-1] + (4,))
							# 		plt.close()

							# 		plane_frames.append(labeled_image)  # Add labeled frame to the list

							# 	# Save the frames as a multipage TIFF for the current plane
							# 	tiff_path = Path(r"C:\Users\joaquim\Desktop") / f'{fish_name}_plane_{plane_i}_positive_responses_pairs.tiff'
							# 	tifffile.imwrite(tiff_path, plane_frames, photometric='rgb')









							# # Build one 5D stack for all planes and save a single multipage TIFF
							# # Axes order: P (planes), C (channels: 0=response, 1=anatomy), T (trials), Y, X
							# response_pos_planes = []
							# response_neg_planes = []
							# anatomy_planes = []

							# for plane in all_data.planes:
							# 	response_pos_frames = []
							# 	response_neg_frames = []
							# 	anatomy_frames = []

							# 	for trial in [plane.trials[-1]]:
							# 		resp_pos = np.where(trial.cs_us_vs_pre > 0, trial.cs_us_vs_pre, 0)
							# 		resp_pos_norm = fi.normalize_image(resp_pos, quantiles=(0, 1))

							# 		resp_neg = np.where(trial.cs_us_vs_pre < 0, trial.cs_us_vs_pre, 0)
							# 		resp_neg_norm = fi.normalize_image(-resp_neg, quantiles=(0, 1))

							# 		# Rotate by 90 degrees left
							# 		response_pos_frames.append(np.rot90(resp_pos_norm, k=1))
							# 		response_neg_frames.append(np.rot90(resp_neg_norm, k=1))
							# 		anatomy_frames.append(np.rot90(trial.anatomy, k=1))

							# 	response_pos_planes.append(np.asarray(response_pos_frames, dtype=np.float32))   # (T, Y, X)
							# 	response_neg_planes.append(np.asarray(response_neg_frames, dtype=np.float32))   # (T, Y, X)
							# 	anatomy_planes.append(np.asarray(anatomy_frames, dtype=np.float32))     # (T, Y, X)

							# # responses = np.stack(response_planes, axis=0).astype(np.float32)   # (P, T, Y, X)
							# # anatomies = np.stack(anatomy_planes, axis=0).astype(np.float32)    # (P, T, Y, X)

							# # combined_all = np.stack([responses, anatomies], axis=1)            # (P, C=2, T, Y, X)

							# tiff_path = Path(r"C:\Users\joaquim\Desktop") / f"{fish_name}_all_planes_anatomy.tiff"
							# tifffile.imwrite(tiff_path, anatomy_planes)

							# tiff_path = Path(r"C:\Users\joaquim\Desktop") / f"{fish_name}_all_planes_positiveResponses.tiff"
							# tifffile.imwrite(tiff_path, response_pos_planes)

							# tiff_path = Path(r"C:\Users\joaquim\Desktop") / f"{fish_name}_all_planes_negativeResponses.tiff"
							# tifffile.imwrite(tiff_path, response_neg_planes)




							# # 	break

							# # fig.subplots_adjust(hspace=0, wspace=0)
							# # fig.tight_layout()
							# # fig.suptitle(f'{fish_name} Positive responses to CS_pairs of trials', fontsize=10, y=1)

							# # fig.savefig(results_figs_path_save / '6.2. Positive responses to CS.png', dpi=600, facecolor='white', bbox_inches='tight')


							# # #* CS negative response
							# # fig, axs = plt.subplots(num_planes, num_pairs_trials, figsize=(9 * image_dim_1/614, 7 * image_dim_2/350), squeeze=False)
							# # for plane_i, plane in enumerate(all_data.planes):

							# # 	for trial_pair_i in range(num_pairs_trials):

							# # 		anatomy = plane.anatomies[trial_pair_i]
							# # 		cs_us_vs_pre = plane.cs_us_vs_pre[trial_pair_i]

							# # 		response = np.where(cs_us_vs_pre < 0, cs_us_vs_pre, 0)
							# # 		response_frame = fi.normalize_image(-response, quantiles=(0, 1))
							# # 		# response_thr = np.quantile(response_frame, 0.99)
							# # 		axs[plane_i,trial_pair_i].imshow(fi.add_colors_to_world_improved_2(anatomy, response_frame, colormap='inferno', activity_threshold=0.5, alpha=1), interpolation='none')
							# # 		axs[plane_i,trial_pair_i].axis('off')
							# # fig.subplots_adjust(hspace=0, wspace=0)
							# # fig.tight_layout()
							# # fig.suptitle(f'{fish_name}\nNegative responses to CS_pairs of trials', fontsize=10, y=1)

							# # fig.savefig(results_figs_path_save / '7.2. Negative responses to CS.png', dpi=600, facecolor='white', bbox_inches='tight')


							# # exec(open('5.Save_data_as_HDF5.py').read())


							# print('END')





							# # #region Create anatomical scaffold for every plane
							# # plane_anatomy = [[] for _ in range(num_planes)]


							# # fig, axs = plt.subplots(num_planes, len(all_data.planes[0].trials)//2, figsize=(10, 50), squeeze=False)

							# # for plane_i, plane in tqdm(enumerate(all_data.planes)):

							# # 	plane_anatomies = np.stack([all_data.planes[plane_i].trials[i].template_image for i in range(len(all_data.planes[plane_i].trials))])

							# # 	plane_anatomies = [np.mean(plane_anatomies[start_index : start_index + p.step], axis=0) for start_index in range(0, num_trials_per_plane, p.step)]
							# # 	# plt.figure('Processed Anatomy Channel')
							# # 	plt.imshow(plane_anatomies[0], interpolation='none')
							# # 	plt.title(f'Plane {plane_i} Processed Anatomy Channel')
							# # 	plt.colorbar(shrink=0.5)
							# # 	plt.show()

							# # 	# for trial_i in tqdm(range(len(all_data.planes[0].trials))):



							# # 	# 	break
							# # 	# break

							# # 		# trial_images = all_data.planes[plane_i].trials[trial_i].images.copy()
							# # 		# trial_images.values = fi.align_frames(trial_images.to_numpy(),
							# # 		# 								np.stack([trial_images['shift correction in X'].to_numpy(),
							# # 		# 								trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))
							# # 		# trial_images = trial_images.sel({'Time (ms)':trial_images['mask good frames']})

							# # 		# # plane_anatomy[plane_i].append(trial_images.values)



							# # 	plane_anatomies = [
							# # 		fi.calculate_anatomy(
							# # 			np.concatenate(plane_anatomies[start_index : start_index + p.step], axis=0).astype(np.float64),
							# # 			border_size
							# # 		)
							# # 		for start_index in range(0, num_trials_per_plane, p.step)
							# # 	]

							# # 	all_data.planes[plane_i].anatomy_channel = plane_anatomies





							# # 	for i in range(len(plane_anatomies)):
									
							# # 		im = axs[plane_i,i].imshow(plane_anatomies[i], interpolation='none')
							# # 		# fig.colorbar(im, ax=axs[plane_i,], shrink=0.5)

							# # 		axs[plane_i,i].set_xticks([])
							# # 		axs[plane_i,i].set_yticks([])
							# # 		axs[plane_i,i].set_xticklabels([])
							# # 		axs[plane_i,i].set_yticklabels([])

							# # 		axs[plane_i,i].set_title(f"Plane {plane_i}")

							# # 	# break


							# # fig.set_size_inches(3, 28)
							# # fig.subplots_adjust(hspace=0, wspace=0)
							# # fig.savefig(results_figs_path_save / '7. Planes anatomy.png', dpi=600, facecolor='white', bbox_inches='tight', facecolor='white')



							# # # #* Using a general template per plane

							# # # for plane_i, plane in enumerate(all_data.planes):

							# # # 	plane_baseline_color = plane.template

							# # # 	for trial_i, trial in enumerate(plane.trials):

							# # # 		color_frame_original = np.clip(trial.cs_us_vs_pre, 0, 1)

							# # # 		trial_red_channel = plane_baseline_color * (1 - color_frame_original)
							# # # 		trial_green_channel = trial_red_channel + color_frame_original * color_frame_original
							# # # 		trial_blue_channel = trial_red_channel

							# # # 		trial_cs_response = np.stack([trial_red_channel,trial_green_channel,trial_blue_channel])


							# # # 		trial_cs_response = np.moveaxis(trial_cs_response, 0, -1)

							# # # 		plt.imshow(Image.fromarray((trial_cs_response*256).astype(np.uint8)))
							# # # 		plt.show()

									
							# # # 		# break
							# # # 	break

							# # # plane_numbers = np.zeros((15,4))

							# # # all_data = c.Data(all_data.planes, anatomical_stack_images)

							# # #endregion







							# # #* Save the images as a multipage tiff
							# # #region Save the images as a multipage tiff
							# # tifffile = []
							# # for trial_i, trial in enumerate(all_data.planes[2].trials):
							# # 	tifffile.append(Image.fromarray(trial.cs_positive_response))

							# # # Save tifffile as a multipage tiff
							# # tiff_path = imaging_path_ / 'cs_positive_response_multipage new 1.tiff'
							# # tifffile[0].save(tiff_path, save_all=True, append_images=tifffile[1:])



							# # # Load the TIFF file
							# # tiff_path = imaging_path_ / 'cs_positive_response_multipage new 1.tiff'


							# # # Label a specific frame in the TIFF stack
							# # frame_index = 0  # Change this to the index of the frame you want to label
							# # labeled_frame = tifffile[1]

							# # # Display the labeled frame
							# # plt.imshow(labeled_frame, cmap='gray')
							# # plt.title(f'Labeled Frame {1}')
							# # plt.colorbar()
							# # plt.show()



















							# # # region Voxel analysis
							# # #* Voxel analysis

							# # #* Bin the 2D images.

							# # fig, axs = plt.subplots(num_planes, len(all_data.planes[0].trials), figsize=(15 * image_dim_1/614, 7 * image_dim_2/350), squeeze=False)

							# # for plane_i, plane in enumerate(all_data.planes):
							# # 	for trial_i, trial in enumerate(plane.trials):


							# # 		protocol = trial.protocol

							# # 		cs_beg_time = protocol.loc[protocol['CS beg']!=0, 'Time (ms)'].values[0]
							# # 		cs_end_time = protocol.loc[protocol['CS end']!=0, 'Time (ms)'].values[0]
											
							# # 		if (us_beg_time := protocol.loc[protocol['US beg']!=0, 'Time (ms)']).empty:
							# # 			us_beg_time = cs_beg_time + interval_between_cs_onset_us_onset*1000
							# # 			us_end_time = us_beg_time + 100
							# # 		else:
							# # 			us_beg_time = us_beg_time.values[0]
							# # 			us_end_time = protocol.loc[protocol['US end']!=0, 'Time (ms)'].values[0]


							# # 		trial_images = trial.images.copy().fillna(0)

							# # 		trial_images.values = fi.align_frames(trial_images.to_numpy(), np.stack([trial_images['shift correction in X'].to_numpy(), trial_images['shift correction in Y'].to_numpy()]).swapaxes(0,1))


							# # 		trial_images = block_reduce(trial_images, block_size=(1, p.voxel_bin_size, p.voxel_bin_size), func=np.mean, cval=0)


							# # 		trial_images['mask before cs'] = trial_images['Time (ms)'] < cs_beg_time
							# # 		trial_images['mask cs-us'] = (trial_images['Time (ms)'] > cs_beg_time) & (trial_images['Time (ms)'] < us_beg_time)
							# # 		trial_images['mask after us'] = trial_images['Time (ms)'] > us_beg_time


							# # 		plt.imshow(np.mean(trial_images, axis=0), interpolation='none')

							# # 		break
							# # 	break








							# # 	deltaF = []
							# # 	deltaF_ratio = []

							# # 	for i in range(len(cs_indices)):

							# # 		baseline = np.nanmean(trial_images_good_binned[[cs_indices[i, 0] - 20, cs_indices[i, 0]]], axis=0)
									
							# # 		during_cs = np.nanmean(trial_images_good_binned[[cs_indices[i, 0], cs_indices[i, 1]]], axis=0)

							# # 		deltaF_ratio.append((during_cs - baseline) / baseline)

							# # 		if i == 0:
										
							# # 			deltaF.append((trial_images_good_binned[ : plane_trials_number_images[0]] - baseline) / baseline)

							# # 		elif i < len(cs_indices)-1:

							# # 			deltaF.append((trial_images_good_binned[np.cumsum(plane_trials_number_images)[i-1] : np.cumsum(plane_trials_number_images)[i]] - baseline) / baseline)

							# # 		else:
							# # 			deltaF.append((trial_images_good_binned[np.cumsum(plane_trials_number_images)[i-1] : ] - baseline) / baseline)

							# # 	deltaF = np.concatenate(deltaF)

							# # 	deltaF = np.where(np.isnan(deltaF), 0, deltaF)

							# # 	deltaF_ratio = np.array(deltaF_ratio)


							# # 	for i in range(len(cs_indices)):
							# # 		plt.imshow(deltaF_ratio[i], interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
							# # 		plt.colorbar(shrink=0.5)
							# # 		plt.title('DeltaF_SR')
							# # 		plt.show()


							# # 	A = np.mean(np.array([deltaF_ratio[0], deltaF_ratio[1]]), axis=0)
							# # 	B = np.mean(np.array([deltaF_ratio[2], deltaF_ratio[3]]), axis=0)



							# # 	plt.imshow(A, interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
							# # 	plt.colorbar(shrink=0.5)
							# # 	plt.title('DeltaF_SR A')
							# # 	plt.show()

							# # 	plt.imshow(B, interpolation='none', vmin=-10, vmax=10, cmap='RdBu_r')
							# # 	plt.colorbar(shrink=0.5)
							# # 	plt.title('DeltaF_SR B')
							# # 	plt.show()

							# # 	plt.imshow(B/A, interpolation='none', vmin=-100, vmax=100, cmap='RdBu_r')
							# # 	plt.colorbar(shrink=0.5)
							# # 	plt.title('DeltaF_SR B / DeltaF_SR A')
							# # 	plt.savefig(imaging_path_ /  fish_name / (fish_name + '_deltaF_SR_voxels_plane ' + str(plane_i) + '.tif'))
							# # 	plt.show()

							# # 	deltaF_ratio = np.concatenate(deltaF_ratio)

							# # #!!!
							# # 	# deltaF_ = np.empty(tuple([plane_trials_all_images.shape[0]] + list(deltaF.shape[1:]))) * np.nan
							# # 	# deltaF_[~plane_trials_mask_good_frames, :, :] = deltaF

							# # 	# deltaF = deltaF_.copy()

							# # 	# del deltaF_

							# # 	for i in range(len(cs_indices)):
							# # 		deltaF[cs_indices[i,0]:cs_indices[i,1],:20,-20:] = -100

							# # 	plt.imshow(np.nanmean(deltaF, axis=0))
							# # 	plt.colorbar(shrink=0.5)

							# # 	#* Save rois_zscore_over_time as a TIFF file.
							# # 	tifffile.imwrite(imaging_path_ /  fish_name / (fish_name + '_deltaF_voxels_plane ' + str(plane_i) + '.tif'), deltaF.astype('float32'))

							# # 	# endregion





							# # # #* Plot the position in the anatomical stack.
							# # # # region Position in the anatomical stack
							# # # try:
							# # # 	A = []
							# # # 	B = []

							# # # 	C = []
							# # # 	D = []

							# # # 	for i in range(num_planes):

							# # # 		for j in range(2):

							# # # 			A.append(all_data.planes[i].trials[j].position_anatomical_stack)

							# # # 			C.append(all_data.planes[i].trials[j].template_image)

							# # # 		for l in range(2,4):

							# # # 			B.append(all_data.planes[i].trials[l].position_anatomical_stack)

							# # # 			D.append(all_data.planes[i].trials[l].template_image)


							# # # 	A = np.array(A)
							# # # 	B = np.array(B)

							# # # 	C = np.array(C)
							# # # 	D = np.array(D)


							# # # 	sns.set_style('whitegrid')


							# # # 	path_ = path_home / fish_name


							# # # 	plt.xlabel('Trial before or after initial train')
							# # # 	plt.ylabel('Plane number in anatomical stack')
							# # # 	plt.plot(A, 'blue')
							# # # 	plt.plot(B, 'red')
							# # # 	plt.legend(['Before initial train', 'After initial train'])
							# # # 	plt.savefig(path_ / ('Where in the anatomical stack' + '.png'), dpi=300, bbox_inches='tight')


							# # # 	plt.xlabel('Trial before or after initial train')
							# # # 	plt.ylabel('Difference between planes imaged\n before and after initial train (μm)')
							# # # 	plt.plot(A-B, 'k')
							# # # 	plt.ylim(-10, 10)
							# # # 	plt.savefig(path_ / ('Difference when revisiting planes' + '.png'), dpi=300, bbox_inches='tight')


							# # # 	sns.set_style('white')
								
							# # # except:
							# # # 	pass
















							# # # 	# except:
							# # # 	# 	continue

							# # # # compression_level = 4
							# # # # compression_library = 'zlib'

							# # # # with pd.HDFStore(path_pkl, complevel=compression_level, complib=compression_library) as store:
								
							# # # # 	store.append(fish_name, planes_list, data_columns=[cs, us], expectedrows=len(fish.raw_data), append=False)

							# # # # 	store.get_storer(fish.dataset_key()).attrs['metadata'] = fish.metadata._asdict()





							# # # #* For correlation map.

							# # # ##* Preparing the data for the correlation map.

							# # # # for plane_i, plane in enumerate(Data.planes):
							# # # # 	for trial_i, trial in enumerate(plane):

							# # # # A = [Data.planes[plane_i].trials[trial_i].images.values for trial_i, trial in enumerate(plane.trials) for plane_i, plane in enumerate(Data.planes)]

							# # # # B = np.sum([np.sum(x, axis=0) for x in A], axis=0)

							# # # # plt.imshow(B)


							# # # eye_mask = np.ones(all_data.planes[0].trials[0].images.shape[1:], dtype='bool')

							# # # #!
							# # # eye_mask[350:, 350:450] = False
							# # # eye_mask[:50, 350:450] = False
							# # # plt.imshow(eye_mask)

							# # # # # A = ndimage.uniform_filter(trial_images_good, size=(30, 30), axes=(1,2))
							# # # # ndimage.gaussian_filter(trial_images_good, sigma=gaussian_filter_sigma, axes=(1,2))

							# # # # plt.imshow(np.mean(A, axis=0))
							# # # # plt.colorbar()









							# # # for plane_i, plane in enumerate(all_data.planes):

							# # # 	# if plane_i not in [0,1,3,6,8,9,10,13]:
							# # # 	# 	continue
							# # # 	# break
							# # # #!
							# # # 	# plane.trials = plane.trials



							# # # 	#!!!!!!!!!!!!!!!!!!!!!!!!! DO ALL OF THIS FOR SINGLE TRIAL AND THEN CONCATENATE TO GET PLANE DATA




							# # # 	#* To get a correlation map for the whole plane data, we need to concatenate all the images of the trials.
							# # # 	# plane_trials_all_images = np.concatenate([t.images.values for t in plane.trials])
							# # # 	plane_trials_all_images = plane.get_all_images()

							# # # 	plt.title('All images from plane')
							# # # 	plt.imshow(np.mean(plane_trials_all_images, axis=0))
							# # # 	plt.colorbar
							# # # 	plt.show()


							# # # 	#* Get the number of images per trial.
							# # # 	plane_trials_number_images = np.array([t.images.shape[0] for t in plane.trials])


							# # # 	#* Get the indices of the cs in the images of the trials.
							# # # 	cs_indices = np.array([trial.get_stim_index(cs) for trial in plane.trials])

							# # # 	cs_indices[1:,0] += np.cumsum(plane_trials_number_images[:-1])
							# # # 	cs_indices[1:,1] += np.cumsum(plane_trials_number_images[:-1])









							# # # 	#* Discard good frames due to motion, gating of the PMT or plane change.
							# # # 	plane_trials_mask_good_frames = np.concatenate([t.mask_good_frames for t in plane.trials])
							# # # 	plane_bad_frames_index = np.where(plane_trials_mask_good_frames)[0]
							# # # 	trial_images_good = plane_trials_all_images[~plane_trials_mask_good_frames].copy()

							# # # 	plt.title('All good images from plane')
							# # # 	plt.imshow(np.mean(trial_images_good, axis=0))
							# # # 	plt.colorbar
							# # # 	plt.show()










							# # # 	#* Filter in space.
							# # # 	trial_images_good_filtered = ndimage.gaussian_filter(trial_images_good, sigma=gaussian_filter_sigma, axes=(1,2))

							# # # 	plt.title('All good images from plane filtered')
							# # # 	plt.imshow(np.mean(trial_images_good_filtered, axis=0))
							# # # 	plt.colorbar

							# # # #!!!!!!!!!!!!!!!!! move it further down
							# # # 	#* Calcultate the correlation map.
							# # # 	# Inspired in Suit2p. There, the function that computes the correlation map is celldetect2.getVmap.
							# # # 	correlation_map = np.linalg.norm(ndimage.gaussian_filter(trial_images_good, sigma=correlation_map_sigma, axes=(1,2)), axis=0)**2 / ndimage.gaussian_filter(np.linalg.norm(trial_images_good, axis=0), sigma=correlation_map_sigma)**2

							# # # 	plt.figure('Correlation map')
							# # # 	plt.imshow(correlation_map)
							# # # 	plt.colorbar(shrink=0.5)
							# # # 	plt.show()






							# # # 	#* Subtract the background.
							# # # 	# Pixel values equal to 0 are ignored to discard the artificial edges of the images that were introduced during the motion correction.
							# # # 	images_mean = np.nanmean(np.where(trial_images_good == 0, np.nan, trial_images_good), axis=(1,2))

							# # # 	images_mean = np.nanmean(trial_images_good, axis=(1,2))
							# # # 	for image_i in range(trial_images_good.shape[0]):
							# # # 		trial_images_good[image_i] -= images_mean[image_i]

							# # # 	del images_mean

							# # # 	#* Mask the background.
							# # # 	plane_images_mask_fish = np.where(np.median(trial_images_good, axis=0) <= 0, 0, 1).astype(dtype='bool')
							
							# # # 	#* Mask the background and the eyes.
							# # # 	plane_images_mask_fish_without_eyes = plane_images_mask_fish & eye_mask


							# # # 	#* Set to 0 the pixels that are not part of the fish in the images.
							# # # 	trial_images_good = np.where(plane_images_mask_fish_without_eyes, trial_images_good, 0)

							# # # 	plt.title('All good images from plane masked background')
							# # # 	plt.imshow(np.mean(trial_images_good, axis=0))
							# # # 	plt.colorbar(shrink=0.5)
							# # # 	plt.show()

							# # # 	#* Set to 0 the pixels that are not part of the fish in the correlation map.
							# # # 	correlation_map = np.where(plane_images_mask_fish_without_eyes, correlation_map, 0)

							# # # 	plt.title('Correlation map masked background')
							# # # 	plt.imshow(np.where(plane_images_mask_fish_without_eyes, correlation_map, 0))
							# # # 	plt.colorbar(shrink=0.5)
							# # # 	plt.show()





							# # # 	#* ROIs

							# # # 	all_traces, all_rois, used_pixels, correlation_map_ = get_ROIs(Nrois=100, correlation_map=correlation_map, images=trial_images_good_images_filtered, threshold=0.3, max_pixels=60)

							# # # 	images_times = trial_images.time.values


							# # # 	trial_time_ref = images_times[0]

							# # # 	trial_protocol = trial.protocol

							# # # 	cs_times = trial_protocol[trial_protocol[cs]!=0]
							# # # 	cs_times = cs_times.iloc[[0,-1]] if cs_times.shape[0] > 1 else cs_times

							# # # 	us_times = trial_protocol[trial_protocol[us]!=0]
							# # # 	us_times = us_times.iloc[[0,-1]] if us_times.shape[0] > 1 else us_times


							# # # 	images_times = images_times - trial_time_ref
							# # # 	cs_times = cs_times['Time (ms)'].values - trial_time_ref
							# # # 	us_times = us_times['Time (ms)'].values - trial_time_ref


							# # # 	number_traces = 50

							# # # 	fig, axs = plt.subplots(number_traces, 1, sharex=True)
							# # # 	# figsize=(10, 8)

							# # # 	for i in range(number_traces):

							# # # #!
							# # # 			axs[i].plot(images_times[:110], all_traces[i+50][:110])

							# # # 			if cs_times.shape[0] > 0:
							# # # 				axs[i].axvline(x=cs_times[0], color='g', linestyle='-')
							# # # 				axs[i].axvline(x=cs_times[1], color='g', linestyle='--')
										
							# # # 			if us_times.shape[0] > 0:
							# # # 				axs[i].axvline(x=us_times[0], color='r', linestyle='-')
							# # # 				axs[i].axvline(x=us_times[1], color='r', linestyle='--')

							# # # 		fig.show()


							# # # 		plt.imshow(zscore(all_traces, 1), aspect="auto", vmin=-3, vmax=3, cmap="RdBu_r")

							# # # 		plt.show()
							# # # 		plt.imshow(all_rois)
							# # # 		plt.show()
							# # # 		plt.imshow(correlation_map_)
							# # # 		plt.show()
							# # # 		plt.imshow(np.sum(trial_images_good_images, axis=0))
							# # # 		plt.show()
							# # # 		plt.imshow(original_correlation_map)
							# # # 		plt.show()
							# # # # fig,(ax1,ax2,ax3,ax4)= plt.subplots(1,4)
							# # # # ax1 = plt.subplot(121)
							# # # # img=ax1.imshow(zscore(all_traces, 1), aspect="auto", vmin=-3, vmax=3, cmap="RdBu_r")
							# # # # ax1.set_ylabel("trace ROI number")
							# # # # ax1.set_xlabel("frame number")
							# # # # fig.colorbar(img,ax=ax1)
							# # # # ax2 = plt.subplot(322)
							# # # # ax2.imshow(all_rois)
							# # # # ax3 = plt.subplot(324)
							# # # # ax3.imshow(correlation_map_)
							# # # # ax4 = plt.subplot(326)
							# # # # ax4.imshow(original_correlation_map)
							# # # # # plt.show()
							# # # # fig.tight_layout()


							# # # a = Data.planes[0].trials[0].images.values

							# # # plt.imshow(np.mean(a, axis=0))


							# # # b = Data.planes[0].trials[3].images.values

							# # # plt.imshow(np.mean(b, axis=0))


							# # # plt.imshow(np.mean(a, axis=0) - np.mean(b, axis=0), cmap='viridis')
							# # # plt.colorbar(shrink=0.5)


							# # # __dict__.keys()