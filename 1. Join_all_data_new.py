"""
Data joining and preprocessing pipeline.

This script synchronizes camera timestamps, protocol timing, galvo triggers,
tail tracking, and raw imaging frames into a unified dataset that is split by
plane and trial. The resulting Data object is saved for downstream motion
correction and analysis.
"""

# Imports
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
from my_general_variables import *
from my_paths_new import fish_name, path_home

# Configuration
RELOAD_MODULES = True
USE_PLOTLY_DARK = True

PANDAS_OPTIONS = {
 "mode.copy_on_write": True,
 "compute.use_numba": True,
 "compute.use_numexpr": True,
 "compute.use_bottleneck": True,
}


def configure_environment(use_plotly_dark: bool) -> None:
	for option, value in PANDAS_OPTIONS.items():
		pd.set_option(option, value)
	if use_plotly_dark:
		pio.templates.default = "plotly_dark"
	plotting_style.set_plot_style(use_constrained_layout=False)


def is_summer_time(fish_name: str) -> bool:
	date = int(fish_name.split('_')[0][4:6])
	return 4 <= date <= 10


if RELOAD_MODULES:
	reload(fi)
	reload(c)
	reload(p)

configure_environment(USE_PLOTLY_DARK)

# Paths and dataset selection
path_results_save = Path(r'F:\Results (paper)') / path_home.stem


fish_ID = '_'.join(fish_name.split('_')[:2])


behavior_path_home = path_home / 'Tail'
imaging_path_home = path_home / 'Neurons' / fish_name
results_figs_path_save = path_results_save / 'Neurons' / fish_name
results_figs_path_save.mkdir(parents=True, exist_ok=True)


whole_data_path_save = Path(r'H:\2-P imaging') / path_home.stem / fish_name
whole_data_path_save.mkdir(parents=True, exist_ok=True)

path_pkl_before_motion_correction = whole_data_path_save / (fish_ID + '_1. Before motion correction.pkl')


if path_pkl_before_motion_correction.exists():
	print('Already preprocessed: ', fish_name)
	print(path_pkl_before_motion_correction)

print('Analyzing fish: ', fish_name)

protocol_path = behavior_path_home / (fish_name + '_stim control.txt')
camera_path = behavior_path_home / (fish_name + '_cam.txt')
tracking_path = behavior_path_home / (fish_name + '_mp tail tracking.txt')

galvo_path = imaging_path_home / 'Imaging' / 'signalsfeedback.xls'
images_path = imaging_path_home / 'Imaging' / (fish_name + '_green.tif')

anatomy_1_path = imaging_path_home / 'Anatomical stack 1' / 'Anatomical stack 1.tif'

# Trial indexing rules depend on acquisition layout and path conventions.
summer_time = is_summer_time(fish_name)
relevant_cs = []
index_list = []

match str(path_home):

	case r'D:\2024 03_Delay 2-P 15 planes top part' | r'D:\2024 10_Delay 2-P 15 planes bottom part' | r'D:\2024 10_Delay 2-P 15 planes ca8 neurons':

		number_imaged_planes = 15
		number_reps_plane_consective = 2
		relevant_cs = [range(5,35), range(45,75)]

		index_list = [np.concatenate([[i+number_reps_plane_consective*x*number_imaged_planes, i+number_reps_plane_consective*x*number_imaged_planes+1] for x in range(len(relevant_cs))]) for i in range(0, number_reps_plane_consective * number_imaged_planes, number_reps_plane_consective)]

	case r'D:\2024 10_Delay 2-P single plane':
		number_imaged_planes = 1
		number_reps_plane_consective = 80
		relevant_cs = [np.arange(5,85)]

		index_list = relevant_cs

	case r'D:\2024 09_Delay 2-P 4 planes JC neurons':


		num_planes = 4
		num_reps_per_plane = 2

		relevant_cs = [range(5,15),
		   range(15,25), range(25,55), range(55,45), range(45,55),
		   range(55,65), range(65,75), range(75,85)]


		index_list = [
			np.concatenate([
				[i + num_reps_per_plane * x * num_planes, i + num_reps_per_plane * x * num_planes + 1]
				for x in range(len(relevant_cs))
			])
			for i in range(0, num_reps_per_plane * num_planes, num_reps_per_plane)
		]


if not relevant_cs:
	raise ValueError(f"Unsupported path_home for trial mapping: {path_home}")

relevant_cs = np.concatenate(relevant_cs)


# Load behavioral camera timestamps and frame numbers.
data = fi.read_camera(camera_path)
if data is None:
	raise FileNotFoundError(f"Camera file could not be read: {camera_path}")


data['AbsoluteTime'] = data['AbsoluteTime'].astype('float64')


print('Behavior camera started: ', pd.to_datetime(data['AbsoluteTime'].iat[0], unit='ms'))


# Estimate the effective acquisition rate and identify a stable reference frame.
predicted_framerate, reference_frame_id = fi.framerate_and_reference_frame(data)

if ela_time in data.columns:
	data = data.drop(columns=ela_time)

# Drop frames acquired before the stable reference frame.
data = data[data['Frame number'] >= reference_frame_id]


# Replace capture timestamps with an evenly spaced time base to reduce jitter.
abs_time_start = data['AbsoluteTime'].iat[0]
abs_time_stop = abs_time_start + len(data) * (1000 / predicted_framerate)
data['AbsoluteTime'] = np.linspace(abs_time_start, abs_time_stop, len(data))


protocol = fi.read_protocol(protocol_path)
if protocol is None:
	raise FileNotFoundError(f"Protocol file could not be read: {protocol_path}")


# Label each camera frame with CS and US trial indices from the protocol.
data = fi.identify_trials(data, protocol)


# Load galvo trigger signal used to mark imaging frame starts.
galvo = pd.read_csv(
	galvo_path,
	sep='\t',
	decimal=',',
	usecols=[0, 1],
	names=['AbsoluteTime', 'GalvoValue'],
	dtype={'GalvoValue': 'float64'},
	parse_dates=['AbsoluteTime'],
	date_format=r'%d/%m/%Y  %H:%M:%S,%f',
	skip_blank_lines=True,
	skipinitialspace=True,
	nrows=p.nrows,
).dropna(axis=0)
galvo = galvo.reset_index(drop=True)

# Convert galvo timestamps to Unix milliseconds for alignment.
galvo['AbsoluteTime'] = galvo['AbsoluteTime'].astype('int64') / 10**6


print('Galvo started: ', pd.to_datetime(galvo['AbsoluteTime'].iat[0], unit='ms'))

if summer_time:
	galvo['AbsoluteTime'] -= 60*60*1000

# Remove consecutive duplicate galvo values to simplify peak detection.
galvo = galvo[galvo[galvo_value].ne(galvo[galvo_value].shift())]


galvo = galvo.reset_index(drop=True)


peaks_initial = signal.find_peaks(galvo[galvo_value], height=0.5, prominence=0.05)[0]
if peaks_initial.size == 0:
	raise ValueError("No galvo peaks detected; cannot align imaging frames.")
beg_first_image = peaks_initial[0]
beg_first_image_time = galvo['AbsoluteTime'].iat[beg_first_image]

peaks = (
	signal.find_peaks(
		galvo[galvo_value].iloc[beg_first_image + 1000:],
		height=[0.5, 5],
		distance=100,
		prominence=[0.5, 5],
		width=20,
	)[0]
	+ beg_first_image
	+ 1000
)
if peaks.size == 0:
	raise ValueError("No galvo frame-start peaks detected after initial offset.")
number_peaks = len(peaks)

galvo_sub = galvo.loc[0:10000]

fig, axs = plt.subplots(nrows=4)
axs[0].set_title('')
axs[0].plot(galvo_sub['AbsoluteTime'].to_numpy(), galvo_sub[galvo_value].to_numpy(), 'k')
axs[0].plot(beg_first_image_time, 1, 'ro')
axs[0].plot(galvo_sub['AbsoluteTime'].iloc[peaks[:5]], galvo_sub[galvo_value].iloc[peaks[:5]], 'bo')
axs[0].set_xlabel('Interframe interval (ms)')
axs[0].set_ylabel('Galvo value')


# Estimate interframe interval from consecutive galvo peaks.
interframe_interval_array = galvo['AbsoluteTime'].iloc[peaks].diff()
interframe_interval = interframe_interval_array.median()


print('The median of the interframe interval is:', interframe_interval)
print('Min and max interframe interval:', interframe_interval_array.min(), interframe_interval_array.max())


beg_image_to_consider_index = interframe_interval_array.index[np.where(interframe_interval_array == interframe_interval)[0][0] - 1]
beg_image_to_consider_time = galvo['AbsoluteTime'].iat[beg_image_to_consider_index]

peaks = peaks[peaks >= beg_image_to_consider_index]


number_images_before_first_image_to_consider = round((beg_image_to_consider_time - beg_first_image_time) / interframe_interval)


# Read TIFF header to infer image dimensions and offsets.
bytes_header, height, width = fi.get_bytes_header_and_image(images_path)
bytes_header_and_image = bytes_header + height * width * 2


number_images = fi.get_number_images(images_path, bytes_header_and_image)


print('Number of galvo peaks identified:', number_peaks)
print('Number of images in the tiff:', number_images)


# Mark detected frame starts and align count to the imaging TIFF.
galvo['Frame beg'] = np.nan
galvo.loc[galvo.iloc[peaks].index[:number_images-number_images_before_first_image_to_consider], 'Frame beg'] = 1


galvo_sub = galvo.iloc[0:5000].copy()


# Insert expected frame-start timestamps for frames missing at the beginning.
galvo = pd.merge_ordered(
	galvo,
	pd.DataFrame({
		'AbsoluteTime': np.arange(
			beg_image_to_consider_time - number_images_before_first_image_to_consider * interframe_interval,
			beg_image_to_consider_time,
			interframe_interval,
		),
		'Frame beg': np.ones(number_images_before_first_image_to_consider),
	}),
)


axs[1].plot(galvo_sub['AbsoluteTime'].to_numpy(), galvo_sub[galvo_value].to_numpy(), 'k')
axs[1].plot(galvo_sub['AbsoluteTime'].to_numpy(), galvo_sub['Frame beg'].to_numpy(), 'ro')
axs[1].plot(np.arange(beg_image_to_consider_time - number_images_before_first_image_to_consider * interframe_interval, beg_image_to_consider_time, interframe_interval), np.ones(number_images_before_first_image_to_consider)*2, 'yo')
axs[1].plot(galvo.iloc[0:5000]['AbsoluteTime'].to_numpy(), galvo.iloc[0:5000]['Frame beg'].to_numpy()*3, 'bo')
axs[1].set_xlabel('Interframe interval (ms)')
axs[1].set_ylabel('Galvo value')


galvo_sub = galvo.iloc[-5000:]

galvo_sub.loc[galvo_sub['Frame beg'].notna(), 'Frame beg'] = 1


axs[2].plot(galvo_sub['AbsoluteTime'].to_numpy(), galvo_sub[galvo_value].to_numpy(), 'k')
axs[2].plot(galvo_sub['AbsoluteTime'].to_numpy(), galvo_sub['Frame beg'].to_numpy(), 'ro')
axs[2].set_xlabel('Interframe interval (ms)')
axs[2].set_ylabel('Galvo value')


del galvo_sub


axs[3].plot(interframe_interval_array, 'k.')
axs[3].set_xlabel('Interframe interval (ms)')
axs[3].set_ylabel('Galvo value')


fig.savefig(results_figs_path_save / '1. Galvo signal and frames.png')


# Load tail tracking angles and align to camera frames later.
behavior = fi.read_tail_tracking_data(tracking_path)
if behavior is None:
	raise FileNotFoundError(f"Tail tracking file could not be read: {tracking_path}")
behavior = behavior.astype('float32')


# Align galvo and camera timelines, trimming to shared acquisition window.
if (first_timepoint_galvo := galvo['AbsoluteTime'].iat[0]) <= (first_timepoint_data := data['AbsoluteTime'].iat[0]):


	galvo = galvo[galvo['AbsoluteTime'] >= first_timepoint_data]


	first_timepoint = galvo['AbsoluteTime'].iat[0]

	data['AbsoluteTime'] -= first_timepoint
	galvo['AbsoluteTime'] -= first_timepoint

	data = data[data['AbsoluteTime'] >= 0]
	galvo = galvo[galvo['AbsoluteTime'] <= data['AbsoluteTime'].iat[-1]]

	print('Galvo signal started before the tracking.')


first_timepoint = galvo['AbsoluteTime'].iat[0]

data['AbsoluteTime'] -= first_timepoint
galvo['AbsoluteTime'] -= first_timepoint

data = data[data['AbsoluteTime'] >= 0]
galvo = galvo[galvo['AbsoluteTime'] <= data['AbsoluteTime'].iat[-1]]


number_images = len(galvo.loc[galvo['Frame beg'].notna(),:])


# Merge camera timing with galvo-triggered frame starts, then attach behavior frames.
data = pd.merge_ordered(data, galvo, on=abs_time, how='outer')

del galvo, protocol


# Attach tail tracking by frame number to the merged timeline.
data = pd.merge(data, behavior, on='Frame number', how='left')

data.rename(columns={'AbsoluteTime' : 'Time (ms)'}, inplace=True)


# Load imaging frames from the raw TIFF into memory as float32 for processing.
images = np.array([
	fi.get_image_from_tiff(images_path, image_i, bytes_header, height, width).astype('float32')
	for image_i in tqdm(range(number_images))
])


# Quick alignment check: compare image means against galvo and protocol timing.
images_mean = [image.mean() for image in images]


data[[cs,us]] = data[[cs,us]].sparse.to_dense()
data = data.dropna(subset=['Frame number', 'Frame beg', cs, us], how='all')


data['Image mean'] = np.nan

data.loc[data['Frame beg'].notna(), 'Image mean'] = images_mean[:len(data.loc[data['Frame beg'].notna(), 'Image mean'])]


data = data.reset_index(drop=True)


data[[cs, us]] = data[[cs, us]].fillna(0)


# Visual alignment check: plot image mean around each US event.
stim_numbers = data.loc[data[us] != 0, us].unique()

stim_numbers = stim_numbers.astype('int')


fig, axs = plt.subplots(stim_numbers.size, 1, figsize=(10, 50), sharex=True, sharey=True)


for stim_number_i, stim_number in enumerate(stim_numbers):


	data_ = data.loc[data[us] == stim_number, 'Time (ms)']

	us_beg_ = data_.iat[0]

	us_end_ = data_.iat[-1]


	data_plot = data.loc[data['Time (ms)'].between(us_beg_-5000, us_end_+5000)]

	axs[stim_number_i].plot(data_plot['Time (ms)'].to_numpy() - us_beg_, data_plot['Image mean'].to_numpy(), 'k.')


	axs[stim_number_i].axvline(x=us_beg_ - us_beg_, color='g', linestyle='--')
	axs[stim_number_i].axvline(x=us_end_ - us_beg_, color='r', linestyle='--')
	axs[stim_number_i].set_title(f"Stimulus Number: {stim_number}")


fig.savefig(results_figs_path_save / '2. US and frames alignment.png')


del data_, data_plot


data.drop(columns=['Image mean', 'GalvoValue'], inplace=True, errors='ignore')


# Split merged table into protocol, behavior, and imaging streams.
protocol = data[['Time (ms)', cs, us]].copy()
protocol[['CS', 'US']] = protocol[['CS', 'US']].astype('int')
protocol = protocol[((protocol[cs]!=0) | (protocol[us]!=0))]

behavior = data[['Time (ms)'] + ['Angle of point {} (deg)'.format(i) for i in range(15)]].dropna().rename(columns={'Frame number' : 'Frame number (behavior)'}).copy()


# Build an imaging DataArray with time coordinates aligned to detected frame starts.
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


# Map each relevant CS index to the corresponding protocol row for trial slicing.
cs_onset_index = np.array([protocol.loc[protocol[cs] == relevant_cs[i], :].index[0] for i in range(len(relevant_cs))])


# Load anatomical reference stack for plane identification.
# Group CS indices by plane based on the acquisition pattern.
try:
	planes_cs_onset_indices = [cs_onset_index[[j for j in i]] for i in index_list]
except Exception:
	planes_cs_onset_indices = [cs_onset_index]


del cs_onset_index


# Build plane and trial objects by slicing time windows around each CS onset.
all_data = []


i = 0
for plane_i, plane_cs_onset_indices in tqdm(enumerate(planes_cs_onset_indices)):

	trials_list = []
	for trial_i, trial_cs_onset_index in enumerate(plane_cs_onset_indices):

		time_start = protocol.loc[trial_cs_onset_index, 'Time (ms)'] - p.time_bef_cs_onset
		time_end = protocol.loc[trial_cs_onset_index, 'Time (ms)'] + p.time_aft_cs_onset


		trial_protocol = protocol[protocol['Time (ms)'].between(time_start, time_end)]

		protocol_ = trial_protocol[['Time (ms)', 'CS']]

		protocol_ = protocol_[protocol_['CS'] > 0]


		# Extract CS/US onset and offset markers within the trial window.
		cs_beg_time = protocol_[protocol_['CS'].ne(protocol_['CS'].shift(periods=1))]
		cs_beg_time.rename(columns={'CS' : 'CS beg'}, inplace=True)

		cs_end = protocol_[protocol_['CS'].ne(protocol_['CS'].shift(periods=-1))]
		cs_end.rename(columns={'CS' : 'CS end'}, inplace=True)


		protocol_ = trial_protocol[['Time (ms)', 'US']]

		protocol_ = protocol_[protocol_['US'] > 0]

		us_beg_time = protocol_[protocol_['US'].ne(protocol_['US'].shift(periods=1))]
		us_beg_time.rename(columns={'US' : 'US beg'}, inplace=True)

		us_end = protocol_[protocol_['US'].ne(protocol_['US'].shift(periods=-1))]
		us_end.rename(columns={'US' : 'US end'}, inplace=True)

		trial_protocol = pd.concat([cs_beg_time, cs_end, us_beg_time, us_end])

		trial_protocol = trial_protocol.sort_values(by='Time (ms)').fillna(0)
		trial_protocol[['CS beg', 'CS end', 'US beg', 'US end']] = trial_protocol[['CS beg', 'CS end', 'US beg', 'US end']].astype(pd.SparseDtype("int", 0), copy=False)


		# Extract trial window and apply spatial median filtering to reduce noise.
		trial_images = imaging.loc[time_start:time_end, :, :]
		trial_images.values = ndimage.median_filter(trial_images, size=p.median_filter_kernel, axes=(1, 2))

		trials_list.append(c.Trial(i, trial_protocol, behavior[behavior['Time (ms)'].between(time_start, time_end)], trial_images))

		i += 1

	all_data.append(c.Plane(trials_list))


fig, axs = plt.subplots(len(all_data), len(all_data[0].trials), figsize=(10, 40), squeeze=False)
fig.patch.set_facecolor('white')

for i in range(len(all_data)):
	for j in range(len(all_data[0].trials)):

		axs[i,j].imshow(np.mean(all_data[i].trials[j].images, axis=0), vmin=0, vmax=500)
		axs[i,j].set_xticks([])
		axs[i,j].set_yticks([])


fig.set_size_inches(10, 28)
fig.subplots_adjust(hspace=0, wspace=0)

fig.savefig(results_figs_path_save / '3. Summary of imaged planes.png', facecolor='white')


try:
	anatomical_stack_images = tifffile.imread(anatomy_1_path).astype('float32')
except:
	anatomical_stack_images = tifffile.imread(imaging_path_home / 'Anatomical stack 1.tif').astype('float32')


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

# Apply spatial median filtering to reduce noise in the reference stack.
anatomical_stack_images = ndimage.median_filter(anatomical_stack_images, size=p.median_filter_kernel, axes=(1, 2))


all_data = c.Data(all_data, anatomical_stack_images)


# Persist the full dataset for downstream motion correction.
with open(path_pkl_before_motion_correction, 'wb') as file:
	pickle.dump(all_data, file)


print('END')
