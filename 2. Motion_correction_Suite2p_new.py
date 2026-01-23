"""
Motion correction pipeline using Suite2p.

This script performs rigid/non-rigid registration per trial and then aligns
planes sequentially to build consistent templates across depth. It also flags
bad frames using correlation to the mean image, attaches template images to
each trial, and saves the corrected dataset for downstream analysis.

Analysis flow:
1) Load preprocessed planes/trials from the pickle created in step 1.
2) Subtract background (1st percentile) from anatomical stack and trial images.
3) Write each trial to a Suite2p-compatible binary file.
4) Run Suite2p registration per trial and save registered binaries.
5) Align planes sequentially using the previous plane template as reference.
6) Detect bad frames by correlation to the mean frame per trial.
7) Assign template images and anatomical stack positions to each trial.
8) Save the updated Data object for analysis scripts.
"""

# Imports
import os
import pickle
from collections import defaultdict
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import tifffile
import xarray as xr
from natsort import natsorted
from suite2p import default_ops, io, registration
from suite2p.io.binary import BinaryFile
from suite2p.registration.register import compute_reference
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


if RELOAD_MODULES:
	reload(fi)
	reload(c)
	reload(p)

configure_environment(USE_PLOTLY_DARK)

# Paths and dataset selection
path_results_save = Path(r'F:\Results (paper)') / path_home.stem


fish_ID = '_'.join(fish_name.split('_')[:2])


results_figs_path_save = path_results_save / 'Neurons' / fish_name
results_figs_path_save.mkdir(parents=True, exist_ok=True)

whole_data_path_save = Path(r'H:\2-P imaging') / path_home.stem / fish_name
whole_data_path_save.mkdir(parents=True, exist_ok=True)

path_pkl_before_motion_correction = whole_data_path_save / (fish_ID + '_1. Before motion correction.pkl')

path_pkl_after_motion_correction = whole_data_path_save / (fish_ID + '_2. After motion correction_Suite2p.pkl')


if path_pkl_after_motion_correction.exists():

    print('Already preprocessed: ', fish_name)
    print(path_pkl_after_motion_correction)


print('Analyzing fish: ', fish_name)

# Load the preprocessed dataset.
if not path_pkl_before_motion_correction.exists():
	raise FileNotFoundError(f"Missing input pickle: {path_pkl_before_motion_correction}")
with open(path_pkl_before_motion_correction, 'rb') as file:
    all_data = pickle.load(file)


shape_ = all_data.planes[0].trials[0].images.shape[1:]

# Define a background region used by downstream inspection (dataset-specific).
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


# Normalize anatomical stack by subtracting a low percentile background.
anatomical_stack_images = all_data.anatomical_stack

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

anatomical_stack_images = anatomical_stack_images - anatomical_stack_images.quantile(0.01, dim=('y', 'x'))
anatomical_stack_images = anatomical_stack_images.clip(0, None)
anatomical_stack_images = anatomical_stack_images.drop_vars('quantile')

# Save a quick QC image of the background-subtracted anatomy stack.
plt.imshow(np.mean(anatomical_stack_images, axis=0))
plt.colorbar(shrink=0.5)
plt.savefig(results_figs_path_save / '4. Mean of anatomical stack without background.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


# Subtract per-trial background to improve registration stability.
for plane_i, plane in enumerate(all_data.planes):
    for trial_i, trial in enumerate(plane.trials):
        trial.images = trial.images - trial.images.quantile(0.01, dim=('y', 'x'))
        trial.images = trial.images.clip(0, None)
        trial.images = trial.images.drop_vars('quantile')
        all_data.planes[plane_i].trials[trial_i].images = trial.images


# Configure Suite2p registration options for rigid + non-rigid correction.
ops = default_ops()
ops.update({
    'nonrigid': True,
    'two_step_registration': True,
    'maxregshift': 0.1,
    'maxregshiftNR': 5,
    'nimg_init': 180,
    'th_badframes': 0.7,
    'batch_size': 1,
})


# Output locations for registered binaries and summary TIFFs.
output_tif = os.path.join(whole_data_path_save, 'motion_corrected_all_planes_within.tif')
reg_dir = os.path.join(whole_data_path_save, "registeredBin")
aligned_dir = os.path.join(whole_data_path_save, "alignedBin")
final_reg_dir = os.path.join(whole_data_path_save, "registeredBinFinal")


os.makedirs(reg_dir, exist_ok=True)
os.makedirs(aligned_dir, exist_ok=True)
os.makedirs(final_reg_dir, exist_ok=True)


# Write each trial to a Suite2p-compatible binary file.
for plane_i, plane in tqdm(enumerate(all_data.planes)):
    for trial_i, trial in tqdm(enumerate(plane.trials)):
        bin_file_path = whole_data_path_save / f'plane_{plane_i}_trial_{trial_i}.bin'
        os.makedirs(bin_file_path.parent, exist_ok=True)

        # Suite2p expects int16 frames; clip to avoid overflow.
        trial_array = np.asarray(trial.images, dtype=np.float32)
        trial_array = np.clip(trial_array, 0, (2**15 - 2)).astype(np.int16)
        trial_array.tofile(str(bin_file_path))


# Determine frame geometry for binary parsing.
merged_dir = str(whole_data_path_save)
Ly = int(all_data.planes[0].trials[0].images.shape[1])
Lx = int(all_data.planes[0].trials[0].images.shape[2])
bytes_per_pixel = np.dtype(np.int16).itemsize
frame_bytes = Ly * Lx * bytes_per_pixel


# Run Suite2p per binary file and write a combined TIFF for inspection.
with tifffile.TiffWriter(output_tif, bigtiff=True) as tif_writer:


    bin_files = sorted([f for f in os.listdir(merged_dir) if f.endswith('.bin')])
    if len(bin_files) == 0:
        print(f"No .bin files found in {merged_dir}")
    for bin_filename in bin_files:
        full_bin = os.path.join(merged_dir, bin_filename)
        size_bytes = os.path.getsize(full_bin)
        if frame_bytes == 0 or (size_bytes % frame_bytes) != 0:
            print(f"Skipping {bin_filename}: size not divisible by frame bytes (Ly={Ly}, Lx={Lx}).")
            continue


        n_frames = size_bytes // frame_bytes
        ops['nimg_init'] = min(ops.get('nimg_init', 100), int(n_frames))
        print(f"\nProcessing {bin_filename} ... size_bytes={size_bytes}, frame_bytes={frame_bytes}, n_frames={n_frames}")

        # Register frames in this binary file and write the registered output.
        f_raw = io.BinaryFile(Ly=Ly, Lx=Lx, filename=full_bin, n_frames=int(n_frames))
        base_name = os.path.splitext(bin_filename)[0]
        reg_filename = os.path.join(reg_dir, f'{base_name}_reg.bin')


        if os.path.exists(reg_filename):
            try:
                os.remove(reg_filename)
            except PermissionError:
                print(f"Reg file in use, skipping {reg_filename}")
                continue

        f_reg = io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_filename, n_frames=int(n_frames))

        print("memmap shapes -> raw:", getattr(f_raw, "file", {}).shape, "reg:", getattr(f_reg, "file", {}).shape)
        if f_reg.file.shape[0] != int(n_frames):
            print(f"Shape mismatch for {reg_filename}: expected {n_frames}, got {f_reg.file.shape[0]}; skipping file.")
            continue


        # Suite2p registration produces rigid/non-rigid offsets and bad frame flags.
        refImg, rmin, rmax, meanImg, rigid_offsets, \
        nonrigid_offsets, zest, meanImg_chan2, badframes, \
        yrange, xrange = registration.registration_wrapper(
            f_reg, f_raw=f_raw, f_reg_chan2=None, f_raw_chan2=None,
            refImg=None, align_by_chan2=False, ops=ops)


        try:
            f_reg.file.flush()
        except Exception:

            pass


        badframe_indices = np.where(badframes)[0]
        if len(badframe_indices) > 0:
            print(f"{len(badframe_indices)} frames were dropped")
            print("Dropped frame indices:", badframe_indices)
        else:
            print("No frames dropped.")

        badframes_filename = os.path.join(reg_dir, f'{base_name}_badframes.npy')
        np.save(badframes_filename, badframe_indices)


        for i in range(int(n_frames)):
            tif_writer.write(f_reg[i])


# Align planes sequentially to ensure cross-plane spatial consistency.
output_tif = os.path.join(aligned_dir, 'motion_corrected_all_planes_between.tif')
badframes_all_planes_file = os.path.join(aligned_dir, 'badFrames_all_planes.npy')


all_reg_files = natsorted([f for f in os.listdir(reg_dir) if f.endswith("_reg.bin")])

plane_dict = defaultdict(list)
for f in all_reg_files:
    plane_num = int(f.split('_')[1])
    plane_dict[plane_num].append(f)

reg_files = [plane_dict[i] for i in sorted(plane_dict.keys())]


# Sequential between-plane alignment using the previous plane template.
with tifffile.TiffWriter(output_tif, bigtiff=True) as tif_writer:
    total_badframes_all_planes = []
    for reg_files_plane in reg_files:
        aligned_templates = []
        total_badframes_plane_list = []

        for j, fname in enumerate(reg_files_plane):
            path_in = os.path.join(reg_dir, fname)
            path_out = os.path.join(final_reg_dir, fname)


            plane_number = int(fname.split('_')[1])
            trial_number = int(fname.split('_')[3])

            frames_per_plane = int(all_data.planes[plane_number].trials[trial_number].images.shape[0])


            print(f"\nAligning {fname} ...")


            f_raw = io.BinaryFile(Ly=Ly, Lx=Lx, filename=path_in, n_frames=frames_per_plane)
            f_aligned = io.BinaryFile(Ly=Ly, Lx=Lx, filename=path_out, n_frames=frames_per_plane)


            badframes_cell5_file = path_in.replace('_reg.bin', '_badframes.npy')
            badframes_cell5 = np.load(badframes_cell5_file) if os.path.exists(badframes_cell5_file) else np.array([], dtype=int)


            # Use the first trial to build an initial template; reuse templates across trials.
            if j == 0:
                good_idx_plane = np.setdiff1d(np.arange(frames_per_plane), badframes_cell5)
                if good_idx_plane.size >= 2:
                    frames = np.array([f_raw[i] for i in good_idx_plane])
                else:
                    frames = np.array([f_raw[i] for i in range(frames_per_plane)])
                refImg = compute_reference(frames, ops) if frames.size else np.zeros((Ly, Lx), dtype=np.float32)
            else:
                refImg = aligned_templates[-1]


            refImg, rmin, rmax, meanImg, rigid_offsets, \
            nonrigid_offsets, zest, meanImg_chan2, badframes_cell6, \
            yrange, xrange = registration.registration_wrapper(
                f_aligned,
                f_raw=f_raw,
                f_reg_chan2=None,
                f_raw_chan2=None,
                refImg=refImg,
                align_by_chan2=False,
                ops=ops
            )


            # Combine Suite2p badframes from within-plane and cross-plane alignment.
            total_badframes_plane = np.union1d(badframes_cell5, np.where(badframes_cell6)[0])
            total_badframes_plane_list.append(total_badframes_plane)


            for l in range(frames_per_plane):
                tif_writer.write(f_aligned[l])


            good_idx_plane = np.setdiff1d(np.arange(frames_per_plane), total_badframes_plane)
            if len(good_idx_plane) > 0:
                frames = np.array([f_aligned[m] for m in good_idx_plane])
                template = compute_reference(frames, ops)
            else:
                template = aligned_templates[-1] if aligned_templates else np.zeros((Ly, Lx), dtype=np.float32)
            aligned_templates.append(template)


            print(f"Plane {plane_number + 1}: {len(total_badframes_plane)} total bad frames")
            if len(total_badframes_plane) > 0:
                print("Dropped frame indices:", total_badframes_plane)


        f_aligned.file.flush()

        # Store per-plane badframe indices for later inspection.
        total_badframes_all_planes.append(total_badframes_plane_list)

    badframes_dict = {f"plane_{i+1:02d}": total_badframes_all_planes[i] for i in range(len(total_badframes_all_planes))}
    np.save(badframes_all_planes_file, badframes_dict, allow_pickle=True)
    print(f"\nAll planes aligned and saved to BigTIFF: {output_tif}")


# Detect bad frames using correlation to the mean image for each trial.
badframes_outfile = os.path.join(whole_data_path_save, "badFrames_corr_all_planes.npy")


threshold = 2.5


reg_files = natsorted([f for f in os.listdir(reg_dir) if f.endswith("_reg.bin")])


badframes_dict = {}

for i, fname in enumerate(reg_files):


    plane_number = int(fname.split('_')[1])
    trial_number = int(fname.split('_')[3])

    frames_per_plane = int(all_data.planes[plane_number].trials[trial_number].images.shape[0])

    path_in = os.path.join(reg_dir, fname)
    f_raw = BinaryFile(Ly=Ly, Lx=Lx, filename=path_in, n_frames=frames_per_plane)

    # Load frames for correlation-based QC.
    images = [f_raw[j] for j in range(frames_per_plane)]


    trial = all_data.planes[plane_number].trials[trial_number]


    if images[0].shape == tuple(trial.images.shape[1:]) and len(images) == trial.images.shape[0]:
        trial.images.values = np.asarray(images)
    else:
        trial.images = xr.DataArray(
            np.asarray(images),
            dims=trial.images.dims,
            coords={trial.images.dims[0]: trial.images[trial.images.dims[0]].values[:len(images)],
                'y': trial.images['y'],
                'x': trial.images['x']},
            name='images')


    avg_img = np.mean(images, axis=0)


    corr_list = []
    badframes_plane = []
    for j in range(frames_per_plane):
        frame = f_raw[j]
        corr = np.corrcoef(frame.flatten(), avg_img.flatten())[0, 1]
        corr_list.append(corr)


    for k in range(frames_per_plane):
        corr = corr_list[k]
        mean_corr = np.mean(corr_list)
        sd_corr = np.std(corr_list)
        if corr < mean_corr-threshold*sd_corr:
            badframes_plane.append(k)

    badframes_dict[fname] = np.array(badframes_plane)
    print(f"Plane {fname}: {len(badframes_plane)}/{frames_per_plane} bad frames detected")


    # Mark bad frames and store a template image built from good frames only.
    mask_good_frames = np.ones(frames_per_plane, dtype=bool)
    mask_good_frames[badframes_plane] = False


    trial.images = trial.images.assign_coords({'mask good frames': ('Time (ms)', mask_good_frames)})

    trial.template_image = trial.images.isel({'Time (ms)': trial.images['mask good frames']}).mean(dim='Time (ms)')


    all_data.planes[plane_number].trials[trial_number] = trial


np.save(badframes_outfile, badframes_dict)
print(f"\nSaved correlation-based bad frames for all planes to {badframes_outfile}")


all_data = c.Data(all_data.planes, anatomical_stack_images)


with open(path_pkl_after_motion_correction, 'wb') as file:
    pickle.dump(all_data, file)


# Normalize the anatomical stack to aid template matching.
anatomical_stack_images = np.stack([fi.normalize_image(plane, (0.05, 0.95)) for plane in anatomical_stack_images])


# Estimate plane positions by matching trial templates to the anatomical stack.
plane_numbers = np.zeros((len(all_data.planes), len(all_data.planes[0].trials)), dtype='int32')

for plane_i, plane in enumerate(all_data.planes):


    for trial_i, trial in enumerate(plane.trials):

        # Match each trial template to the anatomical stack to estimate plane position.
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


print(plane_numbers)


all_data = c.Data(all_data.planes, anatomical_stack_images)


# Save the final dataset with corrected frames, templates, and plane positions.
with open(path_pkl_after_motion_correction, 'wb') as file:
    pickle.dump(all_data, file)


exec(open('3.1.Analysis_of_imaging_data_pixels_Suite2p.py').read())

print('END')
