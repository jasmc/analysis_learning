"""
Motion Correction Pipeline using Suite2p
==========================================

This script performs rigid and non-rigid motion correction on multi-plane 2-photon calcium imaging data.

Workflow:
---------
1. Load preprocessed imaging data (from 1.Join_all_data.py output)
2. Background subtraction: Remove 1st percentile background from all frames
3. Within-plane motion correction: Apply Suite2p registration to each plane independently
   - Uses rigid + non-rigid registration
   - Builds reference image from initial frames
   - Outputs registered binary files
4. Between-plane sequential alignment: Align subsequent planes to previous plane templates
   - Creates iterative templates for sequential reference matching
   - Maintains frame alignment across all planes
5. Bad frame detection: Identify corrupted frames using correlation analysis
   - Computes correlation of each frame with mean image
   - Marks frames deviating >2.5 SD from mean as bad
6. Plane position identification: Match template images to anatomical stack
   - Uses template matching to locate each plane in the anatomical reference
7. Save corrected data: Store motion-corrected trial images with masks and templates

Output:
-------
- Motion-corrected trial images (xarray DataArray with 'mask good frames' coordinate)
- Template images for each trial (mean of good frames)
- Bad frame indices per plane
- Plane position in anatomical stack
- Pickle file: {fish_ID}_2. After motion correction_Suite2p.pkl

Key dependencies:
- suite2p: Registration algorithms
- xarray: Data structure for imaging frames
- tifffile: Large-scale output (BigTIFF)
"""

# Save all data in a single pickle file.
# Anatomical stack images and imaging data are median filtered.


#* Imports

##   
# region Imports
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
from suite2p.registration import register
from suite2p.registration.register import compute_reference, register_frames
from suite2p.run_s2p import run_s2p
from tqdm import tqdm

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
# path_home = Path(r'D:\2024 09_Delay 2-P 4 planes JC neurons')
# Path(r'D:\2024 10_Delay 2-P 15 planes ca8 neurons')

path_results_save = Path(r'F:\Results (paper)') / path_home.stem



# fish_name = r'20241009_01_delay_2p-7_mitfaminusminus,elavl3h2bgcamp6f_5dpf'
# '20241007_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241008_02_delay_2p-5_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241008_03_delay_2p-6_mitfaMinusMinus,elavl3H2BGCaMP6f_6dpf'
# '20241008_01_delay_2p-4_mitfaminusminus,elavl3h2bgcamp6f_6dpf'
# '20241007_03_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
# '20241015_03_delay_2p-9_mitfaminusminus,ca8e1bgcamp6s_6dpf'


fish_ID = '_'.join(fish_name.split('_')[:2])


behavior_path_home = path_home / 'Tail'
imaging_path_home = path_home / 'Neurons' / fish_name

behavior_path_save = path_results_save / 'Tail'
results_figs_path_save = path_results_save / 'Neurons' / fish_name

whole_data_path_save = Path(r'H:\2-P imaging') / path_home.stem / fish_name
os.makedirs(whole_data_path_save, exist_ok=True)

path_pkl_before_motion_correction = whole_data_path_save / (fish_ID + '_1. Before motion correction.pkl')

path_pkl_after_motion_correction = whole_data_path_save / (fish_ID + '_2. After motion correction_Suite2p.pkl')






# for fish_i, fish_name in enumerate(fish_names_list):

#     try:

# imaging_path_ = imaging_path / 'Imaging'

if path_pkl_after_motion_correction.exists():

    print('Already preprocessed: ', fish_name)
    print(path_pkl_after_motion_correction)
    
    # continue

print('Analyzing fish: ', fish_name)


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
        #     pass
    #     break
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


# _, y_dim, x_dim = np.array(anatomical_stack_images.shape)

# # x_dim = int(x_dim * p.xy_movement_allowed/2)
# # y_dim = int(y_dim * p.xy_movement_allowed/2)









#!!!!!!!!!!!!!!!!!!!!!!


# Suite2p options and load files

ops = default_ops()
ops.update({
    'nonrigid': True,            # enable non-rigid registration
    'two_step_registration': True,
    'maxregshift': 0.1,
    'maxregshiftNR': 5,
    'nimg_init': 180,            # frames to build reference
#    'do_bidiphase': True,       # estimate bidirectional offset
#    'bidiphase': 0,             # 0 → use estimated offset
#    'bidi_corrected': True 
    'th_badframes': 0.7,   # more stringent: will exclude frames with severe distortion
    'batch_size': 1,
})

# # List all merged bins
# bin_files = sorted([f for f in os.listdir(whole_data_path_save) if f.endswith('.bin')])
# print(f"Found {len(bin_files)} bin files in {merged_dir}")

#tif_files = sorted([f for f in os.listdir(merged_dir) if f.endswith('.tif')])
#print(f"Found {len(tif_files)} TIFFs in {merged_dir}")


all_data.planes[0].trials[0].images.shape





output_tif = os.path.join(whole_data_path_save, 'motion_corrected_all_planes_within.tif')
reg_dir = os.path.join(whole_data_path_save, "registeredBin")
aligned_dir = os.path.join(whole_data_path_save, "alignedBin")
final_reg_dir = os.path.join(whole_data_path_save, "registeredBinFinal")


os.makedirs(reg_dir, exist_ok=True) 
os.makedirs(aligned_dir, exist_ok=True)
os.makedirs(final_reg_dir, exist_ok=True)


#* Save each plane data in a tiff binary file for Suite2p.
for plane_i, plane in tqdm(enumerate(all_data.planes)):
    for trial_i, trial in tqdm(enumerate(plane.trials)):
        bin_file_path = whole_data_path_save / f'plane_{plane_i}_trial_{trial_i}.bin'
        os.makedirs(bin_file_path.parent, exist_ok=True)
        # Write Suite2p-compatible int16 (clip to avoid overflow)
        trial_array = np.asarray(trial.images, dtype=np.float32)
        trial_array = np.clip(trial_array, 0, (2**15 - 2)).astype(np.int16)
        trial_array.tofile(str(bin_file_path))




#* Run Suite2p motion correction (within plane)


# Open TIFF writer for final multi-plane output
# Ensure required variables are defined and point to the folder where .bin files were written.
merged_dir = str(whole_data_path_save)  # directory where .bin files were saved earlier
# Use frame dimensions from variables available earlier in the script
Ly = int(all_data.planes[0].trials[0].images.shape[1])
Lx = int(all_data.planes[0].trials[0].images.shape[2])

bytes_per_pixel = np.dtype(np.int16).itemsize
frame_bytes = Ly * Lx * bytes_per_pixel

# # Determine frames per binary (use first trial as reference)
# try:
#     frames_per_plane = int(all_data.planes[0].trials[0].images.shape[0])
# except Exception:
#     # Fallback to a safe default if structure is unexpected
#     frames_per_plane = 1

with tifffile.TiffWriter(output_tif, bigtiff=True) as tif_writer:

    # Iterate through all .bin files written to merged_dir (robust to naming)
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

        f_raw = io.BinaryFile(Ly=Ly, Lx=Lx, filename=full_bin, n_frames=int(n_frames))
        base_name = os.path.splitext(bin_filename)[0]
        reg_filename = os.path.join(reg_dir, f'{base_name}_reg.bin')

        # Delete stale reg file so shape matches n_frames
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


        refImg, rmin, rmax, meanImg, rigid_offsets, \
        nonrigid_offsets, zest, meanImg_chan2, badframes, \
        yrange, xrange = registration.registration_wrapper(
            f_reg, f_raw=f_raw, f_reg_chan2=None, f_raw_chan2=None,
            refImg=None, align_by_chan2=False, ops=ops)

        # ensure all frames are flushed to disk
        try:
            f_reg.file.flush()
        except Exception:
            # Some BinaryFile implementations may differ; ignore flush errors
            pass

        # 🔎 Inspect dropped frames
        badframe_indices = np.where(badframes)[0]
        if len(badframe_indices) > 0:
            print(f"⚠️ {len(badframe_indices)} frames were dropped")
            print("Dropped frame indices:", badframe_indices)
        else:
            print("✅ No frames dropped.")

        badframes_filename = os.path.join(reg_dir, f'{base_name}_badframes.npy')
        np.save(badframes_filename, badframe_indices)

        # Step 4: Save registered frames to combined TIFF
        for i in range(int(n_frames)):
            tifffile.imwrite  # no-op to guide editor folding
            tif_writer.write(f_reg[i])



#* Post-hoc plane-to-plane alignment (sequential) with combined BigTIFF
# Use iteratively aligned template instead of average plane 

output_tif = os.path.join(aligned_dir, 'motion_corrected_all_planes_between.tif')
badframes_all_planes_file = os.path.join(aligned_dir, 'badFrames_all_planes.npy')

# Collect registered binaries
reg_files = natsorted([f for f in os.listdir(reg_dir) if f.endswith("_reg.bin")])


# Collect registered binaries organized by plane
all_reg_files = natsorted([f for f in os.listdir(reg_dir) if f.endswith("_reg.bin")])

# Group files by plane using dictionary comprehension, then convert to list of lists
plane_dict = defaultdict(list)

for f in all_reg_files:
    # Extract plane number from filename (assumes format: plane_{i}_trial_{j}_reg.bin)
    plane_num = int(f.split('_')[1])
    plane_dict[plane_num].append(f)

# Convert to list of lists, sorted by plane number
reg_files = [plane_dict[i] for i in sorted(plane_dict.keys())]


#!!!!!!!!!!!!!!!! different to Manuel's code
# ops.update({'nonrigid': False})




# Open BigTIFF writer
with tifffile.TiffWriter(output_tif, bigtiff=True) as tif_writer:
    for reg_files_plane in reg_files:

        aligned_templates = []           # store mean images for sequential reference
        total_badframes_all_planes = []  # list of arrays for each plane


        for j, fname in enumerate(reg_files_plane):
            path_in = os.path.join(reg_dir, fname)
            path_out = os.path.join(final_reg_dir, fname)  # aligned binary output

            # Extract plane and trial numbers from the filename
            plane_number = int(fname.split('_')[1])  # Assuming the second part of the filename is the plane number
            trial_number = int(fname.split('_')[3])  # Assuming the third part of the filename is the trial number

            frames_per_plane = int(all_data.planes[plane_number].trials[trial_number].images.shape[0])


            print(f"\n🔹 Aligning {fname} ...")

            # Step 1: Load registered binary
            f_raw = io.BinaryFile(Ly=Ly, Lx=Lx, filename=path_in, n_frames=frames_per_plane)
            f_aligned = io.BinaryFile(Ly=Ly, Lx=Lx, filename=path_out, n_frames=frames_per_plane)

            # Step 2: Load bad frames from cell5
            badframes_cell5_file = path_in.replace('_reg.bin', '_badframes.npy')
            badframes_cell5 = np.load(badframes_cell5_file) if os.path.exists(badframes_cell5_file) else np.array([], dtype=int)

            # Step 3: Determine reference image
            if j == 0:
                # For first plane, use mean of good frames
                good_idx_plane = np.setdiff1d(np.arange(frames_per_plane), badframes_cell5)
                frames = np.array([f_raw[i] for i in good_idx_plane[:-1]])
                refImg = compute_reference(frames, ops)
            else:
                # Reference is previous plane template
                refImg = aligned_templates[-1]

            # Step 4: Align all frames in this plane to reference
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

            # Step 5: Combine all bad frames
            total_badframes_plane = np.union1d(badframes_cell5, np.where(badframes_cell6)[0])
            total_badframes_all_planes.append(total_badframes_plane)

            # Step 6: Save all aligned frames to BigTIFF (including bad frames)
            for l in range(frames_per_plane):
                tif_writer.write(f_aligned[l])

            # Step 7: Save mean of good frames for next sequential reference
            good_idx_plane = np.setdiff1d(np.arange(frames_per_plane), total_badframes_plane)
            if len(good_idx_plane) > 0:
                frames = np.array([f_aligned[m] for m in good_idx_plane])
                template = compute_reference(frames, ops)
            else:
                template = aligned_templates[-1] if aligned_templates else np.zeros((Ly, Lx), dtype=np.float32)
            aligned_templates.append(template)

            # Optional: report dropped frames
            print(f"⚠️ Plane {i+1}: {len(total_badframes_plane)} total bad frames")
            if len(total_badframes_plane) > 0:
                print("Dropped frame indices:", total_badframes_plane)

            # Flush BinaryFile to disk
            f_aligned.file.flush()

    # Step 8: Save cumulative bad frames for all planes
    badframes_dict = {f"plane_{i+1:02d}": total_badframes_all_planes[i] for i in range(len(total_badframes_all_planes))}
    np.save(badframes_all_planes_file, badframes_dict, allow_pickle=True)
    print(f"\n✅ All planes aligned and saved to BigTIFF: {output_tif}")
    #print(f"💾 Cumulative bad frames saved to: {badframes_all_planes_file}")










#* Find out bad frames through correlation

# Paths
badframes_outfile = os.path.join(whole_data_path_save, "badFrames_corr_all_planes.npy")  # final output

# Parameters
threshold = 2.5  # number of standard deviations below the mean 

# Get all registered binaries and sort naturally
reg_files = natsorted([f for f in os.listdir(reg_dir) if f.endswith("_reg.bin")])

# Dictionary to store bad frames per plane
badframes_dict = {}
corr_dict = {}
# Loop over planes
for i, fname in enumerate(reg_files):

    # Extract plane and trial numbers from the filename
    plane_number = int(fname.split('_')[1])  # Assuming the second part of the filename is the plane number
    trial_number = int(fname.split('_')[3])  # Assuming the third part of the filename is the trial number

    frames_per_plane = int(all_data.planes[plane_number].trials[trial_number].images.shape[0])

    path_in = os.path.join(reg_dir, fname)
    f_raw = BinaryFile(Ly=Ly, Lx=Lx, filename=path_in, n_frames=frames_per_plane)

    images = [f_raw[j] for j in range(frames_per_plane)]



    trial = all_data.planes[plane_number].trials[trial_number]

    # If shapes match, assign directly; else rebuild DataArray
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
        


    # Compute plane average
    avg_img = np.mean(images, axis=0)

    # plt.imshow(np.mean(images, axis=0), cmap='gray', vmin=0, vmax=np.percentile(images, 99), interpolation='none')
    # plt.show()

    # Compute correlation of each frame with average
    corr_list=[]
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
    corr_dict[fname] = np.array(corr_list)
    print(f"Plane {fname}: {len(badframes_plane)}/{frames_per_plane} bad frames detected")




    # Create boolean mask: True for good frames, False for bad frames
    mask_good_frames = np.ones(frames_per_plane, dtype=bool)
    mask_good_frames[badframes_plane] = False


    trial.images = trial.images.assign_coords({'mask good frames' : ('Time (ms)', mask_good_frames)})

    trial.template_image = trial.images.isel({'Time (ms)': trial.images['mask good frames']}).mean(dim='Time (ms)')


    all_data.planes[plane_number].trials[trial_number] = trial


# Save dictionary
np.save(badframes_outfile, badframes_dict)
print(f"\n💾 Saved correlation-based bad frames for all planes to {badframes_outfile}")




all_data = c.Data(all_data.planes, anatomical_stack_images)



with open(path_pkl_after_motion_correction, 'wb') as file:
    pickle.dump(all_data, file)






#endregion





####################### TODO #########################




anatomical_stack_images = np.stack([fi.normalize_image(plane, (0.05, 0.95)) for plane in anatomical_stack_images])

#* Identify the plane number of the trial.
plane_numbers = np.zeros((len(all_data.planes), len(plane.trials)), dtype='int32')

for plane_i, plane in enumerate(all_data.planes):
    # if plane_i == 1:
    #     break
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
        #     break

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
    
#     all_data.planes[plane_i].template_image_position_anatomical_stack = int(plane_position_stack[plane_i])



# fig, axs = plt.subplots(len(all_data.planes), len(plane.trials)+1, figsize=(10, 50), squeeze=False)

# for plane_i in range(len(all_data.planes)):
#     for trial_i in range(len(plane.trials)):

#         anatomical_plane = anatomical_stack_images[planes_[plane_i],:,:]
        
#         axs[plane_i,trial_i+1].imshow(all_data.planes[plane_i].trials[trial_i].template_image, interpolation='None')
#         axs[plane_i,trial_i+1].set_xticks([])
#         axs[plane_i,trial_i+1].set_yticks([])
#         # axs.title(f'Plane {plane_i}, Trial {trial_i}, Anat. Pos. {plane_numbers[plane_i, trial_i]}')
#     axs[plane_i,0].imshow(anatomical_plane, interpolation='None', vmin=np.quantile(anatomical_plane, 0.05), vmax=np.quantile(anatomical_plane, 0.99))
#     axs[plane_i,0].set_xticks([])
#     axs[plane_i,0].set_yticks([])

#     # break

# fig.set_size_inches(10, 20)
# fig.subplots_adjust(hspace=0, wspace=0)

# fig.savefig(results_figs_path_save / '5. Template images.png')








# fig, axs = plt.subplots(len(all_data.planes), len(plane.trials)+1, figsize=(10, 50), squeeze=False)

# for plane_i in range(len(all_data.planes)):
#     for trial_i in range(len(plane.trials)):

#         anatomical_plane = anatomical_stack_images[planes_[plane_i],:,:]
        
#         axs[plane_position_stack[plane_i],trial_i+1].imshow(all_data.planes[plane_i].trials[trial_i].template_image, interpolation='None')
#         axs[plane_position_stack[plane_i],trial_i+1].set_xticks([])
#         axs[plane_position_stack[plane_i],trial_i+1].set_yticks([])
#         # axs.title(f'Plane {plane_i}, Trial {trial_i}, Anat. Pos. {plane_numbers[plane_i, trial_i]}')
#     axs[plane_position_stack[plane_i],0].imshow(anatomical_plane, interpolation='None', vmin=np.quantile(anatomical_plane, 0.05), vmax=np.quantile(anatomical_plane, 0.99))
#     axs[plane_position_stack[plane_i],0].set_xticks([])
#     axs[plane_position_stack[plane_i],0].set_yticks([])

#     break

# fig.set_size_inches(10, 20)
# fig.subplots_adjust(hspace=0, wspace=0)

# # fig.savefig(results_figs_path_save / '5. Template images.png')




# fig, axs = plt.subplots(len(all_data.planes), 1, figsize=(10, 50), squeeze=False)
# for plane_i in range(len(all_data.planes)):
#     axs[plane_i,0].imshow(anatomical_stack_images[planes_[plane_i], :, :], interpolation='None')

# #* Save the coordinates of the black box.
# # all_data.black_box = [x_black_box_beg, x_black_box_end, y_black_box_beg, y_black_box_end]




exec(open('3.1.Analysis_of_imaging_data_pixels_Suite2p.py').read())

print('END')






#                 # fig, axs = plt.subplots(15, 2, figsize=(10, 50))

#                 # for i in range(30):
#                 #     if i<=14:
#                 #         im = axs[i,0].imshow(C[i*2], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
#                 #         axs[i,0].axis('off')

#                 #     if i>14 and i<=29:
#                 #         axs[i-15,1].imshow(C[(i-15)*2+1], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
#                 #         axs[i-15,1].axis('off')

#                 # fig.tight_layout()
#                 # # fig.suptitle('Templates Before Correction', fontsize=16)
#                 # fig.savefig(r'H:\My Drive\PhD\Lab meetings\templates before.png', dpi=300, bbox_inches='tight')



#                 # fig, axs = plt.subplots(15, 2, figsize=(10, 50))

#                 # for i in range(30):
#                 #     if i<=14:
#                 #         axs[i,0].imshow(D[i*2], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
#                 #         axs[i,0].axis('off')

#                 #     if i>14 and i<=29:
#                 #         axs[i-15,1].imshow(D[(i-15)*2+1], interpolation='none', cmap='RdBu_r', vmin=80, vmax=500)
#                 #         axs[i-15,1].axis('off')

#                 # fig.tight_layout()
#                 # # fig.suptitle('Templates After Correction', fontsize=16)
#                 # fig.savefig(r'H:\My Drive\PhD\Lab meetings\templates after .png', dpi=300, bbox_inches='tight')

# #  endregion

# #* Load the data.
# path_pkl = path_home / 'Imaging' / fish_name / (fish_name + '_2.pkl')
# # path_pkl = r"E:\2024 03_Delay 2-P multiple planes\20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf\20240415_02_delay_2p-2_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf.pkl"

# with open(path_pkl, 'rb') as file:
#     all_data = pickle.load(file)



# # Create a new HDF5 file
# h5_file = h5py.File(path_home / 'Imaging' / fish_name / (fish_name + '.h5'), 'w')

# # Create a group for the planes data
# planes_group = h5_file.create_group('planes')

# # Loop through each plane in all_data
# for plane_i, plane in enumerate(all_data.planes):
#     # Create a group for the current plane
#     plane_group = planes_group.create_group(f'plane_{plane_i}')
    
#     # Loop through each trial in the plane
#     for trial_i, trial in enumerate(plane.trials):
#         # Create a group for the current trial
#         trial_group = plane_group.create_group(f'trial_{trial_i}')
        
#         trial_group.create_dataset('trial_number', data=trial.trial_number)
#         trial_group.create_dataset('protocol', data=trial.protocol)
#         trial_group.create_dataset('behavior', data=trial.behavior)
#         trial_group.create_dataset('images', data=trial.images)
#         try:
#             trial_group.create_dataset('mask_good_frames', data=trial.mask_good_frames)
#             trial_group.create_dataset('template_image', data=trial.template_image)
#             trial_group.create_dataset('position_anatomical_stack', data=trial.position_anatomical_stack)
#         except:
#             pass

# # Close the HDF5 file
# h5_file.close()
