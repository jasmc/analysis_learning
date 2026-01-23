"""
Imaging and Behavior Utilities
==============================

Collection of utility functions used across the imaging pipeline.
"""

import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.ndimage import shift

# Global variable imports for column names (assumed from my_general_variables)
from my_general_variables import *


# region Data Loading

def read_camera(camera_path):
    try:
        start = timer()
        # Read camera file space-separated
        camera = pd.read_csv(camera_path, engine='pyarrow', sep=' ', header=0, decimal='.')
        # Remove last line if potentially incomplete
        camera = camera.iloc[:-1, :]
        camera.rename(columns={'FrameID': frame_id}, inplace=True)
        print('Time to read cam.txt: {} (s)'.format(timer() - start))
        return camera
    except Exception as e:
        print(f'Issues in the camera log file: {e}')
        return None


def framerate_and_reference_frame(camera):
    # Using 'ela_time' from imported variables, assuming it's 'ElapsedTime' or similar
    camera = camera.drop(columns=abs_time, errors='ignore')
    camera_diff = camera[ela_time].diff()

    print('Max IFI: {} ms'.format(camera_diff.max()))
    ifi = camera_diff.median()
    print('First estimate of IFI: {} ms'.format(ifi))

    camera_diff_index_correct_IFI = np.where(abs(camera_diff - ifi) <= max_interval_between_frames)[0]
    camera_diff_index_correct_IFI_diff = np.diff(camera_diff_index_correct_IFI)

    reference_frame_id = 0
    last_frame_id = 0

    # Find stable region at start
    for i in range(1, len(camera_diff_index_correct_IFI_diff)):
        if camera_diff_index_correct_IFI_diff[i - 1] == 1 and camera_diff_index_correct_IFI_diff[i] == 1:
            reference_frame_id = camera[frame_id].iloc[camera_diff_index_correct_IFI[i] - 1]
            break

    # Find stable region at end
    for i in range(len(camera_diff_index_correct_IFI_diff) - 1, 0, -1):
        if camera_diff_index_correct_IFI_diff[i - 1] == 1 and camera_diff_index_correct_IFI_diff[i] == 1:
            last_frame_id = camera[frame_id].iloc[camera_diff_index_correct_IFI[i] - 1]
            break

    # Refine IFI estimate
    if last_frame_id > reference_frame_id:
        start_idx = reference_frame_id - camera[frame_id].iloc[0]
        end_idx = last_frame_id - camera[frame_id].iloc[0]
        ifi = camera_diff.iloc[start_idx:end_idx].mean()

    print('Second estimate of IFI: {} ms'.format(ifi))
    predicted_framerate = 1000 / ifi
    print('Estimated framerate: {} FPS'.format(predicted_framerate))

    return predicted_framerate, reference_frame_id


def read_tail_tracking_data(data_path):
    try:
        start = timer()
        data = pd.read_csv(data_path, engine='pyarrow', sep=' ', usecols=cols_to_use_orig, header=0, decimal='.',
                           na_filter=False, names=[frame_id] + data_cols)
        data = data.iloc[:-1, :]
        
        # Renaissance of headers if needed (pyarrow might ignore names=)
        data.rename(columns=dict(zip(cols_to_use_orig, [frame_id] + data_cols)), inplace=True)
        print('Time to read tail tracking .txt: {} (s)'.format(timer() - start))

        # Convert radians to degrees
        data.loc[:, angle_cols] *= (180 / np.pi)
        return data
    except Exception as e:
        print(f'Issues in tail tracking data: {e}')
        return None


def read_protocol(protocol_path):
    if Path(protocol_path).exists():
        protocol = pd.read_csv(protocol_path, engine='pyarrow', sep=' ', header=0, decimal='.', names=['Type', beg, end])
    else:
        print('The stim log file does not exist')
        return None

    if protocol.empty:
        print('The stim log file is empty')
        return None

    if protocol.iloc[0, 0] == 0:
        print('The stim log file is corrupted')
        return None

    protocol.rename(columns={'Beg': beg, 'End': end}, inplace=True)
    protocol['Type'] = protocol['Type'].replace({'Cycle': cs, 'Reinforcer': us})
    protocol.sort_values(by=beg, inplace=True)
    protocol = protocol.set_index('Type')

    return protocol


def identify_trials(data, protocol):
    data[[cs, us]] = [0, 0]

    for cs_us in [cs, us]:
        if cs_us in protocol.index:
            # Handle single vs multiple rows
            protocol_subset = protocol.loc[[cs_us]]
            protocol_vals = protocol_subset[[beg, end]].to_numpy()
            
            for i, p in enumerate(protocol_vals):
                data.loc[data[abs_time].between(p[0], p[1]), cs_us] = i + 1

    data = data.set_index(abs_time)
    data.loc[:, data_cols] = data.loc[:, data_cols].interpolate(kind='slinear')
    data = data.reset_index(drop=True).dropna()
    data[time_experiment_f] = data[time_experiment_f].astype('int64')
    
    # Sparse conversion for memory efficiency
    data[[cs, us]] = data[[cs, us]].astype(pd.SparseDtype("int", 0))

    # Fix dtypes to categorical
    for cs_us in [cs, us]:
        data[cs_us] = data.loc[:, cs_us].astype(pd.CategoricalDtype(categories=data[cs_us].unique(), ordered=True))

    return data


# region Image Handling

def get_bytes_header_and_image(images_path):
    byte_number = 0
    # Header format skipping (byte_order, arbitrary, IGD1off...)
    byte_number += 8  # 2+2+4
    number_fields = np.fromfile(images_path, dtype=np.uint16, count=1, offset=byte_number)[0].byteswap()
    byte_number += 2
    # Field skips (tag, type, count, offset/value)
    byte_number += 8 # 2+6
    width = np.fromfile(images_path, dtype=np.uint32, count=1, offset=byte_number)[0].byteswap()
    byte_number += 4
    byte_number += 8 # 2+6
    height = np.fromfile(images_path, dtype=np.uint32, count=1, offset=byte_number)[0].byteswap()
    byte_number += 4
    
    # Skip remaining fields
    for n in range(number_fields - 2):
        byte_number += 12
    byte_number += 20 # 4 + 16 (next_offset + resolution)

    bytes_header = number_fields * 12 + 2 + 4 + 16
    return bytes_header, height, width


def get_image_from_tiff(images_path, image_i, bytes_header, height, width):
    offset_image = 2 + (image_i) * (int(bytes_header) + int(height) * int(width) * 2) + int(bytes_header)
    image_data = np.fromfile(images_path, dtype=np.uint16, count=height * width, offset=offset_image).byteswap().reshape((height, width))
    return image_data


def get_number_images(images_path, bytes_header_and_image):
    total_tif_size = os.path.getsize(images_path)
    number_images = (total_tif_size - 2) // int(bytes_header_and_image)
    return number_images


def normalize_image(image, quantiles=(0.01, 0.99)):
    """Normalize image to 0-1 range based on quantiles."""
    if isinstance(quantiles, tuple):
        q_min, q_max = np.quantile(image, quantiles)
    else:
        q_min, q_max = quantiles
    
    image = np.clip(image, q_min, q_max)
    if q_max > q_min:
        image = (image - q_min) / (q_max - q_min)
    else:
        image = np.zeros_like(image)
    return image


def find_plane_in_anatomical_stack(anatomical_stack_images, template_image):
    """
    Find the best matching plane in an anatomical stack using cross-correlation.
    Returns index of best match.
    """
    correlations = []
    for plane_img in anatomical_stack_images:
        # Simple correlation coefficient
        if np.std(plane_img) > 0 and np.std(template_image) > 0:
            corr = np.corrcoef(plane_img.flat, template_image.flat)[0, 1]
        else:
            corr = 0
        correlations.append(corr)
    
    best_match_idx = np.argmax(correlations)
    return [best_match_idx]


def add_colors_to_world_improved_2(background, foreground, colormap='inferno', activity_threshold=0.2, alpha=1.0):
    """
    Overlay a foreground activity map onto a background anatomy image with transparency.
    """
    norm_bg = normalize_image(background)
    norm_fg = normalize_image(foreground)
    
    # Create RGB background (grayscale)
    rgb_bg = np.stack([norm_bg]*3, axis=-1)
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    rgb_fg = cmap(norm_fg)[:, :, :3]
    
    # Mask low activity
    mask = norm_fg < activity_threshold
    
    # Blend
    combined = rgb_bg.copy()
    # Where mask is False (high activity), blend foreground
    # But strictly, if we want an overlay:
    # Here we typically replace pixels or blend them.
    # Simple replacement where activity is high:
    combined[~mask] = (1 - alpha) * combined[~mask] + alpha * rgb_fg[~mask]
    
    return combined

# endregion
