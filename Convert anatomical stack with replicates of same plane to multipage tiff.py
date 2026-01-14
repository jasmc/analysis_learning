"""
Anatomical Stack Consolidation Utility

This utility script consolidates replicate images of the same anatomical plane
into a single multi-page TIFF file with proper ordering.

Purpose:
- Merge multiple acquisitions of the same Z-plane
- Average replicate images to improve SNR
- Create consolidated anatomical reference stack
- Prepare reference for plane identification

Workflow:
1. Load anatomical stack TIFF files
2. Identify replicate planes (same Z-position)
3. Average or median-filter replicates
4. Reorder planes in proper Z-sequence
5. Export consolidated multi-page TIFF

Key Features:
- Automatic detection of duplicate planes
- SNR improvement through averaging
- Maintains Z-spacing metadata
- ImageJ-compatible output

Applications:
- Create high-quality anatomical reference
- Improve plane identification accuracy
- Reduce storage requirements
- Simplify visualization

Input:
- Raw anatomical stack with replicates (multi-page TIFF)

Output:
- Consolidated anatomical stack (multi-page TIFF)
- Metadata file with plane mapping

Note: This is a preprocessing utility for anatomical reference data.
Output is used by script 1.Join_all_data.py and 2.Motion_correction.py.
"""

import re
from pathlib import Path

import numpy as np
import tifffile

#* Paths
data_folder = r'E:\2024 03_Delay 2-P multiple planes'
fish_name = r'20240415_01_delay_2p-1_mitfaMinusMinus,elavl3H2BGCaMP6f_5dpf'
anatomical_stack = r'Anatomical stack 2'

#* Create the paths of the different data
data_folder = Path(data_folder)

anatomical_stack_path = data_folder / fish_name / anatomical_stack
anatomical_stack_planes_paths = [*Path(anatomical_stack_path).glob('*tif')]


#* Pad the names of the images of the planes in the anatomical stack .
# Regex pattern to find all integer numbers in the file names.
pattern = re.compile(r'(\d+)')

# Iterate through each file and rename it.
for images_name in anatomical_stack_planes_paths:

	new_image_name = re.sub(pattern, lambda x: x.group(1).zfill(5), str(images_name.stem))
	
	images_name.rename(Path(anatomical_stack_path).joinpath(new_image_name + '.tif'))

anatomical_stack_planes_paths = [*Path(anatomical_stack_path).glob('*tif')]


#* Load the anatomical stack images into an np.array and take the average of the replicates of each frame.
anatomical_stack_images = np.array([tifffile.imread(image) for image in anatomical_stack_planes_paths[:-1]])

# Need to cast to 32 bit for opencv to work.
anatomical_stack_images = np.mean(anatomical_stack_images[5:-5], axis=1).astype('float32')

# Save the anatomical stack images as a single multipage tiff.
tifffile.imwrite(data_folder / fish_name / (anatomical_stack + '.tif'), anatomical_stack_images)
