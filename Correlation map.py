"""
Correlation Map Generation (Utility Script)

This standalone utility generates spatial correlation maps from 2-photon imaging data
to identify regions of correlated activity (potential neurons or ROIs).

Purpose:
--------
- Compute pixel-wise temporal correlation maps
- Visualize spatial patterns of coherent activity
- Assist in manual ROI identification and validation
- Support alternative cell detection approaches

Algorithm (Suite2p-inspired):
-----------------------------
1. Load motion-corrected imaging stack
2. Apply spatial filtering (gaussian blur)
3. Calculate correlation strength for each pixel:
   - Temporal variance of gaussian-filtered data
   - Normalized by spatial smoothing of raw variance
   - Result: high values indicate coherent temporal patterns
4. Generate correlation map: values range 0-1

Mathematical Formula:
correlation_map = ||gaussian_filter(images, sigma)||² / gaussian_filter(||images||, sigma)²

where:
- images: 3D array (time, y, x)
- sigma: spatial scale for correlation (typically 3-5 pixels)
- ||·||: L2 norm along time axis

Key Parameters:
---------------
- correlation_map_sigma: Spatial scale for correlation calculation (default: 3)
- gaussian_filter_sigma: Pre-filtering smoothing (default: 1-2)
- threshold: Minimum correlation for ROI detection (default: 0.3-0.5)

Output:
-------
- 2D correlation map (same spatial size as input images)
- Higher values = stronger temporal coherence (likely cell bodies)
- Lower values = noisy/neuropil regions

Use Cases:
----------
- Quick quality check of imaging data
- Visual identification of cell bodies vs. neuropil
- Parameter tuning for automated ROI detection
- Comparison with Suite2p's built-in correlation maps
- Preprocessing for custom segmentation algorithms

Workflow Integration:
---------------------
This script can be used:
1. After motion correction (script 2.x) for QC
2. Before ROI detection (script 5) for parameter tuning
3. Standalone for exploratory analysis

Note: For production ROI detection, use Suite2p's full pipeline (see 5. Cell detection.py)
or the pixel/voxel analysis methods in scripts 3.x.

References:
-----------
- Suite2p: Pachitariu et al., 2017 (bioRxiv)
- Correlation-based segmentation: celldetect2.getVmap() in Suite2p source
"""

import multiprocessing

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import median_filter
from scipy.signal import convolve2d


def CorrelationMap5D(mat5d_motioncorrected_IN, mathandle_correlation_map_OUT, volumes_to_process, correlation_radius, detrend_frames):
    t, c, z, w, h = mat5d_motioncorrected_IN['imagedata'].shape

    if volumes_to_process is None:
        volumes_to_process = np.arange(t)

    vertadd = correlation_radius * 2
    zthickness = vertadd * 2 + 1

    myfilter = np.ones(detrend_frames) * (-1 / detrend_frames)
    myfilter[detrend_frames // 2] += 1

    corrmapall = np.zeros((w, h, z), dtype=np.float32)

    def process_plane(plane):
        print(f"Processing Plane {plane + 1} of {z - zthickness + 1}")
        timeslice = np.zeros((w, h, zthickness, len(volumes_to_process)), dtype=np.float32)
        for i, volume in enumerate(volumes_to_process):
            timeslice[..., i] = np.squeeze(mat5d_motioncorrected_IN['imagedata'][volume, :, plane:plane + zthickness, :, :])

        timeslice = np.transpose(timeslice, (2, 3, 1, 0))  # whz't

        filtered_timeslice = np.zeros_like(timeslice)
        for volume in volumes_to_process:
            filtered_timeslice[..., volume] = gaussian_filter(np.squeeze(timeslice[..., volume]), correlation_radius)

        timeslice = timeslice - np.mean(timeslice, axis=3, keepdims=True)
        filtered_timeslice = filtered_timeslice - np.mean(filtered_timeslice, axis=3, keepdims=True)

        for nn in range(zthickness):
            for n in range(w):
                timeslice[n, :, nn, :] = convolve2d(np.squeeze(timeslice[n, :, nn, :]), myfilter, mode='same', boundary='wrap')
                filtered_timeslice[n, :, nn, :] = convolve2d(np.squeeze(filtered_timeslice[n, :, nn, :]), myfilter, mode='same', boundary='wrap')

        indnorm = np.linalg.norm(timeslice, axis=3)
        meannorm = np.linalg.norm(filtered_timeslice, axis=3)
        meannormsq = np.power(meannorm, 2)
        findnorm = gaussian_filter(indnorm, correlation_radius)
        findnormsq = np.power(findnorm, 2)
        corrmap = meannormsq / findnormsq
        thisplanecorr = corrmap[:, :, vertadd]
        return thisplanecorr

    with multiprocessing.Pool(processes=6) as pool:
        corrmapall_list = pool.map(process_plane, range(z - zthickness + 1))

    for plane, thisplanecorr in enumerate(corrmapall_list):
        corrmapall[:, :, plane + vertadd] = thisplanecorr

    mincorr = np.min(corrmapall, axis=2)
    for n in range(corrmapall.shape[2]):
        corrmapall[:, :, n] -= mincorr
        corrmapall[:4, :, n] = 0
        corrmapall[-4:, :, n] = 0
        corrmapall[:, :4, n] = 0
        corrmapall[:, -4:, n] = 0

    correlationMapFiltered = median_filter(corrmapall, size=(3, 3, 3))

    mathandle_correlation_map_OUT.create_dataset('corrmapall', data=corrmapall)
    mathandle_correlation_map_OUT.create_dataset('correlationMapFiltered', data=correlationMapFiltered)

# Example usage
mat5d_motioncorrected_IN = h5py.File("E:/20230803_01_shortFixed3strace_SCAPE-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_7dpf/motion_corrected.h5", "r")
index_of_volumes_to_use = np.arange(0, mat5d_motioncorrected_IN['imagedata'].shape[0], 5)

ignore = np.array([1], dtype=np.int64)
correlation_radius = 1
detrend_frames = 30

with h5py.File("E:/20230803_01_shortFixed3strace_SCAPE-1_mitfaMinusMinus,elavl3GFF,10UASGCaMP6fEF05_7dpf/corrmap_finalx.h5", "w") as mathandle_correlation_map_OUT:
    mathandle_correlation_map_OUT.create_dataset("ignore", data=ignore, dtype=np.int64)

    CorrelationMap5D(mat5d_motioncorrected_IN, mathandle_correlation_map_OUT, index_of_volumes_to_use, correlation_radius, detrend_frames)
