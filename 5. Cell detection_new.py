"""
5. Cell Detection
=================

Runs Suite2p's ROI detection and neuropil extraction on motion-corrected imaging data.
Generates binary files for Suite2p input and saves the resulting ROI masks and traces.
"""

# Imports
import pickle
import shutil
from importlib import reload
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
from suite2p import default_ops
from suite2p.run_s2p import run_s2p

import my_classes_new as c
import my_functions_imaging_new as fi
import my_parameters_new as p
import plotting_style_new as plotting_style
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

# Paths
fish_ID = "_".join(fish_name.split("_")[:2])

whole_data_path_save = Path(r"H:\2-P imaging") / path_home.stem / fish_name
whole_data_path_save.mkdir(parents=True, exist_ok=True)

path_pkl_responses = whole_data_path_save / (fish_ID + "_3. Responses_Suite2p.pkl")
path_pkl_suite2p_rois = whole_data_path_save / (fish_ID + "_5. Suite2p_ROIs.pkl")

# Suite2p working/output root
suite2p_root = whole_data_path_save / "suite2p"
suite2p_root.mkdir(parents=True, exist_ok=True)


# Helper Functions
def _write_plane_binary_streaming(bin_path: Path, plane_obj) -> int:
    """
    Write a single int16 binary for one plane by streaming all trials.
    Returns total number of frames written.
    """
    total_frames = 0
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bin_path, "wb") as fbin:
        for trial in plane_obj.trials:
            # trial.images is xarray DataArray: (Time, y, x)
            frames = np.asarray(trial.images.values, dtype=np.float32)
            frames = np.clip(frames, 0, (2**15 - 2)).astype(np.int16)
            frames.tofile(fbin)
            total_frames += int(frames.shape[0])
    return total_frames


def _infer_nframes_from_bin(bin_path: Path, ly: int, lx: int, dtype=np.int16) -> int:
    """Infer number of frames from raw binary file size."""
    bytes_per_frame = int(ly) * int(lx) * np.dtype(dtype).itemsize
    size_bytes = bin_path.stat().st_size
    if bytes_per_frame <= 0:
        return 0
    if size_bytes % bytes_per_frame != 0:
        raise ValueError(
            f"Binary size not divisible by frame bytes: {bin_path} "
            f"(size={size_bytes}, bytes_per_frame={bytes_per_frame}, Ly={ly}, Lx={lx})"
        )
    return int(size_bytes // bytes_per_frame)


def _load_suite2p_outputs(plane_out_dir: Path) -> dict:
    """Load canonical Suite2p outputs for plane0."""
    p0 = plane_out_dir / "suite2p" / "plane0"
    out = {
        "suite2p_plane_dir": p0,
        "ops": np.load(p0 / "ops.npy", allow_pickle=True).item(),
        "stat": np.load(p0 / "stat.npy", allow_pickle=True),
        "iscell": np.load(p0 / "iscell.npy", allow_pickle=True),
        "F": np.load(p0 / "F.npy", allow_pickle=True),
        "Fneu": np.load(p0 / "Fneu.npy", allow_pickle=True),
    }
    spks = p0 / "spks.npy"
    if spks.exists():
        out["spks"] = np.load(spks, allow_pickle=True)
    return out


def _reset_suite2p_plane_output(plane_workdir: Path) -> None:
    """
    Clean stale outcomes and recreate folder structure to force Suite2p binary mode.
    """
    s2p_dir = plane_workdir / "suite2p"
    if s2p_dir.exists():
        shutil.rmtree(s2p_dir, ignore_errors=True)
    (s2p_dir / "plane0").mkdir(parents=True, exist_ok=True)


def _write_ops_npy_for_binary_run(plane_workdir: Path, ops_plane: dict) -> Path:
    """Write ops.npy to the expected location."""
    ops_path = plane_workdir / "suite2p" / "plane0" / "ops.npy"
    np.save(ops_path, ops_plane, allow_pickle=True)
    return ops_path


# Main Execution
print(f"Loading data from: {path_pkl_responses}")
with open(path_pkl_responses, "rb") as f:
    all_data = pickle.load(f)

# Frame geometry
Ly = int(all_data.planes[0].trials[0].images.shape[1])
Lx = int(all_data.planes[0].trials[0].images.shape[2])

# Suite2p Options
ops = default_ops()
ops.update({
    "nplanes": 1,
    "nchannels": 1,
    "functional_chan": 1,
    "fs": 30.0,
    "do_registration": False,  # Already corrected
    "roidetect": True,
    "neuropil_extract": True,
    "input_format": "binary",
    "diameter": 10,
    "threshold_scaling": 1.0,
    "max_overlap": 0.75,
    "sparse_mode": True,
    "classifier_path": None,
    "soma_crop": True,
})

planes_to_process = list(range(len(all_data.planes)))
results = {
    "fish_name": fish_name,
    "path_home": str(path_home),
    "whole_data_path_save": str(whole_data_path_save),
    "planes": {},
}

for plane_i in planes_to_process:
    print(f"\nProcessing plane {plane_i}...")
    plane = all_data.planes[plane_i]

    plane_workdir = suite2p_root / f"plane_{plane_i:02d}"
    plane_workdir.mkdir(parents=True, exist_ok=True)

    # Prepare environment
    _reset_suite2p_plane_output(plane_workdir)

    # Create input binary
    input_bin = plane_workdir / f"plane_{plane_i:02d}_all_trials.bin"
    _write_plane_binary_streaming(input_bin, plane)

    n_frames = _infer_nframes_from_bin(input_bin, Ly, Lx)

    if n_frames < 50:
        print(f"[SKIP] Plane {plane_i}: insufficient frames ({n_frames} < 50).")
        results["planes"][plane_i] = {
            "skipped": True,
            "reason": "n_frames < 50"
        }
        continue

    # Configure Suite2p
    db = {
        "data_path": [str(plane_workdir)],
        "save_path0": str(plane_workdir),
        "fast_disk": str(plane_workdir),
    }

    ops_plane = ops.copy()
    ops_plane.update({
        "input_format": "binary",
        "raw_file": str(input_bin),
        "Ly": int(Ly),
        "Lx": int(Lx),
        "Lys": int(Ly),
        "Lxs": int(Lx),
        "nframes": int(n_frames),
        "frames_include": -1,
        "save_path0": str(plane_workdir),
        "save_path": str(plane_workdir / "suite2p"),
    })

    _write_ops_npy_for_binary_run(plane_workdir, ops_plane)

    # Run Suite2p
    try:
        run_s2p(ops=ops_plane, db=db)
    except Exception as e:
        if "no tiffs" in str(e):
            raise RuntimeError(
                "Suite2p failed to locate binary input. Check ops['input_format'] "
                "and folder structure."
            ) from e
        raise

    # Collect Results
    out = _load_suite2p_outputs(plane_workdir)
    results["planes"][plane_i] = {
        "input_bin": str(input_bin),
        "n_frames": int(n_frames),
        "suite2p_plane_dir": str(out["suite2p_plane_dir"]),
        "ops": out["ops"],
        "stat": out["stat"],
        "iscell": out["iscell"],
        "F": out["F"],
        "Fneu": out["Fneu"],
        "spks": out.get("spks"),
        "skipped": False,
    }

# Save final output
with open(path_pkl_suite2p_rois, "wb") as f:
    pickle.dump(results, f)

print(f"\nSaved Suite2p ROI results -> {path_pkl_suite2p_rois}")
print("END")
