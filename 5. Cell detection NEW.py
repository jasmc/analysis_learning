"""
5. ROI / Cell Detection (Suite2p packaged)

Goal
----
Run Suite2p's *packaged* ROI detection + neuropil extraction + (optional) classifier
on the already motion-corrected imaging movies produced by:
    2.Motion_correction_Suite2p.py  ->  *_2. After motion correction_Suite2p.pkl

What this script does
---------------------
1) Loads motion-corrected imaging data (per-plane, per-trial xarray DataArray movies).
2) For each plane, concatenates all trials (streaming) into a single Suite2p-compatible
   int16 binary movie (Ly x Lx frames).
3) Runs Suite2p via `suite2p.run_s2p` with:
   - do_registration = False  (movies already registered)
   - roidetect = True         (ROI detection)
   - neuropil_extract = True  (Fneu)
4) Saves a compact pickle containing (per plane):
   - stat, iscell, ops, F, Fneu (and paths to the Suite2p output folders)

Outputs
-------
- Suite2p output folders:
    {whole_data_path_save}/suite2p/plane_{XX}/suite2p/plane0/{stat.npy, iscell.npy, F.npy, Fneu.npy, ops.npy, ...}
- Pickle summary for downstream analysis:
    {whole_data_path_save}/{fish_ID}_5. Suite2p_ROIs.pkl

Notes
-----
- This is intentionally per-plane (1-plane runs) to match your existing data structure.
- If you want *per-trial* ROI extraction later, keep the same ROIs and re-extract traces
  by slicing frames belonging to each trial (not implemented here).
"""

# %% Imports
import pickle
import shutil
from importlib import reload
from pathlib import Path

import numpy as np
from suite2p import default_ops
from suite2p.run_s2p import run_s2p

import my_classes as c
import my_functions_imaging as fi
import my_parameters as p
from my_paths import fish_name, path_home

reload(fi)
reload(c)
reload(p)

# %% Paths (keep dataset context in my_paths.py)
fish_ID = "_".join(fish_name.split("_")[:2])

whole_data_path_save = Path(r"H:\2-P imaging") / path_home.stem / fish_name
whole_data_path_save.mkdir(parents=True, exist_ok=True)

# path_pkl_after_motion_correction = whole_data_path_save / (fish_ID + "_2. After motion correction_Suite2p.pkl")
path_pkl_after_motion_correction = whole_data_path_save / (fish_ID + "_3. Responses_Suite2p.pkl")

path_pkl_suite2p_rois = whole_data_path_save / (fish_ID + "_5. Suite2p_ROIs.pkl")

# Suite2p working/output root for this fish
suite2p_root = whole_data_path_save / "suite2p"
suite2p_root.mkdir(parents=True, exist_ok=True)

# %% Load motion-corrected data
with open(path_pkl_after_motion_correction, "rb") as f:
	all_data = pickle.load(f)

# Frame geometry (assumes consistent across planes/trials)
Ly = int(all_data.planes[0].trials[0].images.shape[1])
Lx = int(all_data.planes[0].trials[0].images.shape[2])

# %% Suite2p ops (packaged pipeline; registration disabled because already corrected)
ops = default_ops()
ops.update(
	{
		# --- core ---
		"nplanes": 1,
		"nchannels": 1,
		"functional_chan": 1,
		"fs": 30.0,  # adjust if needed
		# --- skip registration ---
		"do_registration": False,
		# --- do detection/extraction ---
		"roidetect": True,
		"neuropil_extract": True,
		# IMPORTANT: input is a prebuilt binary (otherwise Suite2p looks for tiffs and raises "no tiffs")
		"input_format": "binary",
		# --- ROI detection tuning (start conservative; adjust per dataset) ---
		"diameter": 10,              # soma diameter in pixels (tune!)
		"threshold_scaling": 1.0,    # higher -> fewer ROIs
		"max_overlap": 0.75,
		"sparse_mode": True,
		# --- classifier ---
		"classifier_path": None,     # set path to a classifier if you have one
		"soma_crop": True,
	}
)

# %% Helpers
def _write_plane_binary_streaming(bin_path: Path, plane_obj) -> int:
	"""
	Write a single int16 binary for one plane by streaming all trials in order.
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
		# Keep it strict: Suite2p expects exact frames; caller can inspect mismatch.
		raise ValueError(
			f"Binary size not divisible by frame bytes: {bin_path} "
			f"(size={size_bytes}, bytes_per_frame={bytes_per_frame}, Ly={ly}, Lx={lx})"
		)
	return int(size_bytes // bytes_per_frame)


def _load_suite2p_outputs(plane_out_dir: Path) -> dict:
	"""
	Load the canonical Suite2p outputs for plane0 inside a suite2p run directory.
	"""
	p0 = plane_out_dir / "suite2p" / "plane0"
	out = {
		"suite2p_plane_dir": p0,
		"ops": np.load(p0 / "ops.npy", allow_pickle=True).item(),
		"stat": np.load(p0 / "stat.npy", allow_pickle=True),
		"iscell": np.load(p0 / "iscell.npy", allow_pickle=True),
		"F": np.load(p0 / "F.npy", allow_pickle=True),
		"Fneu": np.load(p0 / "Fneu.npy", allow_pickle=True),
	}
	# Optional files (don’t fail if missing)
	spks = p0 / "spks.npy"
	if spks.exists():
		out["spks"] = np.load(spks, allow_pickle=True)
	return out


def _reset_suite2p_plane_output(plane_workdir: Path) -> None:
	"""
	We run Suite2p in "binary input" mode (we already wrote the .bin ourselves).
	However, Suite2p's run_s2p() decides whether to look for TIFFs based on whether
	it finds existing output plane folders under: {save_path0}/suite2p/plane0.

	If that folder does NOT exist, run_s2p() may call io.tiff_to_binary() and fail with:
	    Exception: no tiffs

	So we (1) delete stale outputs, then (2) recreate the expected folder structure
	BEFORE calling run_s2p().
	"""
	s2p_dir = plane_workdir / "suite2p"
	if s2p_dir.exists():
		shutil.rmtree(s2p_dir, ignore_errors=True)
	# IMPORTANT: recreate plane0 folder so run_s2p() does NOT try to find TIFFs
	(s2p_dir / "plane0").mkdir(parents=True, exist_ok=True)


def _write_ops_npy_for_binary_run(plane_workdir: Path, ops_plane: dict) -> Path:
	"""
	Write Suite2p's expected ops.npy into {plane_workdir}/suite2p/plane0/.
	This is the key that forces run_s2p() into the "binary already exists" code path.
	"""
	ops_path = plane_workdir / "suite2p" / "plane0" / "ops.npy"
	np.save(ops_path, ops_plane, allow_pickle=True)
	return ops_path


# %% Run Suite2p per plane (consistent ROIs across trials within plane)
planes_to_process = list(range(len(all_data.planes)))  # edit for debugging, e.g. [2]

results = {
	"fish_name": fish_name,
	"path_home": str(path_home),
	"whole_data_path_save": str(whole_data_path_save),
	"planes": {},
}

for plane_i in planes_to_process:
	plane = all_data.planes[plane_i]

	plane_workdir = suite2p_root / f"plane_{plane_i:02d}"
	plane_workdir.mkdir(parents=True, exist_ok=True)

	# 0) Clean stale outputs but recreate suite2p/plane0/ (required for binary mode)
	_reset_suite2p_plane_output(plane_workdir)

	# 1) Build input binary for this plane (concatenate all trials)
	input_bin = plane_workdir / f"plane_{plane_i:02d}_all_trials.bin"
	_write_plane_binary_streaming(input_bin, plane)

	# 2) Authoritative frame count from file size (ensures we don't lie to Suite2p)
	n_frames = _infer_nframes_from_bin(input_bin, Ly, Lx, dtype=np.int16)

	# Suite2p hard-requires >= 50 frames
	if n_frames < 50:
		print(f"[SKIP] plane {plane_i:02d}: only {n_frames} frames in {input_bin} (Suite2p requires >= 50).")
		results["planes"][plane_i] = {
			"input_bin": str(input_bin),
			"n_frames": int(n_frames),
			"skipped": True,
			"skip_reason": "precheck_nframes<50",
		}
		continue

	# 3) Configure db (paths only; Suite2p will read per-plane ops from ops.npy)
	db = {
		"data_path": [str(plane_workdir)],
		"save_path0": str(plane_workdir),
		"fast_disk": str(plane_workdir),
	}

	# 4) Per-plane ops must point to the binary and include required geometry fields
	ops_plane = ops.copy()
	ops_plane.update(
		{
			# Force binary input
			"input_format": "binary",
			"raw_file": str(input_bin),
			# Geometry
			"Ly": int(Ly),
			"Lx": int(Lx),
			# Some Suite2p versions use these in binary mode
			"Lys": int(Ly),
			"Lxs": int(Lx),
			# Frame count (Suite2p enforces >=50 internally)
			"nframes": int(n_frames),
			# Include all frames
			"frames_include": -1,
			# Ensure outputs go where we expect
			"save_path0": str(plane_workdir),
			"save_path": str(plane_workdir / "suite2p"),
		}
	)

	# 5) CRITICAL: write ops.npy into suite2p/plane0 so run_s2p won't search for TIFFs
	_write_ops_npy_for_binary_run(plane_workdir, ops_plane)

	# 6) Run Suite2p (defensive error handling with actionable message)
	try:
		run_s2p(ops=ops_plane, db=db)
	except Exception as e:
		if "no tiffs" in str(e):
			raise RuntimeError(
				"Suite2p tried to look for TIFFs. This usually means it did not detect the "
				"pre-created suite2p/plane0 folder and/or ops.npy for binary input. "
				f"Check that {plane_workdir / 'suite2p' / 'plane0' / 'ops.npy'} exists "
				f"and that ops['input_format']=='binary' and ops['raw_file']=={input_bin!s}."
			) from e
		raise

	# 7) Collect outputs
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
		**({} if "spks" not in out else {"spks": out["spks"]}),
		"skipped": False,
	}

# %% Save results
with open(path_pkl_suite2p_rois, "wb") as f:
	pickle.dump(results, f)

print("Saved Suite2p ROI results ->", path_pkl_suite2p_rois)