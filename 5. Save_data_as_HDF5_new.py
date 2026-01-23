

import pickle
from pathlib import Path
from typing import Any, Dict, Union

import h5py

import numpy as np
import pandas as pd
import plotly.io as pio
import xarray as xr

import plotting_style_new as plotting_style
from my_general_variables import *
from my_paths_new import fish_name, path_home


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


configure_environment(USE_PLOTLY_DARK)


FISH_NAME_OVERRIDE = None


if FISH_NAME_OVERRIDE:
	fish_name = FISH_NAME_OVERRIDE

fish_ID = '_'.join(fish_name.split('_')[:2])

whole_data_path_save = Path(r'H:\2-P imaging') / path_home.stem / fish_name

path_pkl_responses = whole_data_path_save / (fish_ID + '_3. Responses_Suite2p.pkl')

path_h5_save = whole_data_path_save / (fish_ID + '_4. Activity_maps_data.h5')

whole_data_path_save.mkdir(parents=True, exist_ok=True)


with open(path_pkl_responses, 'rb') as file:
	all_data = pickle.load(file)


for plane in all_data.planes:
	for trial in plane.trials:
		try:
			trial.images = trial.images.drop_vars('quantile')
			delattr(trial, 'mask_good_frames')
		except:
			pass


print("Type of all_data:", type(all_data))
print("all_data attributes:", dir(all_data))
if hasattr(all_data, '__dict__'):
	print("all_data.__dict__:", all_data.__dict__)
if hasattr(all_data, 'planes'):
	print("Type of all_data.planes:", type(all_data.planes))
	print("Number of planes:", len(all_data.planes))
	if len(all_data.planes) > 0:
		print("Type of first plane:", type(all_data.planes[0]))
		print("First plane attributes:", dir(all_data.planes[0]))
		if hasattr(all_data.planes[0], 'trials'):
			print("Type of first plane.trials:", type(all_data.planes[0].trials))
			print("Number of trials in first plane:", len(all_data.planes[0].trials))
			if len(all_data.planes[0].trials) > 0:
				trial0 = all_data.planes[0].trials[0]
				print("Type of first trial:", type(trial0))
				print("First trial attributes:", dir(trial0))
				if hasattr(trial0, '__dict__'):
					print("First trial __dict__:", trial0.__dict__)

				for attr in ['protocol', 'behavior', 'images', 'mask_good_frames', 'template_image', 'shift_correction']:
					if hasattr(trial0, attr):
						val = getattr(trial0, attr)
						print(f"trial0.{attr}: type={type(val)}", end='')
						if hasattr(val, 'shape'):
							print(f", shape={val.shape}")
						elif hasattr(val, '__len__') and not isinstance(val, str):
							print(f", len={len(val)}")
						else:
							print()


num_planes = len(all_data.planes)
num_trials_per_plane = len(all_data.planes[0].trials)


def _save_attrs(h5_item: Union[h5py.Group, h5py.Dataset], attrs: Dict[str, Any]):
	"""Saves dictionary items as HDF5 attributes, handling None and complex types."""
	for key, value in attrs.items():
		try:
			if value is None:
				h5_item.attrs[key] = 'None'
			elif isinstance(value, (str, int, float, np.number, np.bool_)):
				h5_item.attrs[key] = value
			elif isinstance(value, (list, tuple)):
				try:
					arr_val = np.array(value)
					if arr_val.dtype.kind in ('i', 'u', 'f', 'b', 'S', 'U'):
						if arr_val.nbytes < 64000:
							h5_item.attrs[key] = arr_val
						else:
							print(f"    Warning: Attribute '{key}' data size exceeds limit, saving as string.")
							h5_item.attrs[key] = str(value)
					else:
						h5_item.attrs[key] = str(value)
				except Exception:
					h5_item.attrs[key] = str(value)
			elif isinstance(value, (np.ndarray)):
				if value.nbytes < 64000:
					h5_item.attrs[key] = value
				else:
					print(f"    Warning: Attribute '{key}' data size exceeds limit, saving as string.")
					h5_item.attrs[key] = str(value)
			else:
				h5_item.attrs[key] = str(value)
		except Exception as e:
			print(f"    Warning: Could not save attribute '{key}' (value: {value}). Error: {e}. Saving as string.")
			try:
				h5_item.attrs[key] = str(value)
			except Exception as e_str:
				print(f"    Error: Could not even save attribute '{key}' as string. Error: {e_str}")

def _save_basic_dataset(h5_group: h5py.Group, data: Any, name: str, **kwargs):
	"""Saves basic data types (like numpy arrays) as HDF5 datasets."""
	if data is None:
		h5_group.attrs[f'{name}_is_None'] = True
		return
	try:
		if isinstance(data, (list, tuple)):
			data = np.asarray(data)

		if isinstance(data, np.ndarray):
			compression = kwargs.get('compression', None)
			compression_opts = kwargs.get('compression_opts', None)
			chunks = kwargs.get('chunks', True if compression else None)

			ds = h5_group.create_dataset(name, data=data,
			     compression=compression,
			     compression_opts=compression_opts,
			     chunks=chunks)
			print(f"    Saved {name} (numpy array like). Shape: {data.shape}, Dtype: {data.dtype}")
			return ds
		else:
			print(f"    Warning: Data type for '{name}' ({type(data)}) not directly saved as dataset. Skipping.")

	except Exception as e:
		print(f"    Error saving {name} as basic dataset: {e}")
	return None

def _save_xarray(h5_group: h5py.Group, xr_data: xr.DataArray, name: str, **kwargs):
	"""Saves an xarray.DataArray to an HDF5 group."""
	if xr_data is None:
		h5_group.attrs[f'{name}_is_None'] = True
		return
	try:
		xr_group = h5_group.create_group(name)
		xr_group.attrs['__xarray__'] = True

		compression = kwargs.get('compression', "gzip")
		compression_opts = kwargs.get('compression_opts', 4)
		chunks = kwargs.get('chunks', None)
		if chunks is None and xr_data.ndim == 3:
			chunks = (1, xr_data.shape[1] // 4, xr_data.shape[2] // 4)
			chunks = tuple(max(1, c) for c in chunks)

		data_ds = xr_group.create_dataset('data', data=xr_data.values,
		       compression=compression,
		       compression_opts=compression_opts,
		       chunks=chunks)
		print(f"    Saved {name} data. Shape: {xr_data.shape}, Dtype: {xr_data.dtype}")

		coords_group = xr_group.create_group('coords')
		coord_names = list(xr_data.coords.keys())
		coords_group.attrs['coord_names'] = coord_names
		for coord_name, coord_array in xr_data.coords.items():
			coord_ds = coords_group.create_dataset(coord_name, data=coord_array.values)
			_save_attrs(coord_ds, coord_array.attrs)

		_save_attrs(xr_group, xr_data.attrs)
		print(f"    Saved {name} (xarray structure).")

	except Exception as e:
		print(f"    Error saving {name} (xarray): {e}")
		if hasattr(xr_data, 'values') and isinstance(xr_data.values, np.ndarray):
			print(f"    Attempting fallback: saving only data for {name}.")
			_save_basic_dataset(h5_group, xr_data.values, f"{name}_data_fallback", **kwargs)

def _save_dataframe(h5_group: h5py.Group, df: pd.DataFrame, name: str, **kwargs):
	"""Saves a pandas.DataFrame to an HDF5 group using a custom structure."""
	if df is None:
		h5_group.attrs[f'{name}_is_None'] = True
		return
	try:
		df_group = h5_group.create_group(name)
		df_group.attrs['__dataframe__'] = True

		compression = kwargs.get('compression', "gzip")
		compression_opts = kwargs.get('compression_opts', 4)
		chunks = kwargs.get('chunks', None)
		if chunks is None and df.ndim == 2:
			chunks = (min(1000, df.shape[0]), df.shape[1])
			chunks = tuple(max(1, c) for c in chunks)

			values_data = df.values
			if values_data.dtype == 'object':
				try:
					inferred_values = df.infer_objects().values
					if inferred_values.dtype != 'object':
						values_data = inferred_values
						print(f"    Inferred non-object dtype for DataFrame '{name}'.")
					else:
						values_data = values_data.astype(str)
						print(f"    Warning: Converted object dtype values in DataFrame '{name}' to string.")
				except Exception as e:
					values_data = df.values.astype(str)
					print(f"    Warning: Could not infer types for object dtype in DataFrame '{name}', converting to string. Error: {e}")

				df_group.create_dataset('values', data=values_data,
				     compression=compression,
				     compression_opts=compression_opts,
				     chunks=chunks)
				print(f"    Saved {name} values. Shape: {df.shape}, Dtype: {values_data.dtype}")
			else:
				df_group.create_dataset('values', data=df.values,
				     compression=compression,
				     compression_opts=compression_opts,
				     chunks=chunks)
				print(f"    Saved {name} values. Shape: {df.shape}, Dtype: {df.values.dtype}")

		index_group = df_group.create_group('index')
		if isinstance(df.index, pd.MultiIndex):
			index_group.attrs['__multiindex__'] = True
			index_group.attrs['names'] = [str(n) for n in df.index.names]
			for i, level in enumerate(df.index.levels):
				index_group.create_dataset(f'level_{i}', data=level.to_numpy())
			for i, code in enumerate(df.index.codes):
				index_group.create_dataset(f'code_{i}', data=code)
		else:
			index_data = df.index.to_numpy()
			ds = index_group.create_dataset('index_values', data=index_data)
			ds.attrs['name'] = str(df.index.name)

		columns_group = df_group.create_group('columns')
		if isinstance(df.columns, pd.MultiIndex):
			columns_group.attrs['__multiindex__'] = True
			columns_group.attrs['names'] = [str(n) for n in df.columns.names]
			for i, level in enumerate(df.columns.levels):
				columns_group.create_dataset(f'level_{i}', data=level.to_numpy())
			for i, code in enumerate(df.columns.codes):
				columns_group.create_dataset(f'code_{i}', data=code)
		else:
			try:
				columns_data = df.columns.to_numpy()
			except Exception:
				columns_data = df.columns.astype(str).to_numpy()
				print(f"    Warning: Converted column names in DataFrame '{name}' to string.")

			ds = columns_group.create_dataset('columns_values', data=columns_data)
			ds.attrs['name'] = str(df.columns.name)

		print(f"    Saved {name} (DataFrame structure).")

	except Exception as e:
		print(f"    Error saving {name} (DataFrame): {e}")
		if hasattr(df, 'values'):
			print(f"    Attempting fallback: saving only values for {name}.")
			_save_basic_dataset(h5_group, df.values, f"{name}_values_fallback", **kwargs)

def _save_object_fields(h5_group: h5py.Group, obj: Any, skip_fields=None):
	"""Recursively save all fields of an object as datasets or groups."""
	if skip_fields is None:
		skip_fields = set()
	if hasattr(obj, '__dict__'):
		for key, value in obj.__dict__.items():
			if key in skip_fields:
				continue
			if value is None:
				h5_group.attrs[f'{key}_is_None'] = True
				continue
			if isinstance(value, pd.DataFrame):
				_save_dataframe(h5_group, value, key, compression="gzip", compression_opts=4)
			elif isinstance(value, xr.DataArray):
				_save_xarray(h5_group, value, key, compression="gzip", compression_opts=4)
			elif isinstance(value, np.ndarray):
				_save_basic_dataset(h5_group, value, key, compression="gzip", compression_opts=4)
			elif isinstance(value, (list, tuple)) and all(isinstance(v, (int, float, str, np.number, np.bool_)) for v in value):
				_save_basic_dataset(h5_group, np.array(value), key, compression="gzip", compression_opts=4)
			elif isinstance(value, list) and value and hasattr(value[0], '__dict__'):
				subgroup = h5_group.create_group(key)
				for idx, item in enumerate(value):
					item_group = subgroup.create_group(f'item_{idx}')
					_save_object_fields(item_group, item)
			elif isinstance(value, dict):
				subgroup = h5_group.create_group(key)
				for k, v in value.items():
					if isinstance(v, (int, float, str, np.number, np.bool_)):
						subgroup.attrs[k] = v
					elif isinstance(v, np.ndarray):
						_save_basic_dataset(subgroup, v, k, compression="gzip", compression_opts=4)
					else:
						subgroup.attrs[k] = str(v)
			elif hasattr(value, '__dict__'):
				subgroup = h5_group.create_group(key)
				_save_object_fields(subgroup, value)
			else:
				try:
					h5_group.attrs[key] = value
				except Exception:
					h5_group.attrs[key] = str(value)
	else:
		try:
			_save_basic_dataset(h5_group, obj, 'value', compression="gzip", compression_opts=4)
		except Exception:
			h5_group.attrs['value'] = str(obj)

def save_data_to_hdf5(data_object: Any, filename: Union[str, Path]):
	"""
	Saves any object and its contents to an HDF5 file using h5py.
	Args:
		data_object: The object to save.
		filename: The path to the output HDF5 file (string or Path object).
	"""
	filepath = Path(filename)
	print(f"\n--- Starting HDF5 Save to {filepath} ---")
	filepath.parent.mkdir(parents=True, exist_ok=True)
	with h5py.File(filepath, 'w') as f:
		_save_object_fields(f, data_object)
	print(f"--- Data successfully saved to {filepath} ---")


save_data_to_hdf5(all_data, path_h5_save)
