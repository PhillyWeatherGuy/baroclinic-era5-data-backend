# processERA5.py
import os
import shutil
import datetime
import glob
import json
import math
import sys
import re
import time
import threading
import contextlib
import functools
import concurrent.futures
import numpy as np
import pandas as pd
import xarray as xr
import cv2
import contourpy
import gzip
import struct
import gc

# ===========================================================================
# 🌪️ STORM CONFIGURATION (Edit this section for each storm)
# ===========================================================================
STORM_ID   = "era5_superstorm_1993"          # URL-safe slug (keep 'era5_' prefix for Cloudflare routing)
STORM_NAME = "1993 Superstorm"              # Display title in the UI manifest
START_DATE = "1993-03-10 00:00"             # UTC Start time (YYYY-MM-DD HH:MM)
END_DATE   = "1993-03-16 05:00"             # UTC End time   (YYYY-MM-DD HH:MM)
STEP_HOURS = 1                              # 1 for hourly, 3 for 3-hourly, 6 for 6-hourly

HF_REPO_ID = "PhillyWeatherGuy/baroclinic-era5-data"
CONFIG_FILE_PATH = os.path.join("config", "parametersERA5.json")

# Public Cloud Open Data Bucket (Free, Anonymous, Zero Keys Needed)
ARCO_ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Performance & Concurrency
MAX_CONCURRENT_WORKERS = int(os.environ.get("MAX_WORKERS", "12"))
MAX_CONCURRENT_PARAMS  = 1
GZIP_LEVEL = 6
GZIP_THREADS = 4
FRAMES_PER_VOLUME = 10                       # 10 time frames per binary chunk
BLOCK_HOURS = 24                             # Frames loaded into RAM at once by BlockedVar

CONTOUR_COORD_SCALE = 1000                  # Stored as integer thousandths of a degree
CONTOUR_FORMAT = "CTV2"
INT16_MAX = 32767

os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"
os.environ.setdefault("TQDM_DISABLE", "1")
sys.stdout.reconfigure(line_buffering=True)

CLIMATOLOGY_CACHE = {}

# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------
TIMINGS = {}
_T_LOCK = threading.Lock()

@contextlib.contextmanager
def tm(key):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        with _T_LOCK:
            t = TIMINGS.setdefault(key, [0.0, 0])
            t[0] += dt
            t[1] += 1

def timed(key):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            with tm(key):
                return fn(*a, **k)
        return wrapper
    return deco

def _gzip_to_file(path, raw_bytes):
    with open(path, "wb") as f:
        f.write(gzip.compress(raw_bytes, compresslevel=GZIP_LEVEL))

def _print_timing(pid, phases, total):
    ph = " | ".join(f"{k} {v:.1f}s" for k, v in phases.items())
    work = " | ".join(f"{k} {v[0]:.1f}s/{v[1]}x" for k, v in sorted(TIMINGS.items()))
    print(f"⏱️ [{pid}] total {total:.1f}s || phases: {ph} || thread-summed: {work}", flush=True)

def load_parameter_config(param_key="2t"):
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            params = data.get("parameters", {})
            if param_key in params:
                return params[param_key]
    raise FileNotFoundError(f"Parameter '{param_key}' not found in {CONFIG_FILE_PATH}")


# ---------------------------------------------------------------------------
# Contours: Vectorized extraction + CTV2 binary packing
# ---------------------------------------------------------------------------
def _pack_level(cont_gen, level):
    pts_list, off_list = cont_gen.lines(level)
    pts, off = pts_list[0], off_list[0]
    if pts is None or off is None or len(off) < 2:
        return None

    off = off.astype(np.intp)
    lens = np.diff(off)
    starts = off[:-1]

    x = np.minimum(pts[:, 0], 180.0)
    seam = (np.maximum.reduceat(x, starts) < -179.99) | (np.minimum.reduceat(x, starts) > 179.99)

    q = np.empty((len(pts), 2), dtype=np.int32)
    q[:, 0] = np.rint(x * CONTOUR_COORD_SCALE)
    q[:, 1] = np.rint(pts[:, 1] * CONTOUR_COORD_SCALE)

    keep_line = ~seam
    if not keep_line.all():
        q = q[np.repeat(keep_line, lens)]
        lens = lens[keep_line]
    if len(lens) == 0:
        return None

    n = len(q)
    is_start = np.zeros(n, dtype=bool)
    is_start[np.cumsum(lens) - lens] = True

    if n > 1:
        is_start[1:] |= (np.abs(np.diff(q, axis=0)) > INT16_MAX).any(axis=1)

    seg_starts = np.flatnonzero(is_start)
    seg_lens = np.diff(np.append(seg_starts, n))

    keep_seg = seg_lens >= 2
    if not keep_seg.all():
        q = q[np.repeat(keep_seg, seg_lens)]
        seg_lens = seg_lens[keep_seg]
        seg_starts = np.cumsum(seg_lens) - seg_lens
        if len(seg_lens) == 0:
            return None
        is_start = np.zeros(len(q), dtype=bool)
        is_start[seg_starts] = True

    deltas = np.diff(q, axis=0)
    inner = ~is_start[1:]
    dx = deltas[inner, 0].astype("<i2")
    dy = deltas[inner, 1].astype("<i2")

    return b"".join((
        struct.pack("<I", len(seg_lens)),
        seg_lens.astype("<u4").tobytes(),
        q[seg_starts, 0].astype("<i4").tobytes(),
        q[seg_starts, 1].astype("<i4").tobytes(),
        dx.tobytes(),
        dy.tobytes(),
    ))


def _level_style(level, contours_config):
    name = str(level)
    unit = None
    for c_def in contours_config:
        if c_def.get("dynamic") and c_def.get("unit") in {"dam", "mb", "hPa"}:
            unit = c_def.get("unit")
            name = str(int(level)) if float(level).is_integer() else str(level)
            break

    match = next(
        (c for c in contours_config if c.get("dynamic") or c.get("target") == level),
        {},
    )
    return {
        "name": name,
        "unit": unit or match.get("unit", ""),
        "color": match.get("color", "#000000"),
        "width": match.get("width", 1.6),
        "opacity": match.get("opacity", 0.9),
    }


@timed("contour_math")
def extract_contour_levels(raw_arr_k, contours_config=None, contour_settings=None):
    if not contours_config:
        return []

    try:
        contour_settings = contour_settings or {}
        coarsen_stride = max(1, int(contour_settings.get("coarsen_stride", 1)))
        blur_kernel = tuple(contour_settings.get("blur_kernel", (5, 5)))
        blur_sigma = float(contour_settings.get("blur_sigma", 1.2))

        working_arr = raw_arr_k.astype(np.float32)
        if coarsen_stride > 1:
            working_arr = cv2.resize(
                working_arr,
                (max(1, working_arr.shape[1] // coarsen_stride), max(1, working_arr.shape[0] // coarsen_stride)),
                interpolation=cv2.INTER_AREA,
            )

        frame_h, frame_w = working_arr.shape
        smoothed = cv2.GaussianBlur(working_arr, blur_kernel, blur_sigma)
        smoothed_flipped = np.flipud(smoothed)
        smoothed_cyclic = np.hstack([smoothed_flipped, smoothed_flipped[:, :1]])

        lon_step = 360.0 / frame_w
        lons = np.linspace(-180.0, 180.0 + lon_step, frame_w + 1)
        lats = np.linspace(-90.0, 90.0, frame_h)
        cont_gen = contourpy.contour_generator(
            x=lons, y=lats, z=smoothed_cyclic,
            line_type=contourpy.LineType.ChunkCombinedOffset,
        )

        interval = 6.0
        explicit_levels = []

        for c_def in contours_config:
            if c_def.get("dynamic"):
                interval = float(c_def.get("interval", 6.0))
                valid_mask = np.isfinite(working_arr) & (working_arr > 0)
                if not np.any(valid_mask):
                    explicit_levels = []
                    break

                valid_values = working_arr[valid_mask]
                data_min = float(np.nanmin(valid_values))
                data_max = float(np.nanmax(valid_values))
                if np.isfinite(data_min) and np.isfinite(data_max) and data_max > data_min:
                    start = math.ceil(data_min / interval) * interval
                    end = math.floor(data_max / interval) * interval
                    explicit_levels = [float(v) for v in np.arange(start, end + (interval * 0.5), interval)]
                break
            if "target" in c_def:
                explicit_levels.append(float(c_def["target"]))

        if not explicit_levels:
            return []

        levels = []
        for target_val in sorted(set(explicit_levels)):
            payload = _pack_level(cont_gen, target_val)
            if payload is not None:
                levels.append({"props": _level_style(target_val, contours_config), "payload": payload})

        return levels
    except Exception as e:
        print(f"  ❌ Contour extraction exception: {e}")
        return []


@timed("pack")
def pack_step(levels):
    return {
        "blob": b"".join(l["payload"] for l in levels),
        "props": [l["props"] for l in levels],
    }


def assemble_contour_file(steps):
    parts = [CONTOUR_FORMAT.encode("ascii"), struct.pack("<I", len(steps))]
    for step in sorted(steps):
        entry = steps[step]
        parts.append(struct.pack("<III", int(step), len(entry["props"]), len(entry["blob"])))
        parts.append(entry["blob"])
    return b"".join(parts)


def normalize_array(raw_arr, param_config):
    scaling = param_config.get("scaling", {})
    mode = scaling.get("mode", "linear")

    if mode == "piecewise" and "val_points" in scaling and "byte_points" in scaling:
        multiplier = scaling.get("unit_multiplier", 1.0)
        v = np.nan_to_num(raw_arr, nan=0.0) * multiplier
        val_pts = scaling["val_points"]
        byte_pts = scaling["byte_points"]
        return np.interp(v, val_pts, byte_pts).astype(np.uint8)
    else:
        min_v = scaling.get("min_val", param_config.get("min_val", 0.0))
        max_v = scaling.get("max_val", param_config.get("max_val", 255.0))
        arr = np.nan_to_num(raw_arr, copy=False, nan=min_v)
        np.clip(arr, min_v, max_v, out=arr)
        arr -= min_v
        arr /= (max_v - min_v)
        arr *= 255.0
        return arr.astype(np.uint8)


def get_climatology_grid(param_id, target_date=None, target_shape=(721, 1440), unit="m"):
    month = int(target_date[4:6]) if target_date and len(target_date) >= 6 else 10
    cache_key = f"{param_id}_{month}_{unit}"
    if cache_key in CLIMATOLOGY_CACHE:
        return CLIMATOLOGY_CACHE[cache_key]

    clim_dir = os.path.join("config", "climatology")
    if not os.path.exists(clim_dir):
        clim_dir = "climatology"

    candidate_files = [
        os.path.join(clim_dir, "z500_clim.nc"),
        os.path.join(clim_dir, "era5_z500_climatology.nc"),
        os.path.join(clim_dir, "z500_climatology.nc"),
        os.path.join(clim_dir, f"z500_clim_{month:02d}.npy"),
        os.path.join(clim_dir, "z500_clim.npy"),
    ]

    for c_path in candidate_files:
        if os.path.exists(c_path):
            try:
                if c_path.endswith(".nc"):
                    with xr.open_dataset(c_path) as ds:
                        var = next((v for v in ["z", "gh", "z500", "hgt"] if v in ds), list(ds.data_vars)[0])
                        da = ds[var]
                        if "month" in da.coords:
                            da = da.sel(month=month)
                        grid = np.squeeze(da.values).astype(np.float32)
                elif c_path.endswith(".npy"):
                    grid = np.load(c_path).astype(np.float32)

                if grid.shape != target_shape:
                    grid = cv2.resize(grid, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

                if np.nanmean(grid) > 20000.0:
                    scale = 98.0665 if unit == "dam" else 9.80665
                    grid /= scale
                elif unit == "dam" and np.nanmean(grid) > 1000.0:
                    grid /= 10.0

                CLIMATOLOGY_CACHE[cache_key] = grid
                return grid
            except Exception as e:
                print(f"  ⚠️ Error loading climatology file {c_path}: {e}")

    return None


# ---------------------------------------------------------------------------
# Public Cloud ARCO-ERA5 Data Slicer (Zero accounts, Zero keys)
# ---------------------------------------------------------------------------
_GLOBAL_DS = None
_DS_LOCK = threading.Lock()

def get_public_era5_dataset():
    """Lazily opens the Google Cloud ARCO-ERA5 Zarr store anonymously."""
    global _GLOBAL_DS
    with _DS_LOCK:
        if _GLOBAL_DS is None:
            print("☁️ Connecting anonymously to Public Cloud ERA5 Zarr store...")
            _GLOBAL_DS = xr.open_zarr(
                ARCO_ERA5_ZARR,
                chunks=None,
                storage_options=dict(token="anon")
            )
            print("✅ Connected to Public Cloud ERA5 successfully.")
        return _GLOBAL_DS


def get_variable_slice(ds, var_name, start_dt, end_dt, level=None):
    """Pulls and subsets a 3D/4D variable slice directly from the cloud."""
    var_map = {
        "2t": "2m_temperature",
        "tp": "total_precipitation",
        "sf": "snowfall",
        "msl": "mean_sea_level_pressure",
        "tcwv": "total_column_water_vapour",
        "cape": "convective_available_potential_energy",
        "t": "temperature",
        "z": "geopotential",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "vo": "vorticity"
    }
    cloud_var = var_map.get(var_name, var_name)

    da = ds[cloud_var].sel(time=slice(start_dt, end_dt))
    if level is not None and "level" in da.coords:
        da = da.sel(level=int(level))

    if 'lon' in da.coords: da = da.rename({'lon': 'longitude'})
    if 'lat' in da.coords: da = da.rename({'lat': 'latitude'})
    da = da.sortby('latitude', ascending=False)
    if da.longitude.max() > 180:
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby('longitude')

    # 🌟 MAGIC SPEED FIX + MEMORY FIX: .load() forces Xarray to fetch the entire 3D block
    # over the network into RAM immediately. .astype(np.float32) halves the RAM usage!
    return da.astype(np.float32).load()


class BlockedVar:
    """
    Memory-friendly stand-in for a fully loaded variable slice.

    Loads only `block` hourly frames into RAM at a time (the zarr store is chunked
    one time step per chunk, so this costs no extra efficiency) and exposes the same
    `.sel(time=..., method="nearest")` access the processing loop already uses.
    Works best when frames are requested in chronological order, which they are.
    """
    def __init__(self, ds, var, start, end, level=None, block=BLOCK_HOURS):
        self.ds, self.var, self.level = ds, var, level
        self.end = pd.Timestamp(end)
        self.block = block
        self._da = None
        self._lo = self._hi = None

    def _load(self, t):
        hi = min(t + pd.Timedelta(hours=self.block - 1), self.end)
        self._da = None  # release the previous block before pulling the next one
        gc.collect()
        self._da = get_variable_slice(self.ds, self.var, t, hi, level=self.level)
        self._lo, self._hi = t, hi

    def sel(self, time, method=None):
        t = pd.Timestamp(time)
        if self._da is None or not (self._lo <= t <= self._hi):
            self._load(t)
        return self._da.sel(time=t, method="nearest")

    @property
    def latitude(self):
        return self._da.latitude


class BlockedWind:
    """Blocked wind speed: sqrt(u^2 + v^2), with u and v each loaded a block at a time."""
    def __init__(self, ds, start, end, level=None, block=BLOCK_HOURS):
        self.u = BlockedVar(ds, "u", start, end, level=level, block=block)
        self.v = BlockedVar(ds, "v", start, end, level=level, block=block)

    def sel(self, time, method=None):
        return np.hypot(self.u.sel(time, method), self.v.sel(time, method))


# ---------------------------------------------------------------------------
# Master Pipeline for ERA5 Storm
# ---------------------------------------------------------------------------
def run_era5_storm_pipeline(selected_param_key="2t"):
    param_config = load_parameter_config(selected_param_key)
    patterns = param_config["filename_patterns"]
    output_dist_dir = f"storm_{STORM_ID}_{param_config['id']}"

    TIMINGS.clear()
    t_total = time.perf_counter()
    phases = {}

    start_dt = datetime.datetime.fromisoformat(START_DATE)
    end_dt   = datetime.datetime.fromisoformat(END_DATE)

    total_hours = int((end_dt - start_dt).total_seconds() // 3600)
    time_steps = list(range(0, total_hours + 1, STEP_HOURS))
    date_list = [start_dt + datetime.timedelta(hours=h) for h in time_steps]

    target_date_str = start_dt.strftime("%Y%m%d")
    init_time_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"\n🌪️ [{param_config['id']}] Storm: {STORM_NAME} ({STORM_ID}) | {len(time_steps)} steps ({START_DATE} to {END_DATE})")

    os.makedirs(output_dist_dir, exist_ok=True)

    ds = get_public_era5_dataset()

    # 1. Fetch Main Variable
    t0 = time.perf_counter()
    grib_p = param_config["grib_param"]
    lev = param_config.get("levelist")

    if param_config["id"] == "pva":
        da_u = BlockedVar(ds, "u", start_dt, end_dt, level=lev)
        da_v = BlockedVar(ds, "v", start_dt, end_dt, level=lev)
        da_main = (da_u, da_v)
    elif isinstance(grib_p, list) and "u" in grib_p and "v" in grib_p:
        da_main = BlockedWind(ds, start_dt, end_dt, level=lev)
    else:
        var_name = grib_p if isinstance(grib_p, str) else grib_p[0]
        da_main = BlockedVar(ds, var_name, start_dt, end_dt, level=lev)

    phases["cloud_fetch_main"] = time.perf_counter() - t0

    # 2. Fetch Contour Dependencies (e.g. MSLP, Geopotential Heights, or Thickness)
    contour_sources = param_config.get("contour_sources")
    if not contour_sources and param_config.get("contour_source"):
        contour_sources = [param_config["contour_source"]]

    da_contours = {}
    if contour_sources:
        for c_src in contour_sources:
            c_id = c_src.get("id", c_src.get("grib_param", "contour"))
            if c_src.get("type") == "thickness":
                da_contours[f"{c_id}_500"] = BlockedVar(ds, "z", start_dt, end_dt, level=500)
                da_contours[f"{c_id}_1000"] = BlockedVar(ds, "z", start_dt, end_dt, level=1000)
            else:
                c_lev = c_src.get("levelist")
                da_contours[c_id] = BlockedVar(ds, c_src["grib_param"], start_dt, end_dt, level=c_lev)

    # 3. Fetch P-Type Dependencies (if precipitation rate)
    da_ptype_deps = {}
    if param_config.get("ptype_source"):
        da_ptype_deps["sf"] = BlockedVar(ds, "sf", start_dt, end_dt)
        da_ptype_deps["2t"] = BlockedVar(ds, "2t", start_dt, end_dt)

    # 4. Process all timesteps
    t0 = time.perf_counter()
    results = {}
    contour_steps = {}
    ptype_results = {}

    is_accum = param_config["id"].startswith(("tp", "asnow"))

    for idx, dt in enumerate(date_list):
        step_num = time_steps[idx]

        if param_config["id"] == "pva" and isinstance(da_main, tuple):
            da_u, da_v = da_main
            u_slice = np.squeeze(da_u.sel(time=dt, method="nearest").values)
            v_slice = np.squeeze(da_v.sel(time=dt, method="nearest").values)
            
            lats = da_u.latitude.values
            dy = -27830.0 # 0.25 deg grid spacing (descending latitudes)
            dx = np.maximum((27830.0 * np.cos(np.radians(lats)))[:, None], 1.0)
            
            du_dy = np.gradient(u_slice, axis=0) / dy
            dv_dx = np.gradient(v_slice, axis=1) / dx
            slice_2d = dv_dx - du_dy
        else:
            slice_2d = np.squeeze(da_main.sel(time=dt, method="nearest").values)

        raw_arr_k = slice_2d.copy()
        if "anom" in param_config.get("id", "").lower() or param_config.get("is_anomaly"):
            target_unit = param_config.get("unit", "m").lower()
            scale_div = 98.0665 if target_unit in {"dam"} else 9.80665
            hgt = raw_arr_k / scale_div
            clim = get_climatology_grid(param_config["id"], target_date_str, hgt.shape, unit=target_unit)
            if clim is None:
                clim = np.tile(np.nanmean(hgt, axis=1, keepdims=True), (1, hgt.shape[1]))
            raw_arr_k = hgt - clim
        elif param_config.get("unit") == "dam" or str(param_config.get("grib_param", "")).lower() in {"z", "gh", "hgt"}:
            raw_arr_k = raw_arr_k / 98.0665
        elif param_config.get("unit") in {"mb", "hPa"} or str(param_config.get("grib_param", "")).lower() in {"msl", "mslp"}:
            raw_arr_k = raw_arr_k / 100.0

        c_levels = extract_contour_levels(
            raw_arr_k,
            param_config.get("contours", []),
            param_config.get("contours_settings")
        )

        if contour_sources:
            for c_src in contour_sources:
                c_id = c_src.get("id", c_src.get("grib_param", "contour"))
                if c_src.get("type") == "thickness":
                    z500 = np.squeeze(da_contours[f"{c_id}_500"].sel(time=dt, method="nearest").values)
                    z1000 = np.squeeze(da_contours[f"{c_id}_1000"].sel(time=dt, method="nearest").values)
                    thick_dam = (z500 - z1000) / 98.0665
                    levels = extract_contour_levels(thick_dam, c_src.get("contours", []), c_src.get("contours_settings"))
                else:
                    arr_c = np.squeeze(da_contours[c_id].sel(time=dt, method="nearest").values)
                    if c_src.get("unit") in {"mb", "hPa"}: arr_c = arr_c / 100.0
                    elif c_src.get("unit") == "dam": arr_c = arr_c / 98.0665
                    levels = extract_contour_levels(arr_c, c_src.get("contours", []), c_src.get("contours_settings"))
                c_levels.extend(levels)

        if c_levels:
            packed = pack_step(c_levels)
            if packed and packed["props"]:
                contour_steps[step_num] = packed

        if is_accum:
            results[step_num] = np.nan_to_num(raw_arr_k, nan=0.0).astype(np.float32)
        else:
            results[step_num] = normalize_array(raw_arr_k, param_config)

        if da_ptype_deps:
            sf_val = np.squeeze(da_ptype_deps["sf"].sel(time=dt, method="nearest").values)
            tp_val = slice_2d
            t2m_val = np.squeeze(da_ptype_deps["2t"].sel(time=dt, method="nearest").values) - 273.15

            ptype = np.zeros_like(tp_val, dtype=np.uint8)
            has_precip = tp_val > 0.00005
            snow_ratio = np.divide(sf_val, tp_val, out=np.zeros_like(sf_val), where=tp_val > 0.00001)

            ptype[has_precip & (snow_ratio > 0.75)] = 5
            ptype[has_precip & (snow_ratio <= 0.75) & (t2m_val < 0.0)] = 3
            ptype[has_precip & (snow_ratio <= 0.75) & (t2m_val >= 0.0)] = 1
            ptype_results[step_num] = ptype

    phases["process_frames"] = time.perf_counter() - t0

    # 🌟 AGGRESSIVE GARBAGE COLLECTION (free source arrays before contouring/gzipping)
    del da_main
    da_contours.clear()
    da_ptype_deps.clear()
    gc.collect()

    sorted_steps = sorted(results.keys())

    if is_accum:
        accum_results = {}
        interval_h = param_config.get("interval_hours")
        for s in sorted_steps:
            if interval_h:
                valid_steps = [st for st in sorted_steps if (s - interval_h) < st <= s]
            else:
                valid_steps = [st for st in sorted_steps if st <= s]
            
            if valid_steps:
                accum_arr = sum(results[st] for st in valid_steps)
            else:
                accum_arr = results[s]
                
            accum_results[s] = normalize_array(accum_arr, param_config)
        results = accum_results

    frame_arrays = [results[s] for s in sorted_steps]
    ptype_arrays = [ptype_results.get(s) for s in sorted_steps]
    steps_written = sorted_steps

    if not frame_arrays:
        print(f"❌ [{param_config['id']}] No frames extracted. Exiting.")
        return

    # 6. Save CTV2 Contours Binary & JSON
    t0 = time.perf_counter()
    if contour_steps and "contours_binary" in patterns:
        contour_binary = assemble_contour_file(contour_steps)
        compressed_contours = gzip.compress(contour_binary, compresslevel=6)

        if len(compressed_contours) >= 100:
            c_bin_filename = patterns["contours_binary"].format(storm=STORM_ID, param=param_config["id"])
            c_json_filename = patterns["contours_json"].format(storm=STORM_ID, param=param_config["id"])

            master_contours = {
                "model": STORM_ID,
                "storm": STORM_NAME,
                "parameter": param_config["id"],
                "run": "era5",
                "date": target_date_str,
                "steps": {
                    str(step): {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {"type": "MultiLineString", "coordinates": []},
                                "properties": props,
                            }
                            for props in contour_steps[step]["props"]
                        ],
                    }
                    for step in sorted(contour_steps)
                },
                "binary": {
                    "format": CONTOUR_FORMAT,
                    "scale": CONTOUR_COORD_SCALE,
                    "file": c_bin_filename,
                }
            }

            with open(os.path.join(output_dist_dir, c_json_filename), 'w') as f:
                json.dump(master_contours, f, separators=(",", ":"))
            with open(os.path.join(output_dist_dir, c_bin_filename), "wb") as f:
                f.write(compressed_contours)

            print(f"  📦 [{param_config['id']}] Contours: {len(contour_binary)/1e6:.2f} MB -> {len(compressed_contours)/1e6:.2f} MB gzip")
    phases["contours_file"] = time.perf_counter() - t0

    # 7. Build 3D Volume Chunks [T, H, W]
    t0 = time.perf_counter()
    frame_h, frame_w = frame_arrays[0].shape
    manifest_chunks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=GZIP_THREADS) as gz_pool:
        gz_futures = []
        for chunk_idx, i in enumerate(range(0, len(frame_arrays), FRAMES_PER_VOLUME)):
            chunk_frames = frame_arrays[i:i + FRAMES_PER_VOLUME]
            chunk_steps = steps_written[i:i + FRAMES_PER_VOLUME]
            chunk_ptype = ptype_arrays[i:i + FRAMES_PER_VOLUME] if ptype_results else None

            volume_arr = np.stack(chunk_frames, axis=0).astype(np.uint8)
            volume_filename = patterns["volume"].format(storm=STORM_ID, param=param_config["id"], chunk_idx=chunk_idx)
            volume_filepath = os.path.join(output_dist_dir, volume_filename)

            gz_futures.append(gz_pool.submit(_gzip_to_file, volume_filepath, volume_arr.tobytes()))

            chunk_meta = {
                "file": volume_filename,
                "forecast_steps": chunk_steps,
                "frame_count": len(chunk_frames)
            }

            if chunk_ptype and all(frame is not None for frame in chunk_ptype):
                ptype_filename = volume_filename.replace("_volume_", "_ptype_volume_")
                ptype_arr = np.stack(chunk_ptype, axis=0).astype(np.uint8)
                gz_futures.append(gz_pool.submit(
                    _gzip_to_file,
                    os.path.join(output_dist_dir, ptype_filename),
                    ptype_arr.tobytes(),
                ))
                chunk_meta["ptype_file"] = ptype_filename

            manifest_chunks.append(chunk_meta)

        for fut in gz_futures:
            fut.result()

    phases["volumes_gzip"] = time.perf_counter() - t0

    # 8. Write Manifest File
    manifest = {
        "model": STORM_ID,
        "storm": STORM_NAME,
        "parameter": param_config["id"],
        "name": f"{STORM_NAME} - {param_config.get('name', param_config['id'])}",
        "unit": param_config.get("unit", ""),
        "scaling": param_config.get("scaling", {}),
        "run": "era5",
        "date": target_date_str,
        "init_time": init_time_iso,
        "type": "volume_chunked",
        "total_frames": len(steps_written),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "temp_min_k": param_config.get("min_val", param_config.get("scaling", {}).get("min_val", 0.0)),
        "temp_max_k": param_config.get("max_val", param_config.get("scaling", {}).get("max_val", 255.0)),
        "chunks": manifest_chunks,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
    }

    manifest_filename = patterns["manifest"].format(storm=STORM_ID, param=param_config["id"])
    with open(os.path.join(output_dist_dir, manifest_filename), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n🎉 [{param_config['id']}] Storm assets ready in {output_dist_dir}/")

    # 9. Upload to Hugging Face
    t0 = time.perf_counter()
    upload_to_huggingface(output_dist_dir, repo_id=HF_REPO_ID)
    phases["hf_upload"] = time.perf_counter() - t0

    try:
        shutil.rmtree(output_dist_dir)
        print(f"  ✅ [{param_config['id']}] Cleanup complete.")
    except Exception as e:
        print(f"  ❌ [{param_config['id']}] Failed to delete temp directory: {e}")

    _print_timing(param_config["id"], phases, time.perf_counter() - t_total)


def upload_to_huggingface(folder_path, repo_id=HF_REPO_ID):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN not set in environment. Skipping Hugging Face upload.")
        return

    print(f"\n🤗 Uploading {folder_path} assets to Hugging Face ({repo_id})...")
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    # Several matrix jobs commit to the same repo at once, so retry with jittered backoff.
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload storm {STORM_ID} data: {folder_path}"
            )
            print(f"  ✅ Uploaded to Hugging Face: {folder_path}")
            return
        except Exception as e:
            if attempt == attempts:
                print(f"  ❌ Failed to upload to Hugging Face after {attempts} attempts: {e}")
                raise
            wait = (2 ** attempt) + np.random.uniform(0, 3)
            print(f"  ⚠️ Upload attempt {attempt}/{attempts} failed ({e}); retrying in {wait:.1f}s")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    target_params = sys.argv[1:]

    if not target_params:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                target_params = list(data.get("parameters", {}).keys())
        else:
            target_params = ["2t"]

    print(f"🚀 Launching ERA5 Storm Processing: '{STORM_NAME}' ({STORM_ID})")
    print(f"📋 Parameters: {target_params}")

    batch_size = MAX_CONCURRENT_PARAMS
    t_run = time.perf_counter()
    failed_params = []

    for i in range(0, len(target_params), batch_size):
        batch = target_params[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = math.ceil(len(target_params) / batch_size)
        t_batch = time.perf_counter()

        print(f"\n📦 [Batch {batch_num}/{total_batches}] Processing parameters: {batch}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(run_era5_storm_pipeline, param): param for param in batch}
            for future in concurrent.futures.as_completed(futures):
                param = futures[future]
                try:
                    future.result()
                    print(f"✅ Finished: {param}")
                except Exception as e:
                    print(f"❌ Error on '{param}': {e}")
                    failed_params.append(param)

        print(f"⏱️ Batch {batch_num}/{total_batches} took {time.perf_counter() - t_batch:.1f}s", flush=True)

    if failed_params:
        print(f"\n❌ STORM '{STORM_NAME}' finished with failures: {failed_params}")
        sys.exit(1)

    print(f"\n🎉 STORM '{STORM_NAME}' COMPLETED SUCCESSFULLY in {time.perf_counter() - t_run:.1f}s!")
