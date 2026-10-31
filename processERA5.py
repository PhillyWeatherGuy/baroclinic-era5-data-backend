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
import xarray as xr
import cv2
import boto3
import contourpy
import gzip
import struct

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

# Performance & Concurrency
MAX_CONCURRENT_WORKERS = int(os.environ.get("MAX_WORKERS", "12"))
MAX_CONCURRENT_PARAMS  = 2
GZIP_LEVEL = 6
GZIP_THREADS = 4
FRAMES_PER_VOLUME = 10                       # 10 time frames per binary chunk

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
# ERA5 Downloader: Batches the full storm window in 1 single CDS API request
# ---------------------------------------------------------------------------
def download_era5_storm_dataset(param_config, date_list, output_file):
    """
    Downloads all storm hours in a single efficient CDS API request.
    If the file already exists locally, download is skipped.
    """
    if os.path.exists(output_file):
        return output_file

    import cdsapi
    c = cdsapi.Client()

    is_pl = (param_config.get("levtype") == "pl")
    dataset = "reanalysis-era5-pressure-levels" if is_pl else "reanalysis-era5-single-levels"

    years = sorted(list({dt.strftime("%Y") for dt in date_list}))
    months = sorted(list({dt.strftime("%m") for dt in date_list}))
    days = sorted(list({dt.strftime("%d") for dt in date_list}))
    times = sorted(list({dt.strftime("%H:00") for dt in date_list}))

    var_name = param_config.get("era5_name") or param_config["grib_param"]
    var_list = var_name if isinstance(var_name, list) else [var_name]

    request_params = {
        "product_type": "reanalysis",
        "format": "grib",
        "variable": var_list,
        "year": years,
        "month": months,
        "day": days,
        "time": times,
    }

    if is_pl and "levelist" in param_config:
        lev = param_config["levelist"]
        request_params["pressure_level"] = [str(lev)] if isinstance(lev, (int, str)) else [str(l) for l in lev]

    print(f"  📥 [CDS API] Requesting {param_config['id']} for {len(date_list)} timesteps...")
    c.retrieve(dataset, request_params, output_file)
    return output_file


@timed("decode_all")
def process_era5_step_from_ds(ds, dt_step, param_config, target_date_str, want_raster=True, want_raw=False):
    """Extracts, rotates, and normalizes a single hourly slice from the xarray dataset."""
    try:
        # Time selection
        if 'time' in ds.coords:
            ds_step = ds.sel(time=dt_step, method="nearest")
        else:
            ds_step = ds

        if 'lon' in ds_step.coords:
            ds_step = ds_step.rename({'lon': 'longitude'})
        if 'lat' in ds_step.coords:
            ds_step = ds_step.rename({'lat': 'latitude'})

        ds_step = ds_step.sortby('latitude', ascending=False)
        if ds_step.longitude.max() > 180:
            ds_step = ds_step.assign_coords(
                longitude=(((ds_step.longitude + 180) % 360) - 180)
            ).sortby('longitude')

        grib_var = param_config["grib_param"]
        if "u" in ds_step and "v" in ds_step:
            raw_arr_k = np.hypot(np.squeeze(ds_step["u"].values), np.squeeze(ds_step["v"].values))
        else:
            target_var = grib_var if (isinstance(grib_var, str) and grib_var in ds_step) else list(ds_step.data_vars)[0]
            raw_arr_k = np.squeeze(ds_step[target_var].values)

        if param_config.get("category_mode"):
            return np.nan_to_num(raw_arr_k, nan=0.0).clip(0, 255).astype(np.uint8), []

        # 500mb Height Anomaly
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

        contour_levels = extract_contour_levels(
            raw_arr_k,
            param_config.get("contours", []),
            param_config.get("contours_settings")
        )

        if want_raw:
            return np.nan_to_num(raw_arr_k, nan=0.0).astype(np.float32), contour_levels

        arr_8bit = normalize_array(raw_arr_k, param_config) if want_raster else None
        return arr_8bit, contour_levels

    except Exception as e:
        print(f"  ❌ Error slicing step {dt_step}: {e}")
        return None, []


def process_thickness_from_ds(ds_z, dt_step, contour_config):
    """Calculates 1000-500mb thickness: (Z500 - Z1000) / 98.0665"""
    ds_step = ds_z.sel(time=dt_step, method="nearest")
    if 'lon' in ds_step.coords: ds_step = ds_step.rename({'lon': 'longitude'})
    if 'lat' in ds_step.coords: ds_step = ds_step.rename({'lat': 'latitude'})
    ds_step = ds_step.sortby('latitude', ascending=False)
    if ds_step.longitude.max() > 180:
        ds_step = ds_step.assign_coords(longitude=(((ds_step.longitude + 180) % 360) - 180)).sortby('longitude')

    da = ds_step["z"] if "z" in ds_step else list(ds_step.data_vars)[0]
    level_coord = next((c for c in da.coords if 'isobaric' in c.lower() or 'level' in c.lower()), None)

    if level_coord:
        z500 = da.sel({level_coord: 500}).values.squeeze()
        z1000 = da.sel({level_coord: 1000}).values.squeeze()
    else:
        raw_vals = da.values
        z500, z1000 = raw_vals[0].squeeze(), raw_vals[1].squeeze()

    thickness_dam = (z500 - z1000) / 98.0665
    return extract_contour_levels(
        thickness_dam,
        contour_config.get("contours", []),
        contour_config.get("contours_settings")
    )


def derive_ptype_array(ds_sf, ds_tp, ds_2t, dt_step):
    """
    Derives precipitation type category code (Rain, Freezing Rain, Sleet, Snow)
    from ERA5 surface snowfall (sf), total precip (tp), and 2m temperature (2t).
    """
    try:
        sf = np.squeeze(ds_sf.sel(time=dt_step, method="nearest").values)
        tp = np.squeeze(ds_tp.sel(time=dt_step, method="nearest").values)
        t2m = np.squeeze(ds_2t.sel(time=dt_step, method="nearest").values) - 273.15

        ptype = np.zeros_like(tp, dtype=np.uint8)
        has_precip = tp > 0.00005

        snow_ratio = np.divide(sf, tp, out=np.zeros_like(sf), where=tp > 0.00001)

        # 1 = Rain, 3 = Freezing Rain, 5 = Snow, 7 = Sleet/Ice
        ptype[has_precip & (snow_ratio > 0.75)] = 5
        ptype[has_precip & (snow_ratio <= 0.75) & (t2m < 0.0)] = 3
        ptype[has_precip & (snow_ratio <= 0.75) & (t2m >= 0.0)] = 1

        return ptype
    except Exception as e:
        return None


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

    start_dt = datetime.datetime.fromisoformat(START_DATE).replace(tzinfo=datetime.timezone.utc)
    end_dt   = datetime.datetime.fromisoformat(END_DATE).replace(tzinfo=datetime.timezone.utc)

    total_hours = int((end_dt - start_dt).total_seconds() // 3600)
    time_steps = list(range(0, total_hours + 1, STEP_HOURS))
    date_list = [start_dt + datetime.timedelta(hours=h) for h in time_steps]

    target_date_str = start_dt.strftime("%Y%m%d")
    init_time_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"\n🌪️ [{param_config['id']}] Storm: {STORM_NAME} ({STORM_ID}) | {len(time_steps)} steps ({START_DATE} to {END_DATE})")

    os.makedirs(output_dist_dir, exist_ok=True)
    temp_dir = os.path.join(output_dist_dir, "temp_downloads")
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Download Master File for this parameter
    main_grib = os.path.join(temp_dir, f"{STORM_ID}_{param_config['id']}.grib")
    t0 = time.perf_counter()
    download_era5_storm_dataset(param_config, date_list, main_grib)
    phases["download_main"] = time.perf_counter() - t0

    ds_main = xr.open_dataset(main_grib, engine="cfgrib", backend_kwargs={'errors': 'ignore'})

    # 2. Contour Sources (e.g. MSLP, Geopotential Heights, or Thickness)
    contour_sources = param_config.get("contour_sources")
    if not contour_sources and param_config.get("contour_source"):
        contour_sources = [param_config["contour_source"]]

    ds_contours = {}
    if contour_sources:
        for c_src in contour_sources:
            c_id = c_src.get("id", c_src.get("grib_param", "contour"))
            c_grib = os.path.join(temp_dir, f"{STORM_ID}_{c_id}.grib")
            c_cfg = {**param_config, **c_src}
            download_era5_storm_dataset(c_cfg, date_list, c_grib)
            ds_contours[c_id] = xr.open_dataset(c_grib, engine="cfgrib", backend_kwargs={'errors': 'ignore'})

    # 3. P-Type Source (if precipitation rate)
    ds_ptype_deps = {}
    if param_config.get("ptype_source"):
        for dep_param in ["sf", "2t"]:
            dep_grib = os.path.join(temp_dir, f"{STORM_ID}_{dep_param}.grib")
            dep_cfg = {"grib_param": dep_param, "era5_name": "snowfall" if dep_param == "sf" else "2m_temperature", "levtype": "sfc", "type": "fc" if dep_param == "sf" else "an"}
            download_era5_storm_dataset(dep_cfg, date_list, dep_grib)
            ds_ptype_deps[dep_param] = xr.open_dataset(dep_grib, engine="cfgrib", backend_kwargs={'errors': 'ignore'})

    # 4. Process all timesteps
    t0 = time.perf_counter()
    results = {}
    contour_steps = {}
    ptype_results = {}

    is_interval = bool(param_config.get("interval_hours") or re.search(r'[-_]?(6|24)h(?:r)?$', param_config.get("id", "").lower()))
    has_contour_sources = bool(contour_sources)
    raster_config = {**param_config, "contours": []} if has_contour_sources else param_config

    for idx, dt in enumerate(date_list):
        step_num = time_steps[idx]

        arr, c_levels = process_era5_step_from_ds(
            ds_main, dt, raster_config, target_date_str, want_raw=is_interval
        )
        if arr is not None:
            results[step_num] = arr
            c_levels = list(c_levels)

            # Extra contour layers (MSLP / Thickness)
            if contour_sources:
                for c_src in contour_sources:
                    c_id = c_src.get("id", c_src.get("grib_param", "contour"))
                    if c_src.get("type") == "thickness":
                        levels = process_thickness_from_ds(ds_contours[c_id], dt, c_src)
                    else:
                        c_cfg = {**param_config, **c_src}
                        _, levels = process_era5_step_from_ds(ds_contours[c_id], dt, c_cfg, target_date_str, want_raster=False)
                    c_levels.extend(levels)

            if c_levels:
                packed = pack_step(c_levels)
                if packed and packed["props"]:
                    contour_steps[step_num] = packed

            # P-Type derivation
            if ds_ptype_deps:
                ptype_arr = derive_ptype_array(ds_ptype_deps["sf"], ds_main, ds_ptype_deps["2t"], dt)
                if ptype_arr is not None:
                    ptype_results[step_num] = ptype_arr

    phases["decode_and_contour"] = time.perf_counter() - t0

    ds_main.close()
    for ds_c in ds_contours.values(): ds_c.close()
    for ds_p in ds_ptype_deps.values(): ds_p.close()

    sorted_steps = sorted(results.keys())

    # 5. Interval Differencing (6h / 24h accumulations)
    interval_h = param_config.get("interval_hours")
    if not interval_h:
        match = re.search(r'[-_]?(6|24)h(?:r)?$', param_config.get("id", "").lower())
        if match:
            interval_h = int(match.group(1))

    if interval_h:
        diff_results = {}
        for s in sorted_steps:
            target_prev = s - interval_h
            if target_prev <= 0:
                prev_arr = results.get(0, 0.0)
            else:
                candidates = [st for st in sorted_steps if st <= target_prev]
                prev_s = max(candidates) if candidates else None
                prev_arr = results[prev_s] if prev_s is not None else 0.0

            diff = np.maximum(0.0, results[s] - prev_arr)
            diff_results[s] = normalize_array(diff, param_config)
        results = diff_results

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

    # Clean temporary raw GRIB files before upload
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n🎉 [{param_config['id']}] Storm assets ready in {output_dist_dir}/")

    # 9. Upload to Hugging Face
    t0 = time.perf_counter()
    upload_to_huggingface(output_dist_dir, repo_id=HF_REPO_ID)
    phases["hf_upload"] = time.perf_counter() - t0

    # Cleanup local dist folder
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
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload storm {STORM_ID} data: {folder_path}"
        )
        print(f"  ✅ Uploaded to Hugging Face: {folder_path}")
    except Exception as e:
        print(f"  ❌ Failed to upload to Hugging Face: {e}")


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

        print(f"⏱️ Batch {batch_num}/{total_batches} took {time.perf_counter() - t_batch:.1f}s", flush=True)

    print(f"\n🎉 STORM '{STORM_NAME}' COMPLETED SUCCESSFULLY in {time.perf_counter() - t_run:.1f}s!")
