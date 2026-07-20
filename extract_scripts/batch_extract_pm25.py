"""
Batch extraction of daily PM2.5 (or a related pollutant) concentrations at a list
of geographic coordinates over a date range, from the monthly NetCDF dataset.

Dataset layout expected (one file per month, holding DAILY slices):

    <data-dir>/dataset_<YEAR>/pm25_<YEAR>-<MM>.nc

Each file holds the variable (default "PM2.5") with dims (date, y, x) on the
dataset's own projected grid. The CRS is read directly from the files
(EPSG:5070 in the current dataset).

Example
-------
    python batch_extract_pm25.py \
        --coords sample_coords.csv \
        --start 2005-08-15 --end 2005-09-15 \
        --data-dir ./demo_data \
        --var "PM2.5" \
        --output pm25_extracted.csv \
        --format wide

The coordinates CSV needs latitude and longitude columns (defaults
`latitude` / `longitude`); an optional id column (default `site_id`) is carried
through. Points outside the CONUS box are dropped (or kept as NaN with
--keep-invalid).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor)
from pyproj import CRS, Transformer, network
network.set_network_enabled(False)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *a, **k):
        return iterable if iterable is not None else []


# Approximate CONUS bounding box in lon/lat (padded). Extent sanity check only;
# it also spans some ocean/border area, so NaN extractions flag points outside
# the actual data coverage.
CONUS_BOUNDS = dict(lat_min=24.0, lat_max=50.0, lon_min=-125.0, lon_max=-66.5)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def within_conus(lat: float, lon: float) -> bool:
    return (CONUS_BOUNDS["lat_min"] <= lat <= CONUS_BOUNDS["lat_max"]
            and CONUS_BOUNDS["lon_min"] <= lon <= CONUS_BOUNDS["lon_max"])


def months_spanned(start: str, end: str):
    """Inclusive list of (year, month) covering the [start, end] date range."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e < s:
        raise ValueError(f"--end ({end}) is before --start ({start}).")
    periods = pd.period_range(s.to_period("M"), e.to_period("M"), freq="M")
    return [(p.year, p.month) for p in periods]


def month_file_path(data_dir, year, month, var_prefix, year_subdir, file_template):
    sub = year_subdir.format(year=year)
    fname = file_template.format(prefix=var_prefix, year=year, month=month)
    return os.path.join(data_dir, sub, fname)


def read_dataset_crs(path, var_name):
    """Read the CRS embedded in the file (no hard-coding)."""
    ds = xr.open_dataset(path, decode_coords="all")
    crs = ds.rio.crs
    if crs is None and var_name in ds:
        crs = ds[var_name].rio.crs
    ds.close()
    if crs is None:
        raise ValueError(
            f"No CRS found in {os.path.basename(path)}. "
            f"Pass --crs (e.g. --crs EPSG:5070) to specify it manually."
        )
    return crs


def resolve_var(ds, var_name):
    if var_name in ds:
        return var_name
    candidates = [v for v in ds.data_vars if ds[v].ndim >= 2]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(
        f"Variable '{var_name}' not found. Available 2-D+ variables: {candidates}. "
        f"Specify one with --var."
    )


def _sample_point_nearest_valid(da, xindex, yindex, dx, dy, px, py, max_radius_px):
    """Daily series at the nearest pixel to (px, py); if NaN (e.g. dropped coastline
    cell), snap to the nearest pixel that HAS data, up to max_radius_px.
    Returns (series, snap_distance_m)."""
    ix0 = int(np.abs(xindex - px).argmin())
    iy0 = int(np.abs(yindex - py).argmin())
    center = da.isel(y=iy0, x=ix0)
    if bool(center.notnull().all()):
        return center, 0.0
    r = 1
    while r <= max_radius_px:
        y0, y1 = max(0, iy0 - r), min(da.sizes["y"], iy0 + r + 1)
        x0, x1 = max(0, ix0 - r), min(da.sizes["x"], ix0 + r + 1)
        if bool(da.isel(y=slice(y0, y1), x=slice(x0, x1)).notnull().all("date").any()):
            R = int(np.ceil(r * 2 ** 0.5)) + 1
            y0, y1 = max(0, iy0 - R), min(da.sizes["y"], iy0 + R + 1)
            x0, x1 = max(0, ix0 - R), min(da.sizes["x"], ix0 + R + 1)
            win = da.isel(y=slice(y0, y1), x=slice(x0, x1))
            valid = win.notnull().all("date").values
            yy, xx = np.nonzero(valid)
            yy, xx = yy + y0, xx + x0
            dist = np.hypot((yy - iy0) * dy, (xx - ix0) * dx)
            k = int(np.argmin(dist))
            return da.isel(y=int(yy[k]), x=int(xx[k])), float(dist[k])
        r *= 2
    return center, np.nan


def sample_month(path, var_name, xs, ys, site_ids,
                 fill_nearest_valid=True, max_snap_km=50.0, snapped=None):
    """Return a DataArray of dims (date, points) sampled point-wise at (xs, ys).

    With fill_nearest_valid, points whose nearest pixel is NaN are snapped to the
    nearest pixel that has data (within max_snap_km); snap distances (m) are
    recorded in the optional `snapped` dict keyed by site id."""
    ds = xr.open_dataset(path, decode_coords="all")
    var = resolve_var(ds, var_name)
    da = ds[var]
    if not fill_nearest_valid:
        x_idx = xr.DataArray(np.asarray(xs), dims="points", coords={"points": site_ids})
        y_idx = xr.DataArray(np.asarray(ys), dims="points", coords={"points": site_ids})
        sampled = da.sel(x=x_idx, y=y_idx, method="nearest").load()
    else:
        xindex, yindex = da["x"].values, da["y"].values
        dx = abs(float(xindex[1] - xindex[0]))
        dy = abs(float(yindex[1] - yindex[0]))
        max_radius_px = int(np.ceil(max_snap_km * 1000 / min(dx, dy)))
        cols = []
        for pid, px, py in zip(site_ids, np.asarray(xs), np.asarray(ys)):
            s, dist = _sample_point_nearest_valid(
                da, xindex, yindex, dx, dy, px, py, max_radius_px)
            cols.append(s.load().reset_coords(drop=True))
            if snapped is not None and dist and not np.isnan(dist) and dist > 0:
                snapped[pid] = max(snapped.get(pid, 0.0), dist)
        sampled = (xr.concat(cols, dim="points")
                   .assign_coords(points=list(site_ids)).transpose("date", "points"))
    ds.close()
    return sampled


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Batch-extract daily pollutant concentrations at coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coords", required=True, help="CSV of coordinates.")
    p.add_argument("--start", required=True, help="First date, 'YYYY-MM-DD'.")
    p.add_argument("--end", required=True, help="Last date (inclusive), 'YYYY-MM-DD'.")
    p.add_argument("--data-dir", default=".", help="Base dir containing dataset_<YEAR>/.")
    p.add_argument("--output", required=True, help="Output CSV path.")
    p.add_argument("--var", default="PM2.5", help="NetCDF variable name to sample.")
    p.add_argument("--crs", default=None,
                   help="Override CRS (e.g. 'EPSG:5070'). Default: read from the files.")
    p.add_argument("--lat-col", default="latitude", help="Latitude column in --coords.")
    p.add_argument("--lon-col", default="longitude", help="Longitude column in --coords.")
    p.add_argument("--id-col", default="site_id", help="Optional id column carried through.")
    p.add_argument("--var-prefix", default="pm25", help="Filename prefix.")
    p.add_argument("--year-subdir", default="dataset_{year}", help="Per-year sub-folder pattern.")
    p.add_argument("--file-template", default="{prefix}_{year}-{month:02d}.nc",
                   help="Filename pattern within each year sub-folder.")
    p.add_argument("--format", choices=["wide", "long"], default="wide",
                   help="wide: one column per date; long: one row per site-date.")
    p.add_argument("--keep-invalid", action="store_true",
                   help="Keep out-of-CONUS points (values NaN) instead of dropping.")
    p.add_argument("--no-fill", action="store_true",
                   help="Disable nearest-valid fill; use the strict nearest pixel "
                        "(coastal/boundary points may be NaN).")
    p.add_argument("--max-snap-km", type=float, default=50.0,
                   help="Max distance to snap a NaN point to the nearest valid pixel.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    df = pd.read_csv(args.coords)
    for col in (args.lat_col, args.lon_col):
        if col not in df.columns:
            sys.exit(f"ERROR: column '{col}' not found in {args.coords}. "
                     f"Columns: {list(df.columns)}")
    if args.id_col not in df.columns:
        df[args.id_col] = [f"site_{i}" for i in range(len(df))]

    # --- CONUS extent validity check -------------------------------------- #
    df["_valid"] = [within_conus(la, lo)
                    for la, lo in zip(df[args.lat_col], df[args.lon_col])]
    n_bad = int((~df["_valid"]).sum())
    if n_bad:
        print(f"[warn] {n_bad} coordinate(s) outside CONUS box: "
              f"{df.loc[~df['_valid'], args.id_col].tolist()}")
        if not args.keep_invalid:
            df = df[df["_valid"]].copy()
            print("[warn] dropped them (use --keep-invalid to keep as NaN).")
    if df.empty:
        sys.exit("ERROR: no valid coordinates left to extract.")

    months = months_spanned(args.start, args.end)

    # --- discover CRS from the first existing file ------------------------ #
    first_path = None
    for (y, m) in months:
        cand = month_file_path(args.data_dir, y, m, args.var_prefix,
                               args.year_subdir, args.file_template)
        if os.path.exists(cand):
            first_path = cand
            break
    if first_path is None:
        sys.exit("ERROR: no data files found for the requested range.")
    data_crs = CRS.from_user_input(args.crs) if args.crs else read_dataset_crs(first_path, args.var)
    print(f"[info] dataset CRS: {data_crs.to_string()}")

    # --- reproject lon/lat -> dataset CRS --------------------------------- #
    transformer = Transformer.from_crs("EPSG:4326", data_crs, always_xy=True)
    site_ids = df.loc[df["_valid"], args.id_col].astype(str).tolist()
    xs, ys = transformer.transform(df.loc[df["_valid"], args.lon_col].values,
                                   df.loc[df["_valid"], args.lat_col].values)

    print(f"[info] {len(site_ids)} site(s), {len(months)} month file(s), "
          f"{args.start}..{args.end}, variable '{args.var}'.")

    # --- sample each month, then restrict to the exact date range --------- #
    pieces = []
    snapped = {}
    for (y, m) in tqdm(months, desc="Extracting", unit="file"):
        path = month_file_path(args.data_dir, y, m, args.var_prefix,
                               args.year_subdir, args.file_template)
        if not os.path.exists(path):
            print(f"[warn] missing file, skipping: {path}")
            continue
        pieces.append(sample_month(path, args.var, xs, ys, site_ids,
                                   fill_nearest_valid=not args.no_fill,
                                   max_snap_km=args.max_snap_km, snapped=snapped))

    if not pieces:
        sys.exit("ERROR: no files could be read for the requested range.")
    if snapped:
        print("[info] snapped coastal point(s) to nearest valid pixel:",
              {k: f"{v/1000:.1f} km" for k, v in snapped.items()})

    combined = xr.concat(pieces, dim="date").sortby("date")
    combined = combined.sel(date=slice(args.start, args.end))
    # (date, points) -> DataFrame index=date, columns=site_ids
    values = combined.transpose("date", "points").to_pandas()
    values.index = pd.DatetimeIndex(values.index).strftime("%Y-%m-%d")
    values = values.T  # -> index=site_ids, columns=dates

    # --- assemble output, re-attaching invalid rows as NaN if requested --- #
    date_cols = list(values.columns)
    out = df[[args.id_col, args.lat_col, args.lon_col]].copy().reset_index(drop=True)
    id_series = df[args.id_col].astype(str).reset_index(drop=True)
    for c in date_cols:
        out[c] = id_series.map(values[c]).values

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.format == "wide":
        out.to_csv(args.output, index=False)
    else:
        long = out.melt(id_vars=[args.id_col, args.lat_col, args.lon_col],
                        value_vars=date_cols, var_name="date", value_name=args.var)
        long.to_csv(args.output, index=False)

    print(f"[done] wrote {args.output} "
          f"({args.format}; {len(out)} site(s) x {len(date_cols)} date(s)).")


if __name__ == "__main__":
    main()
