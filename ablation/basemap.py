# -*- coding: utf-8 -*-
"""
Minimal slippy-map tile fetcher: bounding box -> stitched basemap image.

Written by hand rather than pulling in contextily, which would drag along
geopandas, pyproj and rasterio. Everything here needs only requests + PIL.

Two caches, because the animation renders 48 frames on the same background:
  - individual tiles under results/_tiles/
  - the stitched image under results/_tiles/stitched_*.png
So the first render fetches ~25 tiles and every later one reads one PNG. That
also means the scripts still work on a machine with no internet.

Coordinates: tiles are Web Mercator (EPSG:3857). The demo's own projection is
a local flat approximation, which would drift against the streets, so anything
drawn on top of these tiles has to go through lonlat_to_mercator() here.
"""
import io
import math
import os
import time

import requests
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, "results", "_tiles")

R = 6378137.0                      # WGS84 semi-major axis
WORLD = 2 * math.pi * R            # 40075016.7 m

# CARTO's basemaps.cartocdn.com now serves an "API KEY REQUIRED" watermark
# instead of map data -- the request still returns HTTP 200, so this only shows
# up when you look at the image. Both providers below were verified visually.
PROVIDERS = {
    # light grey, minimal -- stays out of the way under many overlapping lines
    "gray": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
             "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
             "Esri, HERE, Garmin, (c) OpenStreetMap contributors"),
    # full colour: roads, water, labels -- closest to the Maps look
    "osm": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "(c) OpenStreetMap contributors"),
    "osm_de": ("https://tile.openstreetmap.de/{z}/{x}/{y}.png",
               "(c) OpenStreetMap contributors"),
}


def lonlat_to_mercator(lon, lat):
    """WGS84 -> EPSG:3857 metres. Works on scalars and numpy arrays."""
    import numpy as np
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = R * np.radians(lon)
    y = R * np.log(np.tan(math.pi / 4 + np.radians(lat) / 2))
    return x, y


def _deg2tile(lon, lat, z):
    n = 2 ** z
    xt = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    yt = (1.0 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return xt, yt


def pick_zoom(lon_min, lat_min, lon_max, lat_max, min_px=1000, zmax=16):
    """Smallest zoom whose stitched image is at least min_px across."""
    for z in range(10, zmax + 1):
        x0, _ = _deg2tile(lon_min, lat_max, z)
        x1, _ = _deg2tile(lon_max, lat_min, z)
        if (x1 - x0) * 256 >= min_px:
            return z
    return zmax


def _fetch(url, tries=3):
    hdr = {"User-Agent": "bike-link-prediction/1.0 (student project)"}
    for i in range(tries):
        try:
            r = requests.get(url, headers=hdr, timeout=15)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(0.6 * (i + 1))
    return None


def basemap(lon_min, lat_min, lon_max, lat_max, provider="positron",
            zoom=None, retina=True, verbose=True):
    """Stitched basemap for the box.

    Returns (PIL.Image, extent) with extent = (x0, x1, y0, y1) in Web Mercator
    metres, ready for matplotlib's imshow(..., extent=extent).
    """
    os.makedirs(CACHE, exist_ok=True)
    url_tpl, attribution = PROVIDERS[provider]
    z = zoom or pick_zoom(lon_min, lat_min, lon_max, lat_max)
    r_suffix = "@2x" if (retina and "{r}" in url_tpl) else ""

    key = f"stitched_{provider}_z{z}{'_2x' if r_suffix else ''}_" \
          f"{lon_min:.4f}_{lat_min:.4f}_{lon_max:.4f}_{lat_max:.4f}.png"
    stitched_path = os.path.join(CACHE, key)
    meta_path = stitched_path.replace(".png", ".txt")
    if os.path.exists(stitched_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            ext = tuple(float(v) for v in f.read().split())
        if verbose:
            print(f"  [basemap] cache hit: {os.path.basename(stitched_path)}")
        return Image.open(stitched_path), ext, attribution

    x0f, y0f = _deg2tile(lon_min, lat_max, z)      # top-left
    x1f, y1f = _deg2tile(lon_max, lat_min, z)      # bottom-right
    xa, xb = int(math.floor(x0f)), int(math.floor(x1f))
    ya, yb = int(math.floor(y0f)), int(math.floor(y1f))
    nx, ny = xb - xa + 1, yb - ya + 1
    ts = 512 if r_suffix else 256

    if verbose:
        print(f"  [basemap] {provider} z={z}: {nx}x{ny} = {nx*ny} Kacheln")
    canvas = Image.new("RGB", (nx * ts, ny * ts), (240, 240, 240))
    missing = 0
    for i, xt in enumerate(range(xa, xb + 1)):
        for j, yt in enumerate(range(ya, yb + 1)):
            tile_path = os.path.join(CACHE, f"{provider}_{z}_{xt}_{yt}{r_suffix}.png")
            if os.path.exists(tile_path):
                data = open(tile_path, "rb").read()
            else:
                url = (url_tpl.replace("{s}", "abc"[(xt + yt) % 3])
                       .replace("{z}", str(z)).replace("{x}", str(xt))
                       .replace("{y}", str(yt)).replace("{r}", r_suffix))
                data = _fetch(url)
                if data:
                    open(tile_path, "wb").write(data)
            if not data:
                missing += 1
                continue
            canvas.paste(Image.open(io.BytesIO(data)).convert("RGB"),
                         (i * ts, j * ts))
    if missing and verbose:
        print(f"  [basemap] WARNUNG: {missing} Kacheln fehlen (offline?)")

    tile_m = WORLD / (2 ** z)
    ext = (-WORLD / 2 + xa * tile_m, -WORLD / 2 + (xb + 1) * tile_m,
           WORLD / 2 - (yb + 1) * tile_m, WORLD / 2 - ya * tile_m)
    canvas.save(stitched_path)
    with open(meta_path, "w") as f:
        f.write(" ".join(f"{v:.4f}" for v in ext))
    return canvas, ext, attribution
