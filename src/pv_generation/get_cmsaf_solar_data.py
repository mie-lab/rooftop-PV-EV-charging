"""
Created on Dec 9, 2019

@author: rene
"""
from _collections import defaultdict
import datetime
import os
import csolar2.solar
import pyproj
from rasterio.crs import CRS
import pytz
import rasterio
import multiprocessing
import numpy as np
import logging
from batzelis2015 import PVPanel, C2K, ModelBatzelis
from tempdata import read_temp_data
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
    filename="solar_radiation_v2_onlysolar.log",
    filemode="w",
)

input_dirs = r"/disk6/pv_mobility_out"
output_dir = r"/disk6/pv_mobility_out/pvmodel2"

cmsaf_dir = r"/disk6/pvmobility/cmsaf/"
proj_epsg_lv03 = pyproj.Proj("+init=EPSG:21781")
proj_epsg_wgs84 = pyproj.Proj("+init=EPSG:4326")


NODATA = -9999.0
tz = pytz.timezone("UTC")


def transform_angle(angle):
    angle = (360.0 - angle) - 90.0
    if angle > 180.0:
        return -180.0 + (angle - 180.0)
    else:
        return angle


# Each band stands for 30 minutes interval
hours_per_ts = 0.5


def calc_solar_rad(input_dir):

    # Read reaster datasets
    def read_ds(path):
        with rasterio.open(path) as src:
            data = src.read(1)
            fwd = src.transform
            nodata = src.nodata
        return data, fwd, nodata

    ds = {}
    files = ["slope", "aspect", "dom.tif"] + [
        "hor_{}".format(w) for w in range(-179, 181, 1)
    ]

    for fname in files:
        d, fwd, nodata = read_ds(os.path.join(input_dir, fname))
        ds[fname.replace(".tif", "")] = {"data": d, "fwd": fwd, "nodata": nodata}

    # Get elevation
    domdata = np.ma.masked_where(
        ds["dom"]["data"] == ds["dom"]["nodata"], ds["dom"]["data"]
    )
    elevation = domdata.mean()

    # Read temperature data
    h, w = ds["slope"]["data"].shape
    cx, cy = (w * 0.5, h * 0.5) * ds["slope"]["fwd"]
    lon, lat = pyproj.transform(proj_epsg_lv03, proj_epsg_wgs84, cx, cy)

    # Read cmsaf data for each 30min
    d = datetime.date(2017, 1, 1)
    bands = defaultdict(int)

    path = Path(input_dir)

    tots = defaultdict(float)

    while d.year < 2018:

        cmsaf_files = [
            ("dni", "DNIin{:04d}{:02d}{:02d}0000003UD10001E1UD.nc"),
            ("sid", "SIDin{:04d}{:02d}{:02d}0000003UD10001E1UD.nc"),
            ("sis", "SISin{:04d}{:02d}{:02d}0000003UD10001E1UD.nc"),
        ]

        for rad_type, fname in cmsaf_files:
            D = datetime.datetime(d.year, d.month, d.day, 0, 0)
            fname = fname.format(d.year, d.month, d.day)
            with rasterio.open(os.path.join(cmsaf_dir, fname)) as src:
                ix, iy = map(int, (lon, lat) * ~src.transform)
                raddata = src.read()

                for b in range(0, src.count):
                    rad = raddata[b, iy, ix]

                    if rad == src.nodata:
                        rad = None
                    else:
                        rad = float(rad)
                        tots[rad_type] += rad * hours_per_ts

                    D += datetime.timedelta(minutes=30)
                    bands[rad_type] += 1

        d += datetime.timedelta(days=1)

    fpath = os.path.join("data", "cmsaf_rad_{}.json".format(path.parts[-1]))
    print(fpath)
    with open(fpath, "w") as f:

        vals = ["sis_Wh", "sid_Wh", "dni_Wh"]
        f.write(",".join(list(map(str, vals))) + "\n")
        vals = [tots["sis"], tots["sid"], tots["dni"]]
        print(vals)
        f.write(",".join(list(map(str, vals))) + "\n")


for input_dir in [
    os.path.join(input_dirs, o)
    for o in os.listdir(input_dirs)
    if os.path.isdir(os.path.join(input_dirs, o))
]:

    try:
        logging.info("process {}".format(input_dir))
        calc_solar_rad(input_dir)

    except Exception as e:
        print(str(e))
        logging.exception("ooops: {}".format(str(e)))
