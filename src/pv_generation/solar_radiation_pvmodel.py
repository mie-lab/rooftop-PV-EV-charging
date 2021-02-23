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
    filename="solar_radiation_v2.log",
    filemode="w",
)

input_dirs = r"/disk6/pv_mobility_out"
output_dir = r"/disk6/pv_mobility_out/pvmodel2"

cmsaf_dir = r"/disk6/pvmobility/cmsaf/"
proj_epsg_lv03 = pyproj.Proj("+init=EPSG:21781")
proj_epsg_wgs84 = pyproj.Proj("+init=EPSG:4326")


NODATA = -9999.0
tz = pytz.timezone("UTC")


scenarios = ["ONLY_SOLAR", "PVMODEL_SPV170", "PVMODEL_JA", "PVMODEL_JINKO"]


def transform_angle(angle):
    angle = (360.0 - angle) - 90.0
    if angle > 180.0:
        return -180.0 + (angle - 180.0)
    else:
        return angle


# Read temperature data

tree, locs, tempdata = read_temp_data()


def get_temp(x03, y03):

    res = tree.query((x03, y03), k=50)

    temps = {}
    dd = datetime.datetime(2017, 1, 1, 0, 0)
    while dd.year == 2017:

        tkey = "_".join(list(map(str, [dd.year, dd.month, dd.day, dd.hour])))

        for i, iloc in enumerate(res[1]):

            stn = locs[iloc]["stn"]
            stn_elev = locs[iloc]["elev"]
            dist = res[0][i]

            # todo elevation correction, average multiple stations

            if stn in tempdata and tkey in tempdata[stn]:

                temps[tkey] = (tempdata[stn][tkey], stn_elev, dist)
                break

        dd += datetime.timedelta(hours=1)

    return temps


panel_spv = PVPanel(
    name="Schuco, SPV 170-SME-1",
    N_p=1.0,
    N_s=72.0,
    V_oc_ref=44.0,
    I_sc_ref=5.15,
    V_mp_ref=35.0,
    I_mp_ref=4.86,
    a_T_ref=None,
    a_Tper_ref=0.055 / 100.0,
    b_T_ref=None,
    b_Tper_ref=-0.37 / 100.0,
    NOCT=C2K(45.0),
    cell_area=125.0 * 125.0,
    panel_area=1580.4 * 808.4,
)

model_spv = ModelBatzelis(panel_spv)


panel_ja = PVPanel(
    name="JA Solar,  JAM60D09-315/BP",
    N_p=1.0,
    N_s=60.0,
    V_oc_ref=40.53,
    I_sc_ref=10.01,
    V_mp_ref=33.23,
    I_mp_ref=9.48,
    a_T_ref=None,
    a_Tper_ref=0.060 / 100.0,
    b_T_ref=None,
    b_Tper_ref=-0.300 / 100.0,
    NOCT=C2K(45.0),
    cell_area=1676.0 * 994.0 / 60.0,
    panel_area=1676.0 * 994.0,
)

model_ja = ModelBatzelis(panel_ja)


panel_jinko = PVPanel(
    name="Jinko, JKM320PP",
    N_p=1.0,
    N_s=72.0,
    V_oc_ref=46.4,
    I_sc_ref=9.05,
    V_mp_ref=37.4,
    I_mp_ref=8.56,
    a_T_ref=None,
    a_Tper_ref=0.060 / 100.0,
    b_T_ref=None,
    b_Tper_ref=-0.300 / 100.0,
    NOCT=C2K(45.0),
    cell_area=156.0 * 156.0,
    panel_area=1956.0 * 992.0,
)

panel_jinko = ModelBatzelis(panel_jinko)


def calc_ts(sis, dni, sid, lat, lon, D, elevation, b, ds, h, w, temperature):
    """
    Calculates the solar irradiation for one time step

    sis: Surface Incoming Solar Radiation W/m2
    dni: Direct Normalized Irradiance (DNI) W/m2
    sid: Surface Direct irradiance W/m2
    lat: latitude degree
    lon: longitude degree
    D: datetime
    elevation: elevation meter
    b:
    ds: data
    h: cell index
    w: cell index

    """

    _results = np.ones((len(scenarios), h, w)) * NODATA

    date = tz.localize(D)
    s = csolar2.solar.Solar(
        lat, lon, date, 0, elevation=elevation, temperature=temperature, pressure=1020
    )

    gamma_deg = s.get_elevation_angle()
    sun_azimuth = int(round(s.get_azimuth(), 0))

    if sun_azimuth == -180:
        sun_azimuth = 180

    if sis is None or dni is None or sid is None:
        return b, _results

    dif = sis - sid

    assert dif >= 0, "negative radiation"

    beta_deg = 0.0
    alpha_deg = 0.0

    for x in range(1, w - 1):
        for y in range(1, h - 1):

            if not ds["slope"]["data"][y, x] == ds["slope"]["nodata"]:
                slope = ds["slope"]["data"][y, x]
                aspect = ds["aspect"]["data"][y, x]
                ds_horizon = ds["hor_{}".format(sun_azimuth)]["data"][y, x]

                # TODO transform aspect
                aspect = transform_angle(aspect)

                beta_deg = slope
                alpha_deg = aspect

                # dif = diffuse radiation
                # dir = direct radiation
                # sis = global radiation
                dir_tilted, dif_tilted, sis_tilted = csolar2.solar.get_tilted_rad(
                    s, sis, sid, 0.2, beta_deg, alpha_deg  # slope
                )  # aspect

                # If sun is below horizon, we only use diffuse radiation
                tot_rad = dif_tilted
                if gamma_deg >= ds_horizon:
                    tot_rad += dir_tilted

                tot_rad = max(0.0, tot_rad)

                _results[scenarios.index("ONLY_SOLAR"), y, x] = tot_rad

                pv_eff = model_spv.module_efficency(temperature, tot_rad)

                _results[scenarios.index("PVMODEL_SPV170"), y, x] = tot_rad * pv_eff

                pv_eff_ja = model_ja.module_efficency(temperature, tot_rad)
                _results[scenarios.index("PVMODEL_JA"), y, x] = tot_rad * pv_eff_ja

                pv_eff_jinko = panel_jinko.module_efficency(temperature, tot_rad)
                _results[scenarios.index("PVMODEL_JINKO"), y, x] = (
                    tot_rad * pv_eff_jinko
                )

    return b, _results


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

    station_temperature = get_temp(cx, cy)

    def adapt_temp(station_temperature, D, elev):

        tkey = "_".join(list(map(str, [D.year, D.month, D.day, D.hour])))

        stn_temp, stn_elev, stn_dist = station_temperature[tkey]

        elev_diff = elev - stn_elev

        # -0.66 degree C per increase in 100 meter elevation
        temp_corr = elev_diff / 100.0 * -0.66

        temp = stn_temp + temp_corr

        # if we are higher than weather station, temperature should decrease
        if elev > stn_elev:
            assert temp < stn_temp, "temp correction is wrong"
        elif elev < stn_elev:
            assert temp > stn_temp, "temp correction is wrong"

        return temp

    # Read cmsaf data for each 30min
    d = datetime.date(2017, 1, 1)
    bands = defaultdict(int)
    radiations = defaultdict(dict)

    while d.year < 2018:

        print(d)

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

                    radiations[(bands[rad_type], D)][rad_type] = rad
                    D += datetime.timedelta(minutes=30)
                    bands[rad_type] += 1

        d += datetime.timedelta(days=1)

    res_bands = bands["sis"]

    # Initialize result datastructure with nodata
    results = {}
    for SCEN in scenarios:
        results[SCEN] = np.ones((res_bands, h, w)) * NODATA

    # For each cell, calculate solar radiation
    h, w = ds["slope"]["data"].shape

    jobs = []
    for b, D in sorted(radiations.keys(), key=lambda x: x[1]):

        if b % 1000 == 0:
            print(b, D)

        sis = radiations[(b, D)]["sis"]
        dni = radiations[(b, D)]["dni"]
        sid = radiations[(b, D)]["sid"]

        temperature = None
        if D.minute == 0:
            temperature = adapt_temp(station_temperature, D, elevation)
        elif D.minute == 30:

            t1 = adapt_temp(
                station_temperature, D - datetime.timedelta(minutes=30), elevation
            )

            if D.month == 12 and D.day == 31 and D.hour == 23:
                t2 = t1
            else:
                t2 = adapt_temp(
                    station_temperature, D + datetime.timedelta(minutes=30), elevation
                )
            temperature = 0.5 * (t1 + t2)
        else:
            assert False, "this should never happen"

        jobs.append((sis, dni, sid, lat, lon, D, elevation, b, ds, h, w, temperature))

    print("Process {} jobs".format(len(jobs)))
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    _rs = pool.starmap(calc_ts, jobs)
    for b, _r in _rs:
        for i, SCEN in enumerate(scenarios):
            results[SCEN][b, :, :] = _r[i]

    pool.close()
    pool.join()

    path = Path(input_dir)
    # Save results
    for SCEN in scenarios:
        fpath = os.path.join(
            output_dir,
            "solar_rad_house_{}_scenario_{}_v1.tif".format(path.parts[-1], SCEN),
        )
        print(fpath)
        with rasterio.open(
            fpath,
            mode="w",
            driver="GTiff",
            crs=CRS.from_epsg(21781),
            nodata=NODATA,
            count=res_bands,
            width=w,
            height=h,
            transform=ds["slope"]["fwd"],
            compress="lzw",
            dtype=rasterio.float32,
        ) as sink:

            sink.write(results[SCEN].astype(rasterio.float32))


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
