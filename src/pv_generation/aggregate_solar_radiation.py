import rasterio
import numpy as np
import os
import json
import gzip
from pathlib import Path
import pprint
from pvlib.pvsystem import pvwatts_ac

# Each cell has size 0.5 * 0.5 m2
cell_area = 0.5 * 0.5

# Each band stands for 30 minutes interval
hours_per_ts = 0.5 


output_dir =  r"/disk6/pv_mobility_out/pvmodel2"

scenarios = ["PVMODEL_SPV170",
            "PVMODEL_JA",
            "PVMODEL_JINKO"]


def aggregate(user_directory):
    
    print("Processing {}".format(user_directory))

    res = {}

    # read mask
    with rasterio.open(os.path.join(user_directory, "mask.tif")) as src:
        mask = src.read(1)

    area = np.sum(mask) * 0.5 * 0.5
    print("area", area)

    path = Path(user_directory)
    
    res = {}
    res['area'] = area
    
    for SCEN in scenarios:
        fpath = os.path.join(output_dir, "solar_rad_house_{}_scenario_{}_v1.tif".format(path.parts[-1], SCEN))
        
        
        res[SCEN] = {}
        
        with rasterio.open(fpath) as src:

            
            bands = src.count
            data = src.read(masked=True)

            max_W = 0.0
            roofWs = {}
            for b in range(bands):
                roof_W = np.sum(data[b,:,:].filled(fill_value=0) * mask) * cell_area
                roofWs[b] = roof_W
                max_W = max(max_W, roof_W)
            
            print(max_W)

            year_Wh_dc = 0.0
            year_Wh_ac = 0.0

            for b in range(bands):

                roof_W = roofWs[b]

                # Convert radiation level per cell to Watt hours
                roof_Wh = roof_W * hours_per_ts
                
                year_Wh_dc += roof_Wh
                
                roof_Wh_ac = pvwatts_ac(pdc=roof_Wh, pdc0=max_W)
                
                year_Wh_ac += roof_Wh_ac
                
                
                if roof_Wh_ac > 0:
                    res[SCEN]["band_{}_Wh".format(b)] = round(float(roof_Wh_ac), 4)
            
            res[SCEN]["max_W"] = max_W
            res[SCEN]["year_Wh_dc"] = year_Wh_dc
            res[SCEN]["year_Wh_ac"] = year_Wh_ac
            res[SCEN]["year_inv_eff"] = round(year_Wh_ac / year_Wh_dc * 100.0, 3)
            print(SCEN, max_W, year_Wh_dc, year_Wh_ac, round(year_Wh_ac / year_Wh_dc * 100.0, 3))
    
    
    pv_sys_effs = list(range(5, 30)) + [100]
    for pv_sys_eff in pv_sys_effs:
        res["ONLY_SOLAR_PVSYSEFF_{}".format(pv_sys_eff)] = {}
        res["ONLY_SOLAR_PVSYSEFF_{}".format(pv_sys_eff)]["max_W"] = 0.0
        res["ONLY_SOLAR_PVSYSEFF_{}".format(pv_sys_eff)]["year_Wh_ac"] = 0.0
    
    
    fpath = os.path.join(output_dir, "solar_rad_house_{}_scenario_{}_v1.tif".format(path.parts[-1], "ONLY_SOLAR"))

    
    with rasterio.open(fpath) as src:
        bands = src.count
        data = src.read(masked=True)


        for b in range(bands):
            roof_W = np.sum(data[b,:,:].filled(fill_value=0) * mask) * cell_area

            for pv_sys_eff in pv_sys_effs:
                roof_Wh_ac = roof_W * hours_per_ts * pv_sys_eff / 100.0
                
                if roof_Wh_ac > 0:
                    res["ONLY_SOLAR_PVSYSEFF_{}".format(pv_sys_eff)]["band_{}_Wh".format(b)] = round(float(roof_Wh_ac), 4)
                    res["ONLY_SOLAR_PVSYSEFF_{}".format(pv_sys_eff)]["year_Wh_ac"] += roof_Wh_ac
                    res["ONLY_SOLAR_PVSYSEFF_{}".format(pv_sys_eff)]["max_W"] = max(roof_W, 
                                                                                    res["ONLY_SOLAR_PVSYSEFF_{}".format(pv_sys_eff)]["max_W"])

    with gzip.GzipFile(os.path.join("data", "solar_rad_{}.json.gz".format(path.parts[-1])), 'w') as f:
        f.write(json.dumps(res).encode('utf-8')) 


input_dirs = r"/disk6/pv_mobility_out"
t = 0
for input_dir in [os.path.join(input_dirs, o) for o in os.listdir(input_dirs) 
                    if os.path.isdir(os.path.join(input_dirs,o))]:
    t += 1
    print(t)
    try:
        aggregate(input_dir)
    except Exception as e:
        print(str(e))

        
