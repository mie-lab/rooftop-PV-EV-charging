import os
from os.path import join as osp
import glob
import shutil
import ntpath

# # #
os.system(f'python ./src/plotting/plot_pv_generation_year.py')
os.system(f'python ./src/plotting/plot_results_ev_energy_demands_year.py')
os.system(f'python ./src/plotting/plot_bev_at_home_over_the_year.py')
os.system(f'python ./src/plotting/plot_scatter_roof_consumption_coverate.py')
os.system(f'python ./src/plotting/plot_cumsum_charging_strategies.py')
os.system(f'python ./src/plotting/plot_co2_usage_over_year.py')
os.system(f'python ./src/plotting/plot_histogram_overallcoverage.py')
os.system(f'python ./src/plotting/plot_user_coverage_over_year.py')
os.system(f'python ./src/plotting/plot_panel_sensitivity.py')
os.system(f'python ./src/plotting/plot_soc_over_time_singleuser_rawdata.py')
os.system(f'python ./src/plotting/plot_power_factor.py')