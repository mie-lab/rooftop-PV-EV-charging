import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.plotting.myplotlib import init_figure, Columnes, Journal, save_figure

fig_out = os.path.join('plots', 'panel_sensitivity', 'plot_panel_sensitivity.png')
output_folder = os.path.join('.', 'data', 'output')
all_pv_models = ["ONLY_SOLAR_PVSYSEFF_5",
                 "ONLY_SOLAR_PVSYSEFF_6", "ONLY_SOLAR_PVSYSEFF_7", "ONLY_SOLAR_PVSYSEFF_8",
                 "ONLY_SOLAR_PVSYSEFF_9", "ONLY_SOLAR_PVSYSEFF_10", "ONLY_SOLAR_PVSYSEFF_11",
                 "ONLY_SOLAR_PVSYSEFF_12", "ONLY_SOLAR_PVSYSEFF_13", "ONLY_SOLAR_PVSYSEFF_14",
                 "ONLY_SOLAR_PVSYSEFF_15", "ONLY_SOLAR_PVSYSEFF_16", "ONLY_SOLAR_PVSYSEFF_17",
                 "ONLY_SOLAR_PVSYSEFF_18", "ONLY_SOLAR_PVSYSEFF_19", "ONLY_SOLAR_PVSYSEFF_20",
                 "ONLY_SOLAR_PVSYSEFF_21", "ONLY_SOLAR_PVSYSEFF_22", "ONLY_SOLAR_PVSYSEFF_23",
                 "ONLY_SOLAR_PVSYSEFF_24", "ONLY_SOLAR_PVSYSEFF_25", "ONLY_SOLAR_PVSYSEFF_26",
                 "ONLY_SOLAR_PVSYSEFF_27", "ONLY_SOLAR_PVSYSEFF_28", "ONLY_SOLAR_PVSYSEFF_29",
                 # "ONLY_SOLAR_PVSYSEFF_100"
                 ]

df_all_list = []
for pv_model_this in all_pv_models:
    # pv_model_this = all_pv_models[0]
    eff_this = pv_model_this.split('_')[-1]
    result_folder_this = os.path.join(output_folder, pv_model_this)

    cov_by_scenario = pd.read_csv(os.path.join(result_folder_this, "coverage_by_scenario.csv"))

    df_b = pd.DataFrame()
    df_s1 = pd.DataFrame()
    df_s2 = pd.DataFrame()
    df_s3 = pd.DataFrame()

    df_b['coverage'] = cov_by_scenario['baseline']
    df_s1['coverage'] = cov_by_scenario['scenario1']
    df_s2['coverage'] = cov_by_scenario['scenario2']
    df_s3['coverage'] = cov_by_scenario['scenario3']

    df_b['scenario'] = 'Baseline'
    df_s1['scenario'] = 'Scenario 1'
    df_s2['scenario'] = 'Scenario 2'
    df_s3['scenario'] = 'Scenario 3'

    df_all = pd.concat((df_b, df_s1, df_s2, df_s3))
    df_all['eff'] = eff_this
    df_all_list.append(df_all)

df_all_all = pd.concat(df_all_list)

journal = Journal.POWERPOINT_A3

fig, ax = init_figure(nrows=1,
                      ncols=1,
                      columnes=Columnes.ONE,
                      journal=journal, sharex=True, sharey=True)
sns.lineplot(data=df_all_all, x="eff", y="coverage", hue="scenario",
             linewidth=3, ax=ax)
plt.ylabel("Average coverage [\%]", labelpad=20)
plt.xlabel("System efficiency [\%]", labelpad=20)
# legend stuff
handles, labels = ax.get_legend_handles_labels()
for handle in handles:
    handle.set_linewidth(3)
    # handle._linewidth = 50
leg = ax.legend(handles=handles, labels=labels,
                loc='upper left', frameon=False, bbox_to_anchor=(-0.15, -0.15),
                ncol=4)

for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

bbox_extra_artists = [fig, leg]
save_figure(fig_out, bbox_extra_artists=bbox_extra_artists)
plt.close(fig)
