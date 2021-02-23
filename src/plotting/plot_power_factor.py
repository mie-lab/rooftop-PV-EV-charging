import os

import matplotlib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from statsmodels.formula.api import ols

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.plotting.myplotlib import init_figure, Columnes, Journal, save_figure
from src.db_login import DSN

try:
    df = pd.read_csv(os.path.join('.', 'data', 'power_factor_data.csv'))

except FileNotFoundError:
    engine = create_engine('postgresql://{user}:{password}@{host}' +
                           ':{port}/{dbname}'.format(**DSN))

    query = """SELECT soc_customer_start, soc_customer_end,
                consumed_electric_energy_total
                from bmw where zustand = 'fahrt'"""

    df = pd.read_sql_query(query, con=engine)
    df.to_csv(os.path.join('.', 'data', 'power_factor_data.csv'))

# data_PV_Solar preparation
df['dsoc'] = df['soc_customer_start'] - df['soc_customer_end']
df['pow'] = df['consumed_electric_energy_total']

# filter all that is zero
df = df.drop(df[(df['dsoc'] <= 0) | (df['pow'] <= 0)].index)

# create plot
f, ax = init_figure(nrows=1,
                    ncols=1,
                    columnes=Columnes.ONE,
                    journal=Journal.POWERPOINT_A3)

hb = ax.hexbin(df['dsoc'],
               df['pow'],
               gridsize=100,
               bins='log',
               cmap=plt.get_cmap('bone_r'),
               mincnt=1,
               linewidths=0.2,
               edgecolors='slategray')

cb = f.colorbar(hb, ax=ax)
cb.set_label(r'log10(n), n = Number of points in cell', labelpad=20)

ax.set_ylabel(r"Consumed power [\si{\kilo\watthour}]", labelpad=20)
ax.set_xlabel(r'Change in state of charge ($\Delta_{{SoC}}$) [$\%$]', labelpad=20)

ax.set_xlim(0, 100)
ax.set_ylim(0, 32)

# Fit sqrt model
model = ols("pow ~ dsoc + np.sqrt(dsoc) - 1", df)
results = model.fit()
r2_sqrt = results.rsquared
params_sqrt = results.params

# Fit linear model
model_lin = ols("pow ~ dsoc -1", df)
results_lin = model_lin.fit()
r2_lin = results_lin.rsquared
params_lin = results_lin.params

# Plot regression
x = np.linspace(0, 100, 30)
y = x * params_sqrt['dsoc'] + np.sqrt(x) * params_sqrt['np.sqrt(dsoc)']
y_lin = x * params_lin['dsoc']

ln_sqrt, = ax.plot(x, y,
                   linestyle='--',
                   color='r',
                   linewidth=3,
                   label=r"${} * \Delta_{{SoC}}  + {} * ".format(
                       round(params_sqrt['dsoc'], 3),
                       round(
                           params_sqrt['np.sqrt(dsoc)'], 3)) + \
                         r"\sqrt{{ \Delta_{{SoC}} }}$, $R^2 = {}$".format(
                             round(r2_sqrt, 2))
                   )
#
ln_lin, = ax.plot(x, y_lin,
                  linestyle=':',
                  color='orange',
                  linewidth=3,
                  label='linear fit',
                  zorder=10  # draw this line on top
                  )

leg = ax.legend(handles=[ln_sqrt, ln_lin], loc=2, frameon=False)
leg.set_zorder(20)
frame = leg.get_frame()
frame.set_facecolor('#ebefe8')

ax.text(0.99, 0.01, r"N = \num{{ {} }}".format(df.shape[0]), \
        horizontalalignment='right', verticalalignment='bottom', \
        transform=ax.transAxes)

save_figure(os.path.join(".", "plots", "power_factor", "power_factor.png"))