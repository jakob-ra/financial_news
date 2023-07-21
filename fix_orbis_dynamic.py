import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('C:/Users/Jakob/Downloads/lexis_alliances_orbis_dynamic.csv')
df.loc[df.year >= 2012, 'Research & Development expenses'] = df.loc[df.year >= 2012, 'Research & Development expenses'] * 1000
df.loc[df.year >= 2012, 'Added value'] = df.loc[df.year >= 2012, 'Added value'] * 1000
df.loc[df.year >= 2013, 'Sales'] = df.loc[df.year >= 2013, 'Sales'] * 1000
df.replace(0, np.nan, inplace=True)

# changes from base year
to_plot = df.groupby('year').mean()
for col in to_plot.columns:
    to_plot[col] = to_plot[col] / to_plot[col].iloc[0] * 100
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(to_plot.index, to_plot['Research & Development expenses'], label='R&D expenses')
ax.plot(to_plot.index, to_plot['Added value'], label='Added value')
ax.plot(to_plot.index, to_plot['Sales'], label='Sales')
ax.plot(to_plot.index, to_plot['Number of employees'], label='Number of employees')
ax.set_xlabel('Year')
ax.set_ylabel('Mean value')
ax.legend()
plt.show()

# make subplots for each variable
to_plot = df.groupby('year').agg(lambda x: x.median(skipna=True))
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()
for i, col in enumerate(['Research & Development expenses', 'Added value', 'Sales', 'Number of employees']):
    axes[i].plot(to_plot.index, to_plot[col])
    axes[i].set_title(col)
    axes[i].set_xlabel('Year')
    axes[i].set_ylabel('Median value')
plt.tight_layout()
plt.show()


orbis = pd.read_csv('C:/Users/Jakob/Downloads/BvDids-orbis-world-min-5-empl-has-url-turnover.csv')

orbis.bvdid.str[:2].value_counts().head(40)
