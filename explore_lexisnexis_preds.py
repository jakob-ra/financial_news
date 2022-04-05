import pandas as pd
import matplotlib.pyplot as plt

output_path = '/Users/Jakob/Documents/financial_news_data/output/lexis_preds'

important_labels = ['StrategicAlliance', 'JointVenture', 'Marketing', 'Manufacturing',
                    'ResearchandDevelopment', 'Licensing']

df = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/lexisnexis_preds_robust_vortex_99.pkl')

df.drop(columns=['index_x', 'index_y'], inplace=True)


df = df[['publication', 'publication_date', 'firms', 'rels_pred']]

df.publication.str.split(',').str.len().describe()

# Time distribution of deals
plt.figure(figsize=(10,7))
df.groupby(df.publication_date.dt.to_period('M')).size().plot(kind="bar")
plt.xticks(rotation=45, ha='right')
plt.title('Number of recorded deals per year')
plt.locator_params(nbins=30)
# plt.savefig(os.path.join(output_path, 'PartIndustries.pdf'))
plt.show()

# types of deals over time
to_plot = df.explode('rels_pred').groupby(df.publication_date.dt.to_period('Y')).rels_pred.value_counts()
to_plot = to_plot.to_frame(name='count')
to_plot.reset_index(level='rels_pred', inplace=True)
to_plot = to_plot.pivot(columns='rels_pred', values='count').fillna(0)
to_plot[important_labels].plot()
plt.xticks(rotation=45, ha='right')
plt.title('Number of deals per year')
plt.show()

# most important participants

