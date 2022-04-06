import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import pandas as pd

output_path = '/Users/Jakob/Documents/financial_news_data/output/lexis_preds'

important_labels = ['StrategicAlliance', 'JointVenture', 'Marketing', 'Manufacturing',
                    'ResearchandDevelopment', 'Licensing']

df = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/lexisnexis_preds_robust_vortex_99.pkl')

df.drop(columns=['index_x', 'index_y'], inplace=True)


df = df[['publication', 'publication_date', 'firms', 'rels_pred', 'country', 'industry']]

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
to_plot = df.firms.explode().value_counts()
to_plot.head(50).plot(kind='bar')
plt.xticks(rotation=45, ha='right', size=6)
plt.title('Most frequent deal participants')
plt.tight_layout()
plt.show()

# most important countries
to_plot = df.country.explode().value_counts()
to_plot.head(50).plot(kind='bar')
plt.xticks(rotation=45, ha='right', size=6)
plt.title('Most frequent countries')
plt.tight_layout()
plt.show()

# most important industries
to_plot = df.industry.explode().value_counts()
to_plot.head(50).plot(kind='bar')
plt.xticks(rotation=45, ha='right', size=6)
plt.title('Most frequent industries')
plt.tight_layout()
plt.show()

# most important sources
df['publication'] = df.publication.str.split('\'')
df['publication'] = df.publication.apply(lambda x: x[1] if len(x) > 1 else x[0])
to_plot = df.publication.value_counts()
to_plot.head(50).iloc[::-1].plot(kind='barh')
plt.title('Most frequent sources')
plt.yticks(rotation=45, ha='right', size=6)
plt.tight_layout()
plt.show()


# replace firm names with ORBIS ID (ID of LARGEST firm with that name in terms of employees)
orbis_path = '/Users/Jakob/Documents/Orbis/lexis_firms_matched_orbis'
all_orbis = glob.glob(os.path.join(orbis_path, "*.xlsx"))

orbis = pd.concat((pd.read_excel(f) for f in all_orbis), ignore_index=True)

orbis.drop(columns=['Unnamed: 0'], inplace=True)

orbis.dropna(subset=['Company name Latin alphabet', 'BvD ID number'], inplace=True)

# orbis['BvD ID number'] = orbis['BvD ID number'].ffill()
# orbis['Company name Latin alphabet'] = orbis['Company name Latin alphabet'].ffill()
#
# orbis.drop_duplicates(subset=['Company name Latin alphabet', 'BvD ID number'], inplace=True)

def isnull(val):
    nan_strings = ['', 'NaN', 'NA', 'na', 'n.a.', 'nan']
    return pd.isnull(val) or val in nan_strings
#
# def agg_list(vals):
#     res = [val for val in vals if not isnull(val)]
#     if len(res) == 1:
#         return res[0]
#     else:
#         return res

orbis = orbis.groupby(['Company name Latin alphabet', 'BvD ID number']).agg(list)
orbis.reset_index(inplace=True)

orbis.to_pickle('orbis_michael_preprocessed.pkl')
orbis = pd.read_pickle('orbis_michael_preprocessed.pkl')

def first_not_null_in_list(vals: list):
    for val in vals:
        if not isnull(val):
            return val
    return np.nan

for col in orbis.columns[2:]:
    orbis[col] = orbis[col].apply(first_not_null_in_list)

from firm_name_matching import firm_name_clean
orbis['cleaned_name'] = orbis['Company name Latin alphabet'].apply(firm_name_clean)

# sort by size (employees) first, then value added, then number of NA columns
orbis['count_na'] = orbis.isnull().sum(1)
orbis.sort_values(by=['Number of employees\n2020', 'Added value\nth USD 2020', 'count_na'],
                          inplace=True, ascending=False)
# now keep only the biggest firms / firms with most complete records among those with the same name
orbis.drop_duplicates(subset=['cleaned_name'], keep='first', inplace=True)


lexis_firm_names_clean = df.firms.explode().apply(firm_name_clean).value_counts()\
    .index.to_frame(name='cleaned_name').reset_index(drop=True)

names_ids = lexis_firm_names_clean.merge(orbis[['cleaned_name', 'BvD ID number']],
                                         on='cleaned_name', how='left')
names_ids = names_ids.dropna().set_index('cleaned_name').squeeze().to_dict()

df['cleaned_firms'] = df.firms.apply(lambda firms: [firm_name_clean(firm) for firm in firms])

rels = df[['publication_date', 'cleaned_firms', 'rels_pred']].copy()
rels['firm_a'] = rels.cleaned_firms.str[0]
rels['firm_b'] = rels.cleaned_firms.str[1]
rels.drop(columns=['cleaned_firms'], inplace=True)

rels['firm_a'] = rels.firm_a.map(names_ids)
rels['firm_b'] = rels.firm_b.map(names_ids)

rels.dropna(inplace=True)

# remove terminated
rels = rels[rels.rels_pred.apply(lambda rels: 'Terminated' not in rels)]

# remove firms where both participants are the same
rels = rels[rels.firm_a != rels.firm_b]

rels.rels_pred.explode().value_counts()



