import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
from firm_name_matching import firm_name_clean

output_path = '/Users/Jakob/Documents/financial_news_data/output/lexis_preds'

important_labels = ['StrategicAlliance', 'JointVenture', 'Marketing', 'Manufacturing',
                    'ResearchandDevelopment', 'Licensing']

# read lexis nexis articles with detected orgs and relations
df = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/lexisnexis_preds_robust_vortex_99.pkl')

df.drop(columns=['index_x', 'index_y'], inplace=True)


df = df[['publication', 'publication_date', 'firms', 'rels_pred', 'country', 'industry']]

df['cleaned_firms'] = df.firms.apply(lambda firms: [firm_name_clean(firm) for firm in firms])


## Exploratory plots

# Time distribution of deals
plt.figure(figsize=(10,7))
df.groupby(df.publication_date.dt.to_period('Y')).size().plot(kind="bar")
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
let

# most important sources
df['publication'] = df.publication.str.split('\'')
df['publication'] = df.publication.apply(lambda x: x[1] if len(x) > 1 else x[0])
to_plot = df.publication.value_counts()
to_plot.head(50).iloc[::-1].plot(kind='barh')
plt.title('Most frequent sources')
plt.yticks(rotation=45, ha='right', size=6)
plt.tight_layout()
plt.show()

## match with orbis
orbis = pd.read_pickle('C:/Users/Jakob/Documents/Orbis/orbis_michael_lexis_2.pkl')

lexis_firm_names_clean = df.cleaned_firms.explode().value_counts()\
    .index.to_frame(name='cleaned_name').reset_index(drop=True)

to_replace = {'amex': 'american express', 'gsk': 'glaxosmithkline', 'mhi': 'mitsubishi heavy industries',
              'ge'  : 'general electric', 'vw': 'volkswagen', 'ibm': 'international business machines',
              'l&t' : 'larsen & toubro', 'arw': 'arrow electronics', 'BMW': 'bayerische motoren werke',
              '(u.k.)': '', '(south Africa)': '', '(pty)': '', 'Merck S/a': 'Merck & Co., Inc.',
              'Sanofi-aventis (suisse) Sa': 'Sanofi'}

lexis_firm_names_clean['cleaned_name'] = lexis_firm_names_clean.cleaned_name.replace(to_replace)

names_ids = lexis_firm_names_clean.merge(orbis[['cleaned_name', 'BvD ID number']],
                                         on='cleaned_name', how='left')

names_ids['BvD ID number'].to_csv('/Users/Jakob/Documents/financial_news_data/lexis_alliances_orbis_ids.csv',
                                  index=False)

## add years before 2012 from KOF Orbis data
orbis_kof = pd.read_csv('C:/Users/Jakob/Documents/Orbis/orbis_kof_merged_michael.csv')
orbis_kof['Closing date'] = orbis_kof['Closing date'].apply(pd.to_datetime)

orbis_kof['Closing date'].dt.month.plot(kind='hist')
plt.show()

orbis_kof['year'] = orbis_kof['Closing date'].dt.year
orbis_kof.drop(columns='Closing date', inplace=True)

orbis_kof_2012 = orbis_kof[orbis_kof['year'] == 2012]
orbis_kof = orbis_kof[orbis_kof['year'] < 2012]

orbis_kof.year.plot(kind='hist', bins=25)
plt.show()

orbis_kof.year.value_counts()

orbis_kof = orbis_kof[orbis_kof['year'] > 1989]

# sort by num of missing values, drop duplicates with more missing values
orbis_kof = orbis_kof.loc[orbis_kof.isnull().sum(1).sort_values(ascending=True).index]
orbis_kof.drop_duplicates(['BvD ID number', 'year'], inplace=True, keep='first')

orbis.sort_values(by='count_na', inplace=True)
orbis.drop_duplicates('BvD ID number', keep='first', inplace=True)

orbis_long = pd.wide_to_long(orbis,
                stubnames=['Added value\nth USD ', 'Number of employees\n',
                                  'Research & Development expenses\nth USD ', 'Sales\nth USD '],
                i='BvD ID number',
                j='year')
orbis_long.reset_index(inplace=True)
orbis_long.columns = orbis_long.columns.map(lambda x: x.split('\n')[0])

orbis_cat = pd.concat([orbis_long, orbis_kof])

orbis_cat.set_index(['BvD ID number', 'year'], inplace=True)

orbis_long['BvD ID number'].isin(orbis_kof['BvD ID number']).sum()/len(orbis_long)


dynamic_cols = ['Added value', 'Number of employees', 'Research & Development expenses', 'Sales']

orbis_cat.sort_index(inplace=True)

# add sales data 2012 from kof separately
orbis_kof_2012.set_index(['BvD ID number', 'year'], inplace=True)
orbis_kof_2012.sort_index(inplace=True)
orbis_kof_2012 = orbis_kof_2012[['Sales']]
orbis_kof_2012.dropna(inplace=True)
orbis_kof_2012 = orbis_kof_2012[~orbis_kof_2012.index.duplicated(keep='first')]
orbis_cat = orbis_cat.fillna(orbis_kof_2012)

orbis_cat[dynamic_cols].to_csv('C:/Users/Jakob/Documents/Orbis/lexis_alliances_orbis_dynamic.csv')

static_cols = ['Company name Latin alphabet', 'BvD ID number', 'Country ISO code', 'City\nLatin Alphabet',
       'NACE Rev. 2, core code (4 digits)', 'NACE Rev. 2, secondary code(s)']

orbis.sort_values('BvD ID number', inplace=True)
orbis[static_cols].to_csv('C:/Users/Jakob/Documents/Orbis/lexis_alliances_orbis_static.csv', index=False)


# look at unmatched
# unmatched = names_ids[names_ids['BvD ID number'].isnull()].copy()
#
# orbis2 = pd.read_pickle('C:/Users/Jakob/Documents/Orbis/combined_firm_list.pkl')
# unmatched = unmatched[['cleaned_name']].merge(orbis2[['company', 'bvdidnumber']],
#                                          left_on='cleaned_name', right_on='company', how='left')
# unmatched.drop(columns=['cleaned_name'], inplace=True)
# unmatched.bvdidnumber.explode().to_csv(
#         'C:/Users/Jakob/Documents/Orbis/bvdids_michael_unmatched_3.csv', index=False)

# ## google lookup on residual
# from google_match import google_KG_match
# from tqdm import tqdm
# tqdm.pandas()
#
# api_key = open('google_api_key.txt', 'r').read()
#
# unmatched['google_res'] = unmatched.company.apply(str).progress_apply(
#     lambda x: google_KG_match(x, api_key, type='Corporation'))
#
# res_df = []
# for index, row in unmatched.iterrows():
#     google_res = row.google_res
#     if google_res != None:
#         if 'Corporation' in google_res['result']['@type']:  # google_res['resultScore'] > 100 and
#             res = pd.json_normalize(google_res)
#             row = pd.DataFrame(row).T
#             res.index = row.index
#
#             res_df.append(res)


names_ids = names_ids.dropna().set_index('cleaned_name').squeeze().to_dict()



## compare to SDC
sdc = pd.read_pickle('C:/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')

sdc['DealNumber'] = sdc.DealNumber.astype(str)
sdc['DealNumber'] = sdc.DealNumber.apply(lambda x: x.split('.')[0])

sdc.drop_duplicates(subset=['DealNumber'], inplace=True)
sdc.drop_duplicates(subset=['DealText'], inplace=True)


sdc.set_index('DealNumber', inplace=True, drop=True)

# there are 12 observations that are not tagged as SA or JV
len(sdc) - sdc[['StrategicAlliance', 'JointVentureFlag']].any(axis=1).sum()
# remove them
sdc = sdc[sdc[['StrategicAlliance', 'JointVentureFlag']].any(axis=1)]

# combine licensing, exclusive licensing, crosslicensing
licensing_cols = ['LicensingAgreementFlag', 'ExclusiveLicensingAgreementFlag', 'CrossLicensingAgreement',
                  'RoyaltiesFlag']
sdc['Licensing'] = sdc[licensing_cols].any(axis=1)
sdc.drop(columns=licensing_cols, inplace=True)

# A lot of agreements are pending (announced), some are announced to be terminated
sdc.Status.value_counts()

# one hot encode pending and terminated
sdc['Pending'] = (sdc.Status == 'Pending').astype(int)
sdc['Terminated'] = (sdc.Status == 'Terminated').astype(int)

# make column names easier to work with
sdc.columns = sdc.columns.str.replace('Flag', '')
sdc.columns = sdc.columns.str.replace('Agreement', '')
sdc.rename(columns={'DealNumber'                               : 'ID', 'AllianceDateAnnounced': 'Date',
                    'DealText'                                 : 'Text',
                    'ParticipantsinVenture/Alliance(ShortName)': 'Participants'}, inplace=True)

labels = ['JointVenture', 'Marketing', 'Manufacturing', 'ResearchandDevelopment', 'Licensing', 'Supply',
          'Exploration', 'TechnologyTransfer', 'Pending', 'Terminated']

sdc = sdc[['Date', 'Text', 'Participants'] + labels]

# take examples with at least 2 participants
sdc = sdc[sdc.Participants.str.len() > 1]

# convert to bool
sdc[labels] = sdc[labels].apply(lambda x: pd.to_numeric(x, downcast='integer')).astype(bool)

sdc['StrategicAlliance'] = ~sdc.JointVenture  # every deal that is not a JV is a SA

labels = ['StrategicAlliance'] + labels

# get string names of labels
for label_name in labels:
    sdc[label_name] = sdc[label_name].apply(lambda x: [label_name] if x == 1 else [])
sdc['rels'] = sdc[labels].sum(axis=1)
sdc.drop(columns=labels, inplace=True)
sdc.columns = ['publication_date', 'text', 'firms', 'rels']

# make a row for each two-way combination between participants
sdc['firms'] = sdc.firms.apply(lambda firms: list(itertools.combinations(firms, 2)))
sdc = sdc.explode('firms')
sdc = sdc[sdc.firms.apply(set).str.len() > 1]

sdc['cleaned_firms'] = sdc.firms.apply(lambda firms: [firm_name_clean(firm) for firm in firms])

# convert to sets
sdc['cleaned_firms'] = sdc.cleaned_firms.apply(frozenset)
df['cleaned_firms'] = df.cleaned_firms.apply(frozenset)

sdc = sdc[sdc.cleaned_firms.str.len() > 1]

sdc['year'] = sdc.publication_date.dt.year
sdc.drop_duplicates(['cleaned_firms', 'year'])

sdc[['cleaned_firms', 'rels']].groupby('cleaned_firms').agg(sum)
max_distance = pd.Timedelta('2Y')
sdc["group_diff"] = df.sort_values(['cleaned_firms', 'publication_date'])\
                     .groupby("cleaned_firms")["publication_date"]\
                     .diff()\
                     .gt(max_distance)\
                     .cumsum()

# merge detected relationships on firm name pairs
sdc_merge = sdc.merge(df, on=['cleaned_firms'], how='left')
sdc_merge = sdc.merge(df, on=['cleaned_firms'], how='inner')
# take only relationships detected within 1 year of each other
sdc_merge = sdc_merge[abs(sdc_merge.publication_date_x-sdc_merge.publication_date_y) < pd.Timedelta('2Y')]
sdc_merge = sdc_merge[['cleaned_firms', 'rels', 'rels_pred']]
sdc_merge = sdc_merge.groupby('cleaned_firms').agg(sum)
sdc_merge['rels'] = sdc_merge.rels.apply(frozenset)
sdc_merge['rels_pred'] = sdc_merge.rels_pred.apply(frozenset)

from sklearn.metrics import recall_score, precision_score, f1_score

from sklearn.dummy import DummyClassifier
# precision, reclall, F1 for each relation type

def get_metrics(df: pd.DataFrame, labels: list):
    for rel_name in labels:
        true = df.rels.apply(lambda rels: 1 if rel_name in rels else 0)
        pred = df.rels_pred.apply(lambda rels: 1 if rel_name in rels else 0)
        recall = recall_score(true, pred)
        precision = precision_score(true, pred)
        f1 = f1_score(true, pred)
        print(f'{rel_name}: Recall - {recall:.2f}, Precision - {precision:.2f}, F1 - {f1:.2f}')
        dummy_clf = DummyClassifier(strategy="stratified")
        dummy_clf.fit(pred, true)
        dummy_preds = dummy_clf.predict(pred)
        dummy_recall = recall_score(true, dummy_preds)
        dummy_precision = precision_score(true, dummy_preds)
        dummy_f1 = f1_score(true, dummy_preds)
        print(f'{rel_name}: Dummy recall - {dummy_recall:.2f}, Dummy precision - {dummy_precision:.2f}, Dummy F1 - {dummy_f1:.2f}')


get_metrics(sdc_merge, labels)


## compare to CATI
cati = pd.read_pickle("C:/Users/Jakob/Documents/CATI/cati_clean.pkl")
cati.rename(columns={'Date': 'publication_date', 'rel_type': 'rels'}, inplace=True)
cati['cleaned_firms'] = cati.Participants.apply(lambda firms: [firm_name_clean(firm) for firm in firms]).apply(frozenset)
cati.drop(columns=['Participants'], inplace=True)
cati['publication_date'] = cati.publication_date.apply(lambda x: pd.to_datetime(x, format='%Y'))

# merge detected relationships on firm name pairs
cati_merge = cati.merge(df, on=['cleaned_firms'], how='left')
# take only relationships detected within 1 year of each other
cati_merge = cati_merge[abs(cati_merge.publication_date_x-cati_merge.publication_date_y) < pd.Timedelta('2Y')]
cati_merge = cati_merge[['cleaned_firms', 'rels', 'rels_pred']]
cati_merge = cati_merge.groupby('cleaned_firms').agg(sum)
cati_merge['rels'] = cati_merge.rels.apply(frozenset)
cati_merge['rels_pred'] = cati_merge.rels_pred.apply(frozenset)

get_metrics(cati_merge, labels)




## create edge list for Michael
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

# remove duplicate relationships (same participants, same type, same year)
rels['year'] = rels.publication_date.dt.year
rels = rels.groupby(['firm_a', 'firm_b', 'year']).agg(list)
rels.reset_index(inplace=True)

# from itertools import chain
# rels['rels_pred'] = rels.rels_pred.apply(chain.from_iterable).apply(list).apply(set).apply(list)

rels = rels.explode('rels_pred')

orbis.to_csv(os.path.join(output_path, 'rel_database', 'lexis_orbis_match.csv'), index=False)

# save separate csvs for each relation type
for rel_name in important_labels:
    rels[rels.rels_pred.apply(lambda rel: rel==rel_name)].drop(columns=['rels_pred']).to_csv(
            os.path.join(output_path, 'rel_database', f'{rel_name}_LexisNexis.csv'), index=False)

rels[rels.rels_pred=='ResearchandDevelopment']

# orbis2 = pd.read_pickle('C:/Users/Jakob/Documents/Orbis/combined_firm_list.pkl')
# orbis_minus_michael = orbis2[~orbis2.company.isin(orbis.cleaned_name)]
# name_ids = name_ids[['cleaned_name']].merge(orbis_minus_michael[['company', 'bvdidnumber']],
#                                          left_on='cleaned_name', right_on='company', how='left')
# name_ids.dropna()
# orbis_minus_michael.columns


## read in orbis financials in chunks
from tqdm import tqdm
merge_chunks = []
orbis_reader = pd.read_csv('C:/Users/Jakob/Downloads/orbis_financials/Key_financials-USD.txt',
                               # nrows=100,
                               chunksize=100000,
                               iterator=True,
                               sep='\\t')

for chunk in tqdm(orbis_reader):
    merge_chunk = names_ids.merge(chunk, on='BvD ID number')
    merge_chunks.append(merge_chunk)

orbis_merge = pd.concat(merge_chunks)