import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# read Thomson SDC RND database
sdc = pd.read_pickle('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')

# most frequent participants
sdc['ParticipantsinVenture/Alliance(ShortName)'].explode().value_counts()[:30]

# most frequent countries
sdc['ParticipantNation'].explode().value_counts()[:30]
sdc['ParticipantUltimateParentNation'].explode().value_counts()[:30]
sdc['ParticipantCity'].explode().value_counts()[:30]

# most frequent activities
sdc['ActivityDescription'].explode().value_counts()[:30]

# most frequent participant industries
sdc['ParticipantIndustry'].explode().value_counts()[:30]

# most frequent sources
sdc['Source'].explode().value_counts()[:30]

# Number of participants per entry
sdc['ParticipantsinVenture/Alliance(ShortName)'].str.len().value_counts()

# most frequent status
sdc.Status.value_counts()

# example for pending status deal text
print(sdc[sdc.Status == 'Pending'].DealText.sample().values)

# some incorrect deal texts seem to belong to the application/purpose column. Example text 'To build a ...'
print(sdc[sdc.DealText.str.startswith('To ')].DealText.size)

# Time distribution of deals
plt.figure(figsize=(18,10))
sdc.groupby(sdc['AllianceDateAnnounced'].dt.to_period('Y')).DealNumber.count().plot(kind="bar")
plt.xticks(rotation=45)
plt.title('Number of recorded deals per month')
# plt.locator_params(nbins=30)
plt.show()

# distribution of flags
cols = sdc.columns
flag_columns = [15,19,20,23,24,27,29,30,31,32,35,38,39,45,48,50,51,52,55,56,57,58,59,60,65,67]
flag_columns = cols[flag_columns]

# flags that define a relation
relation_flags = [0,1,3,4,6,8,9,10,11,12,15,16,17,18,21,22]
relation_flags = flag_columns[relation_flags]

# flag frequencies
sdc[relation_flags].sum().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=45, ha='right')
plt.show()

# combine licensing, exclusive licensing, crosslicensing
licensing_cols = ['LicensingAgreementFlag', 'ExclusiveLicensingAgreementFlag', 'CrossLicensingAgreement']
sdc['LicensingFlag'] = sdc[licensing_cols].any(axis=1)
sdc['LicensingFlag'] = sdc['LicensingFlag'].astype(float)

# big flags
big_flags = ['JointVentureFlag','MarketingAgreementFlag','ManufacturingAgreementFlag',
    'ResearchandDevelopmentAgreementFlag','LicensingFlag','TechnologyTransfer','ExplorationAgreementFlag']

# show deal text examples for each category
print(sdc[sdc.Status == 'Pending'].DealText.sample().values)

# flag correlation heatmap
corr = sdc[big_flags].corr()
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
corr = corr.round(decimals=2)
# corr = corr[abs(corr) < 0.01]
corr = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
    annot=True, ax=ax)
plt.xticks(rotation=45, ha='right')
# plt.savefig('corr', dpi=150)
plt.show()
corr[abs(corr) < 0.01]

# distribution of deal text lengths
plt.figure(figsize=(10,10))
sdc.DealText.str.len().hist(bins=30)
plt.title('Length of deal texts')
plt.xlabel('Number of characters')
plt.ylabel('Frequency')
plt.show()

## FILTERING

# filter for only completed or pending status
# sdc = sdc[sdc.Status.isin(['Completed/Signed', 'Pending'])]

# filter out wrong deal texts
# sdc = sdc[~sdc.DealText.str.startswith('To ')]


# read news articles
corpus = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/news.parquet.gzip')

# Time distribution of articles
plt.figure(figsize=(18,10))
corpus.groupby(pd.to_datetime(corpus.date).dt.to_period('M')).date.count().plot(kind="bar")
plt.xticks(rotation=45)
plt.title('Number of news articles per month')
# plt.locator_params(nbins=30)
plt.show()