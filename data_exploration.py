import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read Thomson SDC RND database
kb = pd.read_pickle('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')

# most frequent participants
kb['ParticipantsinVenture/Alliance(ShortName)'].explode().value_counts()[:30]

# most frequent countries
kb['ParticipantNation'].explode().value_counts()[:30]
kb['ParticipantUltimateParentNation'].explode().value_counts()[:30]
kb['ParticipantCity'].explode().value_counts()[:30]

# most frequent activities
kb['ActivityDescription'].explode().value_counts()[:30]

# most frequent participant industries
kb['ParticipantIndustry'].explode().value_counts()[:30]

# most frequent sources
kb['Source'].explode().value_counts()[:30]

# Time distribution of deals
plt.figure(figsize=(18,10))
kb.groupby(kb['AllianceDateAnnounced'].dt.to_period('Y')).DealNumber.count().plot(kind="bar")
plt.xticks(rotation=45)
plt.title('Number of recorded deals per month')
# plt.locator_params(nbins=30)
plt.show()

# distribution of deal text lengths
plt.figure(figsize=(10,10))
kb.DealText.str.len().hist(bins=30)
plt.title('Length of deal texts')
plt.xlabel('Number of characters')
plt.ylabel('Frequency')
plt.show()

# distribution of flags
cols = kb.columns
flag_columns = [15,19,20,23,24,27,29,30,31,32,35,38,39,45,48,50,51,52,55,56,57,58,59,60,65,67]
flag_columns = cols[flag_columns]

# flags that define a relation
relation_flags = [0,1,3,4,6,8,9,10,11,12,15,16,17,18,21,22]
relation_flags = flag_columns[relation_flags]

# flag frequencies
kb[relation_flags].sum().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=45, ha='right')
plt.show()

# flag correlation heatmap
corr = kb[relation_type_flags].corr()
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
corr = corr.round(decimals=2)
corr = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
    annot=True, ax=ax)
plt.xticks(rotation=45, ha='right')
plt.savefig('corr', dpi=150)
plt.show()

# read news articles
corpus = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/news.parquet.gzip')

# Time distribution of articles
plt.figure(figsize=(18,10))
corpus.groupby(pd.to_datetime(corpus.date).dt.to_period('M')).date.count().plot(kind="bar")
plt.xticks(rotation=45)
plt.title('Number of news articles per month')
# plt.locator_params(nbins=30)
plt.show()