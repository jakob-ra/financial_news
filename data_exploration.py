import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from PIL import Image
import glob
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.format': 'png'})

# matplotlib.rcParams.keys()

# read Thomson SDC RND database
sdc = pd.read_pickle('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')

# flag columns
cols = sdc.columns
flag_columns = [15,19,20,23,24,27,29,30,31,32,35,38,39,45,48,50,51,52,55,56,57,58,59,60,65,67]
flag_columns = cols[flag_columns]

# most frequent participants
sdc['ParticipantsinVenture/Alliance(ShortName)'].explode().value_counts()[:30]

# most frequent countries
sdc['ParticipantNation'].explode().value_counts()[:30]
sdc['ParticipantUltimateParentNation'].explode().value_counts()[:30]
sdc['ParticipantCity'].explode().value_counts()[:30]

# plot most frequent countries
plt.figure(figsize=(10,7))
sdc['ParticipantNation'].explode().value_counts()[:20].plot(kind="bar")
plt.xticks(rotation=45, ha='right')
plt.xlabel('Participant Nation')
plt.savefig('PartCountries.png')
plt.show()

# most frequent activities
sdc['ActivityDescription'].explode().value_counts()[:30]

# most frequent participant industries
sdc['ParticipantIndustry'].explode().value_counts()[:30]

# plot most frequent participant industries
plt.figure(figsize=(10,7))
sdc['ParticipantIndustry'].explode().value_counts()[:20].plot(kind="bar")
plt.xticks(rotation=45, ha='right')
plt.xlabel('Participant Industry')
plt.savefig('PartIndustries.png')
plt.show()

# most frequent sources
sdc['Source'].explode().value_counts()[:50]

sdc.Status.value_counts()

# Number of participants per entry
sdc['ParticipantsinVenture/Alliance(ShortName)'].str.len().value_counts()

# most frequent status
sdc.Status.value_counts()

# example for pending status deal text
print(sdc[sdc.Status == 'Pending'].DealText.sample().values)

# Time distribution of deals
plt.figure(figsize=(10,7))
sdc.groupby(sdc['AllianceDateAnnounced'].dt.to_period('Y')).DealNumber.count().plot(kind="bar")
plt.xticks(rotation=45, ha='right')
# plt.title('Number of recorded deals per year')
# plt.locator_params(nbins=30)
plt.xlabel('Deal Year')
plt.savefig('dealsovertime.png')
plt.show()


# flags that define a relation
relation_flags = [0,1,3,4,6,8,9,10,11,12,15,16,17,18,21,22,24,25]
relation_flags = flag_columns[relation_flags]

# flag frequencies
sdc[relation_flags].sum().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=45, ha='right')
plt.show()

# combine licensing, exclusive licensing, crosslicensing
licensing_cols = ['LicensingAgreementFlag', 'ExclusiveLicensingAgreementFlag', 'CrossLicensingAgreement',
    'RoyaltiesFlag']
sdc['LicensingFlag'] = sdc[licensing_cols].any(axis=1)
sdc['LicensingFlag'] = sdc['LicensingFlag'].astype(float)

# how many entries without any flags?
len(sdc) - sdc[relation_flags].any(axis=1).sum()

# how many entries without the big 4 flags?
len(sdc) - sdc[['StrategicAlliance', 'LicensingFlag', 'JointVentureFlag',
    'ResearchandDevelopmentAgreementFlag']].any(axis=1).sum()

# big flags
plot_cross_flags = ['StrategicAlliance','JointVentureFlag','MarketingAgreementFlag',
    'ManufacturingAgreementFlag',
    'ResearchandDevelopmentAgreementFlag','LicensingFlag','TechnologyTransfer','ExplorationAgreementFlag']

# show deal text examples for each category
print(sdc[sdc.StrategicAlliance == 1].DealText.sample().values)
print(sdc[sdc.JointVentureFlag == 1].DealText.sample().values)
print(sdc[sdc.MarketingAgreementFlag == 1].DealText.sample().values)
print(sdc[sdc.ManufacturingAgreementFlag == 1].DealText.sample().values)
print(sdc[sdc.LicensingFlag == 1].DealText.sample().values)
print(sdc[sdc.TechnologyTransfer == 1].DealText.sample().values)
print(sdc[sdc.ExplorationAgreementFlag == 1].DealText.sample().values)
print(sdc[sdc.RoyaltiesFlag == 1].DealText.sample().values)

# flag co-occurences
flag_cooc = sdc[plot_cross_flags].T.dot(sdc[plot_cross_flags])

# flag co-occurence heatmap
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# flag_cooc = flag_cooc.where(np.tril(np.ones(flag_cooc.shape)).astype(np.bool))
sns.heatmap(flag_cooc, xticklabels=flag_cooc.columns, yticklabels=flag_cooc.columns,
    annot=True, ax=ax, fmt='g')
plt.xticks(rotation=45, ha='right')
# plt.savefig('corr', dpi=150)
plt.show()
#
# # technology transfer and marketing orthogonal to other classifications, drop it
# big_flags = ['JointVentureFlag','ManufacturingAgreementFlag',
#     'ResearchandDevelopmentAgreementFlag','LicensingFlag','ExplorationAgreementFlag']
#
# # flag co-occurences
# flag_cooc = sdc[big_flags].T.dot(sdc[big_flags])
# flag_counts = np.diag(flag_cooc)
#
# # flag co-occurence heatmap
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# # flag_cooc = flag_cooc.where(np.tril(np.ones(flag_cooc.shape)).astype(np.bool))
# sns.heatmap((flag_cooc/flag_counts).round(decimals=2), xticklabels=flag_cooc.columns,
#     yticklabels=flag_cooc.columns,
#     annot=True, ax=ax, fmt='g')
# plt.xticks(rotation=45, ha='right')
# # plt.savefig('corr', dpi=150)
# plt.show()

# differentiate 4 flags: SA, JV, R&D, Licensing
fin_flags = ['JointVentureFlag', 'ResearchandDevelopmentAgreementFlag', 'LicensingFlag']

# drop overlap between flags
kb = sdc[sdc[fin_flags].sum(axis=1) == 1]

# distribution of deal text lengths
plt.figure(figsize=(10,10))
sdc.DealText.str.len().hist(bins=30)
plt.title('Length of deal texts')
plt.xlabel('Number of characters')
plt.ylabel('Frequency')
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

# convert all plots to grayscale
plots_path = '/Users/Jakob/Documents/Github/financial_news'
all_plots = glob.glob(os.path.join(plots_path, "*.png"))
all_plots = [x for x in all_plots if not x.endswith('_gray.png')]
for plot in all_plots:
    gray_plot = Image.open(plot).convert("L")
    gray_plot.save(plot.replace('.png', '') + '_gray' + '.png')