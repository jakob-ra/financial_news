import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from PIL import Image
import glob
import os

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('savefig', **{'dpi': 150, 'bbox': 'tight', 'format': 'png'})
# rc('text', usetex=True)

# matplotlib.rcParams.keys()

# read Thomson SDC RND database
sdc = pd.read_pickle('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')

# keyword extraction
# from rake_nltk import Rake
#
# r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
#
# r.extract_keywords_from_text(' '.join(sdc.DealText.values))
#
# r.get_ranked_phrases()[:20] # To get keyword phrases ranked highest to lowest.

#
# from sklearn.feature_extraction.text import CountVectorizer
#
# cv = CountVectorizer(max_df=0.9,min_df=2, stop_words='english')
# dtm  = cv.fit_transform(sdc['DealText'])
#
# from sklearn.decomposition import LatentDirichletAllocation
# lda = LatentDirichletAllocation(n_components=5,random_state=101)
#
# lda_fit  = lda.fit(dtm)
#
# # understanding each topics top 10 common words
# for id_value, value in enumerate(lda_fit.components_):
#    print(f"The topic would be {id_value}")
#   print([cv.get_feature_names()[index] for index in value.argsort()   [-10:]])
#    print("\n")

import re

def pre_process(text):
    # lowercase
    text = text.lower()

    # remove tags
    text = re.sub("", "", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    text = ' '.join([w for w in text.split() if len(w)>1]) # remove one letter words
    return text

sdc.DealText.sample().values

from sklearn.feature_extraction.text import TfidfVectorizer

sdc['cleaned_text'] = sdc.DealText.apply(lambda x: pre_process(x))

vectorizer = TfidfVectorizer(ngram_range = (1, 2), stop_words='english', max_df=0.2, max_features=50000)
vectorizer = TfidfVectorizer(ngram_range = (2, 3), max_features=50000)
df = corpus.text.sample(10000).apply(lambda x: pre_process(x))

for flag in plot_cross_flags:
    deal_texts = ' '.join(sdc.loc[sdc[flag] == 1, 'cleaned_text'].values)
    df = df.append(pd.Series(deal_texts))

vectorizer.fit(df)
features = vectorizer.transform(df.iloc[10000:])
scores = (features.toarray())
feature_names = vectorizer.get_feature_names()

data = []
for col, term in enumerate(feature_names):
    data.append((term, features[8,col] ))
ranking = pd.DataFrame(data, columns = ['term','rank'])
terms = (ranking.sort_values('rank', ascending = False))
for term in terms.term[:200]:
    print(term)

# plot deal text length (in words)
sdc.DealText.str.split(' ').str.len().hist()
plt.show()

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


# Number of participants per entry
sdc['ParticipantsinVenture/Alliance(ShortName)'].str.len().value_counts()

# most frequent status
sdc.Status.value_counts()
sdc[sdc.Status == 'Letter of Intent'].DealText.sample(10).values
sdc[sdc.Status == 'Terminated'].DealText.sample(10).values

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

# there are 12 observations that are not tagged as SA or JV
len(sdc) - sdc[['StrategicAlliance', 'JointVentureFlag']].any(axis=1).sum()

# remove them
sdc = sdc[sdc[['StrategicAlliance', 'JointVentureFlag']].any(axis=1)]

# how many entries without any flags?
len(sdc) - sdc[relation_flags].any(axis=1).sum()

# how many entries without the big 4 flags?
len(sdc) - sdc[['StrategicAlliance', 'LicensingFlag', 'JointVentureFlag',
    'ResearchandDevelopmentAgreementFlag']].any(axis=1).sum()

# show deal text examples for each category
print(sdc[sdc.StrategicAlliance == 1].DealText.sample().values)
print(sdc[sdc.JointVentureFlag == 1].DealText.sample().values)
print(sdc[sdc.MarketingAgreementFlag == 1].DealText.sample(10).values)
print(sdc[sdc.ManufacturingAgreementFlag == 1].DealText.sample(10).values)
print(sdc[sdc.LicensingFlag == 1].DealText.sample().values)
print(sdc[sdc.TechnologyTransfer == 1].DealText.sample().values)
print(sdc[sdc.ExplorationAgreementFlag == 1].DealText.sample().values)
print(sdc[sdc.RoyaltiesFlag == 1].DealText.sample().values)
print(sdc[sdc.SupplyAgreementFlag == 1].DealText.sample(10).values)

# big flags
plot_cross_flags = ['StrategicAlliance','JointVentureFlag','MarketingAgreementFlag',
    'ManufacturingAgreementFlag',
    'ResearchandDevelopmentAgreementFlag','LicensingFlag','TechnologyTransfer','ExplorationAgreementFlag']
plot_cross_flags = ['StrategicAlliance','JointVentureFlag','MarketingAgreementFlag',
    'ManufacturingAgreementFlag',
    'ResearchandDevelopmentAgreementFlag','LicensingFlag','TechnologyTransfer',
    'SupplyAgreementFlag','ExplorationAgreementFlag']

# flag co-occurences
flag_cooc = sdc[plot_cross_flags].T.dot(sdc[plot_cross_flags])

# flag co-occurence heatmap
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# flag_cooc = flag_cooc.where(np.tril(np.ones(flag_cooc.shape)).astype(np.bool))
sns.set_style({'font.family': 'dejavuserif'})
sns.heatmap(flag_cooc, xticklabels=flag_cooc.columns, yticklabels=flag_cooc.columns,
    annot=True, ax=ax, fmt='g', cmap='Greys')
# sns.set_palette('Greys')
plt.xticks(rotation=45, ha='right')
plt.savefig('co_occur.png')
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
news.groupby(pd.to_datetime(news.Date).dt.to_period('M')).Date.count().plot(kind="bar")
plt.xticks(rotation=45)
plt.title('Number of news articles per month')
# plt.locator_params(nbins=30)
plt.show()

# source of documents
corpus['source'] = corpus['link'].str.contains('reuters')
corpus.loc[corpus['source'] == True, 'source'] = 'Reuters'
corpus.loc[corpus['source'] == False, 'source'] = 'Bloomberg'
len(corpus[corpus.source == 'Bloomberg'])/len(corpus)

# plot article sources
plt.figure(figsize=(18,10))
corpus.groupby(corpus.source).date.count().plot(kind="bar")
plt.xticks(rotation=45)
plt.title('Article sources')
# plt.locator_params(nbins=30)
plt.show()

# article length (characters)
plt.figure(figsize=(18,10))
corpus[corpus.text.str.len() < 15000].text.str.len().hist(bins=500)
plt.xticks(rotation=45)
plt.title('Article lengths')
plt.locator_params(nbins=30)
plt.show()

# article lenght (words)
corpus['words'] = corpus['text'].str.count(' ') + 1

# plot lengths (words)
plt.figure(figsize=(18,10))
corpus[corpus.words < 500].words.hist(bins=500)
plt.xticks(rotation=45)
plt.title('Article lengths')
plt.locator_params(nbins=20)
plt.show()

# plot distribution of participants
part_counts = kb.Participants.str.len().value_counts(
    normalize=True).sort_index()
part_counts_larger_10 = part_counts.iloc[11:].sum()
part_counts = part_counts.iloc[:11]
part_counts = part_counts.append(pd.Series({'>10': part_counts_larger_10}))
part_counts.plot(kind='bar')
plt.xlabel('Number of participants')
plt.ylabel('Share of deals')
plt.savefig('PartDist.png')
plt.show()

# convert all plots to grayscale
plots_path = '/Users/Jakob/Documents/Github/financial_news'
all_plots = glob.glob(os.path.join(plots_path, "*.png"))
all_plots = [x for x in all_plots if not x.endswith('_gray.png')]
for plot in all_plots:
    gray_plot = Image.open(plot).convert("L")
    gray_plot.save(plot.replace('.png', '') + '_gray' + '.png')

# find most common bigrams
# from collections import Counter
#
# words = kb.Text.str.cat().split()
# bigrams = zip(words, words[1:])
# counts = Counter(bigrams)
# print(counts.most_common()[:100])

# find keywords (most specific words in kb)
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# vectorizer = TfidfVectorizer()
# transformed_documents = vectorizer.fit_transform([kb.Text.str.cat(), news.Text.str.cat()])
#
# feature_names = vectorizer.get_feature_names()
#
# def sort_coo(coo_matrix):
#     tuples = zip(coo_matrix.col, coo_matrix.data)
#     return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
#
#
# def extract_topn_from_vector(feature_names, sorted_items, topn=10):
#     """get the feature names and tf-idf score of top n items"""
#
#     # use only topn items from vector
#     sorted_items = sorted_items[:topn]
#
#     score_vals = []
#     feature_vals = []
#
#     # word index and corresponding tf-idf score
#     for idx, score in sorted_items:
#         # keep track of feature name and its corresponding score
#         score_vals.append(round(score, 3))
#         feature_vals.append(feature_names[idx])
#
#     # create a tuples of feature,score
#     # results = zip(feature_vals,score_vals)
#     results = {}
#     for idx in range(len(feature_vals)):
#         results[feature_vals[idx]] = score_vals[idx]
#
#     return results
#
# sorted_items = sort_coo(transformed_documents[0].tocoo())
#
# keywords = extract_topn_from_vector(feature_names,sorted_items,100)

# Tokenizer demonstration
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# sent = "Siemens AG and Atos SE formed a strategic alliance to jointly develop new IT products."
# print(tokenizer.tokenize(sent))
# tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))
#
# sdc.DealText.sample().values
# sdc.columns
#
# sdc.colum
#
# sdc[sdc.DealText.str.len()<100].DealText.sample().values
# sdc[sdc.DealText.str.contains('Siemens')].DealText.sample().values
#
# # Attention demonstration
# from transformers import BertTokenizer, BertModel
#
# model = BertModel.from_pretrained('bert-large-cased', output_attentions=True)
# tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=True)  # C
#
# from bertviz import head_view
#
# def show_head_view(model, tokenizer, sentence):  # B
#     input_ids = tokenizer.encode(sentence, return_tensors='pt', add_special_tokens=True)  # C
#     attention = model(input_ids)[-1]  # D
#     tokens = tokenizer.convert_ids_to_tokens(list(input_ids[0]))
#     head_view(attention, tokens)
#
# show_head_view(model, tokenizer, sent)