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

len(corpus[corpus.words < 500])
# convert all plots to grayscale
plots_path = '/Users/Jakob/Documents/Github/financial_news'
all_plots = glob.glob(os.path.join(plots_path, "*.png"))
all_plots = [x for x in all_plots if not x.endswith('_gray.png')]
for plot in all_plots:
    gray_plot = Image.open(plot).convert("L")
    gray_plot.save(plot.replace('.png', '') + '_gray' + '.png')


# convert SDC to proper format
sdc['Deal'] = 1 # flag all SDC examples as positive
sdc.loc[sdc.Status == 'Terminated', 'Deal'] = 0 # except for the terminated ones

labels = ['Deal', 'Status', 'JointVentureFlag', 'MarketingAgreementFlag', 'ManufacturingAgreementFlag',
    'ResearchandDevelopmentAgreementFlag', 'LicensingFlag', 'SupplyAgreementFlag',
    'ExplorationAgreementFlag', 'TechnologyTransfer']

kb = sdc[['AllianceDateAnnounced', 'DealNumber', 'DealText'] + labels].copy()

# make column names easier to work with
kb.columns = kb.columns.str.replace('Flag', '')
kb.columns = kb.columns.str.replace('Agreement', '')
kb.rename(columns={'Status': 'Pending', 'DealNumber': 'ID', 'AllianceDateAnnounced': 'Date', 'DealText': \
    'Text'}, inplace=True)

# recode status
kb['Pending'].replace('Pending', 1, inplace=True)
kb['Pending'].replace('Completed/Signed', 0, inplace=True)
kb['Pending'].replace('Terminated', 0, inplace=True)

# get organizations
import spacy

# Load model
nlp = spacy.load('en_core_web_sm') # spacy model

def get_orgs(text):
    '''
    This function takes a text. Uses the Spacy model.
    The model will tokenize, POS-tag and recognize the entities named in the text.
    Returns a list of entities in the text that were recognized as organizations.
    '''
    # Apply the model
    tags = nlp(text)
    entities = [ent.text.replace('\'', '') for ent in tags.ents if ent.label_=='ORG'] # also remove apostrophes
    entitie_positions = [ent.start for ent in tags.ents if ent.label_=='ORG']
    # Return the list of entities recognized as organizations
    return entities, entitie_positions

get_orgs('Apple and Microsoft plan to form a joint venture for the development of cloud-based '
             'computing infrastrucutre.')

# save recognized organizations in new column
kb['orgs'] = kb.Text.apply(get_orgs)

kb.to_pickle('/Users/Jakob/Documents/financial_news_data/kb.pkl')
kb = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/kb.pkl')

# merge with negative examples from corpus
# corpus_sample = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/corpus_sample.parquet.gzip')
news = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/news.parquet.gzip')
news.rename(columns={'date': 'Date', 'link': 'ID', 'text': 'Text'}, inplace=True)

# this takes ~9 hours
news['orgs'] = news.Text.apply(get_orgs)

news.to_pickle('/Users/Jakob/Documents/financial_news_data/news_orgs.pkl')
news = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/news_orgs.pkl')

# remove year 2012 from KB and Corpus
kb_fil = kb[pd.to_datetime(kb.Date).dt.to_period('Y').astype(str) != '2012']
news_fil = news[pd.to_datetime(news.Date).dt.to_period('Y').astype(str) != '2012']

# remove articles with keywords to get good negative examples
keywords = ['joint venture', 'strategic alliance', 'R&D', 'research and development',
    'manufacturing agreement', 'licensing agreement', 'marketing agreement', 'exploration agreement']

import re
neg = news_fil[~news_fil.Text.str.contains('|'.join(keywords), flags=re.IGNORECASE)]

# fit length (in sentences) distribution to kb dealtext length
import numpy as np
lengths_dist = kb.Text.str.split('.').str.len().value_counts(normalize=True)
np.random.choice(lengths_dist.index, p=lengths_dist.values)

# reduce article length to match the examples in KB (in terms of sentences)
neg['Text'] =  neg.Text.apply(lambda x: ' '.join(re.split(r'(?<=[.:;])\s', x)[:np.random.choice(
    lengths_dist.index, p=lengths_dist.values)]))

neg = neg[neg.Text.str.split('.').str.len() < 50]

# randomly sample documents as negative examples
random_neg = neg.sample(n=len(kb)-len(kb[kb.Deal == 0])).copy()
random_neg.columns
kb.columns
full = kb_fil.append(neg)
full.fillna(0, inplace=True)

full.to_pickle('/Users/Jakob/Documents/financial_news_data/full.pkl')
full = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/full.pkl')

# train-test split
test_size = 0.5
test = full.sample(frac=test_size, random_state=42)
train = full[~full.Text.isin(test.Text)]

# reduce training size for testing purposes
# test = test.sample(n=100)
# train = train.sample(n=100)

# save as csv
test.to_csv('/Users/Jakob/Documents/financial_news_data/model/data/test.csv')
train.to_csv('/Users/Jakob/Documents/financial_news_data/model/data/train.csv')

# sdc = full[full.Deal == 1].copy()
# sdc.drop(columns='Deal', inplace=True)
# sdc.drop(columns=['Marketing', 'Manufacturing', 'Licensing', 'Supply', 'Exploration',
#     'TechnologyTransfer'],
#     inplace=True)
#
# multi-label stratified split
# from skmultilearn.model_selection import iterative_train_test_split
# X = sdc.Text
# y = sdc[['Pending', 'JointVenture', 'Marketing', 'Manufacturing', 'ResearchandDevelopment', 'Licensing',
#     'Supply', 'Exploration', 'TechnologyTransfer']]
# X_train, X_test, y_train, y_test = iterative_train_test_split(X,y, test_size=0.1)

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
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
sent = "Siemens AG and Atos SE formed a strategic alliance to jointly develop new IT products."
print(tokenizer.tokenize(sent))
tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))

sdc.DealText.sample().values
sdc.columns

sdc.colum

sdc[sdc.DealText.str.len()<100].DealText.sample().values
sdc[sdc.DealText.str.contains('Siemens')].DealText.sample().values

# Attention demonstration
from transformers import BertTokenizer, BertModel

model = BertModel.from_pretrained('bert-large-cased', output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=True)  # C

from bertviz import head_view

def show_head_view(model, tokenizer, sentence):  # B
    input_ids = tokenizer.encode(sentence, return_tensors='pt', add_special_tokens=True)  # C
    attention = model(input_ids)[-1]  # D
    tokens = tokenizer.convert_ids_to_tokens(list(input_ids[0]))
    head_view(attention, tokens)

show_head_view(model, tokenizer, sent)