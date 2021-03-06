import pandas as pd
import re
import spacy
import numpy as np

# read Thomson SDC RND database
sdc = pd.read_pickle('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')

# there are 12 observations that are not tagged as SA or JV
len(sdc) - sdc[['StrategicAlliance', 'JointVentureFlag']].any(axis=1).sum()
# remove them
sdc = sdc[sdc[['StrategicAlliance', 'JointVentureFlag']].any(axis=1)]

# combine licensing, exclusive licensing, crosslicensing
licensing_cols = ['LicensingAgreementFlag', 'ExclusiveLicensingAgreementFlag', 'CrossLicensingAgreement',
    'RoyaltiesFlag']
sdc['LicensingFlag'] = sdc[licensing_cols].any(axis=1)
sdc['LicensingFlag'] = sdc['LicensingFlag'].astype(float)
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

# Load spacy model
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

# test run
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

# inference sets
kb_inf = kb[pd.to_datetime(kb.Date).dt.to_period('Y').astype(str) == '2012']
news_inf = news[pd.to_datetime(news.Date).dt.to_period('Y').astype(str) == '2012']

kb_inf.to_pickle('/Users/Jakob/Documents/financial_news_data/kb_inf.pkl')
news_inf.to_pickle('/Users/Jakob/Documents/financial_news_data/news_inf.pkl')

# remove year 2012 from KB and Corpus
kb_fil = kb[pd.to_datetime(kb.Date).dt.to_period('Y').astype(str) != '2012']
news_fil = news[pd.to_datetime(news.Date).dt.to_period('Y').astype(str) != '2012']

# remove articles with keywords to get good negative examples
keywords = ['joint venture', 'strategic alliance', 'R&D', 'research and development',
    'manufacturing agreement', 'licensing agreement', 'marketing agreement', 'exploration agreement']

alliance_news = news[news.Text.str.contains('|'.join(keywords), flags=re.IGNORECASE)]
alliance_news.to_pickle('/Users/Jakob/Documents/financial_news_data/alliance_news.pkl')
neg = news_fil[~news_fil.Text.str.contains('|'.join(keywords), flags=re.IGNORECASE)]

# fit length (in sentences) distribution to kb dealtext length
lengths_dist = kb.Text.str.split('.').str.len().value_counts(normalize=True)
np.random.choice(lengths_dist.index, p=lengths_dist.values)

# reduce article length to match the examples in KB (in terms of sentences)
neg['Text'] =  neg.Text.apply(lambda x: ' '.join(re.split(r'(?<=[.:;])\s', x)[:np.random.choice(
    lengths_dist.index, p=lengths_dist.values)]))
neg = neg[neg.Text.str.split('.').str.len() < 50].copy()

# randomly sample documents as negative examples
random_neg = neg.sample(n=len(kb_fil[kb_fil.Deal == 1])-len(kb_fil[kb_fil.Deal == 0])).copy()

full = kb_fil.append(random_neg)
full.fillna(0, inplace=True)

full.to_pickle('/Users/Jakob/Documents/financial_news_data/full.pkl')
full = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/full.pkl')

# train-test split
test_size = 1/3
full.reset_index(inplace=True)
test = full.sample(frac=test_size, random_state=42)
train = full[~full.index.isin(test.index)]

# compare class distributions train-test
train[train.columns[4:-1]].mean()
test[test.columns[4:-1]].mean()

# reduce training size for testing purposes
# test = test.sample(n=100)
# train = train.sample(n=100)

# save as csv
test.to_csv('/Users/Jakob/Documents/financial_news_data/model/data/test.csv')
train.to_csv('/Users/Jakob/Documents/financial_news_data/model/data/train.csv')

## check performance of firm name recognition
kb = full[full.Deal == 1].copy()
kb.reset_index(inplace=True, drop=True)
kb['orgs_positions'] = kb.orgs.apply(lambda x: x[1]).copy()
kb['orgs'] = kb.orgs.apply(lambda x: x[0]).copy()

# get back participants column from SDC
df_participants = sdc[['DealNumber', 'ParticipantsinVenture/Alliance(ShortName)']]
df_participants.columns = ['ID', 'Participants']
# remove KB entries with less than two identified participants
df_participants = df_participants[df_participants.Participants.str.len() >= 2]
kb = kb.merge(df_participants, how='left', on=['ID'])
kb.dropna(subset=['Participants'], inplace=True)

for col in ['orgs', 'Participants']:
    # lowercase
    kb[col] = kb[col].apply(lambda list: [x.lower() for x in list])

    # strip special characters
    kb[col] = kb[col].apply(lambda list: [re.sub(r"[^\w\s]", "", x) for x in list])

    # remove 'co', 'AG', etc. (Schwenkler & Zheng, wikipedia.org/wiki/List_of_legal_entity_types_by_country)
    legal_ident = ['co', 'inc', 'ag', 'gmbh', 'ltd', 'lp', 'lpc', 'llc', 'pllc', 'llp', 'plc', 'ltd/plc',
        'corp', 'ab', 'cos', 'cia', 'sa', 'as', 'sas', 'corpus', 'reuter', 'reuters', 'based', 'rrb', 'rrb',
        'corporation', 'and']
    kb[col] = kb[col].apply(lambda list: [" ".join([word for word in x.split() if word not in legal_ident])
        for x in list])

    # strip whitespace from both sides
    kb[col] = kb.orgs.apply(lambda list: [x.strip() for x in list])

# determine share of recognized participants
kb['recognized_participants'] = kb.apply(lambda kb: len(set(kb.orgs) & set(kb.Participants)), axis=1)
kb['share_recognized_participants'] = kb.recognized_participants/kb.orgs.str.len() # get share
print(kb.share_recognized_participants.mean()) # take mean over all KB entries

# remove periods after legal identifiers in order to get better sentence splits
legal_ident = ['co', 'inc', 'ag', 'gmbh', 'ltd', 'lp', 'lpc', 'llc', 'pllc', 'llp', 'plc', 'ltd/plc', 'corp',
    'ab', 'cos', 'cia', 'sa', 'as', 'sas', 'corpus', 'reuter', 'reuters', 'based', 'rrb', 'rrb',
    'corporation', 'and']


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