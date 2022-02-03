import pandas as pd
import re
import spacy
import numpy as np
from itertools import permutations
from tqdm import tqdm
tqdm.pandas()

np.random.seed(42) # set random seed

share_train, share_test, share_dev = .6, .2, .2

# read Thomson SDC RND database
sdc = pd.read_pickle('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')

sdc['DealNumber'] = sdc.DealNumber.astype(str)
sdc['DealNumber'] = sdc.DealNumber.apply(lambda x: x.split('.')[0])

sdc.drop_duplicates(subset=['DealText'], inplace=True)

sdc.set_index('DealNumber', inplace=True, drop=False)

# check different name columns
# sdc[['DealText', 'ParticipantsinVenture/Alliance(ShortName)', 'ParticipantsinVenture/Alliance(LongNamShortLine)',
#      'ParticipantsinVenture/Alliance(LongName3Lines)']].sample().values


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

kb = sdc[['AllianceDateAnnounced', 'DealNumber', 'DealText', 'ParticipantsinVenture/Alliance(ShortName)'] + labels].copy()
kb.rename(columns={'ParticipantsinVenture/Alliance(ShortName)': 'Participants'}, inplace=True)

# make column names easier to work with
kb.columns = kb.columns.str.replace('Flag', '')
kb.columns = kb.columns.str.replace('Agreement', '')
kb.rename(columns={'Status': 'Pending', 'DealNumber': 'ID', 'AllianceDateAnnounced': 'Date', 'DealText': \
    'Text'}, inplace=True)

# recode status
kb['Pending'].replace('Pending', 1, inplace=True)
kb['Pending'].replace('Completed/Signed', 0, inplace=True)
kb['Pending'].replace('Terminated', 0, inplace=True)

text = 'United Energy Ltd and Westcoast Energy Australia planned to form a joint venture.'

# take positive examples
kb = kb[(kb.Deal) & (kb.Pending == 0)]
# with at least 2 participants
kb = kb[kb.Participants.str.len() > 1]

flags = ['JointVenture', 'Marketing', 'Manufacturing', 'ResearchandDevelopment',
       'Licensing', 'Supply', 'Exploration', 'TechnologyTransfer']
## does it work for multiple relation extraction?
kb[flags] = kb[flags].astype('bool')
kb['StrategicAlliance'] = ~kb.JointVenture # every deal that is not a JV is a SA
# flags = ['StrategicAlliance', 'JointVenture', 'Marketing', 'Manufacturing', 'ResearchandDevelopment',
#        'Licensing', 'Supply', 'Exploration', 'TechnologyTransfer']
flags = ['StrategicAlliance', 'JointVenture', 'Marketing', 'Manufacturing', 'ResearchandDevelopment',
       'Licensing']

def match_entities(entities, text):
    nlp = spacy.blank('en')
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [{"label": "ORG", "pattern": entity} for entity in entities]
    ruler.add_patterns(patterns)
    doc = nlp(text)
    tokens = []
    for ent in doc.ents:
        tokens.append(
                {'text'       : ent.text, 'start': ent.start_char, 'end': ent.end_char,
                 'token_start': ent.start, 'token_end': ent.end, 'entityLabel': ent.label_})

    return tokens

kb['tokens'] = kb.progress_apply(lambda x: match_entities(x.Participants, x.Text), axis=1)

# keep the exact matches
kb = kb[kb.tokens.apply(lambda tokens: set([token['text'] for token in tokens])) == kb.Participants.apply(set)]
kb = kb[kb.tokens.apply(lambda x: set(token['text'] for token in x)).str.len()>1]


def match_relations(row):
    participants = row.tokens
    # take only first mentions of participants
    participants = pd.DataFrame(participants)
    participants = participants.drop_duplicates(subset=['text'], keep='first')
    participant_names = participants.text.to_list()
    participants.set_index('text', inplace=True)

    if len(participant_names) < 2:
        raise ValueError('Alliances with less than two participants in the provided records.')

    relations = []
    for flag in flags:  # cycle through flags
        if row[flag]:  # for true flags, append relation between each two-way permutation of participants
            for permut in permutations(participant_names):
                start_child = participants.loc[permut[0]].token_start
                start_head = participants.loc[permut[1]].token_start
                relations.append({"child": start_child, "head": start_head, "relationLabel": flag})

    return relations

kb['relations'] = kb.progress_apply(match_relations, axis=1)

kb['document'] = kb.Text

kb['meta'] = kb.ID.apply(lambda x: {'source': 'Thomson SDC alliances - Deal Number ' + str(x)})

# split into train, test, dev
kb = kb.sample(frac=1) # shuffle data set
train, dev, test = np.split(kb, [int(share_train*len(kb)), int((share_train+share_dev)*len(kb))])

def update_meta(meta: dict, new_key: str, value):
    meta[new_key] = value

    return meta

kb.loc[train.index, 'meta'] = kb.loc[train.index, 'meta'].apply(lambda x: update_meta(x, 'split', 'train'))
kb.loc[dev.index, 'meta'] = kb.loc[dev.index, 'meta'].apply(lambda x: update_meta(x, 'split', 'dev'))
kb.loc[test.index, 'meta'] = kb.loc[test.index, 'meta'].apply(lambda x: update_meta(x, 'split', 'test'))

kb[['document', 'tokens', 'relations', 'meta']].to_json('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict.json',
                                                orient='records', lines=True)

kb = pd.io.json.read_json(path_or_buf='/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict.json',
                          orient='records', lines=True)
def remove_relations(input, remove_type: list):
    output = []
    for relation in input:
        if relation['relationLabel'] not in remove_type:
            output.append(relation)

    return output

kb['relations'] = kb.relations.apply(
        lambda x: remove_relations(x, ['Supply', 'Exploration', 'TechnologyTransfer']))

kb[['document', 'tokens', 'relations', 'meta']].to_json('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict_6class.json',
                                                orient='records', lines=True)

kb = pd.io.json.read_json(path_or_buf='/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict_6class.json',
                          orient='records', lines=True)

# kb = kb[kb.meta.apply(lambda x: x['split'] == 'test')]
# kb[['document', 'tokens', 'relations', 'meta']].to_json('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict_6class_test.json',
#                                                 orient='records', lines=True)

def extract_relations(input):
    output = []
    for relation in input:
        output.append(relation['relationLabel'])

    return set(output)

kb['relation_types'] = kb.relations.apply(extract_relations)

# create balance by random undersampling+
rare_flags = {'Marketing', 'Manufacturing', 'Licensing', 'ResearchandDevelopment'}
kb = kb[kb.relation_types.apply(lambda x: len(x.intersection(rare_flags))) > 0] # focus on examples with rare flags
# kb = kb[kb.relation_types.apply(lambda x: len(x.intersection({'StrategicAlliance', 'Manufacturing'}))) < 2] # remove

# drop some marketing & SA examples
kb = kb.drop(kb[kb.relation_types == {'StrategicAlliance', 'Marketing'}].sample(3000).index)
kb = kb.drop(kb[kb.relation_types == {'StrategicAlliance', 'Manufacturing'}].sample(1000).index)
kb = kb.drop(kb[kb.relation_types == {'StrategicAlliance', 'Licensing'}].sample(1000).index)

kb.relation_types.explode().value_counts()
kb.relations.sample().values
kb.relation_types.value_counts()

## add neg examples from news
news = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/news_orgs.pkl')

# fit length (in sentences) distribution to kb dealtext length
lengths_dist = kb.document.str.split('.').str.len().value_counts(normalize=True)

# reduce article length to match the examples in KB (in terms of sentences)
news['Text'] =  news.Text.apply(lambda x: ' '.join(re.split(r'(?<=[.:;])\s', x)[:np.random.choice(
    lengths_dist.index, p=lengths_dist.values)]))
news = news[news.Text.str.split('.').str.len() < 50].copy()

news = news.sample(len(kb))

# remove articles with keywords to get good negative examples
keywords = ['joint venture', 'strategic alliance', 'R&D', 'research and development',
    'manufacturing agreement', 'licensing agreement', 'marketing agreement', 'exploration agreement',
    'alliance venture', 'form joint', 'Formed joint', 'Signed agreement', 'Planned to form',
    'Agreement disclosed', 'venture agreement', 'entered strategic', 'marketing rights', 'Marketing services',
    'Agreed to manufacture', 'development services', 'Research and development', 'Alliance to develop',
    'granted license', 'granted licensing rights', 'exclusive rights', 'License to manufacture',
    'distribution agreement', 'alliance distribution', 'exploration services', 'marketing services',
    'alliance to manufacture', 'alliance to wholesale', 'Agreement to manufacture and market',
    'Agreement to jointly develop', 'Venture to jointly develop']

news = news[~news.Text.str.contains('|'.join(keywords), flags=re.IGNORECASE)]

news['orgs'] = news.orgs.apply(lambda x: x[0])
news = news[news.orgs.str.len() > 1] # take only docs with at least two orgs
news['tokens'] = news.progress_apply(lambda x: match_entities(x.orgs, x.Text), axis=1)


news['meta'] = news.apply(lambda x: {'source': f'Reuters News Dataset - Article ID {str(x.ID)} - {str(x.Date)}'}, axis=1)
news.drop(columns=['Date', 'ID', 'orgs'], inplace=True)
news.rename(columns={'Text': 'document'}, inplace=True)
news['relations'] = news.meta.apply(lambda x: [])

news.to_pickle('/Users/Jakob/Documents/financial_news_data/news_literal_orgs.pkl')

kb = kb.append(news)

# split into train, test, dev
kb.reset_index(drop=True)
kb = kb.sample(frac=1) # shuffle data set
train, dev, test = np.split(kb, [int(share_train*len(kb)), int((share_train+share_dev)*len(kb))])

kb.loc[train.index, 'meta'] = kb.loc[train.index, 'meta'].apply(lambda x: update_meta(x, 'split', 'train'))
kb.loc[dev.index, 'meta'] = kb.loc[dev.index, 'meta'].apply(lambda x: update_meta(x, 'split', 'dev'))
kb.loc[test.index, 'meta'] = kb.loc[test.index, 'meta'].apply(lambda x: update_meta(x, 'split', 'test'))


kb[['document', 'tokens', 'relations', 'meta']].to_json('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict_6class_balanced_negative.json',
                                                orient='records', lines=True)



# {"text": "The up-regulation of IL-1beta message could be mediated by the latent membrane protein-1, EBV nuclear proteins 2, 3, 4, and 6 genes.",
# "spans": [{"text": "IL-1beta", "start": 21, "token_start": 5, "token_end": 5, "end": 29, "type": "span", "label": "GGP"}],
# "meta": {"source": "BioNLP 2011 Genia Shared Task, PMID-9878621.txt"},
# "_input_hash": -400334998,
# "_task_hash": 1861007191,
# "tokens": [{"text": "The", "start": 0, "end": 3, "id": 0, "ws": true, "disabled": true}],
# "_session_id": null,
# "_view_id": "relations",
# "relations": [{"head": 14, "child": 5, "head_span": {"start": 63, "end": 88, "token_start": 12, "token_end": 14, "label": "GGP"}, "child_span": {"start": 21, "end": 29, "token_start": 5, "token_end": 5, "label": "GGP"}, "color": "#ffd882", "label": "Pos-Reg"}],
# "answer": "accept"}

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

# remove KB entries with less than two identified participants
kb = kb[kb.Participants.str.len() >= 2]
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
    kb[col] = kb[col].apply(lambda list: [x.strip() for x in list])

kb.orgs.sample().values

# determine share of recognized participants
kb['recognized_participants'] = kb.Participants.apply(lambda x: len(set(x)))
# kb['recognized_participants'] = kb.apply(lambda kb: len(set(kb.orgs)) == set(kb.Participants)), axis=1)
# kb['share_recognized_participants'] = kb.recognized_participants/kb.orgs.str.len() # get share
print(kb.share_recognized_participants.mean()) # take mean over all KB entries

kb['recognized_participants'] = kb.orgs.apply(set) == kb.Participants.apply(set)
kb.recognized_participants.mean()


# remove periods after legal identifiers in order to get better sentence splits
# legal_ident = ['co', 'inc', 'ag', 'gmbh', 'ltd', 'lp', 'lpc', 'llc', 'pllc', 'llp', 'plc', 'ltd/plc', 'corp',
#     'ab', 'cos', 'cia', 'sa', 'as', 'sas', 'corpus', 'reuter', 'reuters', 'based', 'rrb', 'rrb',
#     'corporation', 'and']


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



# df = pd.DataFrame({'Text': ['Prenetics Ltd and Caterair decided to form a joint venture. Prenetics Ltd will hold.'],
#                    'Participants': [['Prenetics Ltd', 'Caterair']]})

#
#
# test_df = kb.sample()
# participants = test_df.tokens.iloc[0]
# # take only first mentions of participants
# participants = pd.DataFrame(participants)
# participants = participants.drop_duplicates(subset=['text'], keep='first')
#
# participant_names = participants.text.to_list()
#
# relations = []
# for flag in flags: # cycle through flags
#     if test_df[flag].iloc[0]: # for true flags, append relation between each two-way permutation of participants
#         for permut in permutations(participant_names):
#             start_child = participants[participants.text == permut[0]].token_start.iloc[0]
#             start_head = participants[participants.text == permut[1]].token_start.iloc[0]
#             relations.append({"child": start_child, "head": start_head, "relationLabel": flag})