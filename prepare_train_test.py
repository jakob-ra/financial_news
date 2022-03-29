import pandas as pd
import re
import spacy
import numpy as np
from itertools import permutations
from tqdm import tqdm
from firm_name_matching import firm_name_clean
tqdm.pandas()
import matplotlib.pyplot as plt

np.random.seed(42) # set random seed

# read Thomson SDC RND database
sdc = pd.read_pickle('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')

sdc['DealNumber'] = sdc.DealNumber.astype(str)
sdc['DealNumber'] = sdc.DealNumber.apply(lambda x: x.split('.')[0])

sdc.drop_duplicates(subset=['DealText'], inplace=True)

sdc.set_index('DealNumber', inplace=True, drop=False)

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
sdc.rename(columns={'DealNumber': 'ID', 'AllianceDateAnnounced': 'Date', 'DealText': 'Text',
                    'ParticipantsinVenture/Alliance(ShortName)': 'Participants'}, inplace=True)

labels = ['JointVenture', 'Marketing', 'Manufacturing',
    'ResearchandDevelopment', 'Licensing', 'Supply',
    'Exploration', 'TechnologyTransfer', 'Pending', 'Terminated']

kb = sdc[['Date', 'ID', 'Text', 'Participants'] + labels].copy()

# take examples with at least 2 participants
kb = kb[kb.Participants.str.len() > 1]

# convert to bool
kb[labels] = kb[labels].apply(lambda x: pd.to_numeric(x, downcast='integer')).astype(bool)

kb['StrategicAlliance'] = ~kb.JointVenture # every deal that is not a JV is a SA

labels = ['StrategicAlliance'] + labels

def extract_firm_name_and_spans(text, names, clean_names=True):
    if clean_names:
        cleaned_names = [firm_name_clean(name) for name in names]
        names += cleaned_names
    pattern = r'|'.join(re.escape(word.strip()) for word in names)

    res = re.finditer(pattern, text, flags=re.IGNORECASE)

    return [(match.group(), match.span()) for match in res]

kb['ents'] = kb.progress_apply(lambda x: extract_firm_name_and_spans(x.Text, x.Participants), axis=1)

kb = kb[kb.ents.apply(len) > 1] # need at least two identified participants


kb.drop(columns=['Participants'], inplace=True)

# get string names of labels
for label_name in labels:
    kb[label_name] = kb[label_name].apply(lambda x: [label_name] if x == 1 else [])
kb['rels'] = kb[labels].sum(axis=1)
kb.drop(columns=labels, inplace=True)

# add source tag
kb['source'] = kb.ID.apply(lambda x: 'Thomson SDC alliances - Deal Number ' + str(x))

kb.rename(columns={'Text': 'document'}, inplace=True)
kb.drop(columns=['ID'], inplace=True)

# kb = pd.io.json.read_json(path_or_buf='/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict.json',
#                           orient='records', lines=True)

# kb = pd.io.json.read_json(path_or_buf='/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict_6class.json',
#                           orient='records', lines=True)

kb.to_pickle('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_kb_training.pkl')


## add neg examples from news
news = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/news_orgs.pkl')

news['orgs'] = news.orgs.apply(lambda x: x[0]) # take only org names, not their token position

# fit length (in sentences) distribution to kb dealtext length
from nltk.tokenize import sent_tokenize
lengths_dist_kb = kb.document.apply(sent_tokenize).str.len().value_counts(normalize=True)

# sentence splitting
news['sents'] = news.Text.swifter.apply(sent_tokenize)

# remove the first sentence (headline) for training purposes
news['sents'] = news.sents.apply(lambda x: x[1:])

# reduce article length to match the examples in KB (in terms of sentences)
news['sents'] = news.sents.apply(lambda x: x[:np.random.choice(lengths_dist_kb.index, p=lengths_dist_kb.values)])

news['Text'] =  news.sents.apply(lambda x: ' '.join(x))

lengths_dist_news = news.sents.apply(len).value_counts(normalize=True)

# compare
lengths_dist_kb.to_frame(name='KB').merge(
        lengths_dist_news.to_frame(name='News'),
        left_index=True, right_index=True).plot.bar(
        xlabel='Number of sentences', ylabel='Share of documents')
plt.show()

# look at text length distribution in terms of characters
news.Text.apply(len).hist(bins=100)
plt.show()

news[news.Text.apply(len) < 2500].Text.apply(len).hist(bins=100)
plt.show()

# remove very long articles
news = news[news.Text.apply(len) < 1500].copy()

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

extract_firm_name_and_spans('Volkswagen and Tesco PLC announced blah.', ['Volkswagen AG', 'Tesco PLC'], clean_names=True)

news['ents'] = news.progress_apply(lambda row: extract_firm_name_and_spans(row.Text, row.orgs[:10], clean_names=False), axis=1)


## take only entities appearing before max_len* avg of 3 chars pro token = 384 characters
max_len_chars = 800
news['ents'] = news.ents.apply(lambda ents: [ent for ent in ents if ent[1] <= (max_len_chars, max_len_chars)])

def clean_unique_entities(ents):
    seen_ents = []
    res = []
    for ent in ents:
        cleaned_ent = firm_name_clean(ent[0])
        if cleaned_ent not in seen_ents:
            res.append(ent + (cleaned_ent,))
            seen_ents.append(cleaned_ent)


    return res

news['ents'] = news.ents.apply(clean_unique_entities) # take only unique entities per doc


# filter out firms in orbis
orbis = pd.read_pickle('C:/Users/Jakob/Documents/Orbis/orbis_firm_names.pkl')
orbis = set(orbis.company.to_list())
to_remove = ['federal reserve', 'eu', 'fed', 'treasury', 'congress', 'european central bank',
             'international monetary fund', 'central bank', 'senate', 'white house', 'house', 'sec', 'ecb',
             'european commission', 'state', 'un', 'bank of england', 'opec', 'supreme court', 'world bank',
             'pentagon', 'cabinet', 'web service', 'us senate', 'imf', 'defense',
             'federal reserve bank' 'euro', 'house of representatives', 'bank', 'journal',
             'us bankruptcy court', 'medicare', 'american international', 'finance', 's&p', 's&p 500',
             'news', 'united nations', 'nasdaq', 'parliament', 'us treasury department', 'romney', 'draghi',
             'usda', 'cotton', 'district court', 'army']
orbis = {elem for elem in orbis if elem not in to_remove}

pd.Series(list(orbis), name='company').to_pickle('C:/Users/Jakob/Documents/Orbis/orbis_firm_names.pkl')

news['orbis_ents'] = news.ents.apply(lambda ents: [ent for ent in ents if ent[2] in orbis])

news.orbis_ents.explode().size/news.ents.explode().size

news = news[news.ents.apply(len) > 1] # take only docs with at least two unique entities
news['ents'] = news.ents.apply(lambda ents: ents[:2]) # take only exactly two entities per doc (the first two)

news['orgs'] = news.ents.apply(lambda ents: [ent[0] for ent in ents])
news['spans'] = news.ents.apply(lambda ents: [ent[1] for ent in ents])
news['cleaned_orgs'] = news.ents.apply(lambda ents: [ent[2] for ent in ents])


# match via firm list (flasthext)
# read firm list
# firm_list = pd.read_pickle('C:/Users/Jakob/Documents/Orbis/orbis-europe-min-15-empl-16-03-22.pkl')
# firm_list.rename(columns= {'Number of employees\nLast avail. yr': 'emp'}, inplace=True)
# firm_list['emp'] = pd.to_numeric(firm_list.emp, errors='coerce')
# firm_list.emp.describe() # focus on big firms
# firm_list = firm_list[firm_list.emp > 5000]
#
# from flashtext import KeywordProcessor
# keyword_processor = KeywordProcessor()
# keyword_processor.add_keywords_from_list(firm_list['Company name Latin alphabet'].astype(str).to_list())
#
# keyword_processor.extract_keywords('Volkswagen AG and Tesco PLC announced blah.')
#
# news['firms'] = news.Text.str[:max_len_chars].progress_apply(lambda x: keyword_processor.extract_keywords(x, span_info=True))
#
# news['spans'] = news.firms.apply(lambda ents: [ent[1:] for ent in ents])
# news['firms'] = news.firms.apply(lambda ents: [ent[0] for ent in ents])

# news['orgs'] = news.orgs.apply(lambda x: x[0])
#
# news['tokens'] = news.progress_apply(lambda x: match_entities(x.orgs[:1], x.Text[:400]), axis=1)

news['meta'] = news.apply(lambda x: {'source': f'Reuters News Dataset - Article ID {str(x.ID)} - {str(x.Date)}'}, axis=1)
news.drop(columns=['Date', 'ID', 'ents', 'sents', 'cleaned_orgs'], inplace=True)
news.rename(columns={'Text': 'document', 'orgs': 'entities', 'spans': 'entity_spans'}, inplace=True)

news['relation'] = news.meta.apply(lambda x: [])

# select documents where two firms are mentioned in the beginning (first 100 chars)
news = news[news.spans.apply(lambda x: x[1][1]) < 100]

news = news.sample(len(kb))

news.to_pickle('/Users/Jakob/Documents/financial_news_data/news_literal_orgs.pkl')

news = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/news_literal_orgs.pkl')

kb = kb.append(news)

# split into train, test, dev
kb = kb.sample(frac=1)
kb.reset_index(drop=True)
train, dev, test = np.split(kb, [int(share_train*len(kb)), int((share_train+share_dev)*len(kb))])

kb.loc[train.index, 'meta'] = kb.loc[train.index, 'meta'].apply(lambda x: update_meta(x, 'split', 'train'))
kb.loc[dev.index, 'meta'] = kb.loc[dev.index, 'meta'].apply(lambda x: update_meta(x, 'split', 'dev'))
kb.loc[test.index, 'meta'] = kb.loc[test.index, 'meta'].apply(lambda x: update_meta(x, 'split', 'test'))


kb[['document', 'tokens', 'relations', 'meta']].to_json('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict_6class_balanced_negative.json',
                                                orient='records', lines=True)

test = kb[kb.meta.apply(lambda x: x['split'] == 'test')].sample(1000)

test = test[['document', 'tokens', 'relations', 'meta']].to_json('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict_6class_balanced_negative_small_test.json',
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