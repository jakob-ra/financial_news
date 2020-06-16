import pandas as pd
import nltk
import en_core_web_sm
from fuzzywuzzy import process
from multiprocessing import  Pool
# from transformers import pipeline, TokenClassificationPipeline, TFAutoModelForTokenClassification, AutoTokenizer
# from nltk.tag import StanfordNERTagger
# from nltk.tokenize import word_tokenize
# import tensorflow as tf
# import tensorflow_hub as hub

# read Thomson SDC RND database
relevant_colums = [6,13,26]
column_names = ['date', 'text', 'participants'] # this extracts the PARENT COMPANY participants
df = pd.read_excel('SDC_RND_2015_2020.xlsx', skiprows=1, usecols=relevant_colums, names=column_names)

# keep only date (not time)
df['date'] = df['date'].dt.date

# separate multiline Participants cell into tuple
df['participants'] = df['participants'].str.split('\n')

# replace new line character \n with space in deal text
df['text'] = df['text'].str.replace('\n', ' ')

# spacy implementation
# Load model
nlp = en_core_web_sm.load()

def get_entities(text):
    '''
    This function takes a text. Uses the Spacy model.
    The model will tokenize, POS-tag and recognize the entities named in the text.
    Then, the entities are retrieved and saved in a list.
    It outputs a list of entities in the text that were recognized as organizations.
    '''
    # Apply the model
    tags = nlp(text)

    # Return the list of entities recognized as organizations
    return [ent.text.replace('\'', '') for ent in tags.ents if ent.label_=='ORG'] # also remove apostrophes

# save recognized organizations in new column
df['entities'] = df.text.apply(get_entities)

# remove 'co', 'AG', etc. (Schwenkler & Zheng, wikipedia.org/wiki/List_of_legal_entity_types_by_country)
remove_tokens = ['co', 'inc', 'ag', 'gmbh', 'ltd', 'lp', 'lpc', 'llc', 'pllc', 'llp', 'plc', 'ltd/plc',
    'corp', 'ab', 'cos', 'cia', 'sa', 'sas', 'reuters', 'reuter', 'based', 'rrb']

# other way: find word tokens that are most common in database of company names
companies = pd.read_excel('Orbis_US_public.xlsx', names=['name', 'BvDID', 'ISIN', 'ticker', 'vat', 'city',
    'country', 'website'])
for char in ['.', ',']:
    companies['name'] = companies['name'].str.replace(char, '') # remove punctuation

companies['tokens'] = companies['name'].str.split()
words = companies.tokens.tolist()
words = [word for list_ in words for word in list_] # unstack list of lists
print(pd.DataFrame(nltk.FreqDist(words).most_common(50), columns=['Word', 'Frequency']))

# def match(str2match, options):
#     """ Returns a match in list of names that has a similarity ratio of at least 0.99 (fuzzywuzzy) with
#     the input name. In case of multiple such matches, returns shortest match. In case of no match,
#     returns None. """
#     ratios = process.extract(str2match, options)
#     matches = [rat for rat in ratios if rat[1] > 80] # filter for sufficiently good matches
#
#     return min(matches, key=len) # return shortest match


str2Match = "apple inc"
strOptions = ["Apple Inc.","apple park","apple incorporated","iphone", "apple", "Apple Incorporated",
    "APPLE INC"]

def match(str2match , options, threshold=0.99):
    best_match = process.extractOne(str2match, options)
    if best_match[1] > threshold:
        return best_match[0]
    else:
        return None

firm_names = companies.name.to_list() # firm name database to match on
df['matched'] = df['entities'].apply(lambda list: [match(name,firm_names) for name in list])

import numpy as np
def match_df(df):
    df['matched'] = df['entities'].apply(lambda list: [match(name, firm_names) for name in list])

def parallelize_df(df, func, n_cores=12):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

df = parallelize_df(df, match_df)

# read Reuters financial news database
news = pd.read_parquet('reuters.parquet.gzip')

# delete city and source
# news['Article'] = news['Article'].str.split('(Reuters) - ').str[-1]

# delete special characters
# news['Article'] = news['Article'].str.replace(r"\'", "\'")

# small data set to run fast experiments
news = news.head(1000)

# save recognized organizations in new column
news['entities'] = news.Article.apply(get_entities)



# huggingface NER pipeline
# nlp = pipeline('ner')
# sentence = df.iloc[1].text
# print(pd.DataFrame(nlp(sentence))) # gives out tokens and labes
#
# # BERT tokenizer and token classification
# nlp = TokenClassificationPipeline(model=TFAutoModelForTokenClassification.from_pretrained(
#     'bert-large-cased'),
#     tokenizer=AutoTokenizer.from_pretrained('bert-large-cased'), framework='tf', grouped_entities=False)
# print(pd.DataFrame(nlp(sentence)))

# put back together company names
# text = ' '.join([x for x in tokens])
# fine_text = text.replace(' ##', '')

# tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')
# model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased-finetuned-mrpc')


# st = StanfordNERTagger('/Jakob/Downloads/stanford-classifier-4.0.0/english.all.3class.distsim.crf.ser.gz',
# 					   '/usr/share/stanford-ner/stanford-ner.jar',
# 					   encoding='utf-8')



# participants = df.participants.str.split('\n', expand=True)
# df.drop(columns='participants', inplace=True) # drop old participants column
# df = df.join(participants)