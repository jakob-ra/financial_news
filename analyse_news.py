import pandas as pd
import spacy
import nltk
# from fuzzywuzzy import process
# from multiprocessing import pool
# from google_match import google_KG_match

# from nltk.tag import StanfordNERTagger
# from nltk.tokenize import word_tokenize
# import tensorflow as tf
# import tensorflow_hub as hub


import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# set random state
rs = 42

# read Thomson SDC RND database
rnd = pd.read_parquet('rnd_fin.parquet.gzip')

# read news articles
reuters = pd.read_parquet('reuters.parquet.gzip')
bloomberg = pd.read_parquet('bloomberg.parquet.gzip')

# check news corpus for the word 'alliance'
# reuters['sentences'] = reuters.Article.str.split('.') # not ideal, should use ntlk.sent_tokenize
# alliance = reuters[reuters.Article.str.find('alliance') != -1]
# alliance['rel_sentences'] = alliance['sentences'].apply(lambda list: [s for s in list if 'alliance' in s])

# reuters[reuters.Article.str.find('Apple') != -1]

import copy

## positive examples
pos = pd.DataFrame(rnd.text)
pos['label'] = 1

# select first sentence from pos
pos['first_sentence'] = pos.text.apply(lambda x: nltk.sent_tokenize(x)[0])



## negative examples

# randomly sample from news corpus until we have balance
n_pos = len(pos)
neg = reuters[['Article','Headline']].sample(n=n_pos, random_state=rs)

# remove city and source tag
neg['Article'] = neg['Article'].apply(lambda str: str.split('(Reuters) - ')[-1])

# Add headline as sentence to text
neg['text'] = neg['Headline'] + '. ' + neg['Article']
neg.drop(columns=['Headline','Article'], inplace=True)

# label negative
neg['label'] = -1

# full data
full = pos.append(neg)


## split sentences
neg['sentences'] = neg.text.apply(lambda x: nltk.sent_tokenize(x))
neg_sentences = pd.DataFrame([sentence for list in neg.sentences.to_list() for sentence in list],
    columns=['first_sentence'])
neg_sentences['label'] = -1
# neg_sentences = neg_sentences.sample(n=n_pos, random_state=rs)

full = pos[['first_sentence', 'label']].append(neg_sentences)
full.rename(columns={'first_sentence': 'text'}, inplace=True)

# what is distribution of article length?
# import matplotlib.pyplot as plt
# full['length'] = full.text.apply(len)
# full.length.hist()
# plt.show()
#
# quantiles = np.arange(0,1.001,0.001)
# lengths = full.length.quantile(quantiles).values
# plt.scatter(quantiles, lengths)
# plt.show()

# shuffle data
full = full.sample(frac=1, random_state=rs)


## tf-idf logistic regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

X, y = np.array(full.text.to_list()), np.array(full.label.to_list())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)

# pipeline
text_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', LogisticRegression())])

text_clf.fit(X_train, y_train)
text_clf.score(X_test, y_test)

text_clf.predict([sentence])
text_clf.predict(['I took a walk in the park today.'])
text_clf.predict(['Pfizer and Merck will collaborate on the development of new medicine.'])
text_clf.predict(['Pfizer and Merck announced an alliance for the development of new medicine.'])
text_clf.predict(['Pfizer and Merck announced that they will compete in the development of new medicine.'])

neg_sentences['prediction'] = text_clf.predict(neg_sentences.first_sentence)
neg_sentences[neg_sentences.prediction == 1].first_sentence.to_list()
pos['prediction'] = text_clf.predict(pos.first_sentence)
pos[pos.prediction == -1].text.to_list()

reuters['prediction'] = text_clf.predict(reuters['Article'])
reuters[reuters.prediction == 1]

## huggingface
from transformers import DistilBertTokenizerFast

pretrained_model_name = 'distilbert-base-cased'
tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_name)

sentence = 'Apple and Microsoft plan to form a joint venture for the development of cloud-based computing ' \
           'infrastructure.'

test = full.head(10)

tokenizer.batch_encode_plus(test.text.to_list())
# print(tokenizer.encode_plus(sentence))

def encoder(text):
    encoding = tokenizer.encode_plus(text, max_length=None,
    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    return_token_type_ids=False, pad_to_max_length=False, return_attention_mask=False,
    return_tensors='tf')  # Return Tensorflow tensors)

    return encoding

full['encoded'] = encoder(test.text.apply(encoder))
output['input_ids'].shape

tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

## spacy Named Entity Recognition

# Load model
# nlp = spacy.load('en_trf_bertbaseuncased_lg') # huggingface BERT implementation
nlp = spacy.load('en_core_web_sm') # spacy model

def get_orgs(text):
    '''
    This function takes a text. Uses the Spacy model.
    The model will tokenize, POS-tag and recognize the entities named in the text.
    Returns a list of entities in the text that were recognized as organizations.
    '''
    # Apply the model
    tags = nlp(text)

    # Return the list of entities recognized as organizations
    return [ent.text.replace('\'', '') for ent in tags.ents if ent.label_=='ORG'] # also remove apostrophes

# get_entities('Apple and Microsoft plan to form a joint venture for the development of cloud-based '
#              'computing infrastrucutre.')

# save recognized organizations in new column
df['orgs'] = df.text.apply(get_orgs)

# remove 'co', 'AG', etc. (Schwenkler & Zheng, wikipedia.org/wiki/List_of_legal_entity_types_by_country)
remove_tokens = ['co', 'inc', 'ag', 'gmbh', 'ltd', 'lp', 'lpc', 'llc', 'pllc', 'llp', 'plc', 'ltd/plc',
    'corp', 'ab', 'cos', 'cia', 'sa', 'sas', 'reuters', 'reuter', 'based', 'rrb']

# other way: find word tokens that are most common in database of company names
# companies = pd.read_excel('Orbis_US_public.xlsx', names=['name', 'BvDID', 'ISIN', 'ticker', 'vat', 'city',
#     'country', 'website'])
# for char in ['.', ',']:
#     companies['name'] = companies['name'].str.replace(char, '') # remove punctuation

# companies['tokens'] = companies['name'].str.split()
# words = companies.tokens.tolist()
# words = [word for list_ in words for word in list_] # unstack list of lists
# print(pd.DataFrame(nltk.FreqDist(words).most_common(50), columns=['Word', 'Frequency']))

# def match(str2match , options, threshold=99):
#     best_match = process.extractOne(str2match, options)
#
#     if best_match[1] > threshold:
#         return best_match[0]
#     else:
#         return None
#
# firm_names = companies.name.to_list() # firm name database to match on
# import copy
# small = copy.deepcopy(df.head(10))

# match
# small['matched'] = small['entities'].apply(lambda list: [match(name,firm_names) for name in list])

# Google Knowledge Graph Matching
# api_key = open('google_api_key.txt', 'r').read()
# small['matched'] = small['entities'].apply(lambda list: [google_KG_match(name, api_key, type='Corporation')
#     for name in list])
#
# names = [item for sublist in small.entities.to_list() for item in sublist]
# for name in names:
#     print(name)
#     print(google_KG_match(name, api_key, type='Corporation'))

# def match_df(df):
#     df['matched'] = df['entities'].apply(lambda list: [match(name, firm_names) for name in list])
#
#     return df
#
# def parallelize_df(df, func, n_cores=12):
#     df_split = np.array_split(df, n_cores)
#     pool = Pool(n_cores)
#     df = pd.concat(pool.map(func, df_split))
#     pool.close()
#     pool.join()
#     return df
#
# if __name__ == 'name':
#     small = parallelize_df(small, match_df)


# read Reuters financial news database
# news = pd.read_parquet('reuters.parquet.gzip')

# delete city and source
# news['Article'] = news['Article'].str.split('(Reuters) - ').str[-1]

# delete special characters
# news['Article'] = news['Article'].str.replace(r"\'", "\'")

# small data set to run fast experiments
# news = news.head(1000)
#
# # save recognized organizations in new column
# news['entities'] = news.Article.apply(get_entities)



# huggingface NER pipeline
from transformers import TokenClassificationPipeline, TFAutoModelForTokenClassification, AutoTokenizer
# from transformers import pipeline
# nlp = pipeline('ner')
# print(pd.DataFrame(nlp(sentence))) # gives out tokens and labes

sentence = 'Apple and Microsoft plan to form a joint venture for the development of cloud-based computing ' \
           'infrastrucutre.'

## BERT tokenizer and token classification
nlp = TokenClassificationPipeline(model=TFAutoModelForTokenClassification.from_pretrained(
    'distilbert-base-cased'), tokenizer=AutoTokenizer.from_pretrained('distilbert-base-cased'),
    framework='tf')
print(pd.DataFrame(nlp(sentence)))


from transformers import DistilBertTokenizer, TFDistilBertForTokenClassification
import tensorflow as tf
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-cased')
input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
labels = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1
print(model(input_ids))

import numpy as np
from transformers import AutoTokenizer, pipeline, TFDistilBertModel
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
model = TFDistilBertModel.from_pretrained('distilbert-base-cased')
pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
features = pipe('any text data or list of text data', pad_to_max_length=True)
features = np.squeeze(features)


## Sentence classification


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