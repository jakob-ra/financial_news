import pandas as pd
import nltk
import swifter
import spacy
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

# set random state
rs = 42

## construct knowledge base (kb)

# read KB
kb = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/kb.parquet.gzip')

# remove punctuation after firms legal type (makes splitting sentences easier)
legal_tokens = ['Co', 'Inc', 'AG', 'GmbH', 'Ltd', 'lp', 'lpc', 'Llc', 'pllc', 'llp', 'plc', 'ltd/plc',
    'Corp', 'ab', 'cos', 'cia', 'sa', 'sas', 'rrb']
for tok in legal_tokens:
    kb['text'] = kb.text.str.replace(' ' + tok + '.', ' ' + tok, case=False, regex=False)
    kb['headline'] = kb.headline.str.replace(' ' + tok + '.', ' ' + tok, case=False, regex=False)
kb['participants'] = kb.participants.apply(lambda list: [name.replace('.', '') for name in list])
kb['participants'] = kb.participants.apply(lambda list: [name.replace(',', '') for name in list])

# use only first sentence of deal text
kb['text'] = kb['text'].apply(lambda x: nltk.sent_tokenize(x)[0])

# label kb examples positive
kb['label'] = 1

# many alliances with only one participant (other firm not recognized)
print(len(kb[kb.participants.map(len) == 1]))
kb = kb[kb.participants.map(len) > 1]

## construct corpus

# read news articles
corpus = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/news.parquet.gzip')

# take appropriately sized sample of articles from corpus
corpus = corpus.sample(n=len(kb), random_state=rs)

# remove punctuation after firms legal type (makes splitting sentences easier)
for tok in legal_tokens:
    corpus['text'] = corpus.text.str.replace(' ' + tok + '.', ' ' + tok, case=False, regex=False)

# split into sentences
corpus['text'] = corpus['text'].apply(lambda x: nltk.sent_tokenize(x))
corpus = corpus.explode('text')
corpus.reset_index(drop=True, inplace=True)

corpus.to_parquet('/Users/Jakob/Documents/financial_news_data/corpus.parquet.gzip')

# corpus = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/corpus.parquet.gzip')

# corpus = corpus.sample(n=len(kb))

## spacy solution
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

# extract organizations from each sentence
corpus['orgs'] = corpus.text.swifter.apply(get_orgs) # swifter parallelizes this expensive operation

# candidates = subset of sentences where at least two organizations are detected
candidates = corpus[corpus.orgs.str.len() > 1]

# label candidate sentences negative
candidates['label'] = -1

# construct common database to train and evaluate on
ds = kb[['date', 'text', 'participants', 'alliance_id', 'label']].append(candidates)

ds.to_parquet('/Users/Jakob/Documents/financial_news_data/ds.parquet.gzip')

# run classifier on candidates
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score

X, y = np.array(ds.text.to_list()), np.array(ds.label.to_list())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=rs)

# pipeline
text_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', LogisticRegression(class_weight='balanced'))])

text_clf.fit(X_train, y_train)
text_clf.score(X_test, y_test)

ds['prediction'] = text_clf.predict(ds.text)
recall_score(ds.label, ds.prediction)
precision_score(ds.label, ds.prediction)

neg = ds[(pd.isna(ds.alliance_id))]
newpos = neg[neg.prediction == 1]

newpos.text.to_list()


## Active learning
# from modAL.models import ActiveLearner
#
# # initializing the learner
# learner = ActiveLearner(
#     estimator=text_clf,
#     X_training=X_train, y_training=y_train) # sampling strat defaults to maximum uncertainty
#
#
# n_initial = 100
#
# initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
#
# X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
# X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)
#
# n_queries = 1
#
# for i in range(n_queries):
#     query_idx, query_inst = learner.query(X_pool)
#     print(query_inst[0])
#     print("Collaboration? y/n")
#     y_new = np.array([1 if input() == 'y' else -1], dtype=int)
#     learner.teach(query_inst, y_new)
#     X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
#
# learner.score(X_test, y_test)
# learner.predict(['Microsoft and Google announced a collaboration on the development of new computers.'])
#
#
# corpus['pred'] = text_clf.predict(corpus.text)
#
# corpus[corpus.pred == 1].text.to_list()
#
# len(corpus[corpus.pred == 1].text.to_list())


# def get_sequences_with_2_orgs(text, dist=150):
#     ''' Uses spacy NER to identify organisations. If two organizations are detected within dist
#     characters from each other, extracts the sequence
#     '''
#     # Apply the model
#     tags = nlp(text)
#
#     # Return the list of entities recognized as organizations
#     return [ent.text.replace('\'', '') for ent in tags.ents if ent.label_=='ORG'] # also remove apostrophes

# sentence = 'Microsoft and Google announced a collaboration on the development of new computers.'


