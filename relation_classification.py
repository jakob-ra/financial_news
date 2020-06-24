import pandas as pd
import uuid
import nltk
import swifter
import spacy
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

# set random state
rs = 42

## construct knowledge base (kb)

# read Thomson SDC RND database
kb = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/rnd_fin.parquet.gzip')

# give unique alliance IDs
kb['alliance_id'] = kb['date'].apply(lambda x: uuid.uuid4())

## construct corpus

# read news articles
corpus = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/news.parquet.gzip')

corpus_t = corpus.sample(n=100) # small sample

## construct common database to train and evaluate on
db = kb[['date', 'text','alliance_id']].append(corpus_t)

## spacy solution

nlp = spacy.load('en_core_web_sm') # spacy model
sentencizer = nlp.create_pipe("sentencizer")
example = 'This is the first sentence. Another short sentence starts here. Now we are in the third sentence.'
nlp.add_pipe(sentencizer)

doc = nlp(example)
print(doc.sents)

# split into sentences
db['text'] = db['text'].apply(lambda x: nltk.sent_tokenize(x))
db = db.explode('text')
db.reset_index(drop=True, inplace=True)


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

def get_sequences_with_2_orgs(text, dist=150):
    ''' Uses spacy NER to identify organisations. If two organizations are detected within dist
    characters from each other, extracts the sequence
    '''
    # Apply the model
    tags = nlp(text)

    # Return the list of entities recognized as organizations
    return [ent.text.replace('\'', '') for ent in tags.ents if ent.label_=='ORG'] # also remove apostrophes

sentence = 'Microsoft and Google announced a collaboration on the development of new computers.'


# extract organizations from each sentence
db['orgs'] = db.text.swifter.apply(get_orgs) # swifter parallelizes this expensive operation

# candidates = subset of sentences where at least two organizations are detected
candidates = db[db.orgs.str.len() > 1]

# label sentences from KB positive, rest negative
candidates['label'] = -1
candidates.loc[pd.notna(candidates.alliance_id), 'label'] = 1

# run classifier on candidates
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score

X, y = np.array(candidates.text.to_list()), np.array(candidates.label.to_list())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)

# pipeline
text_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', LogisticRegression(class_weight='balanced'))])

text_clf.fit(X_train, y_train)
text_clf.score(X_test, y_test)

candidates['prediction'] = text_clf.predict(candidates.text)
recall_score(candidates.label, candidates.prediction)
precision_score(candidates.label, candidates.prediction)

neg = candidates[(pd.isna(candidates.alliance_id))]
newpos = neg[neg.prediction == 1]

newpos.text.to_list()

## Active learning
from modAL.models import ActiveLearner

# initializing the learner
learner = ActiveLearner(
    estimator=text_clf,
    X_training=X_train, y_training=y_train) # sampling strat defaults to maximum uncertainty


n_initial = 100

initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)

X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)

n_queries = 1

for i in range(n_queries):
    query_idx, query_inst = learner.query(X_pool)
    print(query_inst[0])
    print("Collaboration? y/n")
    y_new = np.array([1 if input() == 'y' else -1], dtype=int)
    learner.teach(query_inst, y_new)
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)

learner.score(X_test, y_test)
learner.predict(['Microsoft and Google announced a collaboration on the development of new computers.'])


corpus['pred'] = text_clf.predict(corpus.text)

corpus[corpus.pred == 1].text.to_list()

bloomberg.Link.iloc[0]


