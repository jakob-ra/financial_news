import pandas as pd
import re
import spacy
import numpy as np
from itertools import permutations
from tqdm import tqdm
tqdm.pandas()

np.random.seed(42) # set random seed


kb = pd.io.json.read_json(path_or_buf='/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict_6class.json',
                          orient='records', lines=True)

kb['entities'] = kb.tokens.apply(lambda ents: [ent['text'] for ent in ents])
kb['entity_spans'] = kb.tokens.apply(lambda ents: [(ent['start'], ent['end']) for ent in ents])
kb['relation'] = kb.relations.apply(lambda relations: [rel['relationLabel'] for rel in relations])
kb['relation'] = kb.relation.apply(set).apply(list)

## add neg examples from news
news = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/news_literal_orgs.pkl')

kb = kb.append(news)

kb['source'] = kb.meta.apply(lambda x: x['source'])

kb = kb[['document', 'entities', 'entity_spans', 'relation']]

kb = kb[kb.entity_spans.str.len() == 2] # take docs with exactly two entities

kb.to_pickle('/Users/Jakob/Documents/financial_news_data/kb_6class_balanced_neg_examples.pkl', protocol=4)

