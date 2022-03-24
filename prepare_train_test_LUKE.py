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


def extract_relations(input):
    output = []
    for relation in input:
        output.append(relation['relationLabel'])

    return set(output)

kb['relation_types'] = kb.relations.apply(extract_relations)

# create balance by random undersampling+
rare_flags = {'Marketing', 'Manufacturing', 'Licensing', 'ResearchandDevelopment'}
kb = kb[kb.relation_types.apply(lambda x: len(x.intersection(rare_flags))) > 0] # focus on examples with rare flags

# drop some marketing & SA examples
kb = kb.drop(kb[kb.relation_types == {'StrategicAlliance', 'Marketing'}].sample(3000).index)
kb = kb.drop(kb[kb.relation_types == {'StrategicAlliance', 'Manufacturing'}].sample(1000).index)
kb = kb.drop(kb[kb.relation_types == {'StrategicAlliance', 'Licensing'}].sample(1000).index)

kb.relation_types.explode().value_counts()
kb.relations.sample().values
kb.relation_types.value_counts()


## add neg examples from news
news = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/news_literal_orgs.pkl')

kb = kb.append(news)

kb['entities'] = kb.tokens.apply(lambda ents: [ent['text'] for ent in ents])
kb['entity_spans'] = kb.tokens.apply(lambda ents: [(ent['start'], ent['end']) for ent in ents])
kb['relation'] = kb.relations.apply(lambda relations: [rel['relationLabel'] for rel in relations])
kb['relation'] = kb.relation.apply(set).apply(list)
kb['source'] = kb.meta.apply(lambda x: x['source'])

kb = kb[['document', 'entities', 'entity_spans', 'relation']]

kb = kb[kb.entity_spans.str.len() == 2] # take docs with exactly two entities

kb.to_pickle('/Users/Jakob/Documents/financial_news_data/kb_6class_balanced_neg_examples.pkl', protocol=4)

