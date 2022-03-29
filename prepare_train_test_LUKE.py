import pandas as pd
import re
import spacy
import numpy as np
from itertools import permutations
from tqdm import tqdm
tqdm.pandas()

np.random.seed(42) # set random seed


kb = pd.read_pickle('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_kb_training.pkl')

## add neg examples from news
news = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/news_literal_orgs.pkl')


# take only first two entities per doc (LUKE can only handle one entity pair as input per example)
kb['firms'] = kb.firms.apply(lambda x: x[:2])
kb['spans'] = kb.spans.apply(lambda x: x[:2])


df = kb.append(news)


df.to_pickle('/Users/Jakob/Documents/financial_news_data/training-data-29-03-22.pkl', protocol=4)

a/sum(a)