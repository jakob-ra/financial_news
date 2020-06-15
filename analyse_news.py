import pandas as pd
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow_hub as hub

df = pd.read_parquet('financial_data.parquet.gzip')

# small data set to run fast experiments
df_size = 1000
df = df.head(df_size)

df.iloc[500].Article

