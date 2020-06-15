import pandas as pd
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow_hub as hub

# read Thomson SDC RND database
relevant_colums = [6,13,26]
column_names = ['date', 'text', 'participants'] # this extracts the PARENT COMPANY participants
df = pd.read_excel('SDC_RND_2015_2020.xlsx', skiprows=1, usecols=relevant_colums, names=column_names)

# keep only date not time
df['date'] = df['date'].dt.date

# separate multiline Participants cell into tuple
df['participants'] = df['participants'].str.split('\n')

# replace new line character \n with space in deal text
df['text'] = df['text'].str.replace('\n', ' ')


# participants = df.participants.str.split('\n', expand=True)
# df.drop(columns='participants', inplace=True) # drop old participants column
# df = df.join(participants)

# read Reuters financial news database
# df = pd.read_parquet('financial_data.parquet.gzip')

# small data set to run fast experiments
# df_size = 1000
# df = df.head(df_size)