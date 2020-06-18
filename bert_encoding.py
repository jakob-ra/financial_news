from bert_serving.client import BertClient
bc = BertClient()
bc.encode(['First do it', 'then do it right', 'then do it better'])


import pandas as pd
relevant_colums = [6,13,26]
column_names = ['date', 'text', 'participants'] # this extracts the PARENT COMPANY participants
df = pd.read_excel('SDC_RND_2015_2020.xlsx', skiprows=1, usecols=relevant_colums, names=column_names)

# keep only date (not time)
df['date'] = df['date'].dt.date

# separate multiline Participants cell into tuple
df['participants'] = df['participants'].str.split('\n')

# replace new line character \n with space in deal text
df['text'] = df['text'].str.replace('\n', ' ')

import timeit
timeit.timeit(df['text'].apply(lambda text: bc.encode([text])))

