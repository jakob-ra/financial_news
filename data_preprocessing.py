import pandas as pd

### Reads, formats, and saves all relevant data for this project as parquet

## Thomson SDC RND database

# rnd_1 = pd.read_excel('SDC_RND_1962_2012.xlsx', skiprows=3, usecols=relevant_colums, names=column_names)
rnd_2 = pd.read_excel('SDC_RND_2010_2016.xls', skiprows=2, usecols=[6,7,8,18], names=['date', 'participants',
     'parent_participants', 'text'])
rnd_3 = pd.read_excel('SDC_RND_2015_2020.xlsx', skiprows=1, usecols=[6,13,21,26], names=['date', 'text',
    'participants', 'parent_participants'])
# rnd_3 = rnd_3[['date', 'participants', 'text',]]

# remove overlap
# rnd_1 = rnd_1[rnd_1['date'] < '2010-01-01']
rnd_2 = rnd_2[rnd_2['date'] < '2015-01-01']

# merge dataframes
rnd = rnd_2.append(rnd_3)

# keep only date (not time)
rnd['date'] = rnd['date'].dt.date

# separate multiline Participants cell into tuple
rnd['participants'] = rnd['participants'].str.split('\n')
rnd['parent_participants'] = rnd['parent_participants'].str.split('\n')

# replace new line character \n with space in deal text
rnd['text'] = rnd['text'].str.replace('\n', ' ')

# save as parquet file
rnd.to_parquet('rnd_fin.parquet.gzip')

##