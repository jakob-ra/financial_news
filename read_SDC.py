import pandas as pd
import os
import glob
import re

### Reads, formats, and saves all SDC data for this project as parquet


## Thomson SDC RND database

sdc_path = '/Users/Jakob/Documents/Thomson_SDC'
all_sdc = glob.glob(os.path.join(sdc_path, "*.xlsx"))

sdc = pd.read_excel(all_sdc[0], skiprows=3)
sdc = pd.concat((pd.read_excel(f, skiprows=3) for f in all_sdc))

# rnd_1 = pd.read_excel('/Users/Jakob/Documents/financial_news_data/SDC_RND_1962_2012.xlsx', skiprows=3, usecols=relevant_colums, names=column_names)
sdc_2 = pd.read_excel('/Users/Jakob/Documents/financial_news_data/SDC_RND_2010_2016.xls', skiprows=2,
    usecols=[6,7,8,12,18], names=['date', 'participants',
     'parent_participants', 'participant_country', 'text'])
sdc_3 = pd.read_excel('/Users/Jakob/Documents/financial_news_data/SDC_RND_2015_2020.xlsx', skiprows=1,
    usecols=[6,13,21,22,26], names=['date', 'text',
    'participants', 'participant_country', 'parent_participants'])
# rnd_3 = rnd_3[['date', 'participants', 'text',]]

# remove overlap
# sdc_1 = sdc_1[sdc_1['date'] < '2010-01-01']
sdc_2 = sdc_2[sdc_2['date'] < '2015-01-01']

# merge dataframes
sdc = sdc_2.append(sdc_3)

# keep only date (not time)
sdc['date'] = sdc['date'].dt.date

# separate multiline Participants cell into list
sdc['participants'] = sdc['participants'].str.split('\n')
sdc['parent_participants'] = sdc['parent_participants'].str.split('\n')
sdc['participant_country'] = sdc['participant_country'].str.split('\n')

# replace new line character \n with space in deal text
sdc['text'] = sdc['text'].str.replace('\n', ' ')

# remove double spaces
sdc['text'] = sdc['text'].str.replace('  ', ' ')

# add source tag
sdc['source'] = 'ThomsonSDC'