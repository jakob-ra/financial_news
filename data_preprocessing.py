import pandas as pd
import os
import glob
import re
import uuid
import nltk

### Reads, formats, and saves all relevant data for this project as parquet

## Capital IQ database
cap_iq_path = '/Users/Jakob/Documents/financial_news_data/capitaliq'
all_cap_iq = glob.glob(os.path.join(cap_iq_path, "*.xls"))

# columns = pd.DataFrame([pd.read_excel(f, skiprows=7).columns for f in all_cap_iq]) # check columns
cap_iq = pd.concat((pd.read_excel(f, skiprows=7, usecols=[0,2,4,6,7,8]) for f in all_cap_iq))

cap_iq.columns = ['date', 'participants', 'participant_HQ_country', 'headline', 'text', 'source']

# keep only date (not time)
cap_iq['date'] = cap_iq['date'].dt.date

# remove ticker names from participants column
cap_iq['participants'] = cap_iq['participants'].apply(lambda x: re.sub(' \(.*\)', '', x))

# turn participants into list
cap_iq['participants'] = cap_iq['participants'].str.split('; ')

# replace new line character \n with space in deal text
cap_iq['text'] = cap_iq['text'].str.replace('\n', ' ')

# remove double spaces
cap_iq['text'] = cap_iq['text'].str.replace('  ', ' ')

# add source tag
cap_iq['source'] = 'CapIQ'

# drop country
cap_iq.drop(columns=['participant_HQ_country'], inplace=True)

## Thomson SDC RND database

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

# find number of unique swiss firms
cap_iq_switz['participant_HQ_country'] = cap_iq_switz['participant_HQ_country'].str.split(';')
cap_iq_switz_parts = cap_iq_switz.explode('participant_HQ_country')
cap_iq_switz_parts = cap_iq_switz_parts[cap_iq_switz_parts['participant_HQ_country'].str.contains(
    'Switzerland')]
cap_iq_switz_parts['suiss_firm'] = cap_iq_switz_parts['participant_HQ_country'].str.split(' \(').str[0]
cap_iq_switz_parts['suiss_firm'].nunique()

# merge SDC and Capital IQ databases
kb = pd.concat((cap_iq, sdc))

# sort by date
kb.sort_values(by='date', inplace=True)
kb.reset_index(drop=True, inplace=True)

# give unique alliance IDs
kb['alliance_id'] = kb.apply(lambda x: uuid.uuid4(), axis=1)
kb['alliance_id'] = kb['alliance_id'].astype('str')

# save as parquet file
kb.to_parquet('/Users/Jakob/Documents/financial_news_data/kb.parquet.gzip')

## News articles

# read news articles
reuters = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/reuters.parquet.gzip')
bloomberg = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/bloomberg.parquet.gzip')

news = reuters.append(bloomberg)
news = news[['Date', 'Link', 'Article','Headline']]
news.columns = [x.lower() for x in news.columns] # lowercase all column names

# sort out dates
news['date'] = pd.to_datetime(news['date'], utc=True, infer_datetime_format=True)
news.sort_values(by='date', inplace=True)
news['date'] = news['date'].dt.date # keep only date not time

# remove city and source tag (only Reuters articles have them)
news['article'] = news['article'].apply(lambda str: str.split('(Reuters) - ')[-1])

# Add headline as sentence to text
news['text'] = news['headline'] + '. ' + news['article']
news.drop(columns=['headline','article'], inplace=True)

# remove empty articles
news = news[news.text.str.len() > 5]

# get rid of useless articles
news = news[~news.text.str.startswith('TRADING TO RESUME AT')]

news.reset_index(drop=True, inplace=True)

news.to_parquet('/Users/Jakob/Documents/financial_news_data/news.parquet.gzip')

# create small sample for negative examples for classifier training
corpus_sample = news.sample(n=20000) # small sample

# split into sentences
corpus_sample['text'] = corpus_sample['text'].apply(lambda x: nltk.sent_tokenize(x))
corpus_sample = corpus_sample.explode('text')
corpus_sample.reset_index(drop=True, inplace=True)

corpus_sample.to_parquet('/Users/Jakob/Documents/financial_news_data/corpus_sample.parquet.gzip')

# # find swiss alliances
# cap_iq_switz = cap_iq[cap_iq['participant_HQ_country'].str.contains('Switzerland')]
#
# # find number of unique swiss firms
# cap_iq_switz['participant_HQ_country'] = cap_iq_switz['participant_HQ_country'].str.split(';')
# cap_iq_switz_parts = cap_iq_switz.explode('participant_HQ_country')
# cap_iq_switz_parts = cap_iq_switz_parts[cap_iq_switz_parts['participant_HQ_country'].str.contains(
#     'Switzerland')]
# cap_iq_switz_parts['suiss_firm'] = cap_iq_switz_parts['participant_HQ_country'].str.split(' \(').str[0]
# cap_iq_switz_parts['suiss_firm'].nunique()

# # find swiss alliances
# sdc_switz = sdc[sdc['participant_country'].apply(lambda x: ','.join(map(str, x))).str.contains(
#     'Switzerland')]