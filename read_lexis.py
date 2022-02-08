import pandas as pd
import os
from ast import literal_eval

path = 'C:/Users/Jakob/Documents/LexisNexis firm alliances new/'

# combine
chunks = os.listdir(path)

chunks = [chunk.replace('_gap_full_', '') for chunk in chunks]

# check all is there
chunk_nums = [int(x.split('new')[-1].split('.csv')[0]) for x in chunks]
missing = [x for x in range(1, sorted(chunk_nums)[-1]) if x not in chunk_nums]
print('Missing chunks: ', missing)

for n, chunk in enumerate(chunks):
    if n == 0:
        col_converters = {col: eval for col in ['country', 'city', 'industry', 'company', 'subject']}
        col_dtypes = {col: str for col in ['title', 'publication', 'author', 'copyright', 'content']}
        col_dtypes.update({'word_count': int})
        date_cols = ['publication_date', 'publication_date_text']
        df = pd.read_csv(os.path.join(path, chunk), compression='gzip', dtype=col_dtypes,
                         converters=col_converters, parse_dates=date_cols)
    else:
        df = df.append(pd.read_csv(os.path.join(path, chunk), compression='gzip', dtype=col_dtypes,
                       converters=col_converters, parse_dates=date_cols))


df.reset_index(inplace=True, drop=True)

df.sort_values('publication_date', inplace=True)

# drop short or NA articles
df = df[df.title.apply(str).apply(len) + df.content.apply(str).apply(len) > 20]

# drop duplicate articles keeping only the first one (in time)
df.drop_duplicates(subset=['title', 'content', 'publication'], keep='first', inplace=True)

# aggregate sources of articles that appear in multiple publications
df = df.groupby(['title', 'content']).publication.apply(list).reset_index().merge(
    df.drop(columns='publication').drop_duplicates(subset=['title', 'content']), on=['title', 'content'])

# clean
df['content'] = df.content.str.strip()

# export
df.to_csv(os.path.join('C:/Users/Jakob/Documents', 'lexisnexis_firm_alliances_combined_new.csv.gzip'), index=False, compression='gzip')

df = pd.read_csv(os.path.join('C:/Users/Jakob/Documents', 'lexisnexis_firm_alliances_combined_new.csv.gzip'), compression='gzip')

# turn these columns into lists
list_tags = ['subject', 'country', 'city', 'person', 'industry', 'company']
for tag in list_tags:
    df[tag] = df[tag].apply(literal_eval)


df.groupby(df.publication_date.dt.to_period('Y')).size()

df[df.publication_date > pd.to_datetime('01-01-2010')].size()

# find gaps in time
df.loc[df.publication_date.diff() > pd.Timedelta("1 days")].publication_date

import matplotlib.pyplot as plt

df.groupby(df.publication_date.dt.to_period('Y')).size().plot(kind="bar")
plt.show()

df[df.publication_date.dt.to_period('Y') == '2016'].groupby(df.publication_date.dt.to_period('M')).size().plot(kind="bar")
plt.show()

df.memory_usage(index=True, deep=True)/1000000

# most frequent tags
df.publication.value_counts().head(20)

df.country.explode().value_counts().head(20)

df.company.explode().value_counts().head(20)

df.subject.explode().value_counts().head(20)

df.industry.explode().value_counts().head(20)

df.company.explode().value_counts()


# focus on news mentioning at least 2 companies
df = df[df.company.str.len() > 1]

from langdetect import detect

# keep only docs with at least one sentence
df = df[df.content.str.split('. ').str.len() > 1]

# run language detection on first 100 chars
df['lang'] = df.content.apply(lambda x: detect(x[:100]))


# rename
# path2 = 'C:/Users/Jakob/Documents/LexisNexis firm alliances new gap/'
# chunks2 = os.listdir(path2)
#
# start_num = sorted(chunk_nums)[-1] + 1
#
# for n, chunk in enumerate(chunks2):
#     new_name = 'Firm_alliances_new' + str(start_num + n) + '.csv.gzip'
#     os.rename(os.path.join(path2, chunk), os.path.join(path2, new_name))