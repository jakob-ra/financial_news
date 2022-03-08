import pandas as pd
import os
from ast import literal_eval
from langdetect import detect
import matplotlib.pyplot as plt
import swifter

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

# format dates
df['publication_date'] = df.publication_date.apply(pd.to_datetime)

# explore date
df.groupby(df.publication_date.dt.to_period('Y')).size()
df[df.publication_date > pd.to_datetime('01-01-2010')].size
# find gaps in time
df.loc[df.sort_values('publication_date').publication_date.diff() > pd.Timedelta("1 days")].publication_date


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


# run language detection on first 100 chars


def lang_detect(text):
    try:
        return detect(text)
    except Exception as e:
        print(e)
        return None

df['lang'] = df.content.swifter.apply(lambda x: lang_detect(x[:100]))

df.lang.value_counts()

df.to_pickle(os.path.join('C:/Users/Jakob/Documents/lexisnexis_firm_alliances_combined_new.pkl')




