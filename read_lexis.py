import pandas as pd
import os
from ast import literal_eval
from langdetect import detect
import swifter
import matplotlib.pyplot as plt
import string
import re
import unidecode

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


###




## read Orbis firm name list
# path = 'C:/Users/Jakob/Documents/Orbis/Full/BvD_ID_and_Name.txt'
# orbis = pd.read_csv(path, sep='\t')
# # path = 'C:/Users/Jakob/Documents/Orbis/Full/Identifiers.txt'
# # identifiers = pd.read_csv(path, sep='\t', nrows=100)
#
# orbis['company'] = orbis.NAME.apply(str)
# orbis.drop(columns=['NAME'], inplace=True)

orbis_cols = ['bvdidnumber', 'companyname']
orbis = pd.read_csv('C:/Users/Jakob/Documents/Orbis/Full/BvD_ID_and_Name.txt', sep='\t')
orbis.columns = orbis_cols

orbis = orbis.append(pd.read_csv('C:/Users/Jakob/Downloads/Orbis_Web_US.csv', sep=';', usecols=orbis_cols))
orbis = orbis.append(pd.read_csv('C:/Users/Jakob/Downloads/All_25_BvDID_NAICS.csv', sep=';', usecols=orbis_cols))

orbis.drop_duplicates(inplace=True)


def firm_name_clean(firm_name, lower=True, remove_punc=True, remove_legal=True, remove_parentheses=True):
    # make string
    firm_name = str(firm_name)
    firm_name = unidecode.unidecode(firm_name)
    # lowercase
    if lower:
        firm_name = firm_name.lower()
    # remove punctuation
    if remove_punc:
        firm_name = firm_name.translate(str.maketrans('', '', '!"#$%\\\'*+,./:;<=>?@^_`{|}~'))
    # remove legal identifiers
    if remove_legal:
        legal_identifiers = ["co", "inc", "ag", "ltd", "lp", "llc", "pllc", "llp", "plc", "ltdplc", "corp",
                             "corporation", "ab", "cos", "cia", "sa", "company", "companies", "consolidated",
                             "stores", "limited", "srl", "kk", "gmbh", "pty", "group", "yk", "bhd",
                             "limitada", "holdings", "kg", "bv", "pte", "sas", "ilp", "nl", "trust",
                             "genossenschaft", "gesellschaft", "aktiengesellschaft", "ltda", "nv", "oao",
                             "holding", "se", "oy", "plcnv", "the"]
        pattern = '|'.join(legal_identifiers)
        pattern = '\\b(' + pattern + ')\\b' # match only word boundaries
        firm_name = re.sub(pattern, '', firm_name)
    # remove parentheses and anything in them: Bayerische Motoren Werke (BMW) -> Bayerische Motoren Werke
    if remove_parentheses:
        firm_name = re.sub(r'\([^()]*\)', '', firm_name)

    # strip
    firm_name = firm_name.strip()

    return firm_name

orbis['company'] = orbis.companyname.apply(firm_name_clean)
orbis = orbis.groupby('company').agg(lambda x: set(x))
orbis.reset_index(inplace=True)

def firm_name_matching(df, df_lookup, firm_name_col='company', clean_lookup=True):
    assert not df[firm_name_col].duplicated().any(), 'Firm names to match contain duplicates!'
    assert not df_lookup[firm_name_col].duplicated().any(), 'Lookup firm list contains duplicates!'

    df['match_col'] = df[firm_name_col].apply(firm_name_clean)
    if clean_lookup:
        df_lookup['match_col'] = df_lookup[firm_name_col].apply(firm_name_clean)
    else:
        df_lookup['match_col'] = df_lookup[firm_name_col]

    res = df.merge(df_lookup, on='match_col', how='left', indicator=True)
    print(f'Matched {(res._merge == "both").sum()/len(df)*100} percent of companies')

    # res = res.drop(columns=['_merge', 'match_col'])

    return res

# Testing on SDC
sdc = pd.read_pickle('C:/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')
sdc_name_cols = ['ParticipantsinVenture/Alliance(ShortName)', 'ParticipantUltimateParentName',
 'ParticipantsinVenture/Alliance(LongNamShortLine)', 'ParticipantsinVenture/Alliance(LongName3Lines)',
 'ParticipantParentName']

sdc['Parti.CUSIP'].explode().value_counts(dropna=False)

sdc['ParticipantsinVenture/Alliance(LongName3Lines)']

sdc_comps = sdc.ParticipantUltimateParentName.explode().value_counts()
sdc_comps = pd.DataFrame(sdc_comps)
sdc_comps.reset_index(inplace=True)
sdc_comps.columns = ['company', 'count']

res_sdc = firm_name_matching(sdc_comps, orbis, clean_lookup=False)
res_sdc_no_match = res_sdc[res_sdc._merge == 'left_only']


# Testing on LexisNexis (detected firms)
df = pd.read_pickle('C:/Users/Jakob/Documents/lexisnexis_firm_alliances_combined_new.pkl')

df = df.company.explode().value_counts()
df = pd.DataFrame(df)
df.reset_index(inplace=True)
df.columns = ['company', 'count']

res_lexis = firm_name_matching(df, orbis, clean_lookup=False)
res_lexis_no_match = res[res._merge == 'left_only']

orbis[orbis.company.str.contains('the')]

sdc_comps[sdc_comps.match_col.str.contains('international')].head(30)

# keep only docs with at least one sentence
df = df[df.content.str.split('. ').str.len() > 1]

# focus on news mentioning at least 2 companies
df = df[df.company.str.len() > 1]




# find most common legal identifiers
orbis_names = orbis.company.apply(firm_name_clean)
orbis_names = pd.Series(orbis_names.unique())
common_tokens = orbis_names.str.split(' ').explode().value_counts()
common_tokens.head(400).index.values


