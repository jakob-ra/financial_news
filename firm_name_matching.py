import pandas as pd
import swifter
import re
import unidecode

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

orbis = orbis.append(pd.read_csv('C:/Users/Jakob/Downloads/Orbis_Web_US.csv', sep=';', usecols=orbis_cols+['websiteaddress']))
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
                             "limitada", "holdings", "kg", "bv", "pte", "sas", "ilp", "nl",
                             "genossenschaft", "gesellschaft", "aktiengesellschaft", "ltda", "nv", "oao",
                             "holding", "se", "oy", "plcnv", "the", "neft", "& co", "&co"]
        pattern = '|'.join(legal_identifiers)
        pattern = '\\b(' + pattern + ')\\b' # match only word boundaries
        firm_name = re.sub(pattern, '', firm_name)
    # remove parentheses and anything in them: Bayerische Motoren Werke (BMW) -> Bayerische Motoren Werke
    if remove_parentheses:
        firm_name = re.sub(r'\([^()]*\)', '', firm_name)
        
    # make hyphens consistent
    firm_name = firm_name.replace(' - ', '-')

    # remove ampersand symbol
    firm_name = firm_name.replace('&amp;', '&')
    firm_name = firm_name.replace('&amp', '&')

    # strip
    firm_name = firm_name.strip()

    return firm_name


orbis['company'] = orbis.companyname.swifter.apply(firm_name_clean)
orbis = orbis.groupby('company').agg(list)
orbis.reset_index(inplace=True)

orbis.to_pickle('C:/Users/Jakob/Documents/Orbis/combined_firm_list.pkl')

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

## google lookup on residual
from google_match import google_KG_match
api_key = open('google_api_key.txt', 'r').read()

res_sdc_no_match_google = res_sdc_no_match.copy(deep=True)
res_sdc_no_match_google['google_res'] = res_sdc_no_match_google.company_x.apply(str).swifter.apply(lambda x: google_KG_match(x, api_key, type='Corporation'))


res_df = []
for index, row in res_sdc_no_match_google.iterrows():
    google_res = row.google_res
    if google_res != None:
        if 'Corporation' in google_res['result']['@type']: #google_res['resultScore'] > 100 and
            res = pd.json_normalize(google_res)
            row = pd.DataFrame(row).T
            res.index = row.index

            res_df.append(res)


res_df = pd.concat(res_df)
res_df = res_sdc_no_match_google.merge(res_df, left_index=True, right_index=True, how='left')
res_df = res_df[['company_x', 'result.name', 'result.url', 'result.description', 'resultScore']]
res_df['company'] = res_df['result.name']
res_df.dropna(subset=['company'], inplace=True)
res_df = res_df[res_df.company.str.len() > 2]
res_df = res_df.groupby(res_df.company).agg(lambda x: set(x)).reset_index()

res_df = firm_name_matching(res_df, orbis, clean_lookup=False)
res_df = res_df[res_df._merge == 'both']

res_sdc_no_match_google = pd.concat([pd.concat([row, json_normalize(row.google_res)], axis=1, ignore_index=True) for index,row in res_sdc_no_match_google.iterrows()])
# res_sdc_no_match_google = res_sdc_no_match_google.merge(res_sdc_no_match, left_index=True, right_index=True)
res_sdc_no_match_google = res_sdc_no_match_google[['company_x', 'result.name', 'result.url', 'result.description', 'resultScore']]

pd.json_normalize(res_sdc_no_match_google.google_res.iloc[0])




# Testing on LexisNexis (detected firms)
df = pd.read_pickle('C:/Users/Jakob/Documents/lexisnexis_firm_alliances_combined_new.pkl')

lexis_firms = df.company.explode().value_counts()
lexis_firms = pd.DataFrame(lexis_firms)
lexis_firms.reset_index(inplace=True)
lexis_firms.columns = ['company', 'count']

res_lexis = firm_name_matching(lexis_firms, orbis, clean_lookup=False)
res_lexis = res_lexis[~ res_lexis.company_y.isin(['', 'joint venture electronics', 'european commission', 'news', 'i'])]


def agg_lexis(x):
    res = set()
    for elem in x:
        if type(elem) == set:
           res = res | elem
        else:
            res.add(elem)
    # if len(res) == 1:
    #     res = list(res)[0]

    return res


res_lexis_group = res_lexis.groupby(res_lexis.company_y).agg(agg_lexis).reset_index()
res_lexis_group['count'] = res_lexis_group['count'].apply(lambda x: sum(x) if type(x) == set else x)
res_lexis_group.sort_values(by='count', ascending=False, inplace=True)

res_lexis_group = res_lexis_group[['company_y', 'bvdidnumber', 'count']].explode('bvdidnumber')

res_lexis_group.drop(columns=['count'], inplace=True)
res_lexis_group.columns = ['company', 'bvdid']

# find firms with most bvdids
res_lexis_group.groupby('company').size().sort_values(ascending=False).head(100).index

res_lexis_group.to_csv('C:/Users/Jakob/Documents/lexis_firms_matched_orbis.csv', index=False)


from flashtext import KeywordProcessor
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(res_lexis_group.company_y)

for content in df.content.sample(10):
    print(content)
    print(keyword_processor.extract_keywords(content, span_info=True))

firms_flashtext = df.content.sample(100000).apply(keyword_processor.extract_keywords)
firms_flashtext = firms_flashtext.explode()
firms_flashtext = firms_flashtext.value_counts()
firms_flashtext = firms_flashtext.reset_index()
firms_flashtext.columns = ['name', 'freq']
freq_names = firms_flashtext[firms_flashtext.freq > 1000]
freq_names = freq_names[~freq_names.name.isin(res_lexis_group[res_lexis_group['count'] > 500].company_y)]
freq_names[freq_names.name.isin(res_lexis_group[res_lexis_group['count'] > 500].company_y)].values
res_lexis_group[res_lexis_group.company_y == 'total'].values




# COMPUSTAT
compustat_global = pd.read_csv('C:/Users/Jakob/Documents/compustat-global.csv.gz')
compustat_na = pd.read_csv('C:/Users/Jakob/Documents/compustat-north-america.csv.gz')
compustat_na = compustat_na[['gvkey', 'fyear', 'datadate', 'emp', 'revt', 'conm', 'conml', 'cusip', 'loc', 'naics', 'weburl']]
compustat = compustat_na.append(compustat_global)


compustat.dropna(subset=['emp'], inplace=True)
compustat = compustat.sort_values('fyear').groupby('gvkey').tail(1) # take latest
compustat.sort_values('emp', inplace=True, ascending=False)
compustat['company'] = compustat.conml
compustat.drop_duplicates('company', keep='first', inplace=True)

firm_name_matching(compustat, orbis, clean_lookup=False)


## google lookup on residual
from google_match import google_KG_match
from tqdm import tqdm
tqdm.pandas()

api_key = open('google_api_key.txt', 'r').read()

res_lexis_no_match_google = res_lexis_no_match.head(1000).copy(deep=True)
res_lexis_no_match_google['google_res'] = res_lexis_no_match_google.company_x.apply(str).progress_apply(lambda x: google_KG_match(x, api_key, type='Corporation'))

res_df = []
for index, row in res_lexis_no_match_google.iterrows():
    google_res = row.google_res
    if google_res != None:
        if 'Corporation' in google_res['result']['@type']: #google_res['resultScore'] > 100 and
            res = pd.json_normalize(google_res)
            row = pd.DataFrame(row).T
            res.index = row.index

            res_df.append(res)


res_df = pd.concat(res_df)
res_df = res_lexis_no_match_google.merge(res_df, left_index=True, right_index=True, how='left')
res_df = res_df[['company_x', 'result.name', 'count', 'result.url', 'result.description', 'result.@type', 'resultScore']]
res_df['company'] = res_df['result.name']
res_df.dropna(subset=['resultScore'], inplace=True)
res_df.rename(columns={'company_x': 'original_names'}, inplace=True)

res_df = res_df.groupby(res_df.company).agg(lambda x: set(x) if len(set(x)) > 1 else x).reset_index()

# sort by count
res_df['count'] = res_df['count'].apply(lambda x: sum(x) if type(x) == set else x)
res_df.sort_values(by='count', ascending=False).iloc[0]
res_df.iloc[0]

res_df_matching = firm_name_matching(res_df, orbis, clean_lookup=False)




res_df = res_df[res_df._merge == 'both']





res_lexis_no_match[res_lexis_no_match.company_x.str.contains('AT&amp')].match_col.values
res_lexis_no_match.columns

orbis[orbis.company.str.contains('peugot')].company.values

sdc_comps[sdc_comps.match_col.str.contains('international')].head(30)

# keep only year 2017
df = df[df.publication_date.dt.year == 2017]

# keep only english
df = df[df.lang == 'en']

# keep only docs with at least one sentence
df = df[df.content.str.split('. ').str.len() > 1]

# focus on news mentioning at least 2 companies
df = df[df.company.str.len() > 1]




# find most common legal identifiers
orbis_names = orbis.company.apply(firm_name_clean)
orbis_names = pd.Series(orbis_names.unique())
common_tokens = orbis_names.str.split(' ').explode().value_counts()
common_tokens.head(400).index.values


df = pd.read_pickle('C:/Users/Jakob/Documents/lexisnexis_firm_alliances_combined_new.pkl')

