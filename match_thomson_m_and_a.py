import pandas as pd
import os
from datetime import timedelta
from firm_name_matching import firm_name_clean
import matplotlib.pyplot as plt
import country_converter as coco

# read all files in folder and concat
path = 'C:/Users/Jakob/Downloads/Thomson SDC Platinum M&A (Extract 24-11-2022)'
files = os.listdir(path)

chunks = []
for i, file in enumerate(files):
    chunk = pd.read_excel(os.path.join(path, file), skiprows=1, converters={'DateAnnounced': lambda x: str(x)})
    chunk.columns = [c.replace('\n', ' ').strip().replace(' ', '') for c in chunk.columns]
    chunks.append(chunk)

df = pd.concat(chunks)

date_columns = ['DateAnnounced']
for col in date_columns:
    df[col] = df[col].astype('str')
    df[col] = df[col].str.replace(' 00:00:00', '')
    df[col] = pd.to_datetime(df[col], errors='coerce')
    # some dates are parsed as wrong century
    df.loc[df[col] > pd.Timestamp(2050, 1, 1), col] -= timedelta(
        days=365.25 * 100)
    df[col] = df[col].dt.date

df.Form.replace('Acq. Maj. Int.', 'Acquisition', inplace=True)
df.sort_values('DateAnnounced', inplace=True)
df.to_pickle('C:/Users/Jakob/Downloads/thomson_m_and_a_clean.pkl')
df = pd.read_pickle('C:/Users/Jakob/Downloads/thomson_m_and_a_clean.pkl')

# compustat match via github EWENS
# sdc_compustat_cusip_gvkey = pd.read_csv('https://github.com/michaelewens/SDC-to-Compustat-Mapping/raw/master/sdc_compustat_cusip_gvkey.csv')
# sdc_compustat_cusip_gvkey = sdc_compustat_cusip_gvkey[['sdc_cusip', 'gvkey_n']].copy()
# sdc_compustat_cusip_gvkey.columns = ['cusip', 'acquiror_gvkey']
# df = df.merge(sdc_compustat_cusip_gvkey, left_on='AcquirorCUSIP', right_on='cusip', how='left')
# sdc_compustat_cusip_gvkey.columns = ['cusip', 'target_gvkey']
# df = df.merge(sdc_compustat_cusip_gvkey, left_on='TargetCUSIP', right_on='cusip', how='left')
# df[df.acquiror_gvkey.notnull() & df.target_gvkey.notnull()]
sdc_compustat_mapping = pd.read_csv('https://github.com/michaelewens/SDC-to-Compustat-Mapping/raw/master/dealnum_to_gvkey.csv').drop(columns=['updated2024'])
df = df.merge(sdc_compustat_mapping, on='DealNumber', how='left', indicator=False)

# compustat match via cusip
cusip_gvkey = pd.read_csv('/Users/Jakob/Downloads/ozka5ehtfm5g81cl.csv.gz', compression='gzip')
cusip_gvkey = cusip_gvkey[['gvkey', 'cusip']].copy()
cusip_gvkey['cusip'] = cusip_gvkey['cusip'].str[:6]
cusip_gvkey.drop_duplicates(subset=['gvkey', 'cusip'], inplace=True)
cusip_gvkey.columns = ['acquiror_gvkey', 'AcquirorCUSIP']
df = df.merge(cusip_gvkey, on='AcquirorCUSIP', how='left')
cusip_gvkey.columns = ['target_gvkey', 'TargetCUSIP']
df = df.merge(cusip_gvkey, on='TargetCUSIP', how='left')
df.agvkey = df.agvkey.fillna(df.acquiror_gvkey)
df.tgvkey = df.tgvkey.fillna(df.target_gvkey)
df.drop(columns=['acquiror_gvkey', 'target_gvkey'], inplace=True)

# match with compustat
compustat_global = pd.read_csv('C:/Users/Jakob/Documents/compustat-global.csv.gz')
compustat_na = pd.read_csv('C:/Users/Jakob/Documents/compustat-north-america.csv.gz')
compustat_na = compustat_na[
    ['gvkey', 'fyear', 'datadate', 'emp', 'revt', 'conm', 'conml', 'cusip', 'loc', 'naics', 'weburl']]
compustat = compustat_na.append(compustat_global)
del compustat_na, compustat_global

compustat.dropna(subset=['emp'], inplace=True)
compustat = compustat.sort_values('fyear').groupby('gvkey').tail(1)  # take latest
compustat.sort_values('emp', inplace=True, ascending=False)
compustat['company'] = compustat.conml
compustat.drop_duplicates('company', keep='first', inplace=True)

compustat_global_sedol_isin = pd.read_excel('C:/Users/Jakob/Documents/compustat-global-sedol-isin.xlsx')
compustat_global_sedol_isin.columns = ['gvkey', 'fyear', 'datadate', 'isin', 'sedol']
compustat_global_sedol_isin.dropna(subset=['isin', 'sedol'], inplace=True)
compustat_global_sedol_isin.drop_duplicates(subset=['gvkey', 'isin', 'sedol'], keep='first', inplace=True)
# this was to try merging via isin and cusip
# compustat_global_sedol_isin['isin_short'] = compustat_global_sedol_isin['isin'].str[2:8]
# to_merge = compustat_global_sedol_isin[['isin_short', 'gvkey']]
# to_merge.columns = ['AcquirorCUSIP', 'AcquirorGvkey']
# test = df.merge(to_merge, on='AcquirorCUSIP', how='left').dropna(subset=['AcquirorGvkey'])
# df[df.AcquirorCUSIP.str[:6].isin(compustat_global_sedol_isin['isin'].astype(str).str[2:8])]

cusip_to_gvkey = compustat.sort_values('emp', ascending=False)[['cusip', 'gvkey']].dropna()
cusip_to_gvkey['cusip'] = cusip_to_gvkey.cusip.astype(str).str[:6]
cusip_to_gvkey = cusip_to_gvkey.drop_duplicates(subset='cusip', keep='first').set_index('cusip').squeeze()
df['TargetGvkey'] = df.TargetCUSIP.astype(str).str[:6].map(cusip_to_gvkey)
df['AcquirorGvkey'] = df.AcquirorCUSIP.astype(str).str[:6].map(cusip_to_gvkey)
df.agvkey = df.agvkey.fillna(df.TargetGvkey)
df.tgvkey = df.tgvkey.fillna(df.AcquirorGvkey)
df.drop(columns=['TargetGvkey', 'AcquirorGvkey'], inplace=True)
df[df.agvkey.notnull() & df.tgvkey.notnull()]

cusip_to_gvkey = pd.concat([df[['AcquirorCUSIP', 'agvkey']].rename(columns={'AcquirorCUSIP': 'cusip', 'agvkey': 'gvkey'}),
                            df[['TargetCUSIP', 'tgvkey']].rename(columns={'TargetCUSIP': 'cusip', 'tgvkey': 'gvkey'})]).dropna().drop_duplicates()
cusip_to_gvkey = cusip_to_gvkey.drop_duplicates(subset='cusip', keep='last').set_index('cusip').squeeze()
df['TargetGvkey'] = df.TargetCUSIP.astype(str).str[:6].map(cusip_to_gvkey)
df['AcquirorGvkey'] = df.AcquirorCUSIP.astype(str).str[:6].map(cusip_to_gvkey)
df.agvkey = df.agvkey.fillna(df.TargetGvkey)
df.tgvkey = df.tgvkey.fillna(df.AcquirorGvkey)
df.drop(columns=['TargetGvkey', 'AcquirorGvkey'], inplace=True)
df.rename(columns={'agvkey': 'AcquirorGvkey', 'tgvkey': 'TargetGvkey'}, inplace=True)
df[df.AcquirorGvkey.notnull() & df.TargetGvkey.notnull()]

# name matching (including country)
country_dict = pd.concat([df.AcquirorNation, df.TargetNation]).value_counts().to_frame()
country_dict['acquiror_country_code'] = coco.convert(country_dict.index, to='ISO3')
country_dict = country_dict[country_dict.acquiror_country_code!='not found'].acquiror_country_code.squeeze()
df['acquiror_country_code'] = df.AcquirorNation.map(country_dict).replace(pd.NA, '').astype(str)
df['acquiror_matching_name'] = df.AcquirorName.astype(str) + ' ' + df.acquiror_country_code
df['target_country_code'] = df.TargetNation.map(country_dict).replace(pd.NA, '').astype(str)
df['target_matching_name'] = df.TargetName.astype(str) + ' ' + df.target_country_code

names = pd.concat([df.acquiror_matching_name.value_counts().reset_index()['index'], df.target_matching_name.value_counts().reset_index()['index']])
names = names.drop_duplicates()
names = names.to_frame('name')

compustat['company_matching_name'] = compustat.conml + ' ' + compustat['loc']
compustat.drop_duplicates('company_matching_name', keep='first', inplace=True)

names_compustat = compustat.sort_values('emp', ascending=False).company_matching_name.drop_duplicates()
names_compustat = names_compustat.to_frame('name')

names['cleaned_name'] = names.name.apply(firm_name_clean)
names_compustat['cleaned_name'] = names_compustat.name.apply(firm_name_clean)
names = names[names.cleaned_name.str.len() > 1]
names_compustat.drop_duplicates('cleaned_name', inplace=True)

merged = names.merge(names_compustat, how='inner', on='cleaned_name', suffixes=['', '_compustat'])
merged.drop(columns=['cleaned_name'], inplace=True)
merged = merged.set_index('name').squeeze()

df['AcquirorNameCompustat'] = df.acquiror_matching_name.map(merged)
df['TargetNameCompustat'] = df.target_matching_name.map(merged)
name_to_gvkey = compustat.sort_values('emp', ascending=False)[['company_matching_name', 'gvkey']].drop_duplicates('company_matching_name', keep='first').dropna().set_index('company_matching_name').squeeze()
df['AcquirorGvkeyFromName'] = df.AcquirorNameCompustat.map(name_to_gvkey)
df['TargetGvkeyFromName'] = df.TargetNameCompustat.map(name_to_gvkey)
df['AcquirorGvkey'] = df.AcquirorGvkey.fillna(df.AcquirorGvkeyFromName)
df['TargetGvkey'] = df.TargetGvkey.fillna(df.TargetGvkeyFromName)
df[df.AcquirorGvkey.notnull() & df.TargetGvkey.notnull()]
df.drop(columns=['AcquirorNameCompustat', 'TargetNameCompustat', 'AcquirorGvkeyFromName', 'TargetGvkeyFromName'], inplace=True)
df.drop(columns=['acquiror_matching_name', 'target_matching_name'], inplace=True)
df.drop(columns=['acquiror_country_code', 'target_country_code'], inplace=True)

df_matched = df.dropna(subset=['TargetGvkey', 'AcquirorGvkey']).copy()
df_matched[['TargetGvkey', 'AcquirorGvkey']] = df_matched[['TargetGvkey', 'AcquirorGvkey']].astype(int)
df_matched.to_csv('C:/Users/Jakob/Downloads/Thomson_SDC_MandA_matched_Compustat.csv', index=False)




#
# df = pd.read_csv('C:/Users/Jakob/Downloads/Thomson_MandA_matched_Compustat.csv')
#
# ids = df[['TargetGvkey', 'AcquirorGvkey']].stack().unique()
# compustat_in_m_and_a = compustat[compustat.gvkey.isin(ids)]
#
#
# ids = lexis_nexis_r_and_d[['firm_a', 'firm_b']].stack().unique()
# test = df[df.AcquirorGvkey.astype(str).isin(ids) & df.TargetGvkey.astype(str).isin(ids)]

# from name_matching.name_matcher import NameMatcher
# import ray
# import numpy as np
# ray.init()
#
# matcher = NameMatcher(ngrams=(2, 5),
#                         top_n=10,
#                         number_of_rows=500,
#                         number_of_matches=1,
#                         lowercase=True,
#                         punctuations=True,
#                         remove_ascii=True,
#                         legal_suffixes=True,
#                         common_words=False,
#                         preprocess_split=False,
#                         verbose=True)
# matcher.set_distance_metrics(['iterative_sub_string', 'editex']) #, 'bag', 'fuzzy_wuzzy_partial_string', 'editex'])
# matcher.load_and_process_master_data('company', names_compustat, transform=True)
#
# @ray.remote
# def match_name_parallel(names, matcher):
#     results = matcher.match_names(to_be_matched=names, column_matching='company')
#     return results
#
# results = []
# for chunk in np.array_split(names, os.cpu_count()-1):
#     results.append(match_name_parallel.remote(chunk, matcher))
#
# matches = pd.concat(ray.get(results))
# good_matches = matches[matches.score > 95]

names.company.str.split().explode().value_counts().iloc[50:100]
names_compustat.company.str.split().explode().value_counts().iloc[50:100]
good_matches.match_index.max()
complete_matched_data = pd.merge(pd.merge(test_names, matches, how='left', right_index=True, left_index=True), adjusted_names, how='left', left_on='match_index_0', right_index=True, suffixes=['', '_matched'])




# match with lexis
df_static = pd.read_pickle('C:/Users/Jakob/Downloads/matching_result_lexis_orbis2023_compustat.pkl')
df_dynamic = pd.read_csv('C:/Users/Jakob/Downloads/lexis_match_orbis2023_compustat_dynamic.csv')

df_dynamic.added_value.count()
df_dynamic.research_and_development_expenses.count()

df_static[(df_static.country_iso_code.str.startswith('CHE')) & (df_static.ussic_primary_code.astype(str).str.startswith('283'))]

cleaned_names_bvdids = df_static[['cleaned_name', 'bvdid']]
cleaned_names_bvdids = cleaned_names_bvdids[cleaned_names_bvdids.bvdid.isin(lexis_nexis_r_and_d.bvdid)]

df['TargetCleanedName'] = df['TargetName'].apply(firm_name_clean)
df['AcquirorCleanedName'] = df['AcquirorName'].apply(firm_name_clean)

df = df.merge(cleaned_names_bvdids.rename(columns={'cleaned_name': 'TargetCleanedName', 'bvdid': 'TargetBvDID'}),
            on='TargetCleanedName', how='left')
df = df.merge(cleaned_names_bvdids.rename(columns={'cleaned_name': 'AcquirorCleanedName', 'bvdid': 'AcquirorBvDID'}),
            on='AcquirorCleanedName', how='left')
df.dropna(subset=['TargetBvDID', 'AcquirorBvDID'], how='any', inplace=True)
df.drop(columns=['TargetCleanedName', 'AcquirorCleanedName'], inplace=True)

df.to_csv('C:/Users/Jakob/Downloads/ResearchandDevelopment_matched_Thomson_MandA.csv', index=False)

# plot number of mergers and acquisitions separately in each year
df['year'] = df.DateAnnounced.apply(lambda x: x.year)
df[df.Form == 'Acquisition'].groupby('year').size().plot(label='Acquisitions')
df[df.Form == 'Merger'].groupby('year').size().plot(label='Merger')
plt.xlabel('')
plt.ylabel('Number of deals')
# add total number of acquisitions and mergers to legend
total_acquisitions = df[df.Form == 'Acquisition'].shape[0]
total_mergers = df[df.Form == 'Merger'].shape[0]
plt.legend()
legend = plt.gca().get_legend()
legend.texts[0].set_text(f'Acquisitions ({total_acquisitions} total)')
legend.texts[1].set_text(f'Mergers ({total_mergers} total)')
plt.show()

# df.pivot_table(index='year', columns='Form', values='AcquirorBvDID', aggfunc='count').plot(kind='bar', stacked=False)
# plt.show()

master_file = pd.read_csv('C:/Users/Jakob/Downloads/id_bvd_compustat_id_name_sic_nace_country_country_code_city_loc_x_loc_y_compustat_source.csv', sep=';')
m_and_a = pd.read_csv('C:/Users/Jakob/Downloads/M&As.csv')
unique_ids = pd.DataFrame(m_and_a[['target_bvd_compustat_id', 'acquiror_bvd_compustat_id']].stack().unique(), columns=['id']).set_index('id')
unique_ids.join(master_file.set_index(' bvd_compustat_id'), how='inner')

unique_ids.sample(20).index
master_file[' bvd_compustat_id'] = master_file[' bvd_compustat_id'].astype(str).str.strip()


lexis_nexis_r_and_d = pd.read_csv("C:/Users/Jakob/Documents/financial_news_data/output/lexis_preds_2023/rel_database/ResearchandDevelopment_LexisNexis.csv")
unique_r_and_d_ids = pd.DataFrame(lexis_nexis_r_and_d[['firm_a', 'firm_b']].stack().unique(), columns=['id']).set_index('id')
df_static = pd.read_csv('C:/Users/Jakob/Downloads/lexis_match_orbis2023_compustat_static.csv')
ids_in_lexis_nexis_r_and_d = lexis_nexis_r_and_d[['firm_a', 'firm_b']].stack().unique()
df_static[df_static.ID.isin(ids_in_lexis_nexis_r_and_d)][['ID', 'website_address']].to_csv('C:/Users/Jakob/Downloads/firms_r_and_d_network_urls.csv', index=False)
df_static = df_static[df_static.ID.isin(ids_in_lexis_nexis_r_and_d)].copy()
df_static = df_static[df_static.country_iso_code == 'USA'].copy()
df_static[df_static.from_compustat.astype(bool)]


df.TargetCUSIP = df.TargetCUSIP.astype(str)
df_dupl_targetcusip = df[df.TargetCUSIP.duplicated(keep=False)].sort_values(['TargetCUSIP', 'DateAnnounced'])
df_dupl_targetcusip = df_dupl_targetcusip[['DateAnnounced', 'Form', 'TargetCUSIP', 'TargetName', 'AcquirorCUSIP', 'AcquirorName']]
df_dupl_targetcusip[df_dupl_targetcusip.AcquirorCUSIP == '43065T']


df.sort_values('DateAnnounced').drop_duplicates('TargetCUSIP', keep='first')



