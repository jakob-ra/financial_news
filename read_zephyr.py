import pandas as pd
import os
import numpy as np
from firm_name_matching import firm_name_clean
import country_converter as coco

# read all files in folder and concat
path = 'C:/Users/Jakob/Documents/Zephyr M&A'
files = os.listdir(path)

date_cols = ['Last deal status date', 'Announced date', 'Completed date']

chunks = []
for i, file in enumerate(files):
    chunk = pd.read_excel(os.path.join(path, file), sheet_name='Results')
    chunk[date_cols] = chunk[date_cols].apply(lambda x: pd.to_datetime(x, errors='coerce', unit='d', origin='1899-12-30'))
    chunks.append(chunk)

df = pd.concat(chunks)
df.columns = [c.lower().replace('\n', ' ').strip().replace(' ', '_').replace('(%)', 'percent') for c in df.columns]
date_cols = [c.lower().replace(' ', '_') for c in date_cols]
df.drop(columns=['unnamed:_0'], inplace=True)
df.sort_values('announced_date', inplace=True)
df.reset_index(drop=True, inplace=True)

# clean final and acquired stake
df.final_stake_percent = df.final_stake_percent.str.split('\n').str[0].str.replace('%', '').str.strip()
df.final_stake_percent = df.final_stake_percent.str.split('.').str[0].str.strip()
df.drop(df[df.final_stake_percent.isin(['Unknown', 'Unknown minority'])].index, inplace=True)
df.final_stake_percent.value_counts().head(50)
df.acquired_stake_percent = df.acquired_stake_percent.str.split('\n').str[0].str.replace('%', '').str.split('.').str[0].str.strip()
df.acquired_stake_percent.value_counts().head(50)

df.to_pickle('C:/Users/Jakob/Documents/Zephyr M&A/Full/zephyr_m_a_full.pkl')

df = pd.read_pickle('C:/Users/Jakob/Documents/Zephyr M&A/Full/zephyr_m_a_full.pkl')
df.deal_status = df.deal_status.astype(pd.CategoricalDtype())
df.acquiror_country_code = df.acquiror_country_code.str.split('\n').str[0]
df.target_country_code = df.target_country_code.str.split('\n').str[0]
df.acquiror_name = df.acquiror_name.str.split('\n').str[0]
df.target_name = df.target_name.str.split('\n').str[0]

# to-do: filter for acquisitions where acquired stake pushes final stake above 50%

df.deal_status.value_counts().head(50)
df.deal_type.value_counts().head(50) # low number of mergers
df[df.acquiror_isin_number.notnull() & df.target_isin_number.notnull()]
df[df.acquiror_bvd_id_number.notnull() & df.target_bvd_id_number.notnull()]
df[df.acquiror_guo_bvd_number.notnull() & df.target_guo_bvd_number.notnull()]




# match with compustat global via ISIN
compustat_global_sedol_isin = pd.read_excel('C:/Users/Jakob/Documents/compustat-global-sedol-isin.xlsx')
compustat_global_sedol_isin.columns = ['gvkey', 'fyear', 'datadate', 'isin', 'sedol']
compustat_global_sedol_isin.dropna(subset=['isin', 'sedol'], inplace=True)
compustat_global_sedol_isin.drop_duplicates(subset=['gvkey', 'isin', 'sedol'], keep='first', inplace=True)
gvkey_isin = compustat_global_sedol_isin[['gvkey', 'isin']].dropna()
gvkey_isin.columns = ['acquiror_gvkey', 'acquiror_isin_number']
df = df.merge(gvkey_isin, on='acquiror_isin_number', how='left')
gvkey_isin.columns = ['target_gvkey', 'target_isin_number']
df = df.merge(gvkey_isin, on='target_isin_number', how='left')

# match with compustat north america via CUSIP
compustat_na = pd.read_csv('C:/Users/Jakob/Documents/compustat-north-america.csv.gz')
compustat_na = compustat_na[
    ['gvkey', 'fyear', 'datadate', 'emp', 'revt', 'conm', 'conml', 'cusip', 'loc', 'naics', 'weburl']]
cusip_to_gvkey = compustat_na.sort_values('emp', ascending=False)[['cusip', 'gvkey']].dropna()
cusip_to_gvkey['cusip'] = cusip_to_gvkey.cusip.astype(str).str[:6]
cusip_to_gvkey = cusip_to_gvkey.drop_duplicates(subset='cusip', keep='first').set_index('cusip').squeeze()
df['acquiror_cusip'] = df.acquiror_isin_number.str[2:8]
df['target_cusip'] = df.target_isin_number.str[2:8]
df.loc[~df.acquiror_isin_number.astype(str).str.startswith(('US', 'CA')), 'acquiror_cusip'] = np.nan
df.loc[~df.target_isin_number.astype(str).str.startswith(('US', 'CA')), 'target_cusip'] = np.nan
df['acquiror_gvkey_from_cusip'] = df.acquiror_cusip.map(cusip_to_gvkey)
df['target_gvkey_from_cusip'] = df.target_cusip.map(cusip_to_gvkey)
df.acquiror_gvkey.fillna(df.acquiror_gvkey_from_cusip, inplace=True)
df.target_gvkey.fillna(df.target_gvkey_from_cusip, inplace=True)
df.drop(columns=['acquiror_gvkey_from_cusip', 'target_gvkey_from_cusip'], inplace=True)

# name matching (including country)
country_dict = pd.concat([df.acquiror_country_code, df.target_country_code]).value_counts().to_frame()
country_dict['acquiror_country_code'] = coco.convert(country_dict.index, to='ISO3')
country_dict = country_dict[country_dict.acquiror_country_code!='not found'].acquiror_country_code.squeeze()
df.acquiror_country_code = df.acquiror_country_code.map(country_dict).replace(pd.NA, '').astype(str)
df['acquiror_matching_name'] = df.acquiror_name + ' ' + df.acquiror_country_code
df.target_country_code = df.target_country_code.map(country_dict).replace(pd.NA, '').astype(str)
df['target_matching_name'] = df.target_name + ' ' + df.target_country_code

names = pd.concat([df.acquiror_matching_name.value_counts().reset_index()['index'], df.target_matching_name.value_counts().reset_index()['index']])
names = names.drop_duplicates()
names = names.to_frame('name')

compustat_global = pd.read_csv('C:/Users/Jakob/Documents/compustat-global.csv.gz')
compustat_na = pd.read_csv('C:/Users/Jakob/Documents/compustat-north-america.csv.gz')
compustat_na = compustat_na[
    ['gvkey', 'fyear', 'datadate', 'emp', 'revt', 'conm', 'conml', 'cusip', 'loc', 'naics', 'weburl']]
compustat = compustat_na.append(compustat_global)
del compustat_na, compustat_global

compustat.dropna(subset=['emp'], inplace=True)
compustat = compustat.sort_values('fyear').groupby('gvkey').tail(1)  # take latest
compustat.sort_values('emp', inplace=True, ascending=False)
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
df['acquiror_gvkey'] = df.acquiror_gvkey.fillna(df.AcquirorGvkeyFromName)
df['target_gvkey'] = df.target_gvkey.fillna(df.TargetGvkeyFromName)
df.drop(columns=['AcquirorNameCompustat', 'TargetNameCompustat', 'AcquirorGvkeyFromName', 'TargetGvkeyFromName'], inplace=True)
df.drop(columns=['acquiror_matching_name', 'target_matching_name'], inplace=True)
df.drop(columns=['acquiror_cusip', 'target_cusip'], inplace=True)

df_matched = df[df.acquiror_gvkey.notnull() & df.target_gvkey.notnull()].copy()
df_matched[['acquiror_gvkey', 'target_gvkey']] = df_matched[['acquiror_gvkey', 'target_gvkey']].astype(int)
df_matched = df_matched[df_matched.acquiror_isin_number != df_matched.target_isin_number].copy()

df_matched.to_csv('C:/Users/Jakob/Downloads/zephyr_m_a_matched_compustat_via_isin_and_name.csv', index=False)



# match with Orbis financials
bvdids_m_a = df_matched[['acquiror_bvd_id_number', 'target_bvd_id_number']].stack().unique()
bvdids_m_a = pd.Series(bvdids_m_a, name='bvdid')
bvdids_m_a.to_csv('C:/Users/Jakob/Downloads/bvdids_zephyr_m_a.csv', index=False)

# upload to S3
imp