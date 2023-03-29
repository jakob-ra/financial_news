import pandas as pd
from name_matching.name_matcher import NameMatcher
import re

matcher = NameMatcher(low_memory=False, distance_metrics=['discounted_levenshtein'])

sdc = sdc[['AcquirorName']].copy(deep=True)
sdc.drop_duplicates(inplace=True)

matcher.load_and_process_master_data('AcquirorName', sdc)
res = matcher.match_names(to_be_matched=orbis.head(10000), column_matching='companyname')

complete_matched_data = pd.merge(pd.merge(sdc, matches, how='left', right_index=True, left_index=True), adjusted_names, how='left', left_on='match_index_0', right_index=True, suffixes=['', '_matched'])


df = pd.read_csv('C:/Users/Jakob/Downloads/Industry-Global_financials_and_ratios/Industry-Global_financials_and_ratios/Industry-Global_financials_and_ratios.txt',
            sep='\t', nrows=1000)

df['number_of_nonempty_columns'] = df.count(axis=1)

df.sort_values('number_of_nonempty_columns', ascending=False, inplace=True)

df.to_csv('C:/Users/Jakob/Downloads/Industry-Global_financials_and_ratios/Industry-Global_financials_and_ratios/Industry-Global_financials_and_ratios_sample.txt', sep='\t', index=False)

df = pd.read_csv('C:/Users/Jakob/Downloads/Key_financials-USD/Key_financials-USD/Key_financials-USD.txt',
            sep='\t', nrows=1000)

df.to_csv('C:/Users/Jakob/Downloads/Key_financials-USD_sample.txt', sep='\t', index=False)

# define a function to make column names compatible with sql
def make_sql_compatible(column_name):

    # remove non-ascci characters
    column_name = column_name.encode('ascii', 'ignore').decode('ascii')

    # replace '(%)' with 'percent'
    column_name = re.sub(r'\(\%\)', 'percent', column_name)

    # replace th with thousands
    column_name = re.sub(r'\(th\)', 'thousands', column_name)

    # replace '/' with 'over
    column_name = re.sub(r'/', 'over', column_name)

    # replace '&' with 'and'
    column_name = re.sub(r'&', 'and', column_name)

    # remove non-alphanumeric characters except for underscore and space
    column_name = re.sub(r'[^a-zA-Z0-9_ ]', '', column_name)

    return column_name.replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '').lower().strip()

print(', \n'.join([make_sql_compatible(col) + ' ' + replace_dict[str(coltype)] for col, coltype in zip(df.columns, df.dtypes)]))


tls201 = pd.read_csv('C:/Users/Jakob/Downloads/patstat_global (1)/tls201_part01.csv', nrows=1000)

orbis_patstat = pd.read_csv('C:/Users/Jakob/Downloads/orbis_patents.csv')

print(', \n'.join([make_sql_compatible(col) + ' ' + replace_dict[str(coltype)] for col, coltype in zip(orbis_patstat.columns, orbis_patstat.dtypes)]))

orbis_patstat.country.value_counts()

orbis_patstat.patentpublicationnumber.str.split('(').str[1].str.split(')').str[0].value_counts()

orbis_patstat.patentpublicationnumber = orbis_patstat.patentpublicationnumber.str.split('(').str[0]

orbis_patstat = orbis_patstat[['patentpublicationnumber', 'currentownersbvdid']].copy(deep=True)
orbis_patstat.dropna(inplace=True)
orbis_patstat.drop_duplicates(inplace=True)

orbis_patstat.to_parquet('C:/Users/Jakob/Downloads/orbis_patstat.parquet', index=False)

# replace type names
replace_dict = {'int64': 'integer', 'float64': 'integer', 'object': 'string', 'datetime64[ns]': 'date'}

', '.join([col + ' ' + replace_dict[str(coltype)] for col, coltype in zip(tls201.columns, tls201.dtypes)])


tls201.to_csv('C:/Users/Jakob/Downloads/tls201_part01_sample.csv', index=False)

tls207 = pd.read_csv('C:/Users/Jakob/Downloads/tls207_part01.csv', nrows=1000)

tls207.to_csv('C:/Users/Jakob/Downloads/tls207_part01_sample.csv', index=False)

', '.join([col + ' ' + replace_dict[str(coltype)] for col, coltype in zip(tls207.columns, tls207.dtypes)])


tls206 = pd.read_csv('C:/Users/Jakob/Downloads/patstat_global/tls206_part01.csv', nrows=1000)

tls206.to_csv('C:/Users/Jakob/Downloads/tls206_part01_sample.csv', index=False)

', '.join([col + ' ' + replace_dict[str(coltype)] for col, coltype in zip(tls206.columns, tls206.dtypes)])








SELECT DISTINCT a.appin_id, appin_nr, publn_nr, appin_filing_date. publn_date, appin_title, nb_applicants, person_ctry_code, person_name, psn_sector, psn_name, publn_auth, appin_auth„ docdb_family_id, publn_kind, appin_kind, granted, vore2.nusber_applt
FROM patstat2016b.dbo.t1s201_appin AS a
JOIN patstat2010b.dbo.t1s207_pers_appin as person ON a.appin_id = person.appin_id JOIN patstat2016b.dbo.t1s206_person ON person.person_id = t1s206_person.person_id JOIN patstat2010b.dbo.t1s202_appin_title ON a.appin_id = t1s202_appin_yitle.appin_id JOIN patstat2016b.dbo.t1s211_pat_publn ON a.appin_id = t1s211_pat_publn.appin_id 17:N ( SELECT appin_id, COLN- (applt_seq_nr) nuaber_applt FROM patstat2016b.dbo.t1s207_pers_appin JOIN patstat2016b.dbo.t1sie6_person ON t1s207_pers_appin.person_id = t1s206_person.person_id kHERE applt_seq_nr > 0 and invt_seq_nr = 0 and psn_sector in ('COMPANY', 'COMPANY GOV NON-PROFIT*, 'COMPANY GOV NON-PROFIT UNIVERSITY*, 'COMPANY HOSPITAL', 'COMPANY UNIVERSITY', *GOV NON-PROFIT', 'GOV NON-PROFIT HOSPITAL', *GOV NON-PROFIT UNIVERSITY*„*HOSPITAL1UNIVERSITY1UNIVERSITY HOSPITACIUNKNOWN") GROUP BY appin_id HAVING coura(applt_seq_nr)>2) AS more2 ON a.appin_id = more2.appin_id
WHERE tisill_part_publn.pubindate BETWEEN '298e-21-81' AND '2815-12-311'
AND ipr_type ='PI' # ip right type is patent
AND appln_kind ='A' # application type is patent
AND person.applt_seq_nr >=1 # retrieves only those persons from TLS207_PERS_APPLN or TLS227_PERS_PUBLN which are applicants.
AND person.invt_seq_nr =0 # exclude inventors
# AND morez.number_applt >= 3
AND (psn_sector ='COMPANY' OR psn_sector ='COMPANY GOV NON-PROFIT' OR psn_sector .--..COMPANY GOV NON-PROFIT UNIVERSITY' 'JR psn_sector ='COMPANY HOSPITAL' OR psn_sector ='GOV NON-PROFIT' OR psn_sector ='GOV NCN-PROFIT HOSPITAL' :sn_sector =.005/ NON-PROFIT UNIVERSITY' OR psn_sector ='HOSPITAL' OR psn_sector ='UNIVERSITY' :sn_sector =*UNEVERSITY HOSPITAL' OR psn_sector ='UNKNOWN')
ORDER BY a.docdb_family_id, a.appin_auth, t1s211_pat_publn.publn_kind DESC;
AND appin_auth IN ('US*) # application granting authority is US patent office

SELECT DISTINCT a.appin_id, appin_nr, publn_nr, appin_filing_date. publn_date, appin_title, nb_applicants, person_ctry_code, person_name, psn_sector, psn_name, publn_auth, appin_auth„ docdb_family_id, publn_kind, appin_kind, granted, vore2.nusber_applt
FROM tls201_appln AS a
JOIN tls207_pers_appin as person ON a.appin_id = person.appin_id
JOIN tls206_person ON person.person_id = tls206_person.person_id
WHERE ipr_type ='PI' # ip right type is patent
AND appln_kind ='A' # application is granted
AND person.applt_seq_nr >=1 # retrieves only those persons from TLS207_PERS_APPLN or TLS227_PERS_PUBLN which are applicants.
AND person.invt_seq_nr =0 # exclude inventors
ORDER BY a.docdb_family_id, a.appin_auth, t1s211_pat_publn.publn_kind DESC;



select count(distinct a.appln_id)
from tls201_appln as a
join tls207_pers_appln as pers_appln on a.appln_id = pers_appln.appln_id
join tls206_person as person on pers_appln.person_id = person.person_id
where psn_sector='"COMPANY"'
AND ipr_type ='"PI"'
AND appln_kind ='"A "'
AND pers_appln.applt_seq_nr >=1
AND pers_appln.invt_seq_nr =0
AND granted = '"Y"'


CREATE EXTERNAL TABLE `patstat-global-spring-2021`.`orbis_patstat`
            (
            patentpublicationnumber string,
            currentownersbvdid string
            )
STORED AS PARQUET
LOCATION 's3://patstat-global-spring-2021/orbis_patstat/'
TBLPROPERTIES ("parquet.compression"="snappy");


create table tls201_appln_merged_orbis as
select * from (select tls201_appln.*, concat(trim(replace(appln_auth, '"', '')), cast(appln_id as varchar), '(', trim(replace(appln_kind, '"', '')), ')') as patentpublicationnumber
from  tls201_appln) as a
inner join orbis_patstat as o
using (patentpublicationnumber)

create table tls201_appln_merged_orbis as
select * from
    (select tls201_appln.*,
    concat(trim(replace(appln_auth, '"', '')), cast(appln_id as varchar)) as patentpublicationnumber
    from  tls201_appln)
inner join
orbis_patstat
using (patentpublicationnumber)

create external table `patstat-global-spring-2021`.`name_bvdid`
            (
            bvdid string,
            firm_name string
            )
STORED AS PARQUET
LOCATION 's3://patstat-global-spring-2021/orbis-name-matching/'
TBLPROPERTIES ("parquet.compression"="snappy");


import pandas as pd
import awswrangler as wr

query = """SELECT distinct(trim(replace(person_name, '"', ''))) as firm_name 
           from tls206_person 
           where psn_sector='"COMPANY"'
           """
df = wr.athena.read_sql_query(query, database="patstat-global-spring-2021")

orbis_lexis = pd.read_csv('C:/Users/Jakob/Downloads/lexis_alliances_orbis_static.csv')
r_and_d = pd.read_csv('C:/Users/Jakob/Downloads/ResearchandDevelopment_LexisNexis.csv')

r_and_d_firm_ids = set(r_and_d['firm_a'].unique()) | set(r_and_d['firm_b'].unique())

orbis_lexis = orbis_lexis[orbis_lexis['BvD ID number'].isin(r_and_d_firm_ids)]

from name_matching.name_matcher import NameMatcher
#
# matcher = NameMatcher(low_memory=False, top_n=5, common_words=False, legal_suffixes=True,
#                       distance_metrics=['editex', 'discounted_levenshtein', 'refined_soundex']) # distance_metrics=['overlap', 'discounted_levenshtein'])

# matcher.load_and_process_master_data('Company name Latin alphabet', orbis_lexis)
#
# res = matcher.match_names(to_be_matched=df.head(10000), column_matching='firm_name')
#
# matches = res[res['score'] > 90]

## split df into batches and use multiprocessing to speed up the process
import numpy as np
import ray
ray.init(ignore_reinit_error=True)

def split_df(df, chunk_size):
    num_chunks = len(df) // chunk_size + 1
    return np.array_split(df, num_chunks)

@ray.remote
def match_names(df, column_matching, column_to_be_matched):
    matcher = NameMatcher(low_memory=False, top_n=5, common_words=False, legal_suffixes=True,
                          distance_metrics=['editex', 'discounted_levenshtein',
                                            'refined_soundex'])
    matcher.load_and_process_master_data(column_to_be_matched, orbis_lexis)
    res = matcher.match_names(to_be_matched=df, column_matching=column_matching)
    matches = res[res['score'] > 90]
    return matches

def match_names_multiprocessing(df, column_matching, column_to_be_matched, chunk_size=10000):
    df_batches = split_df(df, chunk_size)
    futures = [match_names.remote(df, column_matching, column_to_be_matched) for df in df_batches]
    matches = ray.get(futures)
    return pd.concat(matches)

matches = match_names_multiprocessing(df, 'firm_name', 'Company name Latin alphabet', chunk_size=10000)

matches.to_pickle('C:/Users/Jakob/Downloads/matches.pkl')

matches = pd.read_pickle('C:/Users/Jakob/Downloads/matches.pkl')

matches.match_index.max()

matches.reset_index(names='df_index', inplace=True)
matches.sort_values(by=['df_index', 'score'], inplace=True)
matches.drop_duplicates(subset=['df_index'], keep='first', inplace=True)

merged = df.merge(matches[['match_index', 'df_index']], left_index=True, right_on='df_index', how='inner')
merged2 = merged.merge(orbis_lexis.reset_index(drop=True), left_on='match_index', right_index=True) #.drop(columns=['match_index', 'df_index'])


matches.match_index.value_counts()


from name_matching.run_nm import match_names

orbis_lexis['firm_name'] = orbis_lexis['Company name Latin alphabet']
matches = match_names(df, orbis_lexis, column_first='firm_name',
                      column_second='firm_name', case_sensitive=False, punctuation_sensitive=False,
                      special_character_sensitive=False, threshold=95, low_memory=False, top_n=5,
                      common_words=False, legal_suffixes=True,
                      distance_metrics=['editex', 'discounted_levenshtein', 'refined_soundex'])

complete_matched_data = df.merge(matches, how='left', right_index=True, left_index=True).merge(orbis_lexis, how='left', left_on='match_index', right_index=True, suffixes=['', '_matched'])
complete_matched_data.drop(columns=['name_matching_data', 'original_name', 'match_name', 'score', 'match_index', 'name_matching_data_matched'], inplace=True)

CREATE TABLE merged AS
SELECT * from
(select person_name, person_id, TRIM(REPLACE(psn_name, '"', '')) AS patstat_name
FROM tls206_person
where psn_sector='"COMPANY"' or han_name='"COMPANY"')
as p
INNER JOIN name_bvdid as n
ON p.patstat_name = n.firm_name;


# read orbis granted patent firms
orbis_granted_patents = pd.read_csv('C:/Users/Jakob/Downloads/Granted Patents Firms/Orbis_PatTitle_PatID_BvDFirmName_BvDID_BvDCountry.csv', nrows=1000, sep=';')

df = wr.s3.read_csv('s3://patstat-global-spring-2021/tls209_appln_ipc/tls209_part01.csv', nrows=1000)

CREATE EXTERNAL TABLE `patstat-global-spring-2021`.`tls209_appln_ipc`
            (
            appln_id integer,
            ipc_class_symbol string,
            ipc_class_level string,
            ipc_version string,
            ipc_value string,
            ipc_position string,
            ipc_gener_auth string
            )
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
LOCATION 's3://patstat-global-spring-2021/tls209_appln_ipc/'
TBLPROPERTIES ("skip.header.line.count"="1");


create table merged_full as
SELECT DISTINCT a.appln_id,
                appln_auth,
                appln_nr,
                appln_filing_date,
                appln_kind,
                granted,
                nb_applicants,
                docdb_family_id,
                bvdid,
                p.person_id,
                person_ctry_code,
                p.person_name,
                psn_sector,
                psn_name,
                han_name
FROM tls201_appln AS a
JOIN tls207_pers_appln as person ON a.appln_id = person.appln_id
JOIN tls206_person as p ON person.person_id = p.person_id
INNER JOIN merged as m ON TRIM(REPLACE(p.psn_name, '"', '')) = m.firm_name
where p.psn_sector='"COMPANY"' or p.han_name='"COMPANY"'
AND person.applt_seq_nr >=1
AND person.invt_seq_nr =0
ORDER BY appln_auth, appln_filing_date DESC;


query = """SELECT appln_filing_date, appln_kind, granted, psn_name, bvdid, ipc_class_symbol, ipc_class_level from merged_full_ipc"""
df = wr.athena.read_sql_query(query, database='patstat-global-spring-2021')
df.to_csv('C:/Users/Jakob/Downloads/patents_lexis_alliance_firms.csv', index=False)

df.psn_name.value_counts()

df.ipc_class_symbol.str[:3].value_counts()

df[df.granted == '"Y"']

quote_cols = ['appln_kind', 'granted', 'psn_name']

for col in quote_cols:
    df[col] = df[col].str.strip('"')

import pandas as pd

df = pd.read_stata('C:/Users/Jakob/Downloads/domainonly.dta')
df = df[df.domain_all.str.len() > 3].copy(deep=True)
df.domain_all = df.domain_all.str.strip()
df.domain_all.to_csv('C:/Users/Jakob/Downloads/swiss-survey-covid-urls.csv', index=False, header=False)

df2 = pd.read_csv('C:/Users/Jakob/Documents/GitHub/cc-download/swiss-survey-urls.csv', header=None)

df.domain_all.isin(df2[0]).sum()