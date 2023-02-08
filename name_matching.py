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


