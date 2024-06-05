import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
from firm_name_matching import firm_name_clean

output_path = '/Users/Jakob/Documents/financial_news_data/output/lexis_preds_2023'

important_labels = ['StrategicAlliance', 'JointVenture', 'Marketing', 'Manufacturing',
                    'ResearchandDevelopment', 'Licensing']

# read lexis nexis articles with detected orgs and relations
df = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/lexisnexis_preds_robust_vortex_99.pkl')

df.drop(columns=['index_x', 'index_y'], inplace=True)


df = df[['title', 'document', 'publication', 'publication_date', 'firms', 'rels_pred', 'country', 'industry']]

df['cleaned_firms'] = df.firms.apply(lambda firms: [firm_name_clean(firm) for firm in firms])

df_static = pd.read_pickle('C:/Users/Jakob/Downloads/matching_result_lexis_orbis2023_compustat.pkl')
df_dynamic = pd.read_pickle('C:/Users/Jakob/Downloads/matching_result_dynamic_lexis_orbis2023_compustat.pkl')

lexis_firm_names_clean = df.cleaned_firms.explode().value_counts()\
    .index.to_frame(name='cleaned_name').reset_index(drop=True)

lexis_firm_names = df.firms.explode().value_counts()\
    .index.to_frame(name='name').reset_index(drop=True)

to_replace = {'amex': 'american express', 'gsk': 'glaxosmithkline', 'mhi': 'mitsubishi heavy industries',
              'ge'  : 'general electric', 'vw': 'volkswagen', 'ibm': 'international business machines',
              'l&t' : 'larsen & toubro', 'arw': 'arrow electronics', 'BMW': 'bayerische motoren werke',
              '(u.k.)': '', '(south Africa)': '', '(pty)': '', 'Merck S/a': 'Merck & Co., Inc.',
              'Sanofi-aventis (suisse) Sa': 'Sanofi'}

lexis_firm_names_clean['cleaned_name'] = lexis_firm_names_clean.cleaned_name.replace(to_replace)

lexis_firm_names_clean.squeeze().to_csv('C:/Users/Jakob/Downloads/lexis_firm_names_clean.csv', index=False)

names_ids = lexis_firm_names_clean.merge(df_static[['cleaned_name', 'bvdid']],
                                         on='cleaned_name', how='left')

print(f'Share of unmatched firms: {names_ids["bvdid"].isna().sum()/len(names_ids)}')

names_ids = names_ids.dropna().set_index('cleaned_name').squeeze().to_dict()


# save to pickle
rels = df[['title', 'document', 'publication', 'firms', 'country', 'industry', 'publication_date', 'cleaned_firms', 'rels_pred']].copy()
rels['firm_a'] = rels.cleaned_firms.str[0]
rels['firm_b'] = rels.cleaned_firms.str[1]
rels.drop(columns=['cleaned_firms'], inplace=True)

rels['firm_a'] = rels.firm_a.map(names_ids)
rels['firm_b'] = rels.firm_b.map(names_ids)

rels.dropna(inplace=True)

# remove terminated
rels = rels[rels.rels_pred.apply(lambda rels: 'Terminated' not in rels)]

# remove firms where both participants are the same
rels = rels[rels.firm_a != rels.firm_b]

# remove duplicate relationships (same participants, same type, same year)
rels['year'] = rels.publication_date.dt.year
rels = rels.groupby(['firm_a', 'firm_b', 'year']).agg(list)
rels.reset_index(inplace=True)

from itertools import chain
rels['rels_pred'] = rels.rels_pred.apply(chain.from_iterable).apply(list).apply(set).apply(list)

rels.to_pickle(os.path.join(output_path, 'lexisnexis_rel_database_full.pkl'))

# save separate csvs for each relation type

## create edge list for Michael
rels = df[['publication_date', 'cleaned_firms', 'rels_pred']].copy()
rels['firm_a'] = rels.cleaned_firms.str[0]
rels['firm_b'] = rels.cleaned_firms.str[1]
rels.drop(columns=['cleaned_firms'], inplace=True)

rels['firm_a'] = rels.firm_a.map(names_ids)
rels['firm_b'] = rels.firm_b.map(names_ids)

rels.dropna(inplace=True)

# remove terminated
rels = rels[rels.rels_pred.apply(lambda rels: 'Terminated' not in rels)]

# remove firms where both participants are the same
rels = rels[rels.firm_a != rels.firm_b]

# remove duplicate relationships (same participants, same type, same year)
rels['year'] = rels.publication_date.dt.year
rels = rels.groupby(['firm_a', 'firm_b', 'year']).agg(list)
rels.reset_index(inplace=True)

from itertools import chain
rels['rels_pred'] = rels.rels_pred.apply(chain.from_iterable).apply(list).apply(set).apply(list)

rels = rels.explode('rels_pred')

for rel_name in important_labels:
    rels[rels.rels_pred =='ResearchandDevelopment'].drop(columns=['rels_pred']).to_csv(
            os.path.join(output_path, 'rel_database', f'{rel_name}_LexisNexis.csv'), index=False)

rels[rels.rels_pred =='ResearchandDevelopment'].drop_duplicates(['firm_a', 'firm_b', 'year'])

pd.concat([rels[rels.rels_pred =='ResearchandDevelopment'].firm_a, rels[rels.rels_pred =='ResearchandDevelopment'].firm_b]).nunique()

df_static.sort_values(['max_number_of_employees', 'max_turnover'])
df_static.drop(columns=['cleaned_name', 'max_number_of_employees', 'max_turnover'], inplace=True)
df_static.rename(columns={'bvdid': 'ID'}, inplace=True)
df_static.to_csv('C:/Users/Jakob/Downloads/lexis_match_orbis2023_compustat_static.csv', index=False)

df_dynamic['year'] = df_dynamic.closing_date.astype(str).str[:4]
df_dynamic.to_csv('C:/Users/Jakob/Downloads/lexis_match_orbis2023_compustat_dynamic.csv', index=False)

r_and_d_ids = pd.concat([rels[rels.rels_pred =='ResearchandDevelopment'].firm_a, rels[rels.rels_pred =='ResearchandDevelopment'].firm_b]).drop_duplicates()
means = df_dynamic[df_dynamic.bvdid.isin(r_and_d_ids)].groupby('year').mean()

df_dynamic.sort_values('operating_revenue_turnover', inplace=True, ascending=False)

# read rels research and development
rels = pd.read_csv(os.path.join(output_path, 'rel_database', 'ResearchandDevelopment_LexisNexis.csv'))
rels_industries = rels.merge(df_static[['bvdid', 'nace_rev_2_core_code_4_digits']], left_on='firm_a', right_on='bvdid', how='left', suffixes=('', '_a'))
rels_industries = rels_industries.merge(df_static[['bvdid', 'nace_rev_2_core_code_4_digits']], left_on='firm_b', right_on='bvdid', how='left', suffixes=('', '_b'))

# value count of combinations of industries
rels_industries['nace_rev_2_core_code_2_digits'] = rels_industries.nace_rev_2_core_code_4_digits.str[:2]
rels_industries['nace_rev_2_core_code_2_digits_b'] = rels_industries.nace_rev_2_core_code_4_digits_b.str[:2]
rels_industries[(rels_industries.nace_rev_2_core_code_2_digits.str.len()>1) & (rels_industries.nace_rev_2_core_code_2_digits_b.str.len()>1)].groupby(['nace_rev_2_core_code_2_digits', 'nace_rev_2_core_code_2_digits_b'], observed=True).size().sort_values(ascending=False).head(50)