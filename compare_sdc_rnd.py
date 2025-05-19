import pandas as pd

output_path = '/Users/Jakob/Documents/financial_news_data/output/lexis_preds_2023'

rnd = pd.read_csv(os.path.join(output_path, 'rel_database', 'ResearchandDevelopment_LexisNexis.csv'))
rnd = rnd[rnd.year >= 2001].copy()

sdc = pd.read_pickle('C:/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')

sdc['DealNumber'] = sdc.DealNumber.astype(str)
sdc['DealNumber'] = sdc.DealNumber.apply(lambda x: x.split('.')[0])

sdc.drop_duplicates(subset=['DealNumber'], inplace=True)
sdc.drop_duplicates(subset=['DealText'], inplace=True)


sdc.set_index('DealNumber', inplace=True, drop=True)

# there are 12 observations that are not tagged as SA or JV
len(sdc) - sdc[['StrategicAlliance', 'JointVentureFlag']].any(axis=1).sum()
# remove them
sdc = sdc[sdc[['StrategicAlliance', 'JointVentureFlag']].any(axis=1)]

# combine licensing, exclusive licensing, crosslicensing
licensing_cols = ['LicensingAgreementFlag', 'ExclusiveLicensingAgreementFlag', 'CrossLicensingAgreement',
                  'RoyaltiesFlag']
sdc['Licensing'] = sdc[licensing_cols].any(axis=1)
sdc.drop(columns=licensing_cols, inplace=True)

# A lot of agreements are pending (announced), some are announced to be terminated
sdc.Status.value_counts()

# one hot encode pending and terminated
sdc['Pending'] = (sdc.Status == 'Pending').astype(int)
sdc['Terminated'] = (sdc.Status == 'Terminated').astype(int)

# make column names easier to work with
sdc.columns = sdc.columns.str.replace('Flag', '')
sdc.columns = sdc.columns.str.replace('Agreement', '')
sdc.rename(columns={'DealNumber'                               : 'ID', 'AllianceDateAnnounced': 'Date',
                    'DealText'                                 : 'Text',
                    'ParticipantsinVenture/Alliance(ShortName)': 'Participants'}, inplace=True)

labels = ['JointVenture', 'Marketing', 'Manufacturing', 'ResearchandDevelopment', 'Licensing', 'Supply',
          'Exploration', 'TechnologyTransfer', 'Pending', 'Terminated']

# take examples with at least 2 participants
sdc = sdc[sdc.Participants.str.len() > 1]

# convert to bool
sdc[labels] = sdc[labels].apply(lambda x: pd.to_numeric(x, downcast='integer')).astype(bool)

sdc['StrategicAlliance'] = ~sdc.JointVenture  # every deal that is not a JV is a SA

labels = ['StrategicAlliance'] + labels

sdc_rd = sdc[sdc.ResearchandDevelopment & ~sdc.Pending & ~sdc.Terminated].copy()
sdc_rd.Participants = sdc_rd.Participants.apply(lambda x: sorted(x))
sdc_rd['Year'] = sdc_rd.Date.dt.year
sdc_rd['ParticipantsString'] = sdc_rd.Participants.apply(lambda x: ', '.join(x))
sdc_rd.drop_duplicates(subset=['ParticipantsString', 'Year'])

# compustat match via cusip
path = r"C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\compustat_na_cusip_isin.csv".replace('\\', '/')
cusip_gvkey = pd.read_csv(path)
cusip_gvkey = cusip_gvkey[['gvkey', 'cusip']].copy()
cusip_gvkey['cusip'] = cusip_gvkey['cusip'].str[:6]
cusip_gvkey.drop_duplicates(subset=['gvkey', 'cusip'], inplace=True)
sdc_rd_cusip_merge = sdc_rd['Parti.CUSIP'].explode().reset_index()
sdc_rd_cusip_merge.columns = ['DealNumber', 'cusip']
sdc_rd_cusip_merge = sdc_rd_cusip_merge.merge(cusip_gvkey, how='left', on='cusip').dropna()
# change index name to DealNumber
sdc_rd_cusip_merge['gvkey'] = sdc_rd_cusip_merge['gvkey'].astype(int)
sdc_rd_cusip_merge = sdc_rd_cusip_merge.groupby('DealNumber')['gvkey'].agg(list)
sdc_rd_cusip_merge = sdc_rd_cusip_merge[sdc_rd_cusip_merge.str.len() >= 2] # need at least two firms

sdc_rd = sdc_rd.join(sdc_rd_cusip_merge)
sdc_rd_na = sdc_rd[sdc_rd.gvkey.str.len() >= 2].copy()

# export for Chih-Sheng
sdc_rd_na_export = sdc_rd_na[['gvkey', 'Year']].copy()

# separate row for each combination of gvkeys
from itertools import combinations
new_rows = []
for deal, row in sdc_rd_na_export.iterrows():
    gvkeys = row['gvkey']
    year = row['Year']
    for combo in combinations(gvkeys, 2):
        new_rows.append({
            'gvkey1': combo[0],
            'gvkey2': combo[1],
            'Year': year
        })
sdc_rd_na_export = pd.DataFrame(new_rows)

sdc_rd_na_export.to_csv('C:/Users/Jakob/Documents/financial_news_data/sdc_r_and_d_links_NA_compustat.csv', index=False)

sdc_rd_na = sdc_rd_na[sdc_rd_na.Year >= 2001].copy()

# Create the new DataFrame
new_df = pd.DataFrame(new_rows)

# make separate rows for all combinations
import itertools

def generate_combinations(gvkey_list):
    return list(itertools.combinations(gvkey_list, 2))

sdc_rd_na['gvkey_combinations'] = sdc_rd_na['gvkey'].apply(generate_combinations)
sdc_rd_na = sdc_rd_na.explode('gvkey_combinations')

sdc_rd_na[['firm_a', 'firm_b']] = pd.DataFrame(sdc_rd_na['gvkey_combinations'].tolist(), index=sdc_rd_na.index)


rnd_na = rnd[rnd.firm_a.isin(cusip_gvkey.gvkey.astype(str)) & rnd.firm_b.isin(cusip_gvkey.gvkey.astype(str))]
rnd_na = rnd_na[rnd_na.year >= 2001].copy()

pd.concat([rnd_na.firm_a, rnd_na.firm_b]).drop_duplicates()
pd.concat([sdc_rd_na.firm_a, sdc_rd_na.firm_b]).drop_duplicates()

rnd_na_dup.year
rnd_na_dup = pd.concat([rnd_na, rnd_na.rename(columns={'firm_a': 'firm_b', 'firm_b': 'firm_a'})])
sdc_rd_na_dup = sdc_rd_na[['firm_a', 'firm_b', 'Year']].copy()
sdc_rd_na_dup = pd.concat([sdc_rd_na_dup, sdc_rd_na_dup.rename(columns={'firm_a': 'firm_b', 'firm_b': 'firm_a'})])
sdc_rd_na_dup.rename(columns={'Year': 'year'}, inplace=True)
sdc_rd_na_dup = sdc_rd_na_dup.astype(str)
rnd_na_dup = rnd_na_dup.astype(str)

inner = pd.merge(rnd_na_dup, sdc_rd_na_dup, how='inner', on=['firm_a', 'firm_b', 'year'])
