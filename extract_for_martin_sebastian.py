import pandas as pd

# sdc
sdc = pd.read_pickle('C:/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')

swiss_sdc = sdc[((sdc.ParticipantNation.apply(lambda x: 'Switzerland' in str(x))) | (sdc.ParticipantUltimateParentNation.apply(lambda x: 'Switzerland' in str(x)))) & (sdc.ParticipantSICCodes.apply(lambda x: '283' in str(x)))]
swiss_sdc.to_pickle('C:/Users/Jakob/Downloads/thomson_sdc_strategic_alliances_joint_ventures_involving_swiss_pharma_firms.pkl')

# extracted relations
df = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/output/lexis_preds_2023/lexisnexis_rel_database_full.pkl')
df_static = pd.read_csv('C:/Users/Jakob/Downloads/lexis_match_orbis2023_compustat_static.csv')
swiss_ids = df_static[df_static.country_iso_code == 'CHE'].ID
pharma_ids = df_static[df_static.ussic_primary_code.astype(str).str.startswith('283')].ID

swiss_df = df[((df.firm_a.isin(swiss_ids)) | (df.firm_b.isin(swiss_ids))) & ((df.firm_a.isin(pharma_ids)) | (df.firm_b.isin(pharma_ids)))]
swiss_df.to_pickle('C:/Users/Jakob/Downloads/lexisnexis_rel_database_swiss_pharma_firms.pkl')