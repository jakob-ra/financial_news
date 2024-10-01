import pandas as pd

sdc = pd.read_csv('C:/Users/Jakob/Downloads/Thomson_SDC_MandA_matched_Compustat_v2.csv')
zephyr = pd.read_csv('C:/Users/Jakob/Downloads/zephyr_m_a_matched_compustat_via_isin_and_name.csv')

df = zephyr[['announced_date', 'deal_number', 'acquiror_gvkey', 'target_gvkey']]
df.columns = ['DateAnnounced', 'DealNumberZephyr', 'AcquirorGvkey', 'TargetGvkey']
df = pd.concat([df, sdc[['DateAnnounced', 'DealNumber', 'AcquirorGvkey', 'TargetGvkey']].rename(columns={'DealNumber': 'DealNumberSDC'})])
df['year'] = pd.to_datetime(df.DateAnnounced).dt.year
df.drop_duplicates(subset=['AcquirorGvkey', 'TargetGvkey', 'year'], inplace=True)
df = df[['DateAnnounced', 'DealNumberZephyr', 'DealNumberSDC', 'AcquirorGvkey', 'TargetGvkey']].copy()

df.to_csv('C:/Users/Jakob/Downloads/thomson_sdc_zephyr_m_and_a_combined.csv', index=False)