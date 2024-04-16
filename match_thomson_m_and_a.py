import pandas as pd
import os
from datetime import timedelta
from firm_name_matching import firm_name_clean
import matplotlib.pyplot as plt

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

df_static = pd.read_pickle('C:/Users/Jakob/Downloads/matching_result_lexis_orbis2023_compustat.pkl')
lexis_nexis_r_and_d = pd.read_csv('C:/Users/Jakob/Downloads/lexis_match_orbis2023_compustat_dynamic.csv')

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
