import pandas as pd
import os
import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
import numpy as np

# read compustat naics and sic to supplement if missing
path = r"C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\compustat-global-naics-sic.csv".replace('\\', '/')
compustat_global_naics_sic = pd.read_csv(path, usecols=['gvkey', 'naics', 'sic']).drop_duplicates(subset=['gvkey'], keep='last')
path = r"C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\compustat-na-naics-sic.csv".replace('\\', '/')
compustat_na_naics_sic = pd.read_csv(path, usecols=['gvkey', 'naics', 'sic']).drop_duplicates(subset=['gvkey'], keep='last')
compustat_naics_sic = pd.concat([compustat_global_naics_sic, compustat_na_naics_sic], ignore_index=True).drop_duplicates(subset=['gvkey'], keep='last')
del compustat_global_naics_sic, compustat_na_naics_sic
compustat_naics_sic.dropna(subset=['naics', 'sic'], inplace=True, how='all')

# run this the first time to consolidate all tnic data
all_files = os.listdir('/Users/Jakob/Documents/financial_news_data/tnic_orbis')
chunks = [f for f in all_files if f.startswith('Pub')]
df = pd.concat(
        [pd.read_excel(os.path.join('/Users/Jakob/Documents/financial_news_data/tnic_orbis', chunk)) for
         chunk in chunks])

rename_dict = {'Company name Latin alphabet'                 : 'firm_name', 'BvD ID number': 'bvdid',
               'NACE Rev. 2, primary code(s)'                : 'nace_primary',
               'NACE Rev. 2, primary code(s) - description'  : 'nace_primary_description',
               'NACE Rev. 2, secondary code(s)'              : 'nace_secondary',
               'NACE Rev. 2, secondary code(s) - description': 'nace_secondary_description',
               'US SIC, primary code(s)'                     : 'sic_primary',
               'US SIC, secondary code(s)'                   : 'sic_secondary',
               'NAICS 2022, primary code(s)'                 : 'naics_primary',
               'NAICS 2022, secondary code(s)'               : 'naics_secondary',
               'Country ISO code'                            : 'country', 'ISIN number': 'isin',
               'Trade description (English)'                 : 'trade_description',
               'Products & services'                         : 'products_and_services',
               'Specialisation'                              : 'specialisation',
               'Website address'                             : 'website'}

df.rename(columns=rename_dict, inplace=True)
df = df[rename_dict.values()].copy()

# df.sic_primary_code = df.sic_primary_code.str.split('\n').str[0].str.strip().str[:2]
# sic_in_data = df.sic_primary_code.dropna().unique()

# match to gvkey via isin/cusip
compustat_global_sedol_isin = pd.read_excel('C:/Users/Jakob/Documents/compustat-global-sedol-isin.xlsx')
compustat_global_sedol_isin.columns = ['gvkey', 'fyear', 'datadate', 'isin', 'sedol']
compustat_global_sedol_isin.dropna(subset=['isin', 'sedol'], inplace=True)
compustat_global_sedol_isin.drop_duplicates(subset=['gvkey', 'isin', 'sedol'], keep='first', inplace=True)
compustat_global_sedol_isin = compustat_global_sedol_isin[['gvkey', 'isin']].dropna().drop_duplicates()
compustat_global_sedol_isin = compustat_global_sedol_isin.set_index('isin').squeeze()

df['gvkey'] = df['isin'].map(compustat_global_sedol_isin)

compustat_na_cusip = pd.read_csv('C:/Users/Jakob/Downloads/compustat-na-cusip.csv',
                                 usecols=['gvkey', 'cusip', 'loc']).drop_duplicates().dropna()
compustat_na_cusip = compustat_na_cusip[compustat_na_cusip['loc'].isin(['USA', 'CAN'])].copy()
compustat_na_cusip['loc'] = compustat_na_cusip['loc'].replace({'USA': 'US', 'CAN': 'CA'})
compustat_na_cusip['isin_11_digits'] = compustat_na_cusip['loc'] + compustat_na_cusip['cusip'].astype(str)
compustat_na_cusip.to_csv(
    'C:/Users/Jakob/Documents/financial_news_data/tnic_orbis/compustat_na_cusip_isin.csv', index=False)

df['gvkey2'] = df['isin'].str[:11].map(compustat_na_cusip.set_index('isin_11_digits')['gvkey'])

df['gvkey'] = df['gvkey'].fillna(df['gvkey2'])
df.drop(columns=['gvkey2'], inplace=True)

df.dropna(subset=['gvkey'], inplace=True)
df['gvkey'] = df['gvkey'].astype(int)

# add textual descriptions of primary and secondary NAICS, NACE and SIC
path = r"C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\Two-and-Four-Digit-SIC-with-Descriptions.xlsx".replace('\\', '/')
sic_descriptions = pd.read_excel(path, sheet_name='SIC 4 DIGIT')
sic_descriptions.columns = ['sic', 'description']
path = r"C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\sic-descriptions-wiki.xlsx".replace('\\', '/')
sic_descriptions2 = pd.read_excel(path)
sic_descriptions2.columns = ['sic', 'description']
sic_descriptions = pd.concat([sic_descriptions, sic_descriptions2], ignore_index=True).drop_duplicates(subset=['sic'], keep='first')
sic_descriptions.description = sic_descriptions.description.astype(str).str.replace('-', ' ').str.strip()
sic_descriptions.sic = sic_descriptions.sic.astype(str)
sic_descriptions = sic_descriptions.set_index('sic').squeeze().to_dict()

nace_descriptions = pd.read_excel(
    'C:/Users/Jakob/Documents/financial_news_data/tnic_orbis/NACERev2_ISIC4_Table.xlsx',
    usecols=['NACE_CODE', 'NACE_NAME'])
nace_descriptions['NACE_CODE'] = nace_descriptions['NACE_CODE'].astype(str).str.replace('.', '',
                                                                                        regex=False).str.strip()
nace_descriptions = nace_descriptions.set_index('NACE_CODE').squeeze().to_dict()

path = r"C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\2022_NAICS_Structure.xlsx".replace('\\',                                                                                  '/')
naics_descriptions = pd.read_excel(path, skiprows=2, usecols=['2022 NAICS Code', '2022 NAICS Title'])
naics_descriptions.columns = ['naics', 'description']
naics_descriptions.description = naics_descriptions.description.str.strip().replace('T$', '',
                                                                                    regex=True).str.replace(
    '  ', ' ').str.strip()  # remove capital 'T' from right end of string
naics_descriptions.naics = naics_descriptions.naics.astype(str).str.replace('.', '', regex=False).str.strip()
naics_descriptions = naics_descriptions.set_index('naics').squeeze().to_dict()

for code_type in ['nace', 'sic', 'naics']:
    df[f'{code_type}'] = df[f'{code_type}_primary'].str.split('\n') + df[f'{code_type}_secondary'].str.split(
        '\n')
    df[f'{code_type}'] = df[f'{code_type}'].apply(lambda x: [] if not isinstance(x, list) else x)

# take union of codes with compustat naics/sic
df = df.merge(compustat_naics_sic[['gvkey', 'naics', 'sic']], on='gvkey', how='left', suffixes=('', '_compustat'))

df['naics'] = df['naics'] + df['naics_compustat'].apply(lambda x: [str(int(x))] if not pd.isna(x) else [])
df['sic'] = df['sic'] + df['sic_compustat'].apply(lambda x: [str(int(x))] if not pd.isna(x) else [])
df.drop(columns=['naics_compustat', 'sic_compustat'], inplace=True)

for code_type in ['nace', 'sic', 'naics']:
    df[f'{code_type}'] = df[f'{code_type}'].apply(lambda x: list(set(x)))

df = df [df[['nace', 'sic', 'naics']].sum(axis=1).str.len()>0].copy() # drop firms with no industry codes

nace_cov = df.nace.explode().value_counts()
nace_cov = nace_cov[~nace_cov.index.isin(nace_descriptions.keys())]

sic_cov = df.sic.explode().value_counts()
sic_cov = sic_cov[~sic_cov.index.isin(sic_descriptions.keys())]

naics_cov = df.naics.explode().value_counts()
naics_cov = naics_cov[~naics_cov.index.isin(naics_descriptions.keys())]

# this is sometimes not finding the right text!
df['nace_text'] = df['nace'].apply(lambda x: [nace_descriptions[i] for i in x if i in nace_descriptions])
df['sic_text'] = df['sic'].apply(lambda x: [sic_descriptions[i] for i in x if i in sic_descriptions])
df['naics_text'] = df['naics'].apply(lambda x: [naics_descriptions[i] for i in x if i in naics_descriptions])

df['industry_text'] = df[['nace_text', 'sic_text', 'naics_text']].sum(axis=1).apply(lambda x: list(set(x))).apply(
    lambda x: ', '.join(x))

description_cols = ['trade_description', 'products_and_services', 'specialisation', 'industry_text']
for col in description_cols:
    # nan to empty string
    df[col] = df[col].fillna('')
    df[col] = df[col].astype(str)
    remove_stings = [' [source: Bureau van Dijk]', '\n', '\r', '\t']
    for s in remove_stings:
        df[col] = df[col].str.replace(s, ' ')

    # remove double periods, double spaces, double commas
    df[col] = df[col].str.replace('..', '.').str.replace('  ', ' ').str.replace(',,', ',')

    df[col] = df[col].str.strip()

df['description'] = df[description_cols].apply(lambda x: '. '.join(x), axis=1).str.replace('..', '.')

df.drop(columns=['industry_text'], inplace=True)

df.to_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/tnic_orbis_input_full_including_industry_descripions.csv',index=False)
df = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/tnic_orbis_input_full_including_industry_descripions.csv')

gvkey_in_df = df.gvkey.drop_duplicates().sort_values()
print(f'Unique gvkeys: {len(gvkey_in_df)}')

# unique_gvkeys_etnic = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/Etnic2_data/ETNIC2_unique_gvkeys.csv').squeeze()
# df_in_etnic = df[df.gvkey.isin(unique_gvkeys_etnic)].copy() # DON'T DO THIS because we will only end up with competitors in training data

### construct competition matrix based on shared membership of 3-digit primary or secondary NAICS/NACE
path = r"C:\Users\Jakob\Downloads\for Jakob\compustat ID.xlsx".replace('\\', '/')
sample_gvkeys = pd.read_excel(path, header=None, names=['gvkey'])
sample_gvkeys = sample_gvkeys.gvkey.drop_duplicates()

df_sample = df[df.gvkey.isin(sample_gvkeys)].copy()
df_sample = df_sample.sort_values('gvkey').reset_index(drop=True)
n_firms = len(df)
all_edges = []
for i, code_type in enumerate(['nace', 'sic', 'naics']):
    exploded_df = df_sample[['gvkey', code_type]].explode(code_type)
    exploded_df = exploded_df[exploded_df[code_type].str.len() > 0].copy()
    edges = exploded_df.merge(exploded_df, on=code_type, suffixes=('1', '2'), how='inner')
    edges.drop_duplicates(subset=['gvkey1', 'gvkey2'], inplace=True)
    edges = edges[edges.gvkey1 != edges.gvkey2].copy()
    edges = edges[['gvkey1', 'gvkey2']]
    all_edges.append(edges)

edges = pd.concat(all_edges, ignore_index=True).drop_duplicates()
del all_edges
edges['score'] = 1

edges.to_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/sample_competition_links_based_on_industry_codes.csv', index=False)
edges = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/sample_competition_links_based_on_industry_codes_only_nace.csv')

# create sample for fast iteration
path = '/Users/Jakob/Documents/financial_news_data/tnic_orbis/data_sample/tnic_input_files/'
sample = df.sample(5000)
# save each firm as a separate file
for i, row in tqdm.tqdm(sample.iterrows(), total=len(sample)):
    with open(os.path.join(path, str(row.gvkey)), 'w') as f:
        f.write(row.description)

# full training data
path = '/Users/Jakob/Documents/financial_news_data/tnic_orbis/data/tnic_input_files/'
# save each firm as a separate file
for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    with open(os.path.join(path, str(row.gvkey)), 'w') as f:
        f.write(row.description)

# TNIC/ETNIC overlap inference data
df_overlap = df[df.gvkey.isin(unique_gvkeys_etnic)].copy()
path = '/Users/Jakob/Documents/financial_news_data/tnic_orbis/data_overlap/tnic_input_files/'
# save each firm as a separate file
for i, row in tqdm.tqdm(df_overlap.iterrows(), total=len(df_overlap)):
    with open(os.path.join(path, str(row.gvkey)), 'w') as f:
        f.write(row.description)

# inference data for Estimation sample Compustat global + NA sample (1518 firms)
df_sample = df[df.gvkey.isin(sample_gvkeys_orbis_tnic)].copy()
path = '/Users/Jakob/Documents/financial_news_data/tnic_orbis/data_estimation_sample/tnic_input_files/'
# save each firm as a separate file
for i, row in tqdm.tqdm(df_sample.iterrows(), total=len(df_sample)):
    with open(os.path.join(path, str(row.gvkey)), 'w') as f:
        f.write(row.description)

# run this command with replication code
# python train.py --data_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\data --output_dir output --algorithm pv_dbow --negative 0 --min-count 3 --window 15 --hs 1 --vector-size 300 --epochs 50 --threshold 0.2

# debug
# python train.py --data_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\data_sample --output_dir output --algorithm pv_dbow --negative 0 --min-count 3 --window 15 --hs 1 --vector-size 300 --epochs 50 --threshold 0.2

# run this command for inference on overlap with TNIC/ETNIC
# python multi_inference.py --data_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\data_overlap --output_dir output_inference --model_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\replication_code-20220963\PythonCode\output\1742501369605_pv_dbow_dim=300_window=15_epochs=50\models --number_of_processes_run_inference 8

# run this command for inference on estimation sample
# python multi_inference.py --data_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\data_estimation_sample --output_dir output_inference_estimation_sample --model_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\replication_code-20220963\PythonCode\output\1742501369605_pv_dbow_dim=300_window=15_epochs=50\models --number_of_processes_run_inference 8
# python multi_inference.py --data_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\data_estimation_sample --output_dir output_inference_estimation_sample_new_model --model_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\replication_code-20220963\PythonCode\output\1744410151933_pv_dbow_dim=300_window=15_epochs=50\models --number_of_processes_run_inference 8

# read orbis tnic results
orbis_tnic_path = '/Users/Jakob/Documents/financial_news_data/tnic_orbis/replication_code-20220963/PythonCode/output_inference/'
all_files = [f for f in os.listdir(orbis_tnic_path) if f.endswith('.tsv')]
orbis_tnic = pd.read_csv(os.path.join(orbis_tnic_path, all_files[0]), sep='\t')
orbis_tnic.columns = ['gvkey1', 'gvkey2', 'score']

# orbis_tnic = orbis_tnic[orbis_tnic.gvkey1 < orbis_tnic.gvkey2].copy()

gvkeys_orbis_tnic = set(orbis_tnic.gvkey1).union(set(orbis_tnic.gvkey2))

# read etnic3
etnic3 = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/Etnic3_data/ETNIC3_data.txt', sep='\t')
etnic3 = etnic3.sort_values(['year', 'gvkey1', 'gvkey2']).drop_duplicates(subset=['gvkey1', 'gvkey2'], keep='last') # keep most recent

# similarity score is symmetric, remove duplicates
etnic3 = etnic3[etnic3.gvkey1 < etnic3.gvkey2].copy()

gvkeys_etnic3 = set(etnic3.gvkey1).union(set(etnic3.gvkey2))

shared_gvkeys_etnic3_orbis_tnic = gvkeys_orbis_tnic.intersection(gvkeys_etnic3)

orbis_tnic[orbis_tnic.score >= 0.2039].shape[0] / len(orbis_tnic)

# keep only firms that are in our orbis descriptions data
orbis_tnic_shared = orbis_tnic[orbis_tnic.gvkey1.isin(shared_gvkeys_etnic3_orbis_tnic) & orbis_tnic.gvkey2.isin(shared_gvkeys_etnic3_orbis_tnic)].copy()
etnic3_shared = etnic3[etnic3.gvkey1.isin(shared_gvkeys_etnic3_orbis_tnic) & etnic3.gvkey2.isin(shared_gvkeys_etnic3_orbis_tnic)].copy()

merge = orbis_tnic_shared.merge(etnic3_shared, on=['gvkey1', 'gvkey2'], suffixes=('_orbis_tnic', '_etnic3'), how='outer', validate='one_to_one')
merge_in_orbis_tnic = merge[merge.score_orbis_tnic.notna()]

def plot_roc_and_precision_recall(merge_in_orbis_tnic, percentile_threshold_98=0.2039,
                                  true_y_col='high_etnic3_score', plot_title=''):
    # Calculate ROC Curve
    fpr, tpr, thresholds = roc_curve(merge_in_orbis_tnic[true_y_col],
                                     merge_in_orbis_tnic.score_orbis_tnic)
    roc_auc = roc_auc_score(merge_in_orbis_tnic[true_y_col], merge_in_orbis_tnic.score_orbis_tnic)

    # Calculate F1 scores and find the best threshold
    f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]

    # Calculate Precision-Recall Curve
    prec, recall, pr_thresholds = precision_recall_curve(merge_in_orbis_tnic[true_y_col],
                                                         merge_in_orbis_tnic.score_orbis_tnic)
    junk_classifier = np.sum(merge_in_orbis_tnic[true_y_col]) / len(merge_in_orbis_tnic)

    # Start creating subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # let x and y axes in both subplots range from 0 to 1
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)

    # Plot ROC Curve
    ax1.plot(fpr, tpr, label=f'ROC curve (AUC={roc_auc:.2f})', linewidth=2)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) curve')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random guess')

    # Add twin x-axis for thresholds
    ax1_twin = ax1.twiny()
    ax1_twin.set_xlim(ax1.get_xlim())
    ax1_twin.set_xticks(fpr[np.linspace(0, len(thresholds) - 1, num=5, dtype=int)])
    ax1_twin.set_xticklabels(
            [f'{t:.2f}' for t in thresholds[np.linspace(0, len(thresholds) - 1, num=5, dtype=int)]])
    ax1_twin.set_xlabel('Threshold')

    # Determine the x position for the best threshold on the main x-axis
    best_x_pos_roc = fpr[best_threshold_idx]
    ax1.axvline(best_x_pos_roc, color='r', linestyle='--',
                label=f'Threshold maximizing F1-score={best_threshold:.2f}')

    # line for 98th percentile
    best_x_pos_roc98 = fpr[np.argmin(thresholds >= percentile_threshold_98)]
    ax1.axvline(best_x_pos_roc98, color='g', linestyle='--',
                label=f'Top 2% cutoff={percentile_threshold_98:.2f}')

    # Plot Precision-Recall Curve
    ax2.plot(recall, prec, label=f'Precision-recall curve', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall curve')
    ax2.plot([0, 1], [junk_classifier, junk_classifier], 'k--', linewidth=0.8, label='Random guess')

    # Add twin x-axis for PR thresholds
    ax2_twin = ax2.twiny()
    ax2_twin.set_xlim(ax2.get_xlim())
    ax2_twin.set_xticks(recall[np.linspace(0, len(pr_thresholds) - 1, num=5, dtype=int)])
    ax2_twin.set_xticklabels(
            [f'{t:.2f}' for t in pr_thresholds[np.linspace(0, len(pr_thresholds) - 1, num=5, dtype=int)]])
    ax2_twin.set_xlabel('Threshold')

    # Determine the x position for the best threshold and 98th percentile on the PR curve
    best_x_pos_pr = recall[np.argmax(pr_thresholds >= best_threshold)]
    ax2.axvline(best_x_pos_pr, color='r', linestyle='--',
                label=f'Threshold maximizing F1-score={best_threshold:.2f}')

    # Calculate positions for 98th percentile threshold
    best_x_pos_pr98 = recall[np.argmax(pr_thresholds >= percentile_threshold_98)]

    # Line for 98th percentile
    ax2.axvline(best_x_pos_pr98, color='g', linestyle='--',
                label=f'Top 2% cutoff={percentile_threshold_98:.2f}')

    # Combine legends for the ROC plot
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc='lower right')

    # Combine legends for the PR plot
    handles, labels = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(handles + handles2, labels + labels2, loc='upper right')

    plt.suptitle(plot_title)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

    return best_threshold

plot_roc_and_precision_recall(merge_in_orbis_tnic)

etnic2 = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/Etnic2_data/ETNIC2_data.txt', sep='\t')
etnic2 = etnic2.sort_values(['year', 'gvkey1', 'gvkey2']).drop_duplicates(subset=['gvkey1', 'gvkey2'], keep='last') # keep most recent

gvkeys_etnic2 = set(etnic2.gvkey1).union(set(etnic2.gvkey2))
# unique_gvkeys_etnic = pd.Series(list(unique_gvkeys_etnic)).sort_values()
# unique_gvkeys_etnic2.to_csv('/Users/Jakob/Downloads/Etnic2_data/ETNIC2_unique_gvkeys.csv', index=False)
# unique_gvkeys_etnic2 = pd.read_csv('/Users/Jakob/Downloads/Etnic2_data/ETNIC2_unique_gvkeys.csv').squeeze()

# similarity score is symmetric, remove duplicates
etnic2 = etnic2[etnic2.gvkey1 < etnic2.gvkey2].copy()

shared_gvkeys_etnic2_orbis_tnic = gvkeys_orbis_tnic.intersection(gvkeys_etnic2)

etnic2_shared = etnic2[etnic2.gvkey1.isin(shared_gvkeys_etnic2_orbis_tnic) & etnic2.gvkey2.isin(shared_gvkeys_etnic2_orbis_tnic)].copy()

orbis_tnic_shared = orbis_tnic[orbis_tnic.gvkey1.isin(shared_gvkeys_etnic2_orbis_tnic) & orbis_tnic.gvkey2.isin(shared_gvkeys_etnic2_orbis_tnic)].copy()

merge = orbis_tnic_shared.merge(etnic2_shared, on=['gvkey1', 'gvkey2'], suffixes=('_orbis_tnic', '_etnic2'), how='left', validate='one_to_one')
merge['high_orbis_tnic_score'] = merge.score_orbis_tnic >= 0.2039
merge['high_etnic2_score'] = merge.score_etnic2.notna()
merge_in_orbis_tnic = merge[merge.score_orbis_tnic.notna()]


merge_in_orbis_tnic[['score_etnic2', 'score_orbis_tnic']].sample(100)
merge_in_orbis_tnic['high_etnic2_score'].sum() / merge_in_orbis_tnic.shape[0]
merge_in_orbis_tnic['high_orbis_tnic_score'].sum() / merge_in_orbis_tnic.shape[0]

plot_roc_and_precision_recall(merge_in_orbis_tnic, true_y_col='high_etnic2_score')

# same for tnic
# tnic = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/tnic2_data/tnic2_data.txt', sep='\t')
# tnic = tnic.sort_values(['year', 'gvkey1', 'gvkey2']).drop_duplicates(subset=['gvkey1', 'gvkey2'], keep='last') # keep most recent
#
# # similarity score is symmetric, remove duplicates
# tnic = tnic[tnic.gvkey1 < tnic.gvkey2].copy()
#
# tnic.to_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/tnic2_data/tnic2_data_most_recent_non_redundant.csv', index=False)
tnic = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/tnic2_data/tnic2_data_most_recent_non_redundant.csv')

gvkeys_tnic = set(tnic.gvkey1).union(set(tnic.gvkey2))

shared_gvkeys_tnic_orbis_tnic = gvkeys_orbis_tnic.intersection(gvkeys_tnic)

tnic_shared = tnic[tnic.gvkey1.isin(shared_gvkeys_tnic_orbis_tnic) & tnic.gvkey2.isin(shared_gvkeys_tnic_orbis_tnic)].copy()
orbis_tnic_shared = orbis_tnic[orbis_tnic.gvkey1.isin(shared_gvkeys_tnic_orbis_tnic) & orbis_tnic.gvkey2.isin(shared_gvkeys_tnic_orbis_tnic)].copy()

merge = orbis_tnic_shared.merge(tnic_shared, on=['gvkey1', 'gvkey2'], suffixes=('_orbis_tnic', '_tnic'), how='left', validate='one_to_one')
merge['high_orbis_tnic_score'] = merge.score_orbis_tnic >= 0.2039
merge['high_tnic_score'] = merge.score_tnic.notna()
merge_in_orbis_tnic = merge[merge.score_orbis_tnic.notna()]

plot_roc_and_precision_recall(merge_in_orbis_tnic, true_y_col='high_tnic_score')

# consolidate all etnic data
# etnic_all_path = '/Users/Jakob/Downloads/Etnic_all_data'
# all_files = sorted([f for f in os.listdir(etnic_all_path) if f.endswith('.txt')], reverse=True)
# gvkeys_in_etnic = []
# etnic_all = []
# for f in tqdm.tqdm(all_files):
#     chunk = pd.read_csv(os.path.join(etnic_all_path, f), sep='\t')
#     chunk = chunk[chunk.gvkey1 < chunk.gvkey2].copy()
#     chunk = chunk[chunk.gvkey1.isin(gvkey_in_df) & chunk.gvkey2.isin(gvkey_in_df)].copy()
#     chunk = chunk[~chunk.gvkey1.isin(gvkeys_in_etnic) & ~chunk.gvkey2.isin(gvkeys_in_etnic)].copy()
#     gvkeys_in_chunk = list(set(chunk.gvkey1).union(set(chunk.gvkey2)))
#     gvkeys_in_etnic.extend(gvkeys_in_chunk)
#     etnic_all.append(chunk)
# etnic_all = pd.concat(etnic_all)
# etnic_all = etnic_all.sort_values(['year', 'gvkey1', 'gvkey2'])
# etnic_all.to_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/etnic_all_most_recent_in_orbis_descriptions.csv',
#                      index=False)

etnic_all = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/etnic_all_most_recent_in_orbis_descriptions.csv')

gvkeys_etnic_all = set(etnic_all.gvkey1).union(set(etnic_all.gvkey2))
shared_gvkeys_etnic_all_orbis_tnic = gvkeys_orbis_tnic.intersection(gvkeys_etnic_all)

etnic_all_shared = etnic_all[etnic_all.gvkey1.isin(shared_gvkeys_etnic_all_orbis_tnic) & etnic_all.gvkey2.isin(shared_gvkeys_etnic_all_orbis_tnic)].copy()
orbis_tnic_shared = orbis_tnic[orbis_tnic.gvkey1.isin(shared_gvkeys_etnic_all_orbis_tnic) & orbis_tnic.gvkey2.isin(shared_gvkeys_etnic_all_orbis_tnic)].copy()

merge = orbis_tnic_shared.merge(etnic_all_shared, on=['gvkey1', 'gvkey2'], suffixes=('_orbis_tnic', '_etnic_all'), how='inner', validate='one_to_one')
# merge['high_orbis_tnic_score'] = merge.score_orbis_tnic >= 0.2039
# merge['high_etnic_all_score'] = merge.score_etnic_all >= etnic_all.score.quantile(0.98)

for cutoff in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
    merge['high_orbis_tnic_score'] = merge.score_orbis_tnic >= merge.score_orbis_tnic.quantile(cutoff)
    merge['high_etnic_all_score'] = merge.score_etnic_all >= merge.score_etnic_all.quantile(cutoff)

    best_threshold = plot_roc_and_precision_recall(merge, true_y_col='high_etnic_all_score', plot_title=f'cutoff={cutoff:.2f}')

    print(f'F1 score for cutoff={cutoff:.2f}: {f1_score(merge.high_etnic_all_score, merge.high_orbis_tnic_score)}')

    print(
        f'F1 score for cutoff={cutoff:.2f} and best threshold: '
        f'{f1_score(merge.high_etnic_all_score, (merge.score_orbis_tnic >= best_threshold))},'
        f'adjusted F1: {f1_score(merge.high_etnic_all_score, (merge.score_orbis_tnic >= best_threshold))-merge.high_etnic_all_score.mean()}')

# consolidate tnic all data
# tnic_all_path = '/Users/Jakob/Documents/financial_news_data/tnic_orbis/tnic_all'
# all_files = sorted([f for f in os.listdir(tnic_all_path) if f.endswith('.txt')], reverse=True)
# gvkeys_in_etnic = []
# tnic_all = []
# for f in tqdm.tqdm(all_files):
#     chunk = pd.read_csv(os.path.join(tnic_all_path, f), sep='\t')
#     chunk = chunk[chunk.gvkey1 < chunk.gvkey2].copy()
#     chunk = chunk[chunk.gvkey1.isin(gvkey_in_df) & chunk.gvkey2.isin(gvkey_in_df)].copy()
#     chunk = chunk[~chunk.gvkey1.isin(gvkeys_in_etnic) & ~chunk.gvkey2.isin(gvkeys_in_etnic)].copy()
#     gvkeys_in_chunk = list(set(chunk.gvkey1).union(set(chunk.gvkey2)))
#     gvkeys_in_etnic.extend(gvkeys_in_chunk)
#     tnic_all.append(chunk)
# tnic_all = pd.concat(tnic_all)
# tnic_all = etnic_all.sort_values(['year', 'gvkey1', 'gvkey2'])
# tnic_all.to_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/tnic_all_most_recent_in_orbis_descriptions.csv',
#                      index=False)

tnic_all = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/tnic_all_most_recent_in_orbis_descriptions.csv')

gvkeys_tnic_all = set(tnic_all.gvkey1).union(set(tnic_all.gvkey2))
shared_gvkeys_tnic_all_orbis_tnic = gvkeys_orbis_tnic.intersection(gvkeys_tnic_all)

tnic_all_shared = tnic_all[tnic_all.gvkey1.isin(shared_gvkeys_tnic_all_orbis_tnic) & tnic_all.gvkey2.isin(shared_gvkeys_tnic_all_orbis_tnic)].copy()
orbis_tnic_shared = orbis_tnic[orbis_tnic.gvkey1.isin(shared_gvkeys_tnic_all_orbis_tnic) & orbis_tnic.gvkey2.isin(shared_gvkeys_tnic_all_orbis_tnic)].copy()

merge = orbis_tnic_shared.merge(tnic_all_shared, on=['gvkey1', 'gvkey2'], suffixes=('_orbis_tnic', '_tnic_all'), how='inner', validate='one_to_one')
# merge['high_orbis_tnic_score'] = merge.score_orbis_tnic >= 0.2039
# merge['high_tnic_all_score'] = merge.score_tnic_all >= tnic_all.score.quantile(0.98)


for cutoff in [0.9]: #[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
    merge['high_orbis_tnic_score'] = merge.score_orbis_tnic >= merge.score_orbis_tnic.quantile(cutoff)
    merge['high_tnic_all_score'] = merge.score_tnic_all >= merge.score_tnic_all.quantile(cutoff)

    best_threshold = plot_roc_and_precision_recall(merge, true_y_col='high_tnic_all_score', plot_title=f'cutoff={cutoff:.2f}')

    print(f'F1 score for cutoff={cutoff:.2f}: {f1_score(merge.high_tnic_all_score, merge.high_orbis_tnic_score)}')

    print(f'Precision for cutoff={cutoff:.2f}: {precision_score(merge.high_tnic_all_score, merge.high_orbis_tnic_score)}')

    print(f'Recall for cutoff={cutoff:.2f}: {recall_score(merge.high_tnic_all_score, merge.high_orbis_tnic_score)}')

    print(f'Balanced accuracy for cutoff={cutoff:.2f}: {balanced_accuracy_score(merge.high_tnic_all_score, merge.high_orbis_tnic_score)}')

    print(
        f'F1 score for cutoff={cutoff:.2f} and best threshold: '
        f'{f1_score(merge.high_tnic_all_score, (merge.score_orbis_tnic >= best_threshold))},'
        f'adjusted F1: {f1_score(merge.high_tnic_all_score, (merge.score_orbis_tnic >= best_threshold))-merge.high_tnic_all_score.mean()}')


# Count the occurrences of each pair
# etnic3['gvkey_pair'] = etnic3.apply(lambda row: frozenset((int(row['gvkey1']), int(row['gvkey2']))), axis=1)
# pair_counts = etnic3['gvkey_pair'].value_counts()
# pair_counts.describe() # most pairs only occur twice
# etnic3.drop_duplicates(subset=['gvkey_pair'], keep='first', inplace=True) # keep only one of the duplicates

# read full orbis tnic results
# orbis_tnic_path = r'\Users\Jakob\Documents\financial_news_data\tnic_orbis\replication_code-20220963\PythonCode\output\1742501369605_pv_dbow_dim=300_window=15_epochs=50\similarity'.replace('\\', '/')
# all_files = [f for f in os.listdir(orbis_tnic_path) if f.endswith('.tsv')]
# orbis_tnic = pd.read_csv(os.path.join(orbis_tnic_path, all_files[0]), sep='\t')
# orbis_tnic.columns = ['gvkey1', 'gvkey2', 'score']
# orbis_tnic.score.describe().round(3)





orbis_tnic_path = r'\Users\Jakob\Documents\financial_news_data\tnic_orbis\replication_code-20220963\PythonCode\output\1744410151933_pv_dbow_dim=300_window=15_epochs=50\similarity'.replace('\\', '/')
all_files = [f for f in os.listdir(orbis_tnic_path) if f.endswith('.tsv')]
orbis_tnic = pd.read_csv(os.path.join(orbis_tnic_path, all_files[0]), sep='\t')
orbis_tnic.columns = ['gvkey1', 'gvkey2', 'score']
orbis_tnic.score.describe().round(3)

orbis_tnic = orbis_tnic[orbis_tnic.gvkey1 < orbis_tnic.gvkey2].copy()

gvkeys_orbis_tnic = set(orbis_tnic.gvkey1).union(set(orbis_tnic.gvkey2))

cutoff = 0.23025423288345337 # 0.2039
gvkeys_orbis_tnic_high_scores = set(orbis_tnic[orbis_tnic.score>=cutoff].gvkey1).union(set(orbis_tnic[orbis_tnic.score>=cutoff].gvkey2)) # confirm there are no firms without competitors

path = r"C:\Users\Jakob\Downloads\for Jakob\compustat ID.xlsx".replace('\\', '/')
sample_gvkeys = pd.read_excel(path, header=None, names=['gvkey'])
sample_gvkeys = sample_gvkeys.gvkey.drop_duplicates()
sample_gvkeys_orbis_tnic = sample_gvkeys[sample_gvkeys.isin(gvkeys_orbis_tnic)].copy()
sample_gvkeys_orbis_tnic.to_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/compustat_estimation_sample_gvkeys_covered_by_orbis_trade_descriptions.csv', index=False)
sample_gvkeys_orbis_tnic = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/compustat_estimation_sample_gvkeys_covered_by_orbis_trade_descriptions.csv').squeeze()

estimation_sample_orbis_tnic = orbis_tnic[orbis_tnic.gvkey1.isin(sample_gvkeys_orbis_tnic) & orbis_tnic.gvkey2.isin(sample_gvkeys_orbis_tnic)].copy()
# full similarity matrix was not containing redundant links to save space, have to add them back in
estimation_sample_orbis_tnic = pd.concat([estimation_sample_orbis_tnic, estimation_sample_orbis_tnic.rename(columns={'gvkey1': 'gvkey2', 'gvkey2': 'gvkey1'})], ignore_index=True)


path = r"C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\replication_code-20220963\PythonCode\output_inference_estimation_sample_new_model\tnic_input_files_similarity.tsv".replace('\\', '/')
estimation_sample_orbis_tnic = pd.read_csv(path, sep='\t')
estimation_sample_orbis_tnic.columns = ['gvkey1', 'gvkey2', 'score']
estimation_sample_orbis_tnic.sort_values(['gvkey1', 'gvkey2'], inplace=True)
estimation_sample_orbis_tnic.score.describe()

# calculate median score for each firm
medians = {}
for gvkey in sample_gvkeys_orbis_tnic:
    median = estimation_sample_orbis_tnic[estimation_sample_orbis_tnic.gvkey1 == gvkey].score.median()
    medians[gvkey] = median
medians = pd.Series(medians)
medians.describe()

for gvkey in sample_gvkeys_orbis_tnic:
    estimation_sample_orbis_tnic.loc[estimation_sample_orbis_tnic.gvkey1 == gvkey, 'score'] = estimation_sample_orbis_tnic.loc[estimation_sample_orbis_tnic.gvkey1 == gvkey, 'score'] - medians[gvkey]
    estimation_sample_orbis_tnic.loc[estimation_sample_orbis_tnic.gvkey2 == gvkey, 'score'] = estimation_sample_orbis_tnic.loc[estimation_sample_orbis_tnic.gvkey2 == gvkey, 'score'] - medians[gvkey]


cutoff = 0.01384 # 0.17572 # 0.14 # 0.2039
comp_links = estimation_sample_orbis_tnic[estimation_sample_orbis_tnic.score>=cutoff].copy()
comp_links['score'] = comp_links['score'] - cutoff # subtract 0.2039 to get the adjusted score
comp_links_gvkeys = set(comp_links.gvkey1).union(set(comp_links.gvkey2))
comp_links.to_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/orbis_tnic_compustat_estimation_sample_with_textual_industry_codes.csv', index=False)
comp_links.to_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/orbis_tnic_compustat_estimation_sample_with_textual_industry_codes_minus_median.csv', index=False)

pd.concat([comp_links.gvkey1, comp_links.gvkey2]).value_counts().describe()


num_edges = comp_links.shape[0]
num_nodes = len(comp_links_gvkeys)
(2 * num_edges) / (num_nodes * (num_nodes - 1))

num_edges = etnic3.shape[0]
num_nodes = len(gvkeys_etnic3)
(2 * num_edges) / (num_nodes * (num_nodes - 1)) # density is 4 times higher in ETNIC3

num_edges = tnic3.shape[0]
num_nodes = len(gvkeys_tnic3)
(2 * num_edges) / (num_nodes * (num_nodes - 1)) # density is 4 times higher in TNIC3

# etnic3_in_estimation_sample = etnic3[etnic3.gvkey1.isin(sample_gvkeys) & etnic3.gvkey2.isin(sample_gvkeys)]
# gvkeys_etnic3_in_estimation_sample = set(etnic3_in_estimation_sample.gvkey1).union(set(etnic3_in_estimation_sample.gvkey2))





# read Chih-Sheng TNIC data
# path = r"C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\TNIC-2020.mat".replace('\\', '/')
#
# import scipy.io
# tnic = scipy.io.loadmat(path)['ans']
# tnic = pd.DataFrame(tnic)
# tnic.columns = ['gvkey1', 'gvkey2', 'score']
# tnic[['gvkey1', 'gvkey2']] = tnic[['gvkey1', 'gvkey2']].astype(int)
# tnic_sample = tnic[tnic.gvkey1.isin(sample_gvkeys_orbis_tnic) & tnic.gvkey2.isin(sample_gvkeys_orbis_tnic)].copy()

# read TNIC3
path = r"C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\tnic3_data\tnic3_data.txt".replace('\\', '/')
tnic3 = pd.read_csv(path, sep='\t')
tnic3 = tnic3.sort_values(['year', 'gvkey1', 'gvkey2']).drop_duplicates(subset=['gvkey1', 'gvkey2'], keep='last') # keep most recent
tnic3.dropna(subset=['score'], inplace=True)
gvkeys_tnic3 = set(tnic3.gvkey1).union(set(tnic3.gvkey2))
tnic3.score.mean()
tnic3_sample = tnic3[tnic3.gvkey1.isin(sample_gvkeys) & tnic3.gvkey2.isin(sample_gvkeys)].copy()
tnic3_sample_gvkeys = list(set(tnic3_sample.gvkey1).union(set(tnic3_sample.gvkey2)))

# plot histogram of scores
for df, label in [(tnic3_sample, 'TNIC3 sample'), (comp_links, 'Orbis TNIC')]:
    plt.figure(figsize=(10, 6))
    plt.hist(df.score, bins=100)
    plt.xlim(0, 1)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(f'{label} Score Distribution')
    plt.show()

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

tnic_sample_g = nx.from_pandas_edgelist(tnic3_sample, 'gvkey1', 'gvkey2', ['score'])
estimation_sample_orbis_tnic_graph = nx.from_pandas_edgelist(comp_links, 'gvkey1', 'gvkey2', ['score'])

nx.is_directed(tnic_sample_g)
tnic_sample_g.number_of_edges()
tnic_sample_g.number_of_nodes()
nx.density(tnic_sample_g)

nx.is_directed(estimation_sample_orbis_tnic_graph)
estimation_sample_orbis_tnic_graph.number_of_edges()
estimation_sample_orbis_tnic_graph.number_of_nodes()
nx.density(estimation_sample_orbis_tnic_graph)

# plot degree histogram
for g, label in [(tnic_sample_g, 'TNIC3 sample'), (estimation_sample_orbis_tnic_graph, 'Orbis TNIC')]:
    degrees = [g.degree(n) for n in g.nodes()]
    plt.figure(figsize=(10, 6))
    sns.histplot(degrees, bins=200)
    # plt.xlim(0, 100)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title(f'{label} Degree Distribution')
    plt.show()

    # degree_freq = nx.degree_histogram(g)
    # degrees = range(len(degree_freq))
    # plt.figure(figsize=(12, 8))
    # plt.plot(degrees, degree_freq,'go-')
    # plt.xlabel('Degree')
    # plt.ylabel('Frequency')
    # plt.show()

    sdc['Parti.CUSIP'].explode().sample(10)