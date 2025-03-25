import pandas as pd
import os
import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from matplotlib import pyplot as plt
import numpy as np

    # run this the first time to consolidate all tnic data
# all_files = os.listdir('/Users/Jakob/Documents/financial_news_data/tnic_orbis')
# chunks = [f for f in all_files if f.startswith('Pub')]
# df = pd.concat([pd.read_excel(os.path.join('/Users/Jakob/Documents/financial_news_data/tnic_orbis', chunk)) for chunk in chunks])
#
# rename_dict = {'Company name Latin alphabet' : 'firm_name', 'BvD ID number': 'bvdid',
#                'NACE Rev. 2, primary code(s)': 'nace_rev_2_core_code_4_digits',
#                'US SIC, primary code(s)'     : 'sic_primary_code', 'Country ISO code': 'country',
#                'ISIN number'                 : 'isin', 'Trade description (English)': 'trade_description',
#                'Products & services'         : 'products_and_services', 'Specialisation': 'specialisation',
#                'Website address'             : 'website'}
#
# df.rename(columns=rename_dict, inplace=True)
# df = df[rename_dict.values()].copy()
#
# df.sic_primary_code = df.sic_primary_code.str.split('\n').str[0].str.strip().str[:2]
# sic_in_data = df.sic_primary_code.dropna().unique()
#
# description_cols = ['trade_description', 'products_and_services', 'specialisation']
# for col in description_cols:
#     # nan to empty string
#     df[col] = df[col].fillna('')
#     df[col] = df[col].astype(str)
#     remove_stings = [' [source: Bureau van Dijk]', '\n', '\r', '\t']
#     for s in remove_stings:
#         df[col] = df[col].str.replace(s, ' ')
#
#     # remove double periods, double spaces, double commas
#     df[col] = df[col].str.replace('..', '.').str.replace('  ', ' ').str.replace(',,', ',')
#
#     df[col] = df[col].str.strip()
#
# df['description'] = df[description_cols].apply(lambda x: '. '.join(x), axis=1).str.replace('..', '.')
#
# # match to gvkey via isin/cusip
# compustat_global_sedol_isin = pd.read_excel('C:/Users/Jakob/Documents/compustat-global-sedol-isin.xlsx')
# compustat_global_sedol_isin.columns = ['gvkey', 'fyear', 'datadate', 'isin', 'sedol']
# compustat_global_sedol_isin.dropna(subset=['isin', 'sedol'], inplace=True)
# compustat_global_sedol_isin.drop_duplicates(subset=['gvkey', 'isin', 'sedol'], keep='first', inplace=True)
# compustat_global_sedol_isin = compustat_global_sedol_isin[['gvkey', 'isin']].dropna().drop_duplicates()
# compustat_global_sedol_isin = compustat_global_sedol_isin.set_index('isin').squeeze()
#
# df['gvkey'] = df['isin'].map(compustat_global_sedol_isin)
#
# compustat_na_cusip = pd.read_csv('C:/Users/Jakob/Downloads/compustat-na-cusip.csv', usecols=['gvkey', 'cusip', 'loc']).drop_duplicates().dropna()
# compustat_na_cusip = compustat_na_cusip[compustat_na_cusip['loc'].isin(['USA', 'CAN'])].copy()
# compustat_na_cusip['loc'] = compustat_na_cusip['loc'].replace({'USA': 'US', 'CAN': 'CA'})
# compustat_na_cusip['isin_11_digits'] = compustat_na_cusip['loc'] + compustat_na_cusip['cusip'].astype(str)
# compustat_na_cusip.to_csv('C:/Users/Jakob/Documents/financial_news_data/tnic_orbis/compustat_na_cusip_isin.csv', index=False)
#
# df['gvkey2'] = df['isin'].str[:11].map(compustat_na_cusip.set_index('isin_11_digits')['gvkey'])
#
# df['gvkey'] = df['gvkey'].fillna(df['gvkey2'])
# df.drop(columns=['gvkey2'], inplace=True)
# df['gvkey'] = df['gvkey'].astype(int)
#
# df.to_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/tnic_orbis_input_full.csv', index=False)

df = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/tnic_orbis_input_full.csv')
gvkey_in_df = df.gvkey.drop_duplicates().sort_values()
print(f'Unique gvkeys: {len(gvkey_in_df)}')

# unique_gvkeys_etnic = pd.read_csv('/Users/Jakob/Documents/financial_news_data/tnic_orbis/Etnic2_data/ETNIC2_unique_gvkeys.csv').squeeze()
# df_in_etnic = df[df.gvkey.isin(unique_gvkeys_etnic)].copy() # DON'T DO THIS because we will only end up with competitors in training data

# create sample for fast iteration
path = '/Users/Jakob/Documents/financial_news_data/tnic_orbis/data_sample/tnic_input_files/'
sample = df.sample(5000)
# save each firm as a separate file
for i, row in tqdm.tqdm(sample.iterrows(), total=len(sample)):
    with open(os.path.join(path, str(row.gvkey)), 'w') as f:
        f.write(row.description)

path = '/Users/Jakob/Documents/financial_news_data/tnic_orbis/data/tnic_input_files/'
# save each firm as a separate file
for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    with open(os.path.join(path, str(row.gvkey)), 'w') as f:
        f.write(row.description)


df_overlap = df[df.gvkey.isin(unique_gvkeys_etnic)].copy()
path = '/Users/Jakob/Documents/financial_news_data/tnic_orbis/data_overlap/tnic_input_files/'
# save each firm as a separate file
for i, row in tqdm.tqdm(df_overlap.iterrows(), total=len(df_overlap)):
    with open(os.path.join(path, str(row.gvkey)), 'w') as f:
        f.write(row.description)

# run this command with replication code
# python train.py --data_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\data --output_dir output --algorithm pv_dbow --negative 0 --min-count 3 --window 15 --hs 1 --vector-size 300 --epochs 50 --threshold 0.2

# debug
# python train.py --data_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\data_sample --output_dir output --algorithm pv_dbow --negative 0 --min-count 3 --window 15 --hs 1 --vector-size 300 --epochs 50 --threshold 0.2

# run this command for inference
# python multi_inference.py --data_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\data_overlap --output_dir output_inference --model_dir C:\Users\Jakob\Documents\financial_news_data\tnic_orbis\replication_code-20220963\PythonCode\output\1742543273684_pv_dbow_dim=300_window=15_epochs=50\models --number_of_processes_run_inference 8

# read orbis tnic results
orbis_tnic_path = '/Users/Jakob/Documents/financial_news_data/tnic_orbis/replication_code-20220963/PythonCode/output_inference/'
all_files = [f for f in os.listdir(orbis_tnic_path) if f.endswith('.tsv')]
orbis_tnic = pd.read_csv(os.path.join(orbis_tnic_path, all_files[0]), sep='\t')
orbis_tnic.columns = ['gvkey1', 'gvkey2', 'score']
orbis_tnic = orbis_tnic[orbis_tnic.gvkey1 < orbis_tnic.gvkey2].copy()

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

for cutoff in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
    merge['high_orbis_tnic_score'] = merge.score_orbis_tnic >= merge.score_orbis_tnic.quantile(cutoff)
    merge['high_tnic_all_score'] = merge.score_tnic_all >= merge.score_tnic_all.quantile(cutoff)

    best_threshold = plot_roc_and_precision_recall(merge, true_y_col='high_tnic_all_score', plot_title=f'cutoff={cutoff:.2f}')

    print(f'F1 score for cutoff={cutoff:.2f}: {f1_score(merge.high_tnic_all_score, merge.high_orbis_tnic_score)}')

    print(
        f'F1 score for cutoff={cutoff:.2f} and best threshold: '
        f'{f1_score(merge.high_tnic_all_score, (merge.score_orbis_tnic >= best_threshold))},'
        f'adjusted F1: {f1_score(merge.high_tnic_all_score, (merge.score_orbis_tnic >= best_threshold))-merge.high_tnic_all_score.mean()}')


# Count the occurrences of each pair
# etnic3['gvkey_pair'] = etnic3.apply(lambda row: frozenset((int(row['gvkey1']), int(row['gvkey2']))), axis=1)
# pair_counts = etnic3['gvkey_pair'].value_counts()
# pair_counts.describe() # most pairs only occur twice
# etnic3.drop_duplicates(subset=['gvkey_pair'], keep='first', inplace=True) # keep only one of the duplicates
