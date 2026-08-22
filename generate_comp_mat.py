# File to generate different competitor networks from the BERT embeddings
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import glob

path = r"C:\Users\Jakob\Documents\financial_news_data\for Jakob\compustat ID.xlsx".replace('\\', '/')
sample_gvkeys = pd.read_excel(path, header=None, names=['gvkey'])
sample_gvkeys = sample_gvkeys.gvkey.drop_duplicates()
sample_gvkeys.to_csv("/Users/Jakob/Downloads/sample_gvkeys.txt", index=False, header=False)

compustat_global = pd.read_csv('C:/Users/Jakob/Documents/compustat-global.csv.gz')
compustat_na = pd.read_csv('C:/Users/Jakob/Documents/compustat-north-america.csv.gz')
compustat_na = compustat_na[
    ['gvkey', 'fyear', 'datadate', 'emp', 'revt', 'conm', 'conml', 'cusip', 'loc', 'naics', 'weburl']]
compustat = pd.concat([compustat_na, compustat_global])
del compustat_na, compustat_global

path = r"C:\Users\Jakob\Documents\financial_news_data\compustat-na-urls.csv".replace('\\', '/')
compustat_na_urls = pd.read_csv(path)
compustat_na_urls = compustat_na_urls[['gvkey', 'weburl']].drop_duplicates().dropna()
path = r"C:\Users\Jakob\Documents\financial_news_data\compustat-global-urls.csv".replace('\\', '/')
compustat_global_urls = pd.read_csv(path)
compustat_global_urls = compustat_global_urls[['gvkey', 'weburl']].drop_duplicates().dropna()
compustat_urls2 = pd.concat([compustat_na_urls, compustat_global_urls], ignore_index=True)
del compustat_na_urls, compustat_global_urls

compustat_urls = compustat[['gvkey', 'weburl']].drop_duplicates().dropna()
compustat_urls = pd.concat([compustat_urls, compustat_urls2], ignore_index=True).drop_duplicates()
del compustat_urls2

path = r"C:\Users\Jakob\Documents\financial_news_data\compustat_na_lexis.csv".replace('\\', '/')
compustat_na_lexis = pd.read_csv(path)
path = r"C:\Users\Jakob\Documents\financial_news_data\compustat_global_lexis.csv".replace('\\', '/')
compustat_global_lexis = pd.read_csv(path)
compustat_lexis = pd.concat([compustat_na_lexis, compustat_global_lexis], ignore_index=True)
del compustat_na_lexis, compustat_global_lexis
compustat_lexis_urls = compustat_lexis[['gvkey', 'weburl']].drop_duplicates().dropna()

compustat_urls = pd.concat([compustat_urls, compustat_lexis_urls], ignore_index=True).drop_duplicates()

path = r"C:\Users\Jakob\Documents\SECO_COVID_Data\output 2024\data\compustat-quarterly-for-cc-regression-intermediate.pkl".replace('\\', '/')
compustat_quarterly = pd.read_pickle(path)
compustat_quarterly = compustat_quarterly[['gvkey', 'url']].drop_duplicates().dropna()
compustat_quarterly.rename(columns={'url': 'weburl'}, inplace=True)
compustat_urls = pd.concat([compustat_urls, compustat_quarterly], ignore_index=True).drop_duplicates()

path = r"C:\Users\Jakob\Downloads\comp_us.csv".replace('\\', '/')
comp_us = pd.read_csv(path)
comp_us.rename(columns={'Global Company Key': 'gvkey', 'Web URL': 'weburl'}, inplace=True)
comp_us = comp_us[['gvkey', 'weburl']].drop_duplicates().dropna()
compustat_urls = pd.concat([compustat_urls, comp_us], ignore_index=True).drop_duplicates()

# export compustat urls
compustat_urls.to_csv('/Users/Jakob/Documents/financial_news_data/unique_compustat_urls_union_global_north_america.csv', index=False)

path = r"C:\Users\Jakob\Documents\financial_news_data\gvkeys_with_rnd_stock.csv".replace('\\', '/')
compustat_ids_with_r_n_d = pd.read_csv(path).squeeze().drop_duplicates().astype(str)

dict_df = compustat_urls[compustat_urls.gvkey.isin(sample_gvkeys)].copy()
dict_df = compustat_urls[compustat_urls.gvkey.astype(str).isin(compustat_ids_with_r_n_d)].copy()
dict_df['weburl'] = dict_df['weburl'].str.replace('www.', '').str.lower()
dict_df['weburl'] = dict_df['weburl'].str.split('/').str[0]  # only take the host registered domain

dict_df.drop_duplicates(inplace=True)

# read in url-gvkey crosswalk that you want to use to match your data
url_gvkey_dict = dict(zip(dict_df.weburl, dict_df.gvkey))
urls_to_use = dict_df.weburl # for trimming

# reading in all the raw bert embeddings from the files
extension = 'parquet'
embeddings_path = r"C:\Users\Jakob\Downloads\bert_sims_2023".replace('\\', '/')
all_filenames = [os.path.join(embeddings_path, f) for f in os.listdir(embeddings_path) if f.endswith(extension)]
# first, drop any predictions < 0.3 - this is the threshold rule we use to determine
# pages where we cannot make a prediction
df_lst = [pd.read_parquet(i).loc[:,["url_host_registered_domain","probs","embeddings"]] for i in all_filenames]
df = pd.concat([i[i.probs >= 0.3] for i in df_lst])
del(df_lst) # working memory can fill up quickly, so use dels to keep it light

# lets start by only trimming the url_host_registered_domains to the ones we can match 
df = df[df.url_host_registered_domain.isin(urls_to_use)]
mean_pooled_embeddings = df.groupby('url_host_registered_domain')['embeddings'].apply(lambda x: np.mean(np.stack(x), axis=0))

ids = mean_pooled_embeddings.index #
embed_mat = np.stack(mean_pooled_embeddings.values) # now build the matrix
cos_sim_matrix = cosine_similarity(embed_mat)
# now lets see if we can make a cosine similarity matrix out of this
np.fill_diagonal(cos_sim_matrix, 0)
edgelist = []
for i in range(len(ids)):
    for j in range(len(ids)):  # only take upper triangle, exclude self-similarity
        edgelist.append((ids[i], ids[j], cos_sim_matrix[i, j]))
del(cos_sim_matrix)

edgelist_df = pd.DataFrame(edgelist, columns=['source', 'target', 'score'])
#edgelist_df.to_csv()

edgelist_df.source = edgelist_df.source.apply(lambda x: url_gvkey_dict[x])
edgelist_df.target = edgelist_df.target.apply(lambda x: url_gvkey_dict[x])

edgelist_df = edgelist_df[edgelist_df.target != edgelist_df.source]

top_score_percentiles = 5
tot_entries_use = edgelist_df.shape[0]//(100//2)

edgelist_df = edgelist_df.sort_values(by=['score'], ascending=False)
edgelist_df = edgelist_df.iloc[0:tot_entries_use,:]

# to arrange it nicely by gvkey
edgelist_df = edgelist_df.sort_values(by=['source', 'target'], ascending=True)

pd.concat([edgelist_df.source, edgelist_df.target]).nunique()  # check how many unique gvkeys we have

# edgelist_df.to_csv(f"/Users/Jakob/Documents/financial_news_data/bert_sims_compustat_sample_top_{top_score_percentiles}_percent.csv", index=False)
edgelist_df.to_csv(f"/Users/Jakob/Documents/financial_news_data/bert_sims_compustat_with_r_n_d_expenses_top_{top_score_percentiles}_percent.csv", index=False)

pd.Series(mean_pooled_embeddings.index).apply(lambda x: url_gvkey_dict[x]).to_csv("/Users/Jakob/Documents/financial_news_data/bert_sims_compustat_sample_gvkeys_covered.csv", index=False, header=False)
pd.Series(mean_pooled_embeddings.index).apply(lambda x: url_gvkey_dict[x]).to_csv("/Users/Jakob/Documents/financial_news_data/bert_sims_compustat_with_r_n_d_expenses_gvkeys_covered.csv", index=False, header=False)

# gvkeys_in_embeddings = pd.Series(mean_pooled_embeddings.index).apply(lambda x: url_gvkey_dict[x])
# compustat[compustat.gvkey.isin(gvkeys_in_embeddings) & (compustat.fyear==2021.0)]['loc'].value_counts()

## Kieran new files (Gemma)
# reading in all the raw Gemma embeddings from the files
extension = 'parquet'
df = []
for year in range(2018, 2024):
    embeddings_path = f"C:/Users/Jakob/Downloads/jakob_gemma_preds/{year}"
    all_filenames = [os.path.join(embeddings_path, f) for f in os.listdir(embeddings_path) if f.endswith(extension)]
    # first, drop any predictions < 0.3 - this is the threshold rule we use to determine
    # pages where we cannot make a prediction
    df_lst = [pd.read_parquet(i).loc[:,["url_host_registered_domain","probs","embeddings"]] for i in all_filenames]
    df.extend([i[i.probs >= 0.3] for i in df_lst])
    del (df_lst)  # working memory can fill up quickly, so use dels to keep it light

df = pd.concat(df)

compustat_global_urls = pd.read_csv(r"C:\Users\Jakob\Documents\financial_news_data\compustat-global-urls.csv".replace("\\", "/"))
compustat_na_urls = pd.read_csv(r"C:\Users\Jakob\Documents\financial_news_data\compustat-na-urls.csv".replace("\\", "/"))
compustat_urls = pd.concat([compustat_global_urls, compustat_na_urls], ignore_index=True)
del compustat_global_urls, compustat_na_urls

compustat_urls = compustat_urls[['gvkey', 'weburl']].copy()
compustat_urls.drop_duplicates(inplace=True)
compustat_urls.dropna(inplace=True)

compustat_urls['weburl'] = compustat_urls['weburl'].str.replace('www.', '').str.replace('http://', '').str.replace('https://', '').str.replace('www2.', '').str.replace('www3.', '')
compustat_urls = compustat_urls.set_index('weburl').squeeze()

# lets start by only trimming the url_host_registered_domains to the ones we can match
df = df[df.url_host_registered_domain.isin(compustat_urls.index)]
mean_pooled_embeddings = df.groupby('url_host_registered_domain')['embeddings'].apply(lambda x: np.mean(np.stack(x), axis=0))

ids = mean_pooled_embeddings.index #
embed_mat = np.stack(mean_pooled_embeddings.values) # now build the matrix
cos_sim_matrix = cosine_similarity(embed_mat)
# now lets see if we can make a cosine similarity matrix out of this
np.fill_diagonal(cos_sim_matrix, 0)
edgelist = []
for i in range(len(ids)):
    for j in range(len(ids)):  # only take upper triangle, exclude self-similarity
        edgelist.append((ids[i], ids[j], cos_sim_matrix[i, j]))
del(cos_sim_matrix)

edgelist_df = pd.DataFrame(edgelist, columns=['source', 'target', 'score'])
#edgelist_df.to_csv()

top_score_percentiles = 5
tot_entries_use = edgelist_df.shape[0]//(100//top_score_percentiles)
edgelist_df = edgelist_df.sort_values(by=['score'], ascending=False)
edgelist_df = edgelist_df.iloc[0:tot_entries_use,:]

# map to gvkeys
compustat_map = compustat_urls.reset_index()
compustat_map['in_sample'] = compustat_map.gvkey.isin(sample_gvkeys)
compustat_map.sort_values(by=['in_sample', 'gvkey'], ascending=False, inplace=True)
compustat_map = compustat_map.drop_duplicates(subset=['weburl'], keep='first').set_index('weburl').gvkey.squeeze()

edgelist_df.source = edgelist_df.source.map(compustat_map)
edgelist_df.target = edgelist_df.target.map(compustat_map)

edgelist_df = edgelist_df[edgelist_df.target != edgelist_df.source]

# to arrange it nicely by gvkey
edgelist_df = edgelist_df.sort_values(by=['source', 'target'], ascending=True)

pd.concat([edgelist_df.source, edgelist_df.target]).nunique()  # check how many unique gvkeys we have

# edgelist_df.to_csv(f"/Users/Jakob/Documents/financial_news_data/bert_sims_compustat_sample_top_{top_score_percentiles}_percent.csv", index=False)
edgelist_df.to_csv(f"/Users/Jakob/Documents/financial_news_data/gemma_sims_compustat_top_{top_score_percentiles}_percent.csv", index=False)

edgelist_df_sample = edgelist_df[edgelist_df.source.isin(sample_gvkeys) & edgelist_df.target.isin(sample_gvkeys)]

edgelist_df_sample.to_csv(f"/Users/Jakob/Documents/financial_news_data/gemma_sims_compustat_sample_top_{top_score_percentiles}_percent.csv", index=False)

# cut down to top 2 percent
tot_entries_use = int(edgelist_df_sample.shape[0]*2/5)
edgelist_df_sample = edgelist_df_sample.sort_values(by=['score'], ascending=False)
edgelist_df_sample = edgelist_df_sample.iloc[0:tot_entries_use,:]

edgelist_df_sample.to_csv(f"/Users/Jakob/Documents/financial_news_data/gemma_sims_compustat_sample_top_{2}_percent.csv", index=False)

gvkeys_covered = pd.concat([edgelist_df_sample.source, edgelist_df_sample.target]).drop_duplicates()
gvkeys_covered.to_csv("/Users/Jakob/Documents/financial_news_data/gemma_sims_compustat_sample_gvkeys_covered.csv", index=False, header=False)
