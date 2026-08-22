import pandas as pd
import os

path = r"C:\Users\Jakob\Documents\financial_news_data\compustat_llama_summaries".replace("\\", "/")
files = os.listdir(path)

df = pd.concat([pd.read_csv(os.path.join(path, file)) for file in files], ignore_index=True)
for col in ['llm_paras', 'prod_nms']:
    df[col] = df[col].astype(str).str.replace("'", "").str.replace('"', "").str.replace("[", "").str.replace("nan", "").str.replace(",", "")

df['text'] = df['llm_paras'] + " " + df['prod_nms']

df = df.groupby('url_host_registered_domain').text.agg(lambda x: ' '.join(x)) # 34799 firms/domains

df = df.str.replace('(', '').str.replace(')', '')
df = df.str.replace('Please wait while your request is being verified', '')
df = df.str.replace('empty string', '')
df = df.str.replace('Error Page cannot be displayed Please contact your service provider for more details', '')
df = df.str.replace('<relevant text 1> 2> must be in the format: <main product or service>', '')
df = df.str.replace('please prove that you are human', '')
df = df.str.replace('nstream', '')

df = df.str.lower()

def remove_duplicate_words(s):
    if pd.isna(s):
        return s
    seen = set()
    return ' '.join([word for word in s.split() if not (word in seen or seen.add(word))])

df = df.apply(remove_duplicate_words)

def remove_single_letter_words(s):
    if pd.isna(s):
        return s
    return ' '.join([word for word in s.split() if len(word) > 1])

df = df.apply(remove_single_letter_words)

from sklearn.feature_extraction.text import CountVectorizer
corpus = df.tolist()
vectorizer = CountVectorizer(ngram_range=(2, 3), min_df=0.01, stop_words='english', lowercase=True)
X = vectorizer.fit_transform(corpus)
ngram_counts = X.sum(axis=0).A1 # Sum up the counts of each n-gram
ngram_names = vectorizer.get_feature_names_out()

# Create a DataFrame of n-grams and their counts
ngram_df = pd.DataFrame({
    'ngram': ngram_names,
    'count': ngram_counts
}).sort_values(by='count', ascending=False).reset_index(drop=True)

# remove state abbreviations
state_abbr = ['al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'dc', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia',
              'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh',
              'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc',
              'sd','tn','tx','ut','vt','va','wa','wv','wi','wy']

df = df.replace(r'\b(' + '|'.join(state_abbr) + r')\b', '', regex=True)

# remove most common cities
common_cities = ['new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia',
                 'san', 'antonio', 'san', 'diego', 'dallas', 'san', 'jose', 'austin', 'jacksonville',
                 'fort', 'worth', 'columbus', 'cincinnati', 'charlotte', 'indianapolis', 'seattle',
                 'livermore', 'chandler', 'newark', 'ontario', 'usa', 'north america']

df = df.replace(r'\b(' + '|'.join(common_cities) + r')\b', '', regex=True)

# remove all text within <> brackets
df = df.str.replace(r'<[^>]+>', '', regex=True)

# remove all whitespace
df = df.str.replace(r'\s+', ' ', regex=True)
# remove new line characters
df = df.str.replace('\n', ' ').str.replace('\r', ' ').str.strip()

df = df[df.str.len()>5] # 29095 firms/domains with text

compustat_global_urls = pd.read_csv(r"C:\Users\Jakob\Documents\financial_news_data\compustat-global-urls.csv".replace("\\", "/"))
compustat_na_urls = pd.read_csv(r"C:\Users\Jakob\Documents\financial_news_data\compustat-na-urls.csv".replace("\\", "/"))
compustat_urls = pd.concat([compustat_global_urls, compustat_na_urls], ignore_index=True)
del compustat_global_urls, compustat_na_urls

compustat_urls = compustat_urls[['gvkey', 'weburl']].copy()
compustat_urls.drop_duplicates(inplace=True)
compustat_urls.dropna(inplace=True)

compustat_urls['weburl'] = compustat_urls['weburl'].str.replace('www.', '').str.replace('http://', '').str.replace('https://', '').str.replace('www2.', '').str.replace('www3.', '')
compustat_urls = compustat_urls.set_index('weburl').squeeze()

df = df.to_frame().join(compustat_urls, how='left')
df.dropna(inplace=True)
df.set_index('gvkey', inplace=True)
df.index = df.index.astype(int)
df.sort_index(inplace=True)
df = df.squeeze()

df.to_csv(r"C:\Users\Jakob\Documents\financial_news_data\compustat_website_product_descriptions.csv".replace("\\", "/"))


# bert similarities all
path = r"C:\Users\Jakob\Documents\financial_news_data\bert_sims_all.csv".replace("\\", "/")
df = pd.read_csv(path)
df.drop(columns=['Unnamed: 0'], inplace=True)

df.describe().round(2)

# check that reverse is contained
# df_reverse = df.rename(columns={'source': 'target', 'target': 'source'})
# assert len(df.merge(df_reverse, how='inner', on=['source', 'target'], validate='one_to_one')) == len(df)
# del df_reverse

# take top 2 percent of scores, since relations are contained both ways that means cutoff is 96th percentile
cutoff = df.score.quantile(0.96)

df_full_gvkeys = pd.concat([df.source, df.target]).drop_duplicates()
n_firms_full = len(df_full_gvkeys)
df_full_comp = compustat.merge(df_full_gvkeys.rename('gvkey'), how='inner')
df_full_comp['loc'].value_counts()

### construct competition matrix based on shared membership of 3-digit primary or secondary NAICS/NACE
path = r"C:\Users\Jakob\Documents\financial_news_data\for Jakob\compustat ID.xlsx".replace('\\', '/')
sample_gvkeys = pd.read_excel(path, header=None, names=['gvkey'])
sample_gvkeys = sample_gvkeys.gvkey.drop_duplicates()

df_sample = df[df.source.isin(sample_gvkeys) & df.target.isin(sample_gvkeys)].copy()
df_sample = df_sample.sort_values(['source', 'target']).reset_index(drop=True)
assert len(df_sample[df_sample.source!=df_sample.target])==len(df_sample) # check that no self-links contained

n_firms = pd.concat([df_sample.source, df_sample.target]).nunique()
print(f"Number of firms in sample: {len(sample_gvkeys)}, number of firms covered by Bert product description similarities: {n_firms}")

df_sample = df_sample[df_sample.score >= cutoff].copy()

df_sample.to_csv('/Users/Jakob/Documents/financial_news_data/website_product_descriptions_bert_similarities_compustat_sample.csv', index=False)


## tobias synthetic 10 k filings
path = r"C:\Users\Jakob\Documents\financial_news_data\0503-2018-10K_filings-parsed_results.csv"
df = pd.read_csv(path.replace('\\', '/'))

compustat_global = pd.read_csv('C:/Users/Jakob/Documents/compustat-global.csv.gz')
compustat_na = pd.read_csv('C:/Users/Jakob/Documents/compustat-north-america.csv.gz')
compustat_na = compustat_na[
    ['gvkey', 'fyear', 'datadate', 'emp', 'revt', 'conm', 'conml', 'cusip', 'loc', 'naics', 'weburl']]
compustat = pd.concat([compustat_na, compustat_global])
del compustat_na, compustat_global
compustat = compustat[['weburl', 'gvkey', 'loc']]
compustat = compustat.drop_duplicates().dropna()
# compustat.drop_duplicates(subset=['weburl'], inplace=True)
compustat.weburl = compustat.weburl.str.replace('www.', '').str.replace('http://', '').str.replace('https://', '').str.replace('www2.', '').str.replace('www3.', '')

df = df.merge(compustat.reset_index(), left_on='firm_domain', right_on='weburl', how='inner')
df['loc'].value_counts()

