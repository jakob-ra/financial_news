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