import pandas as pd
import matplotlib.pyplot as plt

# read Thomson SDC RND database
kb = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/rnd_fin.parquet.gzip')

# read news articles
corpus = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/news.parquet.gzip')

# distribution of deal text lengths
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
kb.text.str.len().hist(bins=30)
plt.title('Length of deal texts')
plt.xlabel('Number of characters')
plt.ylabel('Frequency')
plt.show()

# Time distribution of articles
plt.figure(figsize=(18,10))
corpus.groupby(pd.to_datetime(corpus.date).dt.to_period('M')).date.count().plot(kind="bar")
plt.xticks(rotation=45)
plt.title('Number of news articles per month')
# plt.locator_params(nbins=30)
plt.show()