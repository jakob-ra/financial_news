import pandas as pd
from nltk.tokenize import sent_tokenize

df = pd.read_pickle('C:/Users/Jakob/Documents/lexisnexis_firm_alliances_combined_new.pkl')

# keep only year 2017
df_2017 = df[df.publication_date.dt.year == 2017]

# keep only english
df_2017 = df_2017[df_2017.lang == 'en']

# combine title and content
# df_2017['text'] = df_2017.title + '. ' + df_2017.content

# split into sentences
df_2017['sentences'] = df_2017.content.apply(sent_tokenize)
df_2017['num_sentences'] = df_2017.sentences.apply(len)

df_2017.num_sentences.describe()

# cut off docs at maximum length (in sentences)
max_len = 5
df_2017['sentences'] = df_2017.sentences.apply(lambda x: x[:max_len])
df_2017['content'] = df_2017.sentences.apply(lambda x: ' '.join(x))
df_2017.drop(columns=['sentences'], inplace=True)

# focus on news mentioning at least 2 companies
df_2017 = df_2017[df_2017.company.str.len() > 1]

df_2017.to_pickle('C:/Users/Jakob/Documents/lexisnexis_firm_alliances_2017_cleaned_min_2_companies.pkl', protocol=4)








# focus on news about JVs, alliances, and agreements
# df.subject.explode().value_counts().head(50)
# df[df.subject.apply(lambda x: [sub for sub in x if sub in ['JOINT VENTURES', 'ALLIANCES & PARTNERSHIPS', 'AGREEMENTS']]).str.len() > 0]

