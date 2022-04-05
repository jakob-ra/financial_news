import pandas as pd
from nltk.tokenize import sent_tokenize
from firm_name_matching import firm_name_clean
from prepare_train_test import extract_firm_name_and_spans, clean_unique_entities

df = pd.read_pickle('C:/Users/Jakob/Documents/lexisnexis_firm_alliances_combined_new.pkl')

# keep only year 2017
df = df[df.publication_date.dt.year == 2017]

# keep only english
df = df[df.lang == 'en']

# combine title and content (not worth it because of repeating entity pairs
# df['text'] = f.title + '. ' + df.content

# cut off docs at 500 chars
df['content'] = df.content.str[:500]

# focus on news mentioning at least 2 companies
df = df[df.company.str.len() > 1]

# df.to_pickle('C:/Users/Jakob/Documents/lexisnexis_firm_alliances_2017_cleaned_min_2_companies.pkl', protocol=4)

orbis = pd.read_csv('C:/Users/Jakob/Documents/Orbis/firm_lookup_list.csv.gzip', compression='gzip')

orbis = set(orbis.company.to_list())

df['matched_ents'] = df.progress_apply(lambda x: extract_firm_name_and_spans(x.content, x.company), axis=1)

df['ents'] = df.matched_ents + df.ents_pred
df.ents.apply(len).mean()

df['ents'] = df.ents.apply(clean_unique_entities)

# filter out entities not in orbis
df['ents'] = df.ents.apply(lambda ents: [ent for ent in ents if ent[2] in orbis])
df.ents.apply(len).mean()

df = df[df.ents.apply(len) > 1].copy() # need at least two unique identified participants
df.shape

# df['ents'] = df.ents.str[:3] # take only first three entities for each article

#We have multiple detected entities in each row. However, LUKE needs a pair of exactly two entities as an
# input. Therefore, we need to create rows for all combinations of entities in a document:
import itertools
df["ent_comb"] = df.ents.apply(lambda ents: [list(comb) for comb in itertools.combinations(ents, 2)])
df = df.explode("ent_comb")
df.shape

df = df[df.ent_comb.str.len() > 1].copy()

df["firms"] = df.ent_comb.apply(lambda ents: [ent[0] for ent in ents])
df["spans"] = df.ent_comb.apply(lambda ents: [ent[1] for ent in ents])

df.drop(columns=['ents_pred', 'firms_pred', 'spans_pred', 'matched_ents', 'ents', 'ent_comb'], inplace=True)
df.rename(columns={'content': 'document', 'ent_comb': 'firms'}, inplace=True)
df
# focus on news about JVs, alliances, and agreements
# df.subject.explode().value_counts().head(50)
# df[df.subject.apply(lambda x: [sub for sub in x if sub in ['JOINT VENTURES', 'ALLIANCES & PARTNERSHIPS', 'AGREEMENTS']]).str.len() > 0]

