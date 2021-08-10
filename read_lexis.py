import pandas as pd
import os

path = 'C:/Users/Jakob/Documents/LexisNexis firm alliances/'

# combine
chunks = os.listdir(path)

# check all is there
# chunk_nums = [int(x.split('_')[-1].split('.csv')[0]) for x in chunks]
# missing = [x for x in range(sorted(chunk_nums)[-1]) if x not in chunk_nums]

for n, chunk in enumerate(chunks):
    if n == 0:
        df = pd.read_csv(os.path.join(path, chunk), compression='gzip')
    else:
        df = df.append(pd.read_csv(os.path.join(path, chunk), compression='gzip'))

df.drop_duplicates(subset=['title', 'content'], keep='first', inplace=True)

df.to_csv(os.path.join(path, 'lexisnexis_firm_alliances_combined.csv.gzip'), index=False, compression='gzip')
