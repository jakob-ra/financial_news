import pandas as pd
import glob
import os

orbis_path = '/Users/Jakob/Documents/Orbis/lexis_firms_matched_orbis'
all_orbis = glob.glob(os.path.join(orbis_path, "*.xlsx"))

orbis = pd.concat((pd.read_excel(f) for f in all_orbis), ignore_index=True)

orbis.drop(columns=['Unnamed: 0'], inplace=True)

orbis.dropna(subset=['Company name Latin alphabet', 'BvD ID number'], inplace=True)

# orbis['BvD ID number'] = orbis['BvD ID number'].ffill() # for aggregating the multi-line NACE codes
# orbis['Company name Latin alphabet'] = orbis['Company name Latin alphabet'].ffill()

# orbis.drop_duplicates(subset=['Company name Latin alphabet', 'BvD ID number'], inplace=True)

def isnull(val):
    nan_strings = ['', 'NaN', 'NA', 'na', 'n.a.', 'nan']
    return pd.isnull(val) or val in nan_strings

orbis = orbis.groupby(['Company name Latin alphabet', 'BvD ID number']).agg(list)
orbis.reset_index(inplace=True)

def first_not_null_in_list(vals: list):
    for val in vals:
        if not isnull(val):
            return val
    return np.nan

for col in orbis.columns[2:]:
    orbis[col] = orbis[col].apply(first_not_null_in_list)

orbis['cleaned_name'] = orbis['Company name Latin alphabet'].apply(firm_name_clean)

# sort by size (employees) first, then value added, then number of NA columns
orbis['count_na'] = orbis.isnull().sum(1)
orbis.sort_values(by=['Number of employees\n2020', 'Added value\nth USD 2020', 'count_na'],
                          inplace=True, ascending=[False, False, True])

# now keep only the biggest firms / firms with most complete records among those with the same name
orbis.drop_duplicates(subset=['cleaned_name'], keep='first', inplace=True)

orbis.to_pickle('C:/Users/Jakob/Documents/Orbis/orbis_michael_lexis_2.pkl')
orbis.to_csv('C:/Users/Jakob/Documents/Orbis/orbis_michael_lexis_2.csv', index=False)

orbis = pd.read_pickle('C:/Users/Jakob/Documents/Orbis/orbis_michael_lexis_2.pkl')

orbis_financials = pd.read_csv('C:/Users/Jakob/Downloads/orbis_financials/Key_financials-USD.txt',
                              nrows=100,
                              sep='\t',
                              )

df = pd.read_csv('C:/Users/Jakob/Documents/Orbis/Full/BvD_ID_and_Name.txt',
                               sep='\t')

