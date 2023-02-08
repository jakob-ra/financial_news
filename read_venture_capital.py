import pandas as pd
import os
from tqdm import tqdm

df = pd.read_stata('C:/Users/Jakob/Downloads/venture_data_2005_2015.dta')
df.dropna(inplace=True)
df.Firm_Name.nunique()


path = 'C://Users/Jakob/Downloads/PreqinVentureDeals_1970_July2018/'
files = os.listdir(path)

chunks = []
for i, file in tqdm(enumerate(files), total=len(files)):
    chunk = pd.read_excel(os.path.join(path, file), skiprows=13)
    chunks.append(chunk)

preqin = pd.concat(chunks)


deals_per_year = df['Equity_Amount_Estimated__USD_Mil'].groupby(df['Investment_Date'].dt.year).size()
equity_inv_per_year = pd.to_numeric(df['Equity_Amount_Estimated__USD_Mil'], errors='coerce').groupby(df['Investment_Date'].dt.year).sum()
equity_inv_per_year

