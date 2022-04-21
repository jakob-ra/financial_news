import pandas as pd
import itertools

df = pd.read_excel("C:/Users/Jakob/Downloads/CATI/CATI/CATI Year Alliances.xlsx")

# make one column with list of participants
part_cols = [col for col in df.columns if "COFI" in col]
df['Participants'] = df[part_cols].apply(list, axis=1)
df['Participants'] = df.Participants.apply(lambda vals: [val for val in vals if not pd.isnull(val)])
df.drop(columns=part_cols, inplace=True)

df.rename(columns={'ESTA': 'Date', 'FORMCOOP': 'rel_type'}, inplace=True)

# make CATI forms of cooperation comparable to SDC
""" Forms of cooperation in Cati: Joint Research Pact (JRP), Joint Development Agreement (JDA), 
R&DContract (RDC), Licensing (L), Cross-Licensing (XL), Standards (S), Mutual Second Sourcing (MSSA),  
Joint Ventures (JV), Research Corporations (RC), Minority Holding (MH), Cross Holding (CH) """
coop_dict = {'JRP' : 'Strategic Alliance, ResearchandDevelopment',
             'JDA' : 'Strategic Alliance, ResearchandDevelopment',
             'RDC' : 'ResearchandDevelopment, Strategic Alliance', 'L': 'Licensing', 'XL': 'Licensing',
             'MSSA': 'Strategic Alliance, TechnologyTransfer', 'JV': 'JointVenture',
             'RC'  : 'JointVenture, ResearchandDevelopment'}
df['rel_type'] = df.rel_type.str.split(r',|, ')
df = df.explode('rel_type')
df['rel_type'] = df.rel_type.str.strip()
df['rel_type'] = df.rel_type.map(coop_dict)
df.dropna(inplace=True)
df['rel_type'] = df.rel_type.str.split(r',|, ')
df = df.explode('rel_type')
df['rel_type'] = df.rel_type.str.strip()
df = df.groupby(df.index).agg({'Date': min, 'rel_type': list, 'Participants': sum})
df['rel_type'] = df.rel_type.apply(set).apply(list)
df.Date.describe()

# replace firm name abbreviations with name
code2name = pd.read_excel("C:/Users/Jakob/Downloads/CATI/CATI/CATI Name Code.xlsx")
code2name = code2name.set_index('COD').squeeze().to_dict()
df = df.explode('Participants')
df['Participants'] = df.Participants.map(code2name)
df = df.groupby(df.index).agg({'Date': min, 'rel_type': min, 'Participants': list})

assert df.Participants.apply(set).apply(len).min() == 2

# make a row for each two-way combination between participants
df['Participants'] = df.Participants.apply(lambda firms: list(itertools.combinations(firms, 2)))
df = df.explode('Participants')
df['Participants'] = df.Participants.apply(list)

df.to_pickle("C:/Users/Jakob/Documents/CATI/CATI/CATI Year Alliances.xlsx")

# def agg(vals: list):
#     res = set(vals)
#     if len(res) > 1:
#         return res
#     else:
#         return next(iter(res))