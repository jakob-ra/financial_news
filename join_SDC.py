import pandas as pd
import os
import glob
from datetime import timedelta
import numpy as np
import datetime

### Joins all years of SDC data together

sdc_path = '/Users/Jakob/Documents/Thomson_SDC'
all_sdc = glob.glob(os.path.join(sdc_path, "*.xlsx"))

sdc = pd.concat((pd.read_excel(f, skiprows=3) for f in all_sdc))

# remove spaces and special characters in column names
sdc.columns = sdc.columns.str.replace(' ', '')
sdc.columns = sdc.columns.str.replace('\n', '')
sdc.columns = sdc.columns.str.replace('-', '')
sdc.columns = sdc.columns.str.replace('.1', 'Short')

# fix dates
date_columns = ['AllianceDateAnnounced', 'DateEffective', 'DateAllianceTerminated']
for col in date_columns:
    sdc[col] = sdc[col].astype('str')
    sdc[col] = sdc[col].str.replace(' 00:00:00', '')
    sdc[col] = pd.to_datetime(sdc[col], errors='coerce')
    # some dates are parsed as wrong century
    sdc.loc[sdc[col] > pd.Timestamp(2050, 1, 1), col] -= timedelta(
        days=365.25 * 100)
    sdc[col] = sdc[col].dt.date

# sort by date announced
sdc.sort_values('AllianceDateAnnounced', inplace=True)

# save as excel file
for col in date_columns:
    sdc[col] = sdc[col].astype('str')
sdc.to_excel(sdc_path + '/Full/SDC_Strategic_Alliances_Full.xlsx', index=False)
for col in date_columns:
    sdc[col] = pd.to_datetime(sdc[col], errors='coerce')

## Format further and save as pickle
cols = sdc.columns
participant_columns = [6,7,9,10,11,12,13,21,68,69,70,71,72,73,74,75,76,77]
participant_columns = cols[participant_columns]
flag_columns = [15,19,20,23,24,27,29,30,31,32,35,38,39,45,48,50,51,52,55,56,57,58,59,60,65,67]
flag_columns = cols[flag_columns]
text_columns = ['DealText','ParticipantBusinessDescriptionLong',
    'LongBusinessDescription', 'ActivityDescription',
    'CapitalizationText', 'ApplicationText']
other_list_columns = ['ActivityDescription', 'Source', 'SourceShort']

# separate multiline Participant cells and other list columns into lists
for col in participant_columns:
    sdc[col] = sdc[col].str.split('\n')

# separate other list columns into lists
for col in other_list_columns:
    sdc[col] = sdc[col].str.split('\n')

# format texts
for col in text_columns:
    # convert to string
    sdc[col] = sdc[col].astype('str')
    # replace new line character \n with space in texts
    sdc[col] = sdc[col].str.replace('\n', ' ')
    # remove double spaces
    sdc[col] = sdc[col].str.replace('  ', ' ')

## fix flag columns - some have wrong values (likely corresponding to other columns)
print({c: sdc[c].unique() for c in sdc[flag_columns]})

# set correct entries to 1/0
flag_valid_entries = dict(yes=1, Yes=1, Y=1, y=1, No=0, no=0,N=0, n=0)
sdc[flag_columns] = sdc[flag_columns].replace(flag_valid_entries)

# convert to numeric and set wrong entries nan
for col in flag_columns:
    sdc[col] = pd.to_numeric(sdc[col], errors='coerce')
    sdc.loc[sdc[col] > 1, col] = np.nan

# check that we are only left with values in (0,1,nan)
print(pd.concat([pd.Series(sdc[c].unique()) for c in sdc[flag_columns]]).unique())

## FILTERING

# sdc = pd.read_pickle('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')

# drop observations with missing values in flags
sdc.dropna(subset=flag_columns, inplace=True)

# filter for only completed or pending status
sdc = sdc[sdc.Status.isin(['Completed/Signed', 'Pending', 'Terminated'])]

# some incorrect deal texts seem to belong to the application/purpose column. Example text 'To build a ...'
print(sdc[sdc.DealText.str.startswith('To ')].DealText.size)

# filter out these wrong deal texts
sdc = sdc[~sdc.DealText.str.startswith('To ')]

# remove rows with no date or text
sdc.dropna(subset=['AllianceDateAnnounced', 'DealText'], how='any', inplace=True)

# filter for years >=1985
sdc = sdc[sdc['AllianceDateAnnounced'].dt.date >= datetime.date(1985, 1, 1)]

# SAVE
sdc.to_pickle(sdc_path + '/Full/SDC_Strategic_Alliances_Full.pkl')

sdc = pd.read_pickle(sdc_path + '/Full/SDC_Strategic_Alliances_Full.pkl')

# filter for Swiss
sdc = pd.read_excel(sdc_path + '/Full/SDC_Strategic_Alliances_Full.xlsx')

sdc_switz = sdc[(sdc['ParticipantNation'].apply(lambda x: str(x)).str.contains('Switzerland')) |
                (sdc['ParticipantUltimateParentNation'].apply(lambda x: str(x)).str.contains('Switzerland'))]

sdc_switz.reset_index(inplace=True, drop=True)

sdc_switz.to_excel(sdc_path + '/Full/SDC_Swiss_Firms.xlsx')

sdc = pd.read_pickle(sdc_path + '/Full/SDC_Strategic_Alliances_Full.pkl')

participant_identifiers = ['ParticipantsinVenture/Alliance(ShortName)', 'Parti.CUSIP', 'ParticipantNation']
parent_identifiers = ['ParticipantUltimateParentName', 'UltimateParentCUSIP', \
    'ParticipantUltimateParentNation']
participants = sdc[participant_identifiers]
participants.columns = ['Name', 'CUSIP', 'Nation']
parents = sdc[parent_identifiers]
parents.columns = ['Name', 'CUSIP', 'Nation']
participants = participants.append(parents)

participants = participants[participants.Name.str.len() == participants.CUSIP.str.len()]
participants = participants[participants.Name.str.len() == participants.Nation.str.len()]
participants = participants[participants.CUSIP.str.len() == participants.Nation.str.len()]

firms = pd.concat([participants.Name.explode(), participants.CUSIP.explode(),
    participants.Nation.explode()], axis=1)

firms.drop_duplicates(inplace=True)

firms.to_csv(sdc_path + '/Full/Firm_List.csv', index=False)

# filter swiss
firms_swiss = firms[firms.Nation == 'Switzerland']
firms_swiss.to_csv(sdc_path + '/Full/Firm_List_Switzerland.csv', index=False)