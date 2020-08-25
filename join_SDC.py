import pandas as pd
import os
import glob
from datetime import datetime

### Joins all years of SDC data together

sdc_path = '/Users/Jakob/Documents/Thomson_SDC'
all_sdc = glob.glob(os.path.join(sdc_path, "*.xlsx"))

sdc = pd.concat((pd.read_excel(f, skiprows=3) for f in all_sdc))

# remove spaces and special characters in column names
sdc.columns = sdc.columns.str.replace(' ', '')
sdc.columns = sdc.columns.str.replace('\n', '')
sdc.columns = sdc.columns.str.replace('-', '')
sdc.columns = sdc.columns.str.replace('.1', '')

# fix dates that are wrongly parsed as far in the future
sdc.loc[sdc['AllianceDateAnnounced'] > pd.Timestamp(2050, 1, 1), 'AllianceDateAnnounced'] -= \
    timedelta(days=365.25*100)

pd.to_datetime(sdc['AllianceDateAnnounced'])
date_columns = ['AllianceDateAnnounced', 'DateEffective', 'DateAllianceTerminated']

# fix dates
wrong_years = [2063 + i for i in range(10)]
for col in date_columns:
    # sdc[col] = sdc[col].astype('str')
    # sdc[col] = sdc[col].str.replace(' 00:00:00', '')
    # for year in wrong_years:
    #     sdc[col] = sdc[col].str.replace(str(year), str(year-100))
    sdc[col] = pd.to_datetime(sdc[col], errors='coerce')

sdc.AllianceDateAnnounced.iloc[0]

# keep only date (not time)
sdc['AllianceDateAnnounced'] = sdc['AllianceDateAnnounced'].dt.date

# separate multiline Participants cell into list
sdc['participants'] = sdc['participants'].str.split('\n')
sdc['parent_participants'] = sdc['parent_participants'].str.split('\n')
sdc['participant_country'] = sdc['participant_country'].str.split('\n')

# replace new line character \n with space in deal text
sdc['text'] = sdc['text'].str.replace('\n', ' ')

# remove double spaces
sdc['text'] = sdc['text'].str.replace('  ', ' ')

# add source tag
sdc['source'] = 'ThomsonSDC'

sdc['AllianceDateAnnounced'].iloc[0]

py_date = xlrd.xldate.xldate_as_datetime(sdc['AllianceDateAnnounced'].iloc[0])

sdc.columns

pd.read_excel(all_sdc[0], skiprows=3, usecols=[0])