import requests
import gzip
import pandas as pd
from io import BytesIO

path = 'C://Users/Jakob/Downloads/jakub_share/compustat_warc_data_05_09_21.csv'

df = pd.read_csv(path)

prefix = 'https://commoncrawl.s3.amazonaws.com/'

warc_filename = df.warc_filename.iloc[0]
offset, offset_end = int(df.warc_record_offset.iloc[0]), int(df.warc_record_end.iloc[0])

resp = requests.get(prefix + warc_filename, headers={'Range': 'bytes={}-{}'.format(offset, offset_end)})

raw_data = BytesIO(resp.content)
f = gzip.GzipFile(fileobj=raw_data).read()

