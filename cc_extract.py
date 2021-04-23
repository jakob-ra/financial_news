import pandas as pd
from comcrawl import IndexClient
from html2text import html2text

url_list = pd.read_csv('/Users/Jakob/Downloads/All companies for profit Corona mentions high heartbeat all languages.csv')
url_list = url_list['Hostname']

client = IndexClient()

client.search(url_list.iloc[1])

client.results = (pd.DataFrame(client.results)
                  .sort_values(by="timestamp")
                  .drop_duplicates("urlkey", keep="last")
                  .to_dict("records"))

client.download()

results = pd.DataFrame(client.results)

html = results.html.iloc[0]

print(html2text(html))


import warcio

from warcio.archiveiterator import ArchiveIterator

segment_path = '/Users/Jakob/Downloads/segment1'

with open(segment_path, 'rb') as stream:
    for record in ArchiveIterator(stream):
        text = str(record.content_stream().read().decode('utf-8'))
        if 'corona' in text:
            print(text)

import warc