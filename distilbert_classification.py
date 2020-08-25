## huggingface
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, DistilBertForSequenceClassification
import torch

distil_bert = 'distilbert-base-cased'
tokenizer = DistilBertTokenizerFast.from_pretrained(distil_bert, do_lower_case=False, add_special_tokens=True,
                                                max_length=256, pad_to_max_length=True)
token_clf = DistilBertForTokenClassification.from_pretrained(distil_bert)
sequence_clf = DistilBertForSequenceClassification.from_pretrained(distil_bert)

sentence = 'Apple and Microsoft plan to form a joint venture for the development of cloud-based computing ' \
           'infrastructure.'

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
token_clf(input_ids)
outputs = model(input_ids)

last_hidden_states = outputs[0]

test = db.sample(n=10)

token_clf(tokenizer.encode_plus(sentence))
tokenizer.batch_encode_plus(test.text.to_list())


## spacy
def get_sequences_with_2_orgs(text, dist=150):
    ''' Uses spacy NER to identify organisations. If two organizations are detected within dist
    tokens from each other, extracts the sequence
    '''
    # Apply the model
    doc = nlp(text)

    org_positions = [doc.ent.idx for ent in doc.ents if ent.label_ == 'ORG']
    orgs = [ent.text.replace('\'', '') for ent in doc.ents if ent.label_ == 'ORG']


    # Return the list of entities recognized as organizations
    return sequence, orgs # also remove apostrophes

sentence = 'Microsoft and Google announced a collaboration on the development of new computers. ' \
           'Volkswagen AG and Walmart Inc. joined them.'

doc = nlp(sentence)

org_positions = [ent.start for ent in doc.ents if ent.label_ == 'ORG']

for idx in range(len(org_positions)):
    if org_positions[idx + 1]
orgs = [ent.text.replace('\'', '') for ent in doc.ents if ent.label_ == 'ORG']

## spacy custom NER
from spacy.lang.en import English
from spacy.matcher import Matcher
import pandas as pd

# reference database of firm names
firm_names = pd.read_csv('/Users/Jakob/Documents/financial_news_data/Orbis_US_public.csv', usecols=[0],
    sep=';', squeeze=True)
firm_names = firm_names.str.lower()
firm_names = firm_names.str.replace('[^\w\s]', '') # remove punctuation
print(firm_names.str.split(' ').explode().value_counts().iloc[:30]) # most common words
for term in ['inc', 'corp', 'corporation', 'group', 'etf', 'co', 'ltd']:
    firm_names = firm_names.append(firm_names[firm_names.str.contains(term)].str.replace(term, ''))
firm_names = firm_names.str.strip()

firm_name_match = [{'LOWER': name} for name in firm_names.head(2000)]

sentence = 'walmart'
nlp = English()
matcher = Matcher(nlp.vocab)
for pattern in firm_name_match:
    matcher.add('FIRM', None, [pattern])
doc = nlp(sentence)
print([doc[start:end] for match_id, start, end in matcher(doc)])
