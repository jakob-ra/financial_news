import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu'

# inputs
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
PRETRAINED_MODEL = 'roberta-large'
tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL)
# BASE_PATH = '/Users/Jakob/Documents/financial_news_data/model'
# # label_cols = ['Pending', 'JointVenture', 'Marketing', 'Manufacturing', 'ResearchandDevelopment', 'Licensing',
# #      'Supply', 'Exploration', 'TechnologyTransfer']
# label_cols = ['Pending', 'JointVenture', 'ResearchandDevelopment']
# NUM_LABELS = len(label_cols)
BASE_PATH = '/Users/Jakob/Downloads/toxic'

# toxic comments
df = pd.read_csv(BASE_PATH + '/train.csv')
df['list'] = df[df.columns[2:]].values.tolist()
df.rename(columns={'comment_text': 'Text'}, inplace=True)
df = df.sample(n=100)
NUM_LABELS = len(df.list.iloc[0])

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.Text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# Creating the dataset and dataloader for the neural network
train_size = 0.8
train_dataset = df.sample(frac=train_size,random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the
# final output for the model.

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained(PRETRAINED_MODEL)
        # self.l2 = torch.nn.Dropout(0.3)
        self.l2 = torch.nn.Linear(1024, NUM_LABELS)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # output_2 = self.l2(output_1)
        output = self.l2(output_1)
        return output


model = BERTClass()
model.to(device)

model.train()

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(EPOCHS):
    train(epoch)

def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")




# fast_bert solution (not working/slow)
# from fast_bert.data_cls import BertDataBunch
# from fast_bert.learner_cls import BertLearner
# from fast_bert.metrics import accuracy
# import logging
# import torch
#
# BASE_PATH = '/Users/Jakob/Documents/financial_news_data/model'
# DATA_PATH = BASE_PATH + '/data'
# LABEL_PATH = BASE_PATH + '/label'
# OUTPUT_DIR = BASE_PATH + '/output'
#
# label_cols = ['Pending', 'JointVenture', 'Marketing', 'Manufacturing', 'ResearchandDevelopment', 'Licensing',
#      'Supply', 'Exploration', 'TechnologyTransfer']
#
# databunch = BertDataBunch(DATA_PATH, LABEL_PATH, tokenizer='roberta-large', train_file='train.csv',
#     val_file='test.csv', label_file='labels.csv', text_col='Text', label_col=label_cols,
#     batch_size_per_gpu=64, max_seq_length=256, multi_gpu=False, multi_label=True, model_type='roberta')
#
# logger = logging.getLogger()
# device = torch.device('cpu')
# metrics = [{'name': 'accuracy', 'function': accuracy}]
#
# learner = BertLearner.from_pretrained_model(databunch, pretrained_path='roberta-large', metrics=metrics,
#     device=device, logger=logger, output_dir=OUTPUT_DIR, finetuned_wgts_path=None, warmup_steps=500,
#     multi_gpu=False, is_fp16=True, multi_label=True, logging_steps=50)
#
# learner.fit(epochs=1, lr=6e-5, validate=True,  # Evaluate the model after each epoch
#     schedule_type='warmup_cosine', optimizer_type='adamw')
#
# from nlp import Dataset
# dataset = Dataset.from_pandas(full[['Text', 'Pending']])


