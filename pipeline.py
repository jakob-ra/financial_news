from absl import app, flags, logging

import torch as th
import pytorch_lightning as pl

import nlp
import transformers

ds =
flags.DEFINE_boolean('debug', True, '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'distilbert-base-cased', '')
flags.DEFINE_integer('seq_length', 64, '')
flags.DEFINE_integer('percent', 50, '')

FLAGS = flags.FLAGS

# ds.to_csv('/Users/Jakob/Documents/financial_news_data/ds.csv')

class RelationClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
        self.loss = th.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        tokenizer = transformers.DistilBertTokenizer.from_pretrained(
            FLAGS.model,
            do_lower_case=False,
            add_special_tokens=True,
            max_length=FLAGS.seq_length,
            pad_to_max_length=True)

        def _tokenize(x):
            x['input_ids'] = tokenizer.batch_encode_plus(x['text'])['input_ids']

            return x

        def _prepare_ds(split):
            # ds = nlp.load_dataset('csv', data_files='/Users/Jakob/Documents/financial_news_data/ds.csv',
            #     cache_dir='/Users/Jakob/Documents/financial_news_data/nlp_cache')
            ds = pd.read_parquet('/Users/Jakob/Documents/financial_news_data/ds.parquet.gzip')
            # prepared_ds = nlp.Dataset.from_dict(ds)
            # ds = nlp.Dataset.from_file('/Users/Jakob/Documents/financial_news_data/ds.parquet.gzip')
            # ds = nlp.load_dataset('imdb', split=f'{split}[:{FLAGS.batch_size if FLAGS.debug else f"{FLAGS.percent}%"}]')
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.test_ds = map(_prepare_ds, ('train', 'test'))

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}

    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_ds,
                batch_size=FLAGS.batch_size,
                drop_last=True,
                shuffle=True,
                )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
                self.test_ds,
                batch_size=FLAGS.batch_size,
                drop_last=False,
                shuffle=True,
                )

    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
        )


def main(_):
    model = RelationClassifier()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='imdb', version=0),
    )
    trainer.fit(model)


if __name__ == '__main__':
    app.run(main)


import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
# import spacy
#
#
# class RE_Pipeline:
#     """ Relation extraction pipeline, used on individual sentences. """
#     def __init__(self, ner_model, clf_model, tokenizer):
#         self.ner_model = ner_model
#         self.clf_model = clf_model
#         self.tokenizer = tokenizer
#
#     # self.device = "cuda" if torch.cuda.is_available() else "cpu"
#     # self.model.to(self.device)
#
#     def __call__(self, sentence: str):
#         orgs = self.extract_orgs(sentence)
#         relations = self.extract_relations(sentence, orgs)
#
#         return relations
#
#     def extract_orgs(self, sentence):
#         ''' Returns a list of entities in the sentence recognized as organizations. SPACY '''
#         # Apply the model
#         tags = nlp(sentence)
#
#         # Return the list of entities recognized as organizations (also remove apostrophes)
#         return [ent.text.replace('\'', '') for ent in tags.ents if ent.label_ == 'ORG']
#
#     def extract_relations(self, sentence, orgs):
#         # replace orgs in relation with #ent1 and #ent2
#
# ner_model = spacy.load('en_core_web_sm')
# clf_model = 'blub'
# tokenizer = 'blab'
#
# pipe = RE_Pipeline(spacy.load('en_core_web_sm'), clf_model, tokenizer)
#
# sentence = 'Microsoft and Google announced a collaboration on the development of new computers.'
# pipe.extract_orgs(sentence)

from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertConfig, TFDistilBertForSequenceClassification

distil_bert = 'distilbert-base-cased' # Pick any desired pre-trained model

# Defining DistilBERT tokonizer
tokenizer = DistilBertTokenizer.from_pretrained(distil_bert, do_lower_case=True, add_special_tokens=True,
                                                max_length=128, pad_to_max_length=True)

config = DistilBertConfig() # dropout=0.2, attention_dropout=0.2
config.output_hidden_states = False
transformer_model = TFDistilBertModel.from_pretrained(distil_bert, config=config)

input_ids_in = tf.keras.layers.Input(shape=(128,), name='input_token', dtype='int32')
input_masks_in = tf.keras.layers.Input(shape=(128,), name='masked_token', dtype='int32')

embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer)
X = tf.keras.layers.GlobalMaxPool1D()(X)
X = tf.keras.layers.Dense(50, activation='relu')(X)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.Dense(6, activation='sigmoid')(X)
model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs = X)

for layer in model.layers[:3]:
  layer.trainable = False

import tensorflow as tf

tokenizer = DistilBertTokenizer.from_pretrained(distil_bert)
model = TFDistilBertForSequenceClassification.from_pretrained(distil_bert)

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1))  # Batch size 1

outputs = model(inputs)
loss, logits = outputs[:2]