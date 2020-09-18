import torch as th
import pytorch_lightning as pl

import nlp
import transformers

DEBUG = True
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-2
MOMENTUM = .9
MODEL = 'bert-base-uncased'
SEQ_LENGTH = 32
PERCENT = 5


class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(MODEL)
        self.loss = th.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(MODEL)

        def _tokenize(x):
            x['input_ids'] = \
            tokenizer.batch_encode_plus(x['text'], max_length=SEQ_LENGTH, pad_to_max_length=True)[
                'input_ids']
            return x

        def _prepare_ds(split):
            ds = nlp.load_dataset('imdb',
                split=f'{split}[:{BATCH_SIZE if DEBUG else f"{PERCENT}%"}]')
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
        return th.utils.data.DataLoader(self.train_ds, batch_size=BATCH_SIZE, drop_last=True,
            shuffle=True, )

    def val_dataloader(self):
        return th.utils.data.DataLoader(self.test_ds, batch_size=BATCH_SIZE, drop_last=False,
            shuffle=True, )

    def configure_optimizers(self):
        return th.optim.SGD(self.parameters(), lr=LR, momentum=MOMENTUM, )


def main():
    model = IMDBSentimentClassifier()
    trainer = pl.Trainer(default_root_dir='logs', gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=EPOCHS, fast_dev_run=DEBUG,
        logger=pl.loggers.TensorBoardLogger('logs/', name='imdb', version=0), )
    trainer.fit(model)


if __name__ == '__main__':
    main()
