import argparse
import random

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer, models, LoggingHandler, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader


# seed 고정
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)


def sts_model(model_name):
    """
    학습과정에 사용되는 모델 생성용 함수
    """
    # pretrained embedding model을 가져옴
    embedding_model = models.Transformer(
            model_name_or_path=model_name, 
            max_seq_length=256,
            do_lower_case=False
            )

    # 문장을 고정된 크기의의 임베딩으로 변환
    pooling_model = models.Pooling(
        embedding_model.get_word_embedding_dimension(), 
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    
    model = SentenceTransformer(modules=[embedding_model, pooling_model])

    return model


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input):
        self.input_examples = input

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        return self.input_examples[idx]

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.input_examples)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()

        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model = sts_model(self.model_name)

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
    
    def preprocessing(self,dataset):
        # 안쓰는 컬럼을 삭제합니다.
        dataset = dataset.drop(columns=self.delete_columns)

        input_examples = []
        for i, data in dataset.iterrows():
            sentence1 = data['sentence_1']
            sentence2 = data['sentence_2']
            try: 
                targets = (data['label'] / 5.0)
            except:     # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
                targets = None
            
            input_examples.append(InputExample(texts=[sentence1, sentence2], label = targets))

        return input_examples

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs = self.preprocessing(val_data)


            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs)
            self.val_dataset = Dataset(val_inputs)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle,collate_fn=model.smart_batching_collate)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,collate_fn=model.smart_batching_collate)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,collate_fn=model.smart_batching_collate)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size,collate_fn=model.smart_batching_collate)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.model = sts_model(self.model_name)
        # sbert의 경우 cosine similarity를 loss로 사
        self.loss_func = losses.CosineSimilarityLoss(model= self.model)

    def forward(self, x):
        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits * 5.0, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        corpus_embeddings = self.model.encode(x.texts[0], convert_to_tensor=True)
        query_embeddings = self.model.encode(x.tests[1], convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(corpus_embeddings, query_embeddings).cpu().detach().numpy()
        
        logits = cosine_scores

        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        corpus_embeddings = self.model.encode(x.texts[0], convert_to_tensor=True)
        query_embeddings = self.model.encode(x.texts[1], convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(corpus_embeddings, query_embeddings).cpu().detach().numpy()
        
        logits = cosine_scores

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        
        corpus_embeddings = self.model.encode(x.texts[0], convert_to_tensor=True)
        query_embeddings = self.model.encode(x.texts[1], convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(corpus_embeddings, query_embeddings).cpu().detach().numpy()
        
        logits = cosine_scores * 5.0

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="klue/roberta-base", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args(args=[])


    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, args.learning_rate)

    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, 'model.pt')

