import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer, models, LoggingHandler, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample


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
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

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
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.model = sts_model(self.model_name)
        # Sbert의 경우 cosinesimilarity 를 loss로 사용
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
        
        corpus_embeddings = self.sentence_transformer_model.encode(x[0], convert_to_tensor=True)
        query_embeddings = self.sentence_transformer_model.encode(x[1], convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(corpus_embeddings, query_embeddings).cpu().detach().numpy()
        
        logits = cosine_scores

        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        corpus_embeddings = self.sentence_transformer_model.encode(x[0], convert_to_tensor=True)
        query_embeddings = self.sentence_transformer_model.encode(x[1], convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(corpus_embeddings, query_embeddings).cpu().detach().numpy()
        
        logits = cosine_scores

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        
        corpus_embeddings = self.sentence_transformer_model.encode(x[0], convert_to_tensor=True)
        query_embeddings = self.sentence_transformer_model.encode(x[1], convert_to_tensor=True)

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
    parser.add_argument('--model_name', default="Huffon/sentence-klue-roberta-base", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
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

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load('model.pt')
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))
    predictions.apply(lambda if )

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('../data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('output.csv', index=False)
