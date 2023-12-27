import argparse
import random

import pandas as pd

from tqdm.auto import tqdm
from sklearn.model_selection import KFold

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


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
    
class KfoldDataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, k, split_seed, num_splits):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits

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

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            total_data = pd.concat([train_data, val_data])
            total_input, total_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_input, total_targets)

            # total_data = self.read_json('train')
            # total_input, total_targets = self.preprocessing(total_data)
            # total_dataset = Dataset(total_input, total_targets)

            kf = KFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.split_seed)
            all_splits = [k for k in kf.split(total_dataset)]

            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            self.train_dataset = [total_dataset[x] for x in train_indexes]
            self.val_dataset = [total_dataset[x] for x in val_indexes]

        else:
            self.test_dataset = self.val_dataset

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

            # test_data = pd.read_csv(self.test_path)
            # test_inputs, test_targets = self.preprocessing(test_data)
            # self.test_dataset = Dataset(test_inputs, test_targets)

            # predict_data = pd.read_csv(self.predict_path)
            # predict_inputs, predict_targets = self.preprocessing(predict_data)
            # self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)

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

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

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
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        #test_output.extend(logits.squeeze())
        #test_difference.extend(logits.squeeze()-y.squeeze())

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    import numpy as np
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--difference_path', default='../data/difference.csv')
    parser.add_argument('--nums_folds', default=3)
    parser.add_argument('--kfold', default=True)
    parser.add_argument('--model_name', default='kykim/electra-kor-base', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=25, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args(args=[])

    model = Model(args.model_name, args.learning_rate)
    # early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, mode="min")

    test_output = []
    test_difference = []
    test_pred0 = []
    test_pred1 = []
    test_pred2 = []
    
    if args.kfold==True:
        nums_folds = args.nums_folds
        split_seed = 12345
        print('k-fold 진행')

        for k in range(nums_folds):
            print(k, '번째 fold 진행')
            model = Model(args.model_name, args.learning_rate)
            kfdataloader = KfoldDataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                              args.test_path, args.predict_path, k=k, split_seed=split_seed, num_splits=nums_folds)
            kfdataloader.prepare_data()
            kfdataloader.setup()

            trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1) # , callbacks=[early_stop_callback]
            trainer.fit(model=model, datamodule=kfdataloader)
            score = trainer.test(model=model, datamodule=kfdataloader)
            
            test_output.extend(score) # 미션 보고 추가
            print(test_output[-1])
            predictions = trainer.predict(model=model, datamodule=kfdataloader)
            predictions = list(round(float(i), 1) for i in torch.cat(predictions))
            test_output.append(predictions)
            if k == 0:
                print('0번째 predictions 저장')
                test_pred0 = predictions
            elif k == 1:
                print('1번째 predictions 저장')
                test_pred1 = predictions
            elif k == 2:
                print('2번째 predictions 저장')
                test_pred2 = predictions

            torch.save(model, f'model_{k}.pt')
        # predictions_mean = np.mean(test_output, axis=0)
        # predictions_result = list(round(float(i), 1) for i in predictions_mean)
    else:
        print('dataloader 진행')
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path, args.test_path, args.predict_path)
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1) #, callbacks=[early_stop_callback]
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
        
        torch.save(model, f'model.pt')
    
    # predictions = trainer.predict(model=model, datamodule=dataloader)
    # predictions = list(round(float(i), 1) for i in torch.cat(predictions))
    print(test_pred0[0], test_pred1[0], test_pred2[0])
    a = []
    for q,w,e in zip(test_pred0, test_pred1, test_pred2):
        a.append(round((q+w+e)/3, 1))
    # print(a)
    output = pd.read_csv('../data/sample_submission.csv')
    output['target'] = a
    
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, args.learning_rate)
    

    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    test_output = []
    test_difference = []

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, 'model.pt')
    predictions = trainer.predict(model=model, datamodule=dataloader)
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))
    
    difference = pd.read_csv(args.test_path)
    difference['output'] = list(round(float(i), 3) for i in test_output)
    difference['output-label'] = list(round(float(i), 3) for i in test_difference)
    difference = difference.sort_values(by='output-label', ascending=True)
    difference.to_csv(args.difference_path, index=False)
    
    # output = pd.read_csv('../data/sample_submission.csv')
    # output['target'] = predictions_result
    output.to_csv('output.csv', index=False)
    # print(predictions[0], predictions2[0], predictions3[0], a[0])
    '''
    #batch 8 + epoch 10 + klue/roberta-large seed==0
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, mode="min")

    parser = argparse.ArgumentParser() # kykim/electra-kor-base
    parser.add_argument('--model_name', default='monologg/koelectra-base-v3-discriminator', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/Label0_re_dochi2.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args(args=[])

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, args.learning_rate)

    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1, callbacks=[early_stop_callback])

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    predictions = trainer.predict(model=model, datamodule=dataloader)
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    output = pd.read_csv('../data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('mnlgg_epo20.csv', index=False)

    
    parser = argparse.ArgumentParser() # monologg/koelectra-base-v3-discriminator
    parser.add_argument('--model_name', default='monologg/koelectra-base-v3-discriminator', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/Label0_re_dochi2.csv')
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
    predictions2 = trainer.predict(model=model, datamodule=dataloader)
    predictions2 = list(round(float(i), 1) for i in torch.cat(predictions2))

    output = pd.read_csv('../data/sample_submission.csv')
    output['target'] = predictions2
    output.to_csv('mnlgg_epo20.csv', index=False)


    parser = argparse.ArgumentParser() # kykim/electra-kor-base
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/Label0_to_Label5_dochi.csv')
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
    predictions3 = trainer.predict(model=model, datamodule=dataloader)
    predictions3 = list(round(float(i), 1) for i in torch.cat(predictions3))
    # print(predictions3)
    # alpha = pd.read_csv('cssdf.csv')
    # predictions = list(alpha['target'])

    output = pd.read_csv('../data/sample_submission.csv')
    output['target'] = predictions3
    output.to_csv('large_epo15_bat_4.csv', index=False)

    a = []
    for q,w,e in zip(predictions, predictions2, predictions3):
        a.append(round((q+w+e)/3, 1))
    # print(a)
    output = pd.read_csv('../data/sample_submission.csv')
    output['target'] = a
    output.to_csv('output.csv', index=False)
    print(predictions[0], predictions2[0], predictions3[0], a[0])
    print(predictions[1], predictions2[1], predictions3[0], a[1])
    print(predictions[2], predictions2[2], predictions3[0], a[2])
    print(predictions[3], predictions2[3], predictions3[0], a[3])
    

'''
parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args(args=[])

    daekeun-ml/koelectra-small-v3-korsts
    monologg/kobigbird-bert-base
    monologg/koelectra-base-v3-discriminator
    klue/roberta-large
    snunlp/KR-SBERT-V40K-klueNLI-augSTS
'''