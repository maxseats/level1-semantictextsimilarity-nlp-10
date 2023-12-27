import pandas as pd
import json
from tqdm import tqdm


def train_data(df):
    train_data = pd.read_csv('../data/train.csv')
    with open('aug_sen.json', 'r', encoding='UTF8') as json_file:
        aug_data = json.load(json_file)

    len_aug,list_sen=[],[]
    for k,v in aug_data.items():
        len_aug.append(len(v))
        list_sen=list_sen+v
    train_data['times']=len_aug
    df_new=train_data.loc[train_data.index.repeat(train_data.times)]
    df_new,train_data = df_new.drop('times',axis=1),train_data.drop('times',axis=1)
    df_new['sentence_2'] = list_sen

    df_new=df_new.drop(['label','binary-label'],axis=1)
    df_new.index=[x for x in range(df_new.shape[0])]

def get_binary_label(df):
    df['binary-label']=[int(x) for x in (df['label']>=2.5).tolist()]
    return df

def remove_eda_label_side(df):
    df=df[(df['label']>=1.0) & (df['label']<=4.0)]
    return df


if __name__ == '__main__':
    df_eng = pd.read_csv('../data/train_translated.csv', encoding='UTF8')
    df_kor = pd.read_csv('../data/train.csv', encoding='UTF8')

    df_kor = df_kor.drop(4706)
    df_kor = df_kor.drop(6000)
    df_eng=df_eng.iloc[:,1:]
    df_eng.columns = ['translated1','translated2']
    df_kor.index = [x for x in range(df_kor.shape[0])]
    df_eng.index = [x for x in range(df_eng.shape[0])]

    df=pd.concat([df_kor,df_eng],axis=1)

    df.to_csv('../data/train_org_trsn.csv')

