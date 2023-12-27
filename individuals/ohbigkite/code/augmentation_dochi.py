import pandas as pd

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('train.csv')

df2['sentence_2'] = df1['sentence_1']
df2['sentence_1'] = df1['sentence_2']

a = pd.concat([df1,df2])

a.to_csv('dochi_train.csv', index = False)