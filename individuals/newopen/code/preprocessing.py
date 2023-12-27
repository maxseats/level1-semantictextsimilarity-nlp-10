import pandas as pd
import numpy as np

train = pd.read_csv("../data/train_translated.csv")
#dev = pd.read_csv("../data/dev_translated.csv")
#test = pd.read_csv("../data/test_translated.csv")

# small_train = train[:2000]
# small_dev = dev[:100]
# small_test = test[:100]

# small_train.to_csv("../data/small_train.csv")
# small_dev.to_csv("../data/small_dev.csv")
# small_test.to_csv("../data/small_test.csv")

import chardet

with open("../data/test_translated.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

encoding = result['encoding']
print(encoding)