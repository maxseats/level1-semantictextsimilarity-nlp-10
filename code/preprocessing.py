import pandas as pd

train = pd.read_csv("../data/train.csv")
dev = pd.read_csv("../data/dev.csv")

small_train = train[:50]
small_dev = dev[:50]

small_train.to_csv("../data/small_train.csv")
small_dev.to_csv("../data/small_dev.csv")