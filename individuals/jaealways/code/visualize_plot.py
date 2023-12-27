import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scatter_plot(name):
    df_dev=pd.read_csv(f'../label_prediction/{name}.csv')
    x,y=df_dev['label'].tolist(),df_dev['prediction'].tolist()
    plt.scatter(x,y)
    plt.xlabel('Origin')
    plt.ylabel('Predict')
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, np.asarray(m)*x+np.asarray(b))

    plt.savefig(f'../plot/{name}.png')

def output_plot(name):
    df_dev=pd.read_csv(f'../output_prediction/{name}.csv')
    x=df_dev['target'].tolist()
    plt.hist(x, rwidth=0.1)

    plt.savefig(f'../plot/output_{name}.png')



if __name__ == '__main__':
    scatter_plot('33')
    # output_plot('32')


