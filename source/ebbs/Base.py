import pandas as pd
import torch
import os
import json
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
import networkx as nx
import dgl
import numpy as np
from sklearn import preprocessing
import pdb
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def normalize_features(X, train_mask, val_mask, test_mask):
    min_max_scaler = preprocessing.MinMaxScaler()
    A = X.to_numpy(copy=True)
    A[train_mask] = min_max_scaler.fit_transform(A[train_mask])
    A[val_mask + test_mask] = min_max_scaler.transform(A[val_mask + test_mask])
    return pd.DataFrame(A, columns=X.columns).astype(float)

def replace_na(X, train_mask):
    if X.isna().any().any():
        return X.fillna(X.iloc[train_mask].min() - 1)
    return X

def encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask):
    from category_encoders import CatBoostEncoder
    enc = CatBoostEncoder()
    A = X.to_numpy(copy=True)
    b = y.to_numpy(copy=True)
    A[np.ix_(train_mask, cat_features)] = enc.fit_transform(A[np.ix_(train_mask, cat_features)], b[train_mask])
    A[np.ix_(val_mask + test_mask, cat_features)] = enc.transform(A[np.ix_(val_mask + test_mask, cat_features)])
    A = A.astype(float)
    return pd.DataFrame(A, columns=X.columns)

def pandas_to_torch(args):
    return torch.from_numpy(args.to_numpy(copy=True)).float().squeeze().to(device)



def plot(metrics, legend, title, output_fn=None, logx=False, logy=False, metric_name='loss'):
    import matplotlib.pyplot as plt
    metric_results = metrics[metric_name]
    xs = [range(len(metric_results))] * len(metric_results[0])
    ys = list(zip(*metric_results))


    with open('momentum.pickle', 'rb') as handle:
        metric_m = pickle.load(handle)

    xs_m = [range(len(metric_m[metric_name]))] * len(metric_m[metric_name][0])
    ys_m = list(zip(*metric_m[metric_name]))



    with open('filename.pickle', 'rb') as handle:
        metric_m = pickle.load(handle)
  

    xs_bgnn = [range(len(metric_m[metric_name]))] * len(metric_m[metric_name][0])
    ys_bgnn = list(zip(*metric_m[metric_name]))




    plt.rcParams.update({'font.size': 40})
    plt.rcParams["figure.figsize"] = (20, 10)
    lss = ['-', '--', '-.', ':']
    colors = ['#4053d3', '#ddb310', '#b51d14', '#00beff', '#fb49b0', '#00b25d', '#cacaca']
    colors = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2),
                (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
    colors = [[p / 255 for p in c] for c in colors]
    # for i in range(1):
    #     plt.plot(xs[i], ys[i], lw=4, color=colors[i])
    #     plt.plot(xs[i], ys_m[i], lw=4, color=colors[i])
    #     plt.plot(xs[i], bgnn[i], lw=4, color=colors[i])
    # plt.legend(legend, loc=1, fontsize=30)
    # plt.title(title)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(xs[0], ys[0], '-', lw=4, color=colors[0])
    ax.plot(xs_m[0], ys_m[0], '-', lw=4, color=colors[1])
    ax.plot(xs_bgnn[0], ys_bgnn[0], '-', lw=4, color=colors[2])
    m1, =  ax.plot([], [], '-', lw=4, color='lightgrey')


    
    leg = ax.legend(['EBBS','EBBS with momentum','BGNN'], loc=1, fontsize=20)

    ax.plot(xs[1], ys[1], '--', lw=4, color=colors[0])
    ax.plot(xs_m[1], ys_m[1], '--', lw=4, color=colors[1])
    ax.plot(xs_bgnn[1], ys_bgnn[1], '--', lw=4, color=colors[2])

    m2, =  ax.plot([], [], '--', lw=4, color='lightgrey')

    # plt.legend(['EBBS','EBBS with momentum'], loc=1, fontsize=30)

    ax.legend([m1,m2],['Training','Validation'], fontsize=30, loc=9)
    ax.add_artist(leg)

    # plt.title(title)

    # plt.xscale('log') if logx else None
    # plt.yscale('log') if logy else None
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    plt.grid()
    plt.tight_layout()

    plt.savefig(output_fn, bbox_inches='tight') if output_fn else None
    # plt.show()