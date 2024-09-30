import argparse
import json
import os
import pdb
# from collections import defaultdict as ddict
from pathlib import Path
 
import dgl
import fire
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

NUM_THREADS = 32
torch.set_num_threads(NUM_THREADS)

DATASET_NAMES = [
    'tolokers-tab',
    'questions-tab',
    'city-reviews',
    'browser-games',
    'hm-categories',
    'web-fraud',
    'city-roads-M',
    'city-roads-L',
    'avazu-devices',
    'hm-prices',
    'web-traffic',
]
 
from Base import *
from EBBS import *
# from AG import *
from dgl import function as fn
from dgl.data import (
    AmazonCoBuyComputerDataset,
    AmazonCoBuyPhotoDataset,
    CiteseerGraphDataset,
    CoraFullDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    RedditDataset,
)
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
def compute_acc(pred, labels):
    return ((torch.argmax(pred, dim=1) == labels[:, 0]).float().sum() / len(pred)).item()
 
 
def normalize_features2(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
 
 
def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.float64)
 
 
def split_dataset(n_samples):
    val_indices = np.random.choice(list(range(n_samples)), size=int(n_samples * 0.2), replace=False)
    left = set(range(n_samples)) - set(val_indices)
    test_indices = np.random.choice(list(left), size=int(n_samples * 0.2), replace=False)
    train_indices = list(left - set(test_indices))
 
    train_mask = get_mask(train_indices, n_samples).astype(np.bool)
    eval_mask = get_mask(val_indices, n_samples).astype(np.bool)
    test_mask = get_mask(test_indices, n_samples).astype(np.bool)
 
    return train_mask, eval_mask, test_mask
 
 
def load_npz(path):
    with np.load(path, allow_pickle=True) as f:
        f = dict(f)
    features = sp.csr_matrix((f["attr_data"], f["attr_indices"], f["attr_indptr"]), shape=f["attr_shape"])
    features = features.astype(np.float64)
    features = normalize_features2(features)
    labels = f["labels"].reshape(-1, 1)
 
    # The adjacency matrix is symmetric for coauthor-cs and coauthor-phy
    adj = sp.csr_matrix((f["adj_data"], f["adj_indices"], f["adj_indptr"]), shape=f["adj_shape"])
    adj += sp.eye(adj.shape[0])  # add self-loops
    # g = dgl.DGLGraph()
    # print(g.ntypes)
    g = dgl.from_scipy(adj)
 
    train_mask, val_mask, test_mask = split_dataset(labels.shape[0])
 
    return np.array(features.todense()), labels, g, train_mask, val_mask, test_mask
 
 
def load_dataset(dataset):
    if dataset == "coauthor-cs":
        return load_npz("dataset/ms_academic_cs.npz")
    if dataset == "coauthor-phy":
        return load_npz("dataset/ms_academic_phy.npz")
 
 
class RunModel:
    def load_graph_x_target(self, data_name, seed):
        # Load data
        # input_folder = Path(__file__).parent.parent / "datasets" / data_name
        # input_folder = Path(__file__).parent / "datasets" / data_name
        input_folder = f"../bgnn/datasets/{data_name}"
        self.X = pd.read_csv(f"{input_folder}/X.csv")
        self.y = pd.read_csv(f"{input_folder}/y.csv")
 
        categorical_columns = []
        if os.path.exists(f"{input_folder}/cat_features.txt"):
            with open(f"{input_folder}/cat_features.txt") as f:
                for line in f:
                    if line.strip():
                        categorical_columns.append(line.strip())
 
        self.cat_features = None
        if categorical_columns:
            columns = self.X.columns
            self.cat_features = np.where(columns.isin(categorical_columns))[0]
 
            for col in list(columns[self.cat_features]):
                self.X[col] = self.X[col].astype(str)
 
        # load mask
        if os.path.exists(f"{input_folder}/masks.json"):
            with open(f"{input_folder}/masks.json") as f:
                self.masks = json.load(f)
 
        ## split train_mask, test_mask and val_mask
        train_mask, val_mask, test_mask = self.masks["train"], self.masks["val"], self.masks["test"]




        # data preprocessing
        encoded_X = self.X.copy()  # n*d
        if self.cat_features is None:
            self.cat_features = []
        if len(self.cat_features):
            encoded_X = encode_cat_features(encoded_X, self.y, self.cat_features, train_mask, val_mask, test_mask)
        encoded_X = normalize_features(encoded_X, train_mask, val_mask, test_mask)
        encoded_X = replace_na(encoded_X, train_mask)
        encoded_X = pandas_to_torch(encoded_X)
        target = torch.from_numpy(self.y.to_numpy(copy=True)).float()
 
        ## load graph structure
        if data_name not in DATASET_NAMES:
            networkx_graph = nx.read_graphml(f"{input_folder}/graph.graphml")
            networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
            graph = dgl.from_networkx(networkx_graph)
        else:
            edgelist = pd.read_csv(f"{input_folder}/edgelist.csv").values
            sources, targets = torch.LongTensor(edgelist[:, 0]), torch.LongTensor(edgelist[:, 1])
            graph = dgl.graph((sources, targets))
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
 
        return graph, encoded_X, target, train_mask, test_mask, val_mask
 
    def load_academic(self, data_name):
        features, labels, g, train_mask, val_mask, test_mask = load_npz(f"./datasets/academic/{data_name}.npz")
 
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
 
        # train_mask, val_mask, test_mask = map(lambda x: np.where(x == True), (train_mask, val_mask, test_mask))
        features = torch.tensor(features).float()
        labels = torch.tensor(labels).float()
 
        # print("#############", g, features, labels, train_mask, test_mask, val_mask)
        return (
            g,
            features,
            labels.reshape(-1),
            train_mask,
            test_mask,
            val_mask,
        )








    def load_data(self, dataset):
        global n_node_feats, n_classes
    
        if dataset in ["ogbn-arxiv", "ogbn-proteins", "ogbn-products"]:
            data = DglNodePropPredDataset(name=dataset)
        elif dataset == "cora":
            data = CoraGraphDataset()
        elif dataset == "citeseer":
            data = CiteseerGraphDataset()
        elif dataset == "pubmed":
            data = PubmedGraphDataset()
        elif dataset == "cora-full":
            data = CoraFullDataset()
        elif dataset == "reddit":
            data = RedditDataset()
        elif dataset == "amazon-co-computer":
            data = AmazonCoBuyComputerDataset()
        elif dataset == "amazon-co-photo":
            data = AmazonCoBuyPhotoDataset()
        else:
            assert False
    
        if dataset in ["ogbn-arxiv", "ogbn-proteins", "ogbn-products"]:
            graph, labels = data[0]
            splitted_idx = data.get_idx_split()
            train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    
            evaluator_ = Evaluator(name=dataset)
            evaluator = lambda pred, labels: evaluator_.eval(
                {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
            )["acc"]
        elif dataset in ["cora", "citeseer", "pubmed", "reddit"]:
            graph = data[0]
            labels = graph.ndata["label"].reshape(-1, 1)
            train_mask, val_mask, test_mask = graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"]
            train_idx, val_idx, test_idx = map(
                lambda mask: torch.nonzero(mask, as_tuple=False).squeeze_(), [train_mask, val_mask, test_mask]
            )
    
            evaluator = compute_acc
        elif dataset == "cora-full":
            graph = data[0]
            labels = graph.ndata["label"].reshape(-1, 1)
        elif dataset in ["amazon-co-computer", "amazon-co-photo"]:
            graph = data[0]
            labels = graph.ndata["label"].reshape(-1, 1)
            train_idx, val_idx, test_idx = None, None, None
    
            evaluator = compute_acc
        else:
            assert False
    
        n_node_feats = graph.ndata["feat"].shape[1]
        n_classes = (labels.max() + 1).item()
    
        print(f"#Nodes: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}, #Classes: {n_classes}")
        print(f"#Train/Val/Test nodes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    
        return graph, labels, train_idx, val_idx, test_idx, evaluator


    def preprocess(self, graph, lam=0):
        global n_node_feats
    
        # make bidirected
        feat = graph.ndata["feat"]
        graph = dgl.to_bidirected(graph)
        graph.ndata["feat"] = feat
        feat0 = feat
    
        # feat_mean = graph.ndata["feat"].mean(dim=0, keepdim=True)
        # feat_std = graph.ndata["feat"].std(dim=0, keepdim=True) + 1e-6
        # graph.ndata["feat"] = (graph.ndata["feat"] - feat_mean) / feat_std
        # graph.ndata["feat"] = graph.ndata["feat"]  / feat_std
    
        # add self-loop
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    
        # # feature smoothing
        # if lam > 0:
        #     for _ in range(7):
        #         degs = graph.out_degrees().float().clamp(min=1)
        #         norm = torch.pow(degs, -0.5)
        #         shp = norm.shape + (1,) * (feat.dim() - 1)
        #         norm = torch.reshape(norm, shp)
        #         feat = feat * norm
    
        #         graph.srcdata["feat"] = feat
        #         graph.update_all(fn.copy_src(src="feat", out="m"), fn.sum(msg="m", out="feat"))
        #         graph.ndata["feat"] = lam / (1 + lam) * graph.ndata["feat"] + 1 / (lam + 1) * feat0
    
        #         degs = graph.out_degrees().float().clamp(min=1)
        #         norm = torch.pow(degs, -0.5)
        #         shp = norm.shape + (1,) * (feat.dim() - 1)
        #         norm = torch.reshape(norm, shp)
        #         feat = feat * norm
    
        # graph.create_formats_()
    
        return graph






 
    def run_one_model(self, data_name, task, seed, X_lam, X_step, y_lam, y_step, lr, momentum, error_smooth, label_smooth):
 
        print("dataset/seed:", data_name, seed)
 
        if data_name in ["academic_cs", "academic_phy"]:
            graph, encoded_X, target, train_mask, test_mask, val_mask = self.load_academic(data_name)
        elif data_name in [
            "house", "county", "vk", "avazu", "wiki", "house_class", "vk_class", "dblp", "slap",
        ] + DATASET_NAMES:
            graph, encoded_X, target, train_mask, test_mask, val_mask = self.load_graph_x_target(data_name, seed)
        else:
            graph, target, train_mask, val_mask, test_mask, evaluator = self.load_data(data_name)
            graph = self.preprocess(graph)
            encoded_X = graph.ndata["feat"]

 
  
        # print("-----------", graph, encoded_X, target)
        print("Start training...")
        # if autogluon == True:
        #     model = AutoGluon(task, seed, data_name, graph, train_mask, test_mask, val_mask, X_lam, X_step, y_lam, y_step, error_smooth, label_smooth)
        #     metrics = model.train(
        #     encoded_X, target, cat_features=None)
        # else:
        model = GBDT(task, seed, data_name, graph, train_mask, test_mask, val_mask, X_lam, X_step, y_lam, y_step, lr, momentum, error_smooth, label_smooth)
        score = model.train(encoded_X, target, cat_features=None)
        return score
 
    def run(self, max_seeds: int = 5):
 
        parser = argparse.ArgumentParser(
            description="Train a GBDT with graph information", formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--datasets", "-s", type=str)
        parser.add_argument("--task", "-t", type=str, choices=["regression", "classification"])
        parser.add_argument('--X_lam', type=float, required=True, help='lambda')
        parser.add_argument('--X_step', type=int, required=True, help='propagation step')
        parser.add_argument('--y_lam', type=float, required=True, help='lambda')
        parser.add_argument('--y_step', type=int, required=True, help='propagation step')
        parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
        parser.add_argument('--momentum', default=False, action='store_true')
        # parser.add_argument('--autoGluon', default=False, action='store_true')
        parser.add_argument('--error_smooth', default=False, action='store_true')
        parser.add_argument('--label_smooth', default=False, action='store_true')
        args = parser.parse_args()
        state = {k: v for k, v in args._get_kwargs()}
        print(state)

 
        aggregated = dict()
        seed_results = []
        for seed in range(max_seeds):
            np.random.seed(seed)
            seed_results.append(self.run_one_model(args.datasets, args.task, str(seed), args.X_lam, args.X_step, args.y_lam, args.y_step, args.lr, args.momentum, args.error_smooth, args.label_smooth))
        aggregated[args.datasets] = (np.mean(seed_results), np.std(seed_results))
 
        # save_path = f"results/autogluon_{args.autoGluon}/{args.datasets}"
        save_path = f"results/{args.datasets}"

        os.makedirs(save_path, exist_ok=True)
        with open(f"{save_path}/seed_results.json", "w+") as f:
            json.dump(seed_results, f)
        with open(f"{save_path}/aggregated.json", "w+") as f:
            json.dump(aggregated, f)
        exit()
 
 
if __name__ == "__main__":
    fire.Fire(RunModel().run)
