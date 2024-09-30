import argparse
 
import dgl
import pandas as pd
import torch
from autogluon.tabular import TabularDataset, TabularPredictor
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
from modules import UnfoldindAndAttention
 
device = None
 
n_node_feats, n_classes = 0, 0
 
freq = None
 
 
def compute_acc(pred, labels):
    return ((torch.argmax(pred, dim=1) == labels[:, 0]).float().sum() / len(pred)).item()
 
 
def load_data(dataset):
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
 
 
def preprocess(args, graph, lam=20):
    global n_node_feats
 
    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat
    feat0 = feat
 
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
 
    # feature smoothing
    if args.lam > 0:
        for _ in range(args.prop_step):
            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = feat * norm
 
            graph.srcdata["feat"] = feat
            graph.update_all(fn.copy_src(src="feat", out="m"), fn.sum(msg="m", out="feat"))
            graph.ndata["feat"] = lam / (1 + lam) * graph.ndata["feat"] + 1 / (lam + 1) * feat0
 
            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = feat * norm
 
    graph.create_formats_()
 
    return graph
 
 
def train(graph, encoded_X, y, train_idx, val_idx, test_idx):

 
    ## cat the features and target
    X = encoded_X.cpu().numpy()
    Column_index = ["Column_{}".format(i) for i in range(X.shape[1])]
    X_cat = pd.DataFrame(X, columns=Column_index)
    y = pd.DataFrame(y.cpu().numpy(), columns=["class"])
    dataset = pd.concat([X_cat, y], axis=1)
 
    # Or loading existing model
    # predictor = TabularPredictor.load('ogb-arxiv')
 
    # autogluon from here 
    # predictor = TabularPredictor(label='class', path=save_path).fit(dataset.iloc[train_mask], presets='best_quality')
    # predictor = TabularPredictor(label='class', path='ogb-arxiv').fit(dataset.iloc[train_idx.tolist() + val_idx.tolist()], num_bag_folds=5, num_bag_sets=1, num_stack_levels=1, time_limit=1000)
    # predictor = TabularPredictor(label="class", path="ogb-arxiv").fit(
    #     dataset.iloc[train_idx.tolist()], presets="best_quality", time_limit=600
    # )
    predictor = TabularPredictor(label="class", path="ogb-arxiv").fit(
        dataset.iloc[train_idx.tolist()], tuning_data=dataset.iloc[val_idx.tolist()], time_limit=50
    )
 
    ## results of train/val/test
    print('results of train/val/test before C&S')
    predictor.evaluate(dataset.iloc[train_idx.tolist()])
    predictor.evaluate(dataset.iloc[val_idx.tolist()])
    predictor.evaluate(dataset.iloc[test_idx.tolist()])


    ## incorporate the graph information (do correct and smooth)
    ## error smoothing
    y_pred = predictor.predict_proba(X_cat)
    y_pred = torch.from_numpy(y_pred.to_numpy()).float().to(device)
    y_true = torch.from_numpy(y.to_numpy()).long().to(device)
    y_one_hot = torch.zeros_like(y_pred).scatter_(1, y_true, 1)
 
    error_smooth = torch.zeros_like(y_pred)
    error_smooth[train_idx] = y_one_hot[train_idx] - y_pred[train_idx]
    assert len(error_smooth.size()) == 2
    propagation_error = UnfoldindAndAttention(lam=1, prop_step=20)
    error_smooth = propagation_error.forward(graph, error_smooth, train_idx, error=True)
    y_pred = y_pred + error_smooth
 

    ## label smoothing
    label_smooth = y_pred.clone()
    label_smooth[train_idx] = y_one_hot[train_idx]
    propagation_label = UnfoldindAndAttention(lam=1, prop_step=20)
    assert len(label_smooth.size()) == 2
    label_smooth = propagation_label.forward(graph, label_smooth, train_idx, label=True)
 

    # results of test after Correct and Smooth
    pred = label_smooth[test_idx]
    target_test = y_true[test_idx].squeeze()
    print('results of test after C&S:')
    print(torch.Tensor([(target_test == pred.max(1)[1]).sum().item() / target_test.shape[0]]))
 
 
def predict(predictor, x):
    # TODO
    ...
 
 
def main():
    global device
 
    argparser = argparse.ArgumentParser("AutoGluon", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # basic settings
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "ogbn-arxiv",
            "ogbn-proteins",
            "ogbn-products",
            "cora",
            "citeseer",
            "pubmed",
            "cora-full",
            "reddit",
            "amazon-co-computer",
            "amazon-co-photo",
        ],
        default="ogbn-arxiv",
        help="dataset",
    )
   
    # training
    argparser.add_argument("--lam", type=float, default=0, help="feature smoothing alpha")
    argparser.add_argument("--prop-step", type=int, default=5, help="prop step")
    args = argparser.parse_args()
  
 
    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(args.dataset)
    graph = preprocess(args, graph)
    x = graph.ndata["feat"]



    # Train begins here
    predictor = train(graph, x, labels, train_idx, val_idx, test_idx)
  

 
if __name__ == "__main__":
    main()
