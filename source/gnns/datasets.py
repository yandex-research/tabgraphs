import yaml
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import dgl
from sklearn.preprocessing import (FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
                                   QuantileTransformer, OneHotEncoder, KBinsDiscretizer)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, r2_score
from utils import cross_entropy_with_soft_labels, get_soft_labels


class Dataset:
    transforms = {
        'none': FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x),
        'standard-scaler': StandardScaler(),
        'min-max-scaler': MinMaxScaler(),
        'robust-scaler': RobustScaler(unit_variance=True),
        'power-transform-yeo-johnson': PowerTransformer(method='yeo-johnson', standardize=True),
        'quantile-transform-normal': QuantileTransformer(output_distribution='normal', subsample=1_000_000_000,
                                                         random_state=0),
        'quantile-transform-uniform': QuantileTransformer(output_distribution='uniform', subsample=1_000_000_000,
                                                          random_state=0)
    }

    def __init__(self, name, add_self_loops=False, use_node_embeddings=False,
                 num_features_imputation_strategy='most_frequent', num_features_transform='none',
                 regression_target_transform='none', regression_by_classification=False, num_regression_target_bins=50,
                 regression_target_binning_strategy='uniform', use_soft_labels=False, device='cpu'):
        print('Preparing data...')
        with open(f'data/{name}/info.yaml', 'r') as file:
            info = yaml.safe_load(file)

        features_df = pd.read_csv(f'data/{name}/features.csv', index_col=0)
        num_features = features_df[info['num_feature_names']].values.astype(np.float32)
        bin_features = features_df[info['bin_feature_names']].values.astype(np.float32)
        cat_features = features_df[info['cat_feature_names']].values.astype(np.float32)
        targets = features_df[info['target_name']].values.astype(np.float32)

        if num_features.shape[1] > 0:
            if info['has_nans_in_num_features']:
                num_features = SimpleImputer(strategy=num_features_imputation_strategy).fit_transform(num_features)

            num_features = self.transforms[num_features_transform].fit_transform(num_features)

        if cat_features.shape[1] > 0:
            cat_features = OneHotEncoder(sparse_output=False, dtype=np.float32).fit_transform(cat_features)

        if use_node_embeddings:
            node_embeddings = np.load(f'data/{name}/node_embeddings.npz')['node_embeds']

        train_mask = pd.read_csv(f'data/{name}/train_mask.csv', index_col=0).values.reshape(-1)
        train_idx = np.where(train_mask)[0]
        val_mask = pd.read_csv(f'data/{name}/valid_mask.csv', index_col=0).values.reshape(-1)
        val_idx = np.where(val_mask)[0]
        test_mask = pd.read_csv(f'data/{name}/test_mask.csv', index_col=0).values.reshape(-1)
        test_idx = np.where(test_mask)[0]

        if info['task'] == 'regression':
            targets_orig = targets.copy()
            labeled_idx = np.concatenate([train_idx, val_idx, test_idx], axis=0)

            if regression_by_classification:
                target_binner = KBinsDiscretizer(n_bins=num_regression_target_bins,
                                                 strategy=regression_target_binning_strategy,
                                                 encode='ordinal',
                                                 subsample=None)
                target_binner.fit(targets[train_idx][:, None])
                targets[labeled_idx] = target_binner.transform(targets[labeled_idx][:, None]).squeeze(1)
                targets = targets.astype(np.int64)
                if use_soft_labels:
                    targets = get_soft_labels(targets, num_bins=num_regression_target_bins, labeled_idx=labeled_idx)

                bin_edges = target_binner.bin_edges_[0]
                bin_preds = (bin_edges[:-1] + bin_edges[1:]) / 2

            else:
                targets_transform = self.transforms[regression_target_transform]
                targets_transform.fit(targets[train_idx][:, None])
                targets[labeled_idx] = targets_transform.transform(targets[labeled_idx][:, None]).squeeze(1)

        if info['task'] == 'binary_classification':
            targets_dim = 1
            loss_fn = F.binary_cross_entropy_with_logits
        elif info['task'] == 'multiclass_classification':
            targets_dim = info['num_classes']
            targets = targets.astype(np.int64)
            loss_fn = F.cross_entropy
        elif info['task'] == 'regression':
            targets_dim = num_regression_target_bins if regression_by_classification else 1
            if regression_by_classification:
                if use_soft_labels:
                    loss_fn = cross_entropy_with_soft_labels
                else:
                    loss_fn = F.cross_entropy
            else:
                loss_fn = F.mse_loss
        else:
            raise ValueError(f'Unknown task: {info["task"]}.')

        edges_df = pd.read_csv(f'data/{name}/edgelist.csv')
        edges = edges_df.values[:, :2]

        features = np.concatenate([num_features, bin_features, cat_features], axis=1)
        if use_node_embeddings:
            features = np.concatenate([features, node_embeddings], axis=1)

        num_features_mask = np.zeros(features.shape[1], dtype=bool)
        num_features_mask[:num_features.shape[1]] = True

        features = torch.from_numpy(features)
        num_features_mask = torch.from_numpy(num_features_mask)
        targets = torch.from_numpy(targets)
        if info['task'] == 'regression':
            targets_orig = torch.from_numpy(targets_orig)
            if regression_by_classification:
                bin_preds = torch.from_numpy(bin_preds)

        edges = torch.from_numpy(edges)
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(features), idtype=torch.int32)

        if info['graph_is_directed']:
            graph = dgl.to_bidirected(graph)

        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        train_idx = torch.from_numpy(train_idx)
        val_idx = torch.from_numpy(val_idx)
        test_idx = torch.from_numpy(test_idx)

        self.name = name
        self.task = info['task']
        self.metric = info['metric']
        self.device = device

        self.graph = graph.to(device)
        self.features = features.to(device)
        self.num_features_mask = num_features_mask.to(device)
        self.targets = targets.to(device)
        if info['task'] == 'regression':
            self.targets_orig = targets_orig.to(device)
            self.regression_by_classification = regression_by_classification
            self.use_soft_labels = use_soft_labels
            if regression_by_classification:
                self.bin_preds = bin_preds.to(device)
                self.target_binner = target_binner
            else:
                self.targets_transform = targets_transform

        self.features_dim = features.shape[1]
        self.targets_dim = targets_dim

        self.loss_fn = loss_fn

        self.train_idx = train_idx.to(device)
        self.val_idx = val_idx.to(device)
        self.test_idx = test_idx.to(device)

    def compute_metrics(self, preds):
        if self.metric == 'accuracy':
            preds = preds.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.targets[self.train_idx]).float().mean().item()
            val_metric = (preds[self.val_idx] == self.targets[self.val_idx]).float().mean().item()
            test_metric = (preds[self.test_idx] == self.targets[self.test_idx]).float().mean().item()

        elif self.metric == 'AP':
            targets = self.targets.cpu().numpy()
            preds = preds.cpu().numpy()

            train_idx = self.train_idx.cpu().numpy()
            val_idx = self.val_idx.cpu().numpy()
            test_idx = self.test_idx.cpu().numpy()

            train_metric = average_precision_score(y_true=targets[train_idx], y_score=preds[train_idx]).item()
            val_metric = average_precision_score(y_true=targets[val_idx], y_score=preds[val_idx]).item()
            test_metric = average_precision_score(y_true=targets[test_idx], y_score=preds[test_idx]).item()

        elif self.metric == 'R2':
            targets_orig = self.targets_orig.cpu().numpy()

            if self.regression_by_classification:
                bin_idx = preds.argmax(axis=1)
                preds_orig = self.bin_preds[bin_idx].cpu().numpy()
            else:
                preds_orig = self.targets_transform.inverse_transform(preds.cpu().numpy()[:, None]).squeeze(1)

            train_idx = self.train_idx.cpu().numpy()
            val_idx = self.val_idx.cpu().numpy()
            test_idx = self.test_idx.cpu().numpy()

            train_metric = r2_score(y_true=targets_orig[train_idx], y_pred=preds_orig[train_idx])
            val_metric = r2_score(y_true=targets_orig[val_idx], y_pred=preds_orig[val_idx])
            test_metric = r2_score(y_true=targets_orig[test_idx], y_pred=preds_orig[test_idx])

        else:
            raise ValueError(f'Unknown metric: {self.metric}.')

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics
