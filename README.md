# TabGraphs: A Benchmark and Strong Baselines for Learning on Graphs with Tabular Node Features

## About the datasets

The proposed TabGraphs benchmark can be downloaded via [our Zenodo record](https://zenodo.org/records/13823102). It is necessary to put the compressed `.zip` files into the `datasets` directory. To unzip a dataset `<dataset_name>`, one can run `unzip <dataset_name>` in their terminal.

In each dataset subfolder, we provide the following files:
- `features.csv` — node features
- `targets.csv` — node targets
- `edgelist.csv` — list of edges in graph
- `train_mask.csv`, `valid_mask.csv`, `test_mask.csv` — split masks

Besides that, we put `info.yaml` with the necessary information about dataset:
- `dataset_name` — dataset name
- `task` — prediction task
- `metric` — metric used to evaluate predictive performance
- `num_classes` — number of classes, if applicable
- `has_unlabeled_nodes` — whether dataset has unlabaled nodes
- `has_nans_in_num_features` — whether dataset has NaNs in numerical features
- `graph_is_directed` — whether graph is directed
- `graph_is_weighted` — whether graph is weighted (if true, then `edgelist.csv` has 3 columns instead of 2)
- `target_name` — target name
- `num_feature_names` — list of numerical feature names
- `cat_feature_names` — list of categorical feature names
- `bin_feature_names` — list of binary feature names

**Note!** The proposed TabGraphs benchmark is released under the CC BY 4.0 International license.

## About the source code

In `source` directory, one can also find the source code for reproducing experiments in our paper. Note that only `gnns` subfolder contains our original code, while subfolders `bgnn`, `ebbs` and `tabular` are taken from open sources and adapted to make them consistent with our experimental setup.

Further, we provide the original sources:
- `tabular` — [github.com/yandex-research/tabular-dl-tabr](https://github.com/yandex-research/tabular-dl-tabr)
- `bgnn` — [github.com/nd7141/bgnn](https://github.com/nd7141/bgnn)
- `ebbs` — [github.com/JiuhaiChen/EBBS](https://github.com/JiuhaiChen/EBBS)

The only changes that were made in the original repositories are related to the logging of experimental results and the metrics used for validation.

## How to reproduce results

1. Run notebook `notebooks/prepare-graph-augmentation.ipynb` to prepare graph-based feature augmentations (NFA) that can be used by tabular baselines from `tabular`.
2. Run notebook `notebooks/prepare-node-embeddings.ipynb` to prepare optional DeepWalk embeddings (DWE) for the proposed datasets that can further improve predictive performance.
3. Run notebook `notebooks/convert-graph-datasets.ipynb` to convert the provided graph datasets (probably with NFA and/or DWE) into the format required by `tabular` baselines and specialized models `bgnn` and `ebbs`.
4. Run experiments according to the instructions provided in the corresponding directories.

**Note!** The source code for `tabular` baselines and `bgnn` model is distributed under the MIT license, and our code for `gnns` is also released under the same MIT license.