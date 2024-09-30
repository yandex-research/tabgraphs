## AG+ ##
If you are already familiary with running autogluon for node classification, i recommend you to start from AG.py

You need to prepare your dataset, including  graph, labels, train_idx, val_idx, test_idx **(you can refer to Line 227 to Line 230, downloading OGB-arxiv dataset)**

For train function in Line 234, including two parts, the first part is exactly same with AutoGluon and the second part is about Correct and Smooth. 

To implement the code  **python AG.py**





## EBBS ##


There should be at least X.csv (node features), y.csv (target labels), graph.graphml (graph in graphml format).

You can also have cat_features.txt specifying names of categorical columns.

You can also have masks.json specifying train/val/test splits.

Then run the command:

python run.py   --datasets  your_datasets   --task regression   --X_lam 20.0 --X_step 5  --y_lam 2.0 --y_step 5  --lr 0.1    --label_smooth  --error_smooth


(Very simple example on our dataset House: **zip datasets** and then run **python run.py   --datasets  house   --task regression   --X_lam 20.0 --X_step 5  --y_lam 2.0 --y_step 5  --lr 0.1    --label_smooth  --error_smooth**)




