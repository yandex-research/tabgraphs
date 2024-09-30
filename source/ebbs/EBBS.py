import sys
import time
LARGE_NUMBER = sys.maxsize
import numpy as np
import torch
import pdb
import torch.nn.functional as F
from modules import UnfoldindAndAttention
import pandas as pd
from Base import *
from collections import defaultdict as ddict
from catboost import Pool, CatBoostClassifier, CatBoostRegressor, sum_models
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn import metrics as M

MULTICLASS_DATASET_NAMES = [
    'hm-categories',
    'browser-games'
]
NUM_THREADS = 32


class GBDT(object):
    def __init__(self, task, seed, data_name, graph, train_mask, test_mask, val_mask, X_lam, X_step, y_lam, y_step, lr, momentum, error_smooth, label_smooth):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.task = task
        self.seed = seed
        self.data_name = data_name
        self.graph = graph.to(self.device)
        self.train_mask = np.array(train_mask)
        self.test_mask = test_mask
        self.val_mask = val_mask

        self.iter_per_epoch = 1
        self.depth = 6
        self.gbdt_lr = lr
        self.momentum = momentum
        self.error_smooth = error_smooth
        self.label_smooth = label_smooth

        self.propagation_X = UnfoldindAndAttention(lam=X_lam, prop_step=X_step)
        self.propagation_y = UnfoldindAndAttention(lam=y_lam, prop_step=y_step)

        
        

    def _calc_data_scores(self, X, epoch):

        if epoch == 0:
            scores = torch.zeros(self.num_samples, self.out_dim)
        else:       
            scores = (1-self.theta) * self.gbdt_model + self.theta * self.h_model
            scores = torch.from_numpy(scores).float().view(self.num_samples, self.out_dim)

        return scores.to(self.device)




    def _calc_gradient(self, scores, labels):
      
        
        scores.requires_grad_()

        
        mask_rate = 0.5
        mask = torch.rand(self.train_mask.shape) < mask_rate
        # pdb.set_trace()
        self.train_labels_idx = self.train_mask[mask]
        self.train_pred_idx = self.train_mask[~mask]
        
        

        with torch.enable_grad():
            ## error_smooth
            if self.error_smooth:
                error = torch.zeros(self.num_samples, self.out_dim).to(self.device)
                error[self.train_labels_idx] = labels[self.train_labels_idx] - scores[self.train_labels_idx]
                assert len(error.size()) == 2
                error = self.propagation_y.forward(self.graph, error)
                scores_correct = scores + error
            else:
                scores_correct = scores

            ## label_smooth
            if self.label_smooth:
                assert len(scores_correct.size()) == 2
                scores_correct[self.train_labels_idx] = labels[self.train_labels_idx]
                scores_correct = self.propagation_y.forward(self.graph, scores_correct)
       

            if self.task == 'regression':
                loss = F.mse_loss(scores_correct[self.train_pred_idx], labels[self.train_pred_idx], reduction='sum')
                # loss = F.mse_loss(scores_correct[self.train_mask], labels[self.train_mask], reduction='sum')

            elif self.task == 'classification':
                loss = F.cross_entropy(scores_correct[self.train_mask], labels[self.train_mask].long(), reduction='sum')
                

        grad = torch.autograd.grad(loss, scores, only_inputs=True)[0]
        grad = grad.detach()


        return  - grad.cpu().numpy() 

      
                    
    def _calc_loss(self, X, y, metrics):


        pred = torch.from_numpy(self.gbdt_model).float().view(self.num_samples, self.out_dim).to(self.device)

        ## error_smooth
        if self.error_smooth:
            error = torch.zeros(self.num_samples, self.out_dim).to(self.device)
            error[self.train_mask] = y[self.train_mask] - pred[self.train_mask]
            # error[self.train_mask] = self.one_hot[self.train_mask] - pred[self.train_mask]
            assert len(error.size()) == 2
            error = self.propagation_y.forward(self.graph, error)
            scores_correct = pred + error
        else:
            scores_correct = pred

        ## label_smooth
        if self.label_smooth:
            assert len(scores_correct.size()) == 2
            scores_correct[self.train_mask] = y[self.train_mask]
            scores_correct = self.propagation_y.forward(self.graph, scores_correct)




        train_results = self.evaluate_model(scores_correct, y, self.train_pred_idx)
        test_results = self.evaluate_model(scores_correct, y, self.test_mask)
        val_results = self.evaluate_model(scores_correct, y, self.val_mask)

        # pdb.set_trace()

        for metric_name in train_results:
            metrics[metric_name].append((train_results[metric_name].detach().item(),
                               val_results[metric_name].detach().item(),
                               test_results[metric_name].detach().item()
                               ))
        return train_results, test_results, val_results


    def evaluate_model(self, logits, target_labels, mask):
        metrics = {}
        y = target_labels[mask]
        with torch.no_grad():
            pred = logits[mask]
            if self.task == 'regression':
                metrics['loss'] = torch.sqrt(F.mse_loss(pred, y))
                metrics['score'] = torch.tensor([
                    M.r2_score(y.cpu().numpy(), pred.cpu().numpy())
                ])
            elif self.task == 'classification':
                metrics['loss'] = F.cross_entropy(pred, y.long())
                if self.data_name in MULTICLASS_DATASET_NAMES:
                    metrics['score'] = torch.Tensor([(y == pred.max(1)[1]).sum().item()/y.shape[0]])
                else:
                    metrics['score'] = torch.tensor([
                        M.average_precision_score(y.cpu().numpy(), pred.cpu().numpy()[:, 1])
                    ])

            return metrics


    def init_gbdt_model(self, num_epochs):

        if self.task == 'regression':
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = 'RMSE'
        else:
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = 'MultiRMSE'


        return catboost_model_obj(iterations=num_epochs,
                                  depth=self.depth,
                                  learning_rate=self.gbdt_lr,
                                  loss_function=catboost_loss_fn,
                                  thread_count=NUM_THREADS,
                                  random_seed=int(self.seed),
                                  nan_mode='Min')





    def fit_gbdt(self, pool, trees_per_epoch):
        gbdt_model = self.init_gbdt_model(trees_per_epoch)
        gbdt_model.fit(pool, verbose=False)
        return gbdt_model

    def append_gbdt_model(self, new_gbdt_model, X, weights):
        if self.gbdt_model is None:
            return new_gbdt_model.predict(X)
        return self.gbdt_model * weights[0] + self.h_model * weights[1] + new_gbdt_model.predict(X)


    def append_h_model(self, new_gbdt_model, X, weights):
        if self.h_model is None:
            return new_gbdt_model.predict(X)
        return self.h_model * weights[0] +  new_gbdt_model.predict(X) * weights[1]



    def train_gbdt(self, gbdt_X_train, gbdt_y_train, cat_features,
                   gbdt_trees_per_epoch, gbdt_alpha):

        pool = Pool(gbdt_X_train, gbdt_y_train, cat_features=cat_features, thread_count=NUM_THREADS)
        epoch_gbdt_model = self.fit_gbdt(pool, gbdt_trees_per_epoch)
        # epoch_gbdt_model = self.fit_gbdt(pool, 1)

        
        self.gbdt_model = self.append_gbdt_model(epoch_gbdt_model, gbdt_X_train, weights=[1-self.theta, self.theta, gbdt_alpha])

        if self.momentum:
            self.h_model = self.append_h_model(epoch_gbdt_model, gbdt_X_train, weights=[1, gbdt_alpha/self.theta])
        else:
            self.h_model = self.append_h_model(epoch_gbdt_model, gbdt_X_train, weights=[1, gbdt_alpha])





    def train(self, encoded_X, target, cat_features=None, num_boost_round=2000, early_stopping_rounds=20):
        self.gbdt_model = None
        self.h_model =None
        self.epoch_gbdt_model = None
        metrics = ddict(list)
        shrinkage_rate = 1.0
        best_iteration = None
        best_val_loss = LARGE_NUMBER
        self.best_iteration = None
        train_start_time = time.time()

        self.num_samples = target.size(0)
        if self.task == 'regression':
            self.out_dim = 1
        elif self.task == 'classification':
            self.out_dim = int(torch.nan_to_num(target).max() + 1)
            # self.out_dim = 4
            # self.out_dim = len(set(target[self.test_mask, 0]))
            # pdb.set_trace()
            # self.one_hot = torch.zeros(self.num_samples, self.out_dim).scatter_(1, target.unsqueeze(1).long(), 1)
            # one_hot_init = torch.zeros(self.num_samples, self.out_dim).to(self.device)
            # self.one_hot = one_hot_init.scatter_(1, target.unsqueeze(1).long(), 1)
            # self.one_hot = torch.zeros(self.num_samples, self.out_dim).to(self.device)
            target = target.squeeze()    



        print("Training until validation scores don't improve for {} rounds.".format(early_stopping_rounds))

  
        ## propagate the feature
        encoded_X = encoded_X.to(self.device)
        assert len(encoded_X.size()) == 2
        corrected_X = self.propagation_X.forward(self.graph, encoded_X.to(self.device))  
        if self.task == 'regression':
            feature = torch.cat((encoded_X, corrected_X), 1).cpu().numpy()
            # feature = encoded_X.cpu().numpy()
        elif self.task == 'classification':
            feature = encoded_X.cpu().numpy()



        for iter_cnt in range(num_boost_round):
            iter_start_time = time.time()

            if self.momentum:
                self.theta = 2 / (iter_cnt+2)
            else:
                self.theta = 0
       
            scores = self._calc_data_scores(feature, iter_cnt)
            grad = self._calc_gradient(scores, target.to(self.device))
            self.train_gbdt(feature, grad, cat_features, self.iter_per_epoch, gbdt_alpha=shrinkage_rate)



            train_metric, test_metric, val_metric = self._calc_loss(feature, target.to(self.device), metrics)
            train_loss = train_metric['loss']
            test_loss = test_metric['loss']
            val_loss = val_metric['loss']
            test_score = test_metric['score']

         

            val_loss_str = '{:.10f}'.format(val_loss) if val_loss else '-'
            print("Iter {:>3}, Train's Loss: {:.10f}, Test's Loss: {}, Valid's Loss: {}, Test's Accuracy: {}, Elapsed: {:.2f} secs"
                  .format(iter_cnt, train_loss, test_loss, val_loss_str, test_score.item(), time.time() - iter_start_time))

            
            
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_iteration = iter_cnt
                best_test_score = test_score


            if iter_cnt - best_iteration >= early_stopping_rounds:
                break


      
        self.best_iteration = best_iteration
        print("Training finished. Elapsed: {:.2f} secs".format(time.time() - train_start_time))



        # import pickle


        # if self.momentum:
        #     with open('momentum.pickle', 'wb') as handle:
        #         pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #         exit()
                


        # plot(metrics, ['train', 'val', 'test'], 'CBS', 'CBS')
        # exit()

        if self.task == 'regression':
            print("iter {:>3}, test score: {:.4f}".format(best_iteration, best_test_score.item()))
            return float(best_test_score.cpu().numpy()[0])

        elif self.task == 'classification':
            print("iter {:>3}, test score: {:.4f}".format(best_iteration, best_test_score.item()))
            return float(best_test_score.cpu().numpy()[0])
