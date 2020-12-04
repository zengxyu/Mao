import os
import shutil
import sys

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append('../input/pytorch-tabnet')

import numpy as np
import torch
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


class TabnetModelHelper:
    def __init__(self, kfold, param, xtrain, ytrain, ytrain_with_non_scored=None, xtest=None, verbose=False):
        self.kfold = kfold
        self.param = param
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.ytrain_with_non_scored = ytrain_with_non_scored
        self.xtest = xtest
        self.num_features = xtrain.shape[1]
        self.num_labels = ytrain.shape[1]
        self.device = param['device']
        self.loss_fn = SmoothBCEwLogits(smoothing=5e-5)
        self.model = None
        self.verbose = verbose

    def train_models(self):
        best_val_losses = []
        for n, (tr, te) in enumerate(self.kfold.split(self.xtrain, self.ytrain)):
            print(f'Train fold {n + 1}')
            x_train, x_val = self.xtrain[tr], self.xtrain[te]
            y_train, y_val = self.ytrain[tr], self.ytrain[te]

            model_save_path = os.path.join(self.param['output'], self.param['model_save_name'].format(n + 1))
            tabnet_fit_params = self.__get_tabnet_fit_params()
            tabnet_params = self.__get_tabnet_params(param_id=self.param['param_id'])
            self.model = MyTabnetRegressor(**tabnet_params)
            self.model.fit(
                x_train, y_train,
                eval_set=[(x_val, y_val)],
                **tabnet_fit_params,
            )
            best_val_losses.append(min(self.model.history['val_logits_ll']))
            self.model.save_model(model_save_path)
        return best_val_losses

    def test_models(self, is_predict_train_data=False):
        if self.verbose:
            print('###==============Test model for {}===============###'.format(
                "train data" if is_predict_train_data else "test data"))
        if is_predict_train_data:
            xtest = self.xtrain
        else:
            xtest = self.xtest

        num_samples = xtest.shape[0]
        total_test_preds = np.zeros((num_samples, self.num_labels, self.param['n_folds']))

        for n in range(self.param['n_folds']):
            print(f'Test fold {n + 1}')
            model_save_path = os.path.join(self.param['output'], (self.param['model_save_name'] + '.zip').format(n + 1))
            if not os.path.exists(model_save_path):
                path = os.path.join(self.param['output'], (self.param['model_save_name'].format(n + 1)))
                model_save_path = os.path.join(self.param['output'], self.param['model_save_name'].format(n + 1))
                shutil.make_archive(model_save_path, "zip", path)
                model_save_path += '.zip'
            if self.verbose:
                print("Predicting---load model from {}".format(model_save_path))
            test_pred_per_fold = self.__predict(xtest, model_save_path)

            total_test_preds[:, :, n] = test_pred_per_fold

        total_test_preds = np.mean(total_test_preds, axis=2)
        return total_test_preds

    def __predict(self, x_test, model_path):
        tabnet_params = self.__get_tabnet_params(param_id=self.param['param_id'])

        self.model = MyTabnetRegressor(**tabnet_params)
        self.model.load_model(model_path)

        loaded_preds = self.model.predict(x_test)
        loaded_preds = 1 / (1 + np.exp(-loaded_preds))

        return loaded_preds

    def __get_tabnet_params(self, param_id):
        if param_id == 0:
            tabnet_params = dict(
                n_d=24, n_a=24, n_steps=1, gamma=1.3,
                lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                mask_type='entmax',
                scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9, ),
                # epoch打印间隔
                verbose=1
            )
        elif param_id == 1:
            tabnet_params = dict(
                n_d=32, n_a=32, n_steps=1, gamma=1.3, seed=20,
                lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                mask_type='entmax',
                scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9, ),
                # epoch打印间隔
                verbose=1
            )
        elif param_id == 2:
            tabnet_params = dict(
                n_d=24, n_a=256, n_steps=1, gamma=1.3, seed=21,
                lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2, weight_decay=1e-6),
                mask_type='entmax',
                scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9, ),
                # epoch打印间隔
                verbose=1
            )
        else:
            tabnet_params = dict(
                n_d=32, n_a=128, n_steps=1, gamma=1.3, seed=42,
                lambda_sparse=0, n_shared=0, n_independent=1, optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                mask_type='entmax',
                scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9, ),
                # epoch打印间隔
                verbose=1
            )

        return tabnet_params

    def __get_tabnet_fit_params(self):
        tabnet_fit_params = dict(
            eval_name=["val"],
            eval_metric=["logits_ll"],
            max_epochs=self.param['n_epochs'],
            patience=15, batch_size=self.param['train_batch_size'], virtual_batch_size=32,
            num_workers=0, drop_last=False,
            # use binary cross entropy as this is not a regression problem
            # loss_fn=torch.nn.functional.binary_cross_entropy_with_logits
            loss_fn=SmoothBCEwLogits(smoothing=5e-5)
        )
        return tabnet_fit_params


class MyTabnetRegressor(TabNetRegressor):

    def smooth(self, y_true, n_classes, smoothing=0.001):
        assert 0 <= smoothing <= 1
        with torch.no_grad():
            y_true = y_true * (1 - smoothing) + torch.ones_like(y_true).to(self.device) * smoothing / n_classes
        return y_true

    def compute_loss(self, y_pred, y_true):
        y_true = self.smooth(y_true, y_pred.shape[1])
        return self.loss_fn(y_pred, y_true)


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
                                           self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class LogitsLogLoss(Metric):
    """
    LogLoss with sigmoid applied
    """

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            LogLoss of predictions vs targets.
        """
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
        return np.mean(-aux)
