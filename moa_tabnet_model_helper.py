import sys

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append('../input/pytorch-tabnet')

import numpy as np
import torch
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

# tabnet_params = dict(
#     n_d=24, n_a=24, n_steps=1, gamma=1.3,
#     lambda_sparse=0, optimizer_fn=torch.optim.Adam,
#     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
#     mask_type='entmax',
#     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
#     scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9, ),
#     # epoch打印间隔
#     verbose=1
# )

tabnet_params = dict(
    n_d=32,
    n_a=32,
    n_steps=1,
    gamma=1.3,
    lambda_sparse=0,
    optimizer_fn=optim.Adam,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    mask_type="entmax",
    scheduler_params=dict(
        mode="min", patience=5, min_lr=1e-5, factor=0.9),
    scheduler_fn=ReduceLROnPlateau,
    seed=42,
    verbose=10
)


class TabnetModelHelper:
    def __init__(self):
        global tabnet_params
        self.model = MyTabnetRegressor(**tabnet_params)
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

    def fit_and_save(self, data, n_epochs=None, patience=None, train_batch_size=None, val_batch_size=None,
                     model_save_path=None):
        x_train, y_train, x_val, y_val = data

        self.model.fit(x_train, y_train,
                       eval_set=[(x_val, y_val)],
                       eval_name=["val"],
                       eval_metric=["logits_ll"],
                       max_epochs=n_epochs,
                       patience=patience, batch_size=train_batch_size, virtual_batch_size=128,
                       num_workers=0, drop_last=False,
                       # use binary cross entropy as this is not a regression problem
                       loss_fn=self.loss_fn)
        self.model.save_model(model_save_path)

    def predict(self, x_test, model_path):
        self.model.load_model(model_path)

        loaded_preds = self.model.predict(x_test)
        loaded_preds = 1 / (1 + np.exp(-loaded_preds))

        return loaded_preds


class MyTabnetRegressor(TabNetRegressor):

    def smooth(self, y_true, n_classes, smoothing=0.001):
        assert 0 <= smoothing <= 1
        with torch.no_grad():
            y_true = y_true * (1 - smoothing) + torch.ones_like(y_true).to(self.device) * smoothing / n_classes
        return y_true

    def compute_loss(self, y_pred, y_true):
        y_true = self.smooth(y_true, y_pred.shape[1])
        return self.loss_fn(y_pred, y_true)


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
