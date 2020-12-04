import copy
import os
import sys
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.modules.loss import _WeightedLoss


class PytorchModelHelper:
    def __init__(self, kfold, param, xtrain, ytrain, ytrain_with_non_scored=None, xtest=None, verbose=False):
        self.kfold = kfold
        self.param = param
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.ytrain_with_non_scored = ytrain_with_non_scored
        self.xtest = xtest
        self.num_features = xtrain.shape[1]
        self.num_labels = ytrain.shape[1]
        self.num_all_labels = ytrain_with_non_scored.shape[1]
        self.device = param['device']

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.fine_tune_scheduler = None
        self.loss_tr = SmoothBCEwLogits(smoothing=0.001)
        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.verbose = verbose

    def train_models(self):
        for n, (tr, te) in enumerate(self.kfold.split(self.ytrain, self.ytrain)):
            print(f'###===============================Train fold {n + 1}===============================###')
            x_train, x_val = self.xtrain[tr], self.xtrain[te]
            y_train, y_val = self.ytrain[tr], self.ytrain[te]
            y_train_with_non_scored, y_val_with_non_scored = self.ytrain_with_non_scored[tr], \
                                                             self.ytrain_with_non_scored[te]
            model_save_path = os.path.join(self.param['output'], self.param['model_save_name'].format(n + 1))

            if 'is_transfer' in self.param.keys() and self.param['is_transfer']:
                WEIGHT_DECAY = {'ALL_TARGETS': 1e-5, 'SCORED_ONLY': 3e-6}
                MAX_LR = {'ALL_TARGETS': 1e-2, 'SCORED_ONLY': 3e-3}
                # Train on scored + nonscored targets
                best_loss, best_model = self.__fit(Model=self.param['model'],
                                                   data=[x_train, y_train_with_non_scored, x_val,
                                                         y_val_with_non_scored],
                                                   num_labels=self.num_all_labels,
                                                   hidden_sizes=self.param['hidden_sizes'],
                                                   dropout_rates=self.param['dropout_rates'],
                                                   lr=MAX_LR['ALL_TARGETS'],
                                                   weight_decay=WEIGHT_DECAY['ALL_TARGETS'],
                                                   n_epochs=self.param['n_epochs'],
                                                   patience=5,
                                                   train_batch_size=self.param[
                                                       'train_batch_size'],
                                                   val_batch_size=self.param[
                                                       'val_batch_size'],
                                                   )
                print(
                    "====================================迁移模型的学习=====================================================")

                best_loss, best_model = self.__fit(Model=self.param['model'],
                                                   data=[x_train, y_train, x_val,
                                                         y_val],
                                                   num_labels=self.num_labels,
                                                   hidden_sizes=self.param['hidden_sizes'],
                                                   dropout_rates=self.param['dropout_rates'],
                                                   lr=MAX_LR['SCORED_ONLY'],
                                                   weight_decay=WEIGHT_DECAY['SCORED_ONLY'],
                                                   n_epochs=self.param['n_epochs'],
                                                   patience=self.param['patience'],
                                                   train_batch_size=self.param[
                                                       'train_batch_size'],
                                                   val_batch_size=self.param[
                                                       'val_batch_size'],
                                                   is_transfer=True,
                                                   model_dict=best_model
                                                   )
            else:
                best_loss, best_model = self.__fit(Model=self.param['model'],
                                                   data=[x_train, y_train, x_val,
                                                         y_val],
                                                   num_labels=self.num_labels,
                                                   hidden_sizes=self.param['hidden_sizes'],
                                                   dropout_rates=self.param['dropout_rates'],
                                                   lr=2e-2,
                                                   weight_decay=1e-5,
                                                   n_epochs=self.param['n_epochs'],
                                                   patience=self.param['patience'],
                                                   train_batch_size=self.param[
                                                       'train_batch_size'],
                                                   val_batch_size=self.param[
                                                       'val_batch_size'],
                                                   )

            torch.save(best_model, model_save_path)

            print("Fold : {} ; best loss : {}".format(n + 1, best_loss))

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
            if self.verbose:
                print(f'Test fold {n + 1}')
            model_save_path = os.path.join(self.param['output'], self.param['model_save_name'].format(n + 1))

            test_pred_per_fold = self.__predict(model_name=self.param['model'],
                                                num_labels=self.num_labels,
                                                hidden_sizes=self.param['hidden_sizes'],
                                                dropout_rates=self.param['dropout_rates'],
                                                data=xtest,
                                                test_batch_size=self.param[
                                                    'test_batch_size'],
                                                model_path=model_save_path)

            total_test_preds[:, :, n] = test_pred_per_fold

        total_test_preds = np.mean(total_test_preds, axis=2)
        return total_test_preds

    def __fit(self, Model, data, num_labels, hidden_sizes, dropout_rates, lr, weight_decay, n_epochs=None,
              patience=None, train_batch_size=None,
              val_batch_size=None, model_dict=None, model_path=None, is_transfer=False):

        self.train_loader, self.val_loader = self.__train_data_factory(data=data,
                                                                       train_batch_size=train_batch_size,
                                                                       val_batch_size=val_batch_size)

        if is_transfer:
            self.fine_tune_scheduler = FineTuneScheduler(epochs=n_epochs)
            # Copy model without the top layer
            self.model = self.fine_tune_scheduler.copy_without_top(self.param['model'], self.device,
                                                                   model_dict,
                                                                   self.num_features,
                                                                   self.num_all_labels,
                                                                   self.num_labels,
                                                                   self.param['hidden_sizes'],
                                                                   self.param['dropout_rates'], )
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3,
                                                                  eps=1e-4,
                                                                  verbose=True)
            if self.verbose:
                print("=====================================================")
                print("Model : {} ;".format(self.model.__class__.__name__))
                print("Model net : {} ;".format(self.model))
                print("Scheduler : {} ;".format(self.scheduler.__class__.__name__))
                print("=====================================================")
        else:
            self.__init_model(Model=Model, num_labels=num_labels, hidden_sizes=hidden_sizes,
                              dropout_rates=dropout_rates, lr=lr,
                              weight_decay=weight_decay)

        best_loss = {'train': np.inf, 'val': np.inf}
        best_model = None
        early_step = 0
        for epoch in range(n_epochs):
            # if is_transfer:
            #     self.fine_tune_scheduler.step(epoch, self.model)

            train_loss = self.__train()
            val_loss = self.__val()
            epoch_loss = {'train': train_loss, 'val': val_loss}
            print("Epoch {}/{}   -   loss: {:5.5f}   -   val_loss: {:5.5f}".format(epoch + 1, n_epochs,
                                                                                   epoch_loss['train'],
                                                                                   epoch_loss['val']))

            self.scheduler.step(epoch_loss['val'])

            if epoch_loss['val'] < best_loss['val']:
                best_loss = epoch_loss
                best_model = copy.deepcopy(self.model.state_dict())
                early_step = 0
            elif patience is not None:
                early_step += 1
                if early_step >= patience:
                    break

        return best_loss, best_model

    def __predict(self, model_name, num_labels, hidden_sizes, dropout_rates, data, test_batch_size, model_path):
        self.test_loader = self.__test_data_factory(data=data, test_batch_size=test_batch_size)
        self.__init_model(Model=model_name, num_labels=num_labels, hidden_sizes=hidden_sizes,
                          dropout_rates=dropout_rates, model_path=model_path)
        test_pred = self.__test()

        return test_pred

    def __init_model(self, Model, num_labels, hidden_sizes, dropout_rates, lr=2e-2, weight_decay=1e-5, model_dict=None,
                     model_path=None):
        self.model = Model(self.num_features, num_labels, hidden_sizes, dropout_rates).to(self.device)
        # 如果有模型就加载模型
        if model_dict is not None:
            self.model.load_state_dict(model_dict)

        elif model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

        else:
            if self.verbose:
                print("Load the model with nothing, this is a new model")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3,
                                                              eps=1e-4,
                                                              verbose=True)
        if self.verbose:
            print("=====================================================")
            print("Model : {} ;".format(self.model.__class__.__name__))
            print("Model net : {} ;".format(self.model))
            print("Scheduler : {} ;".format(self.scheduler.__class__.__name__))
            print("=====================================================")

    def __train(self):
        self.model.train()
        running_loss = 0.0
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(x)
            loss = self.loss_tr(pred, y)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() / len(self.train_loader)
        return running_loss

    def __val(self):
        self.model.eval()
        running_loss = 0.0
        for i, (x, y) in enumerate(self.val_loader):
            x, y = x.to(self.device), y.to(self.device)
            with torch.set_grad_enabled(False):
                pred = self.model(x)
                loss = self.loss_fn(pred, y)

            running_loss += loss.item() / len(self.val_loader)

        return running_loss

    def __test(self):
        preds = []
        self.model.eval()
        for i, x in enumerate(self.test_loader):
            x = x.to(self.device)
            with torch.no_grad():
                batch_pred = torch.sigmoid(self.model(x))
                preds.append(batch_pred)
        preds = torch.cat(preds, dim=0).cpu().numpy()
        return preds

    def __train_data_factory(self, data, train_batch_size, val_batch_size):
        x_train, y_train, x_val, y_val = data
        train_dataset = TrainDataset(x_train, y_train)
        valid_dataset = TrainDataset(x_val, y_val)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=val_batch_size,
                                                 shuffle=False)
        return train_loader, val_loader,

    def __test_data_factory(self, data, test_batch_size):
        x_test = data
        test_dataset = TestDataset(x_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                                  shuffle=False)
        return test_loader


class TrainDataset:
    def __init__(self, features, labels):
        self.features = features.astype(np.float64)
        self.labels = labels.astype(np.float64)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])

    def __len__(self):
        assert len(self.features) == len(self.labels)
        return len(self.features)


class TestDataset:
    def __init__(self, features):
        self.features = features

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx])

    def __len__(self):
        return len(self.features)


class ModelMlp(nn.Module):
    def __init__(self, num_features, n_targets=206, hidden_sizes=None, dropout_rates=None):
        super(ModelMlp, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout_rates = dropout_rates

        arr = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                # input layer
                arr.append((f'batch_norm{i + 1}', nn.BatchNorm1d(num_features)))
                arr.append((f'dense{i + 1}', nn.utils.weight_norm(nn.Linear(num_features, hidden_sizes[i]))))
                arr.append((f'activation{i + 1}', nn.modules.LeakyReLU()))

            else:
                # hidden layer
                arr.append((f'batch_norm{i + 1}', nn.BatchNorm1d(hidden_sizes[i - 1])))
                arr.append((f'dropout{i + 1}', nn.Dropout(self.get_dropout_rate(dropout_rates, i - 1))))
                arr.append((f'dense{i + 1}', nn.utils.weight_norm(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))))
                arr.append((f'activation{i + 1}', nn.modules.LeakyReLU()))
        # output layer
        i = len(hidden_sizes)
        arr.append((f'batch_norm{i + 1}', nn.BatchNorm1d(hidden_sizes[i - 1])))
        arr.append((f'dropout{i + 1}', nn.Dropout(self.get_dropout_rate(dropout_rates, i - 1))))
        arr.append((f'dense{i + 1}', nn.utils.weight_norm(nn.Linear(hidden_sizes[i - 1], n_targets))))

        self.sequential = nn.Sequential(OrderedDict(arr))

    def forward(self, x):
        x = self.sequential(x)
        return x

    def get_dropout_rate(self, dropout_rates, i):
        if isinstance(dropout_rates, list):
            if i >= len(dropout_rates):
                return dropout_rates[len(dropout_rates) - 1]
            else:
                return dropout_rates[i]
        else:
            return dropout_rates


class FineTuneScheduler:
    def __init__(self, epochs):
        self.epochs = epochs
        self.epochs_per_step = 0
        self.frozen_layers = []

    def copy_without_top(self, Model, device, model_dict, num_features, num_all_labels, num_labels, hidden_sizes,
                         dropout_rates):
        self.frozen_layers = []
        model_new = Model(num_features, num_all_labels, hidden_sizes, dropout_rates)
        model_new.load_state_dict(model_dict)

        model_depth = len(model_new.hidden_sizes) + 1
        # Freeze all weights
        for name, param in model_new.named_parameters():
            layer_index = name.split('.')[1][-1]

            if layer_index == model_depth:
                continue

            param.requires_grad = False

            # Save frozen layer names
            if layer_index not in self.frozen_layers:
                self.frozen_layers.append(layer_index)

        self.epochs_per_step = self.epochs // len(self.frozen_layers)

        # Replace the top layers with another ones
        model_new.sequential[-3] = nn.BatchNorm1d(model_new.hidden_sizes[-1])
        model_new.sequential[-2] = nn.Dropout(
            model_new.get_dropout_rate(model_new.dropout_rates, model_depth - 2))
        model_new.sequential[-1] = nn.utils.weight_norm(
            nn.Linear(model_new.hidden_sizes[-1], num_labels))
        model_new.to(device)
        return model_new

    def step(self, epoch, model):
        if len(self.frozen_layers) == 0:
            return

        if epoch % 30 == 0:
            last_frozen_index = self.frozen_layers[-1]

            # Unfreeze parameters of the last frozen layer
            for name, param in model.named_parameters():
                layer_index = name.split('.')[0][-1]

                if layer_index == last_frozen_index:
                    param.requires_grad = True

            del self.frozen_layers[-1]  # Remove the last layer as unfrozen


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


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
