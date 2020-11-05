import copy
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.modules.loss import _WeightedLoss

from sklearn.multiclass import OneVsRestClassifier


class PytorchModelHelper:
    def __init__(self, device=None):
        self.device = device

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.num_features = None

    def fit_with_non_scored(self, model_name, data_with_non_scored, data, n_epochs=None, scheduler=None, patience=None,
                            train_batch_size=None,
                            val_batch_size=None):
        # 预训练model3withnonscored模型
        best_loss_pretrained, best_model_pretrained = self.fit(model_name + "withnonscored", data_with_non_scored,
                                                               n_epochs,
                                                               scheduler,
                                                               patience,
                                                               train_batch_size,
                                                               val_batch_size)
        # 加载预训练模型参数--前两层
        pretrained_first_two_layers = {}
        for k, v in best_model_pretrained.items():
            if str(k).__contains__('1') or str(k).__contains__('2'):
                pretrained_first_two_layers[k] = v

        best_loss = {'train': np.inf, 'val': np.inf}
        best_model = None
        early_step = 0
        self.train_loader, self.val_loader, self.num_features = self.__train_data_factory(data=data,
                                                                                          train_batch_size=train_batch_size,
                                                                                          val_batch_size=val_batch_size)
        self.__init_model(model_name=model_name, n_epochs=n_epochs, scheduler=scheduler,
                          len_train_loader=len(self.train_loader))
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_first_two_layers)
        self.model.load_state_dict(model_dict)
        # 固定前一层参数
        # 用后两层参数训练
        # for name, param in self.model.named_parameters():
        #     print('param.name:', name)
        #     if str(name).__contains__('1'):
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        # 检查参数
        print("Check whether parameters fixed")
        for k, v in self.model.named_parameters():
            print(k, ";", v.requires_grad)

        for epoch in range(n_epochs):
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

    def fit_and_save(self, model_name, data, n_epochs=None, scheduler=None, patience=None, train_batch_size=None,
                     val_batch_size=None, model_save_path=None):
        best_loss = {'train': np.inf, 'val': np.inf}
        best_model = None
        early_step = 0
        self.train_loader, self.val_loader, self.num_features = self.__train_data_factory(data=data,
                                                                                          train_batch_size=train_batch_size,
                                                                                          val_batch_size=val_batch_size)
        self.__init_model(model_name=model_name, n_epochs=n_epochs, scheduler=scheduler,
                          len_train_loader=len(self.train_loader))

        for epoch in range(n_epochs):
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
        torch.save(best_model, model_save_path)

        return best_loss

    def predict(self, model_name, data, test_batch_size, model_path):
        self.test_loader, self.num_features = self.__test_data_factory(data=data, test_batch_size=test_batch_size)
        self.__init_model(model_name=model_name)
        self.__resume_model(model_path=model_path)

        test_pred = self.__test()

        return test_pred

    def __init_model(self, model_name, n_epochs=None, scheduler=None, len_train_loader=None):

        if model_name == "model1":
            self.model = Model(self.num_features).to(self.device)
        elif model_name == 'model2':
            self.model = Model2(self.num_features).to(self.device)
        elif model_name == 'model3':
            self.model = Model3(self.num_features).to(self.device)
        elif model_name == "model3withnonscored":
            self.model = Model3WithNonScored(self.num_features).to(self.device)
        elif model_name == 'model4':
            self.model = Model4(self.num_features).to(self.device)
        elif model_name == "model4withnonscored":
            self.model = Model4WithNonScored(self.num_features).to(self.device)
        else:
            raise NotImplementedError

        if scheduler == "pla":
            self.optimizer = optim.Adam(self.model.parameters(), lr=2e-2, weight_decay=1e-5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3,
                                                                  eps=1e-4,
                                                                  verbose=True)
        elif scheduler == 'cycle':
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
            self.scheduler = optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer, pct_start=0.1, div_factor=1e3,
                                                           max_lr=1e-2, epochs=n_epochs,
                                                           steps_per_epoch=len_train_loader)
        print(len_train_loader)
        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        self.loss_tr = SmoothBCEwLogits(smoothing=0.001)
        print("Model name : {} ;".format(self.model.__class__.__name__))
        print("Model net : {} ;".format(self.model))
        print("Scheduler : {} ;".format(self.scheduler.__class__.__name__))

    def __resume_model(self, model_path=None, model_state_dict=None):
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        elif model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        else:
            print("please provide a model ''!")

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
        num_features = x_train.shape[1]
        return train_loader, val_loader, num_features

    def __test_data_factory(self, data, test_batch_size):
        x_test = data
        test_dataset = TestDataset(x_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                                  shuffle=False)
        num_features = x_test.shape[1]
        return test_loader, num_features


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


class Model(nn.Module):
    def __init__(self, num_features):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, 2048))

        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.utils.weight_norm(nn.Linear(2048, 1024))

        self.batch_norm3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(0.5)
        self.dense3 = nn.utils.weight_norm(nn.Linear(1024, 206))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class Model2(nn.Module):
    def __init__(self, num_features):
        hidden_size = 512
        super(Model2, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, 206))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class Model3(nn.Module):
    def __init__(self, num_features):
        super(Model3, self).__init__()
        hidden_size = 1500
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, 206))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class Model3WithNonScored(nn.Module):
    def __init__(self, num_features):
        super(Model3WithNonScored, self).__init__()
        hidden_size = 1500
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, 608))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class Model4(nn.Module):
    def __init__(self, num_features):
        super(Model4, self).__init__()
        hidden_size = 1500
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, 2048))

        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.34)
        self.dense2 = nn.utils.weight_norm(nn.Linear(2048, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, 206))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = self.dense4(x)

        return x


class Model4WithNonScored(nn.Module):
    def __init__(self, num_features):
        super(Model4WithNonScored, self).__init__()
        hidden_size = 1500
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, 2048))

        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.34)
        self.dense2 = nn.utils.weight_norm(nn.Linear(2048, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, 608))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = self.dense4(x)

        return x


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
