import os
import shutil
import sys
import random

# 试试non-scored数据
import time

sys.path.append('../input/iterative-stratification/iterative-stratification-master')
sys.path.append('../input/pytorch-tabnet')

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from moa_pytorch_model_helper import PytorchModelHelper
from moa_tabnet_model_helper import TabnetModelHelper
from moa_preprocess import MoaPreprocess
import warnings

warnings.filterwarnings('ignore')

param = {
    'model_name': "model3",
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'output': "",
    'model_save_name': "model_{}.pth",
    'root_dir': "../input",
    'base_seed': 2020,
    'n_folds': 10,
    'n_epochs': 200,
    'scheduler': 'pla',  # choose scheduler from ['pla','cycle']
    'patience': 10,
    'train_batch_size': 1024,
    'val_batch_size': 1024,
    'test_batch_size': 1024,
}

pytorch_preprocess_param = {
    'is_drop_ctl_vehicle': True,

    'is_add_square_feature': True,

    'is_delete_feature': False,

    'is_gauss_rank': True,

    'is_pca': True,
    'is_svd': True,
    'n_gene_comp': 600,
    'n_cell_comp': 50,

    'is_filtered_by_var': True,
    'is_filtered_by_var2': False,
    'variance_thresh': 0.8,

    'is_encoding': True,
    'encoding': 'dummy',

    'is_add_statistic_feature': False,
}

tabnet_preprocess_param = {
    'is_drop_ctl_vehicle': True,

    'is_add_square_feature': False,

    'is_delete_feature': False,

    'is_gauss_rank': True,

    'is_pca': True,
    'is_svd': False,
    'n_gene_comp': 80,
    'n_cell_comp': 10,

    'is_filtered_by_var': True,
    'is_filtered_by_var2': False,
    'variance_thresh': 0.7,

    'is_encoding': True,
    'encoding': 'dummy',

    'is_add_statistic_feature': False
}


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_models(kfold, xtrain, ytrain, ytrain_with_non_scored=None):
    for n, (tr, te) in enumerate(kfold.split(ytrain, ytrain)):
        print(f'Train fold {n + 1}')
        x_train, x_val = xtrain[tr], xtrain[te]
        y_train, y_val = ytrain[tr], ytrain[te]

        model_save_path = os.path.join(param['output'], param['model_save_name'].format(n + 1))
        if str(param['model_name']).startswith('model'):
            best_loss = PytorchModelHelper(device=param['device']).fit_and_save(model_name=param['model_name'],
                                                                                data=[x_train, y_train, x_val, y_val],
                                                                                n_epochs=param['n_epochs'],
                                                                                scheduler=param['scheduler'],
                                                                                patience=param['patience'],
                                                                                train_batch_size=param[
                                                                                    'train_batch_size'],
                                                                                val_batch_size=param['val_batch_size'],
                                                                                model_save_path=model_save_path
                                                                                )
            print("Fold : {} ; best loss : {}".format(n + 1, best_loss))
        elif str(param['model_name']).startswith("tabnet"):
            TabnetModelHelper().fit_and_save(data=[x_train, y_train, x_val, y_val],
                                             n_epochs=param['n_epochs'],
                                             patience=param['patience'],
                                             train_batch_size=param['train_batch_size'],
                                             model_save_path=model_save_path)
        else:
            raise NotImplementedError


def test_models(xtest, num_labels):
    num_samples = xtest.shape[0]
    total_test_preds = np.zeros((num_samples, num_labels, param['n_folds']))

    for n in range(param['n_folds']):
        print(f'Test fold {n + 1}')
        if str(param['model_name']).startswith('model'):
            model_save_path = os.path.join(param['output'], param['model_save_name'].format(n + 1))
            test_pred_per_fold = PytorchModelHelper(device=param['device']).predict(model_name=param['model_name'],
                                                                                    data=xtest,
                                                                                    test_batch_size=param[
                                                                                        'test_batch_size'],
                                                                                    model_path=model_save_path)

        elif str(param['model_name']).startswith("tabnet"):
            model_save_path = os.path.join(param['output'], (param['model_save_name'] + '.zip').format(n + 1))
            test_pred_per_fold = TabnetModelHelper().predict(xtest, model_save_path)

        else:
            raise NotImplementedError

        total_test_preds[:, :, n] = test_pred_per_fold

    total_test_preds = np.mean(total_test_preds, axis=2)
    return total_test_preds


def write_result(preds):
    origin_test_data = pd.read_csv(os.path.join(param['root_dir'], 'lish-moa', 'test_features.csv'))
    submit = pd.read_csv(os.path.join(param['root_dir'], 'lish-moa', 'sample_submission.csv'))
    columns = [col for col in submit.columns if col not in ['sig_id']]
    submit[columns] = preds
    # 将cp_type为ctl_vehicle的行设为0
    submit.loc[origin_test_data['cp_type'] == 'ctl_vehicle', columns] = 0
    submit.to_csv('submission.csv', index=False)

    print("Result has been written to submission.csv")


def post_process(preds, ratio0):
    processed_preds = preds.copy()
    temp = processed_preds.flatten()
    temp.sort()
    print("length:", len(temp))
    value0 = temp[int(len(temp) * ratio0)]
    print("value0:", value0)
    preds[preds < value0] = 0
    return preds


def repeat_training_testing():
    seed_everything(param['base_seed'])

    if not os.path.exists(param['output']):
        os.makedirs(param['output'])

    kfold = MultilabelStratifiedKFold(n_splits=param['n_folds'], random_state=param['base_seed'], shuffle=True)
    preprocess_param = pytorch_preprocess_param if param['model_name'].startswith("model") else tabnet_preprocess_param
    # 处理数据
    x_train, y_train, y_train_with_non_scored, x_test, cols_label, num_cols_label = \
        MoaPreprocess(root_dir=param['root_dir']).process(preprocess_param=preprocess_param,
                                                          base_seed=param['base_seed'])
    train_models(kfold=kfold, xtrain=x_train, ytrain=y_train)
    total_test_preds = test_models(xtest=x_test, num_labels=num_cols_label)
    return total_test_preds


if __name__ == '__main__':
    base_seeds = [2020, 42, 1995, 57, 0, 12, 76, 53, 2021, 2016, 2020, 42, 1995, 57, 0, 12, 76, 53, 2021, 2016]
    # model_names = ["model1", "model2", "model3", "tabnet"]
    model_names = ["model2", "model3", "tabnet"]

    start_time = time.time()

    repeat_time = 1
    preds_after_repeat = []

    for i, model_name in enumerate(model_names):
        param['base_seed'] = base_seeds[i]
        param['output'] = 'output_{}_{}'.format(i, model_name)
        print("model name = {}".format(model_name))
        param['model_name'] = model_name
        total_test_preds = repeat_training_testing()
        # 将每次预测后的结果加到list中
        preds_after_repeat.append(total_test_preds)

    print("preds after repeating -- shape:", np.shape(preds_after_repeat))
    preds_after_repeat = np.mean(np.array(preds_after_repeat), axis=0)
    write_result(preds_after_repeat)

    duration_time = time.time() - start_time
    print("duration time : ", duration_time)
