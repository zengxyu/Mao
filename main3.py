import os
import pickle
import shutil
import sys
import random

# 试试non-scored数据
import time

sys.path.append('../input/iterative-stratification/iterative-stratification-master')
sys.path.append('../input/pytorch-tabnet')
sys.path.append('../input/moa-pytorch-script')

import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from moa_pytorch_model_helper import PytorchModelHelper, ModelMlp
from moa_tabnet_model_helper import TabnetModelHelper
from moa_pytorch_preprocess_helper_WX import PytorchPreprocessHelper
from moa_tabnet_preprocess_helper import TabnetPreprocessHelper
from util import seed_everything, write_result, write_val_result, print_seperater, calculate_overall_loss
import pandas as pd

import warnings

"""
有聚类， weight decay = 1e-6
"""
warnings.filterwarnings('ignore')
param = {
    'root_dir': "../input",
    'model': None,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'output_root_dir': "output3",
    'output': "",
    'model_save_name': "model_{}.pth",
    'base_seed': 2020,
    'n_folds': 10,
    'n_epochs': 150,
    'patience': 10,
    'train_batch_size': 1024,
    'val_batch_size': 1024,
    'test_batch_size': 1024,

    'out_data_dir': 'out_tabnet_data_dir',

    'read_directly': False,

    'is_train_data': True,
    'is_train_model': True,
    'compute_val_loss_only': False

}

pytorch_preprocess_param = {
    'is_drop_ctl_vehicle': True,

    'is_add_square_feature': True,

    'is_add_cluster_gene_cell': True,
    'n_clusters_g': 22,
    'n_clusters_c': 4,

    'is_add_cluster_pca': True,
    'n_comp_cluster_pca': 5,

    'is_gauss_rank': True,

    'is_pca': True,
    'is_svd': True,
    'n_gene_comp': 600,
    'n_cell_comp': 50,

    'is_filtered_by_var': True,
    'variance_thresh': 0.8,

    'is_encoding': True,
    'encoding': 'dummy',

}

tabnet_preprocess_param1 = {

    'is_drop_ctl_vehicle': True,

    'is_add_square_feature': False,

    'is_delete_feature': False,

    'is_gauss_rank': True,

    'is_pca': False,
    'is_svd': False,
    'n_gene_comp': 600,
    'n_cell_comp': 50,

    'is_filtered_by_var': False,
    'variance_thresh': 0.8,

    'is_encoding': True,
    'encoding': 'dummy',

}

model_config = [
    ["py_model_4_1500_1250_1000_750", PytorchModelHelper, ModelMlp,
     {"is_transfer": True, "hidden_sizes": [1500, 1250, 1000, 750], "dropout_rates": [0.5, 0.35, 0.3, 0.25],
      "base_seed": 42},
     PytorchPreprocessHelper, pytorch_preprocess_param, 1],

    ["py_model_3_1500_1024_750", PytorchModelHelper, ModelMlp,
     {"is_transfer": True, "hidden_sizes": [1500, 1024, 750], "dropout_rates": [0.5, 0.35, 0.25],
      "base_seed": 2020},
     PytorchPreprocessHelper, pytorch_preprocess_param, 1],

    ["py_model_3_1280_960_720", PytorchModelHelper, ModelMlp,
     {"is_transfer": True, "hidden_sizes": [1280, 960, 720], "dropout_rates": [0.4, 0.3, 0.18],
      "base_seed": 57},
     PytorchPreprocessHelper, pytorch_preprocess_param, 1],

    ["py_model_2_1500_1500", PytorchModelHelper, ModelMlp,
     {"is_transfer": True, "hidden_sizes": [1500, 1500], "dropout_rates": [0.2619422201258426, 0.2619422201258426],
      "base_seed": 2021},
     PytorchPreprocessHelper, pytorch_preprocess_param, 1],

    ["py_model_2_1280_1280", PytorchModelHelper, ModelMlp,
     {"is_transfer": True, "hidden_sizes": [1280, 1280], "dropout_rates": 0.21,
      "base_seed": 1995},
     PytorchPreprocessHelper, pytorch_preprocess_param, 1],

    ["py_model_2_1024_1024", PytorchModelHelper, ModelMlp,
     {"is_transfer": True, "hidden_sizes": [1024, 1024], "dropout_rates": 0.18,
      "base_seed": 51},
     PytorchPreprocessHelper, pytorch_preprocess_param, 1],

    ["py_model_2_720_720", PytorchModelHelper, ModelMlp,
     {"is_transfer": True, "hidden_sizes": [720, 720], "dropout_rates": 0.1,
      "base_seed": 49},
     PytorchPreprocessHelper, pytorch_preprocess_param, 1],

    # ["py_model_1_2048", PytorchModelHelper, ModelMlp,
    #  {"is_transfer": False, "hidden_sizes": [2048], "dropout_rates": [0.2619422201258426], "base_seed": 12},
    #  PytorchPreprocessHelper, pytorch_preprocess_param],

    # ["tb_model_0", TabnetModelHelper, 'Tabnet', {"param_id": 0, "base_seed": 78}, PytorchPreprocessHelper,
    #  tabnet_preprocess_param1],

    ["tb_model_1", TabnetModelHelper, 'Tabnet', {"param_id": 1, "base_seed": 76}, TabnetPreprocessHelper, None, 1],

    ["tb_model_2", TabnetModelHelper, 'Tabnet', {"param_id": 2, "base_seed": 53}, TabnetPreprocessHelper, None, 1],

    # ["tb_model_3", TabnetModelHelper, 'Tabnet', {"param_id": 3, "base_seed": 3000}, TabnetPreprocessHelper, None],
]


def train_test_once(ModelHelper, PreprocessHelper, preprocess_param):
    seed_everything(param['base_seed'])

    if not os.path.exists(param['output']):
        os.makedirs(param['output'])

    kfold = MultilabelStratifiedKFold(n_splits=param['n_folds'], random_state=param['base_seed'], shuffle=True)

    x_train, y_train, y_train_with_non_scored, x_test, _, _ = \
        PreprocessHelper(root_dir=param['root_dir'], out_data_dir=param['out_data_dir'],
                         is_train=param['is_train_data'], read_directly=param['read_directly']).process(
            preprocess_param=preprocess_param,
            base_seed=param['base_seed'])
    print("")
    helper = ModelHelper(kfold, param, x_train, y_train, y_train_with_non_scored, x_test, False)
    if param['is_train_model']:
        helper.train_models()
        total_train_preds = helper.test_models(is_predict_train_data=True)
        total_test_preds = helper.test_models()
    else:
        total_train_preds = helper.test_models(is_predict_train_data=True)
        total_test_preds = helper.test_models()
        # 模型预测所有train data的值都要写入到文件，要获得每个模型的val loss 和所有模型的val loss
    return total_train_preds, total_test_preds


def compuate_val_loss():
    weights = []
    preds_train_on_all_models = []
    for i, [prefix_name, ModelHelper, Model, special_param, PreprocessHelper, preprocess_param, weight] in enumerate(
            model_config):
        print_seperater(prefix_name, Model, special_param)
        param['output'] = os.path.join(param['output_root_dir'], 'model_save/output_{}'.format(prefix_name))

        weights.append(weight)

        submission_csv = os.path.join(param['output'], 'submission_val.csv')

        submission = pd.read_csv(submission_csv)
        del submission['sig_id']
        preds_train_on_all_models.append(np.array(submission.values))

        overall_val_loss = calculate_overall_loss(param['root_dir'], submission_csv)

        print("model :{}; loss:{}".format(prefix_name, overall_val_loss))

    preds_train_on_all_models = np.average(np.array(preds_train_on_all_models, dtype=float), axis=0, weights=weights)

    overall_val_loss = write_val_result(preds_train_on_all_models, root_dir=param['root_dir'])

    print("overall model ; loss:{}".format(overall_val_loss))

    # print("preds_test_on_all_models:{}".format(overall_val_loss))


def train_test_all():
    preds_test_on_all_models = []
    preds_train_on_all_models = []

    model_val_loss_dict = {}
    weights = []

    for i, [prefix_name, ModelHelper, Model, special_param, PreprocessHelper, preprocess_param, weight] in enumerate(
            model_config):
        print_seperater(prefix_name, Model, special_param)
        param['output'] = os.path.join(param['output_root_dir'], 'model_save/output_{}'.format(prefix_name))
        param['model'] = Model

        param.update(special_param)

        param['out_data_dir'] = os.path.join(param['output_root_dir'], "out_data_dir_" + PreprocessHelper.__name__)

        weights.append(weight)
        total_train_preds, total_test_preds = train_test_once(ModelHelper, PreprocessHelper, preprocess_param)

        # 把train数据的预测结果写到文件中
        overall_val_loss = write_val_result(total_train_preds, root_dir=param['root_dir'], output_dir=param['output'])
        model_val_loss_dict[prefix_name] = [overall_val_loss, weight]

        # 将每次预测后的结果加到list中
        preds_test_on_all_models.append(total_test_preds)
        preds_train_on_all_models.append(total_train_preds)

    print("preds after repeating -- shape:", np.shape(preds_test_on_all_models))
    # 求均值

    preds_test_on_all_models = np.average(np.array(preds_test_on_all_models), axis=0, weights=weights)
    preds_train_on_all_models = np.average(np.array(preds_train_on_all_models), axis=0, weights=weights)
    # 将test的预测结果写到文件
    write_result(preds_test_on_all_models, root_dir=param['root_dir'])
    print("End -- model_val_loss_dict:", model_val_loss_dict)
    # 将train的测试结果写到文件
    overall_val_loss = write_val_result(preds_train_on_all_models, root_dir=param['root_dir'])
    model_val_loss_dict["overall"] = overall_val_loss
    # 将所有的val loss写到文件
    with open("val_loss002.pkl", 'wb') as f:
        pickle.dump(model_val_loss_dict, f)


def average(preds_test_on_all_models, model_val_loss_dict):
    losses = np.array([val_loss for k, val_loss in model_val_loss_dict.items()])
    scores = 1. - losses

    sum_scores = sum(scores)

    weights = [x / sum_scores for x in scores]
    return np.average(np.array(preds_test_on_all_models), axis=0, weights=weights)


if __name__ == '__main__':
    # copy models to work directory
    input_model_dir = "../input/data-and-model"
    if not os.path.exists(param['output_root_dir']):
        os.makedirs(param['output_root_dir'])
    if os.path.exists(input_model_dir):
        shutil.rmtree(param['output_root_dir'])
        shutil.copytree(input_model_dir, param['output_root_dir'])

    start_time = time.time()
    if param['compute_val_loss_only']:
        compuate_val_loss()
    else:
        train_test_all()

    duration_time = time.time() - start_time
    print("duration time : ", duration_time)
