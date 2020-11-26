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
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from moa_pytorch_model_helper import PytorchModelHelper, ModelMlp
from moa_tabnet_model_helper import TabnetModelHelper
from moa_pytorch_preprocess_helper import PytorchPreprocessHelper
from moa_tabnet_preprocess_helper import TabnetPreprocessHelper
import warnings

warnings.filterwarnings('ignore')
param = {
    'root_dir': "../input",
    'model': None,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'output_root_dir': "output",
    'output': "",
    'model_save_name': "model_{}.pth",
    'base_seed': 2020,
    'n_folds': 7,
    'n_epochs': 150,
    'patience': 10,
    'train_batch_size': 1024,
    'val_batch_size': 1024,
    'test_batch_size': 1024,

    'out_data_dir': 'out_tabnet_data_dir',

    'read_directly': False,

    'is_train_data': False,
    'is_train_model': True
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
    'is_filtered_by_var2': False,
    'variance_thresh': 0.8,

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


def repeat_training_testing(ModelHelper, PreprocessHelper, preprocess_param):
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
    helper = ModelHelper(kfold, param, x_train, y_train, y_train_with_non_scored, x_test)
    if param['is_train_model']:
        helper.train_models()
        total_test_preds = helper.test_models()
    else:
        total_test_preds = helper.test_models()
    return total_test_preds


if __name__ == '__main__':
    # copy models to work directory
    input_model_dir = "../input/data-and-model"
    if not os.path.exists(param['output_root_dir']):
        os.makedirs(param['output_root_dir'])
    if os.path.exists(input_model_dir):
        shutil.rmtree(param['output_root_dir'])
        shutil.copytree(input_model_dir, param['output_root_dir'])

    model_config = [
        # ["py_model_40", PytorchModelHelper, ModelMlp,
        #  {"is_transfer": True, "hidden_sizes": [1500, 1250, 1000, 750], "dropout_rates": [0.5, 0.35, 0.3, 0.25],
        #   "base_seed": 42},
        #  PytorchPreprocessHelper, pytorch_preprocess_param],
        #
        ["py_model_41", PytorchModelHelper, ModelMlp,
         {"is_transfer": True, "hidden_sizes": [1500, 1250, 1000, 750], "dropout_rates": [0.5, 0.35, 0.3, 0.25],
          "base_seed": 202},
         PytorchPreprocessHelper, pytorch_preprocess_param],
        #
        # ["py_model_30", PytorchModelHelper, ModelMlp,
        #  {"is_transfer": True, "hidden_sizes": [1500, 1024, 750], "dropout_rates": [0.5, 0.35, 0.25],
        #   "base_seed": 2020},
        #  PytorchPreprocessHelper, pytorch_preprocess_param],
        #
        ["py_model_33", PytorchModelHelper, ModelMlp,
         {"is_transfer": True, "hidden_sizes": [1500, 1024, 750], "dropout_rates": [0.5, 0.35, 0.25],
          "base_seed": 1973},
         PytorchPreprocessHelper, pytorch_preprocess_param],

        # ["py_model_31", PytorchModelHelper, ModelMlp,
        #  {"is_transfer": True, "hidden_sizes": [1500, 1250, 1024], "dropout_rates": [0.5, 0.35, 0.3],
        #   "base_seed": 2020},
        #  PytorchPreprocessHelper, pytorch_preprocess_param],
        #
        # ["py_model_32", PytorchModelHelper, ModelMlp,
        #  {"is_transfer": True, "hidden_sizes": [1500, 1000, 500], "dropout_rates": [0.5, 0.35, 0.3],
        #   "base_seed": 2020},
        #  PytorchPreprocessHelper, pytorch_preprocess_param],
        #
        # ["py_model_20", PytorchModelHelper, ModelMlp,
        #  {"is_transfer": True, "hidden_sizes": [1500, 1500], "dropout_rates": [0.2619422201258426, 0.2619422201258426],
        #   "base_seed": 2021},
        #  PytorchPreprocessHelper, pytorch_preprocess_param],
        #
        ["py_model_22", PytorchModelHelper, ModelMlp,
         {"is_transfer": True, "hidden_sizes": [1500, 1500], "dropout_rates": [0.2619422201258426, 0.2619422201258426],
          "base_seed": 1949},
         PytorchPreprocessHelper, pytorch_preprocess_param],
        #
        # ["py_model_21", PytorchModelHelper, ModelMlp,
        #  {"is_transfer": True, "hidden_sizes": [1500, 800], "dropout_rates": [0.5, 0.35],
        #   "base_seed": 1995},
        #  PytorchPreprocessHelper, pytorch_preprocess_param],
        #
        # ["py_model_11", PytorchModelHelper, ModelMlp,
        #  {"is_transfer": False, "hidden_sizes": [2048], "dropout_rates": [0.2619422201258426], "base_seed": 12},
        #  PytorchPreprocessHelper, pytorch_preprocess_param],

        # ["tb_model_0", TabnetModelHelper, 'Tabnet', {"param_id": 0, "base_seed": 78}, PytorchPreprocessHelper,
        #  tabnet_preprocess_param1],

        # ["tb_model_1", TabnetModelHelper, 'Tabnet', {"param_id": 1, "base_seed": 76}, TabnetPreprocessHelper, None],
        #
        # ["tb_model_2", TabnetModelHelper, 'Tabnet', {"param_id": 2, "base_seed": 53}, TabnetPreprocessHelper, None],
        #
        # ["tb_model_3", TabnetModelHelper, 'Tabnet', {"param_id": 3, "base_seed": 3000}, TabnetPreprocessHelper, None],
    ]
    start_time = time.time()

    preds_after_repeat = []

    for i, [prefix_name, ModelHelper, Model, special_param, PreprocessHelper, preprocess_param] in enumerate(
            model_config):
        param['output'] = os.path.join(param['output_root_dir'], 'model_save/output_{}'.format(prefix_name))
        param['model'] = Model

        param.update(special_param)

        param['out_data_dir'] = os.path.join(param['output_root_dir'], "out_data_dir_" + PreprocessHelper.__name__)

        total_test_preds = repeat_training_testing(ModelHelper, PreprocessHelper, preprocess_param)
        # 将每次预测后的结果加到list中
        preds_after_repeat.append(total_test_preds)

    print("preds after repeating -- shape:", np.shape(preds_after_repeat))
    preds_after_repeat = np.mean(np.array(preds_after_repeat), axis=0)
    write_result(preds_after_repeat)

    duration_time = time.time() - start_time
    print("duration time : ", duration_time)
