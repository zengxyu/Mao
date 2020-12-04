import os
import pickle
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss


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


def write_result(preds, root_dir):
    origin_test_data = pd.read_csv(os.path.join(root_dir, 'lish-moa', 'test_features.csv'))
    submit = pd.read_csv(os.path.join(root_dir, 'lish-moa', 'sample_submission.csv'))
    columns = [col for col in submit.columns if col not in ['sig_id']]
    submit[columns] = preds
    # 将cp_type为ctl_vehicle的行设为0
    submit.loc[origin_test_data['cp_type'] == 'ctl_vehicle', columns] = 0
    submit.to_csv('submission.csv', index=False)

    print("Result has been written to submission.csv")


def write_val_result(preds, root_dir, output_dir=""):
    X_train = pd.read_csv(os.path.join(root_dir, 'lish-moa', 'train_features.csv'))
    submit = pd.read_csv(os.path.join(root_dir, 'lish-moa', 'train_targets_scored.csv'))
    X_train = X_train[X_train['cp_type'] != 'ctl_vehicle']

    columns = [col for col in submit.columns if col not in ['sig_id']]

    submit = submit.iloc[X_train.index]
    submit.reset_index(drop=True, inplace=True)
    submit[columns] = preds
    submit.to_csv(os.path.join(output_dir, 'submission_val.csv'), index=False)

    print("Val Result has been written to {}".format(os.path.join(output_dir, 'submission_val.csv')))
    overall_val_loss = calculate_overall_loss(root_dir, os.path.join(output_dir, 'submission_val.csv'))
    print("The overall val loss is {}".format(overall_val_loss))
    return overall_val_loss


def print_seperater(prefix_name, Model, special_param):
    print()
    print()
    print("==========================================================================================")
    model_name = Model if isinstance(Model, str) else Model.__name__
    print("prefix_name = {},  Model = {} ".format(prefix_name, model_name))
    print("special_param = {}".format(special_param))

    print("------------------------------------------------------------------------------------------")


def calculate_overall_loss(root_dir, submission):
    y_true = pd.read_csv(os.path.join(root_dir, 'lish-moa', 'train_targets_scored.csv'))
    X_train = pd.read_csv(os.path.join(root_dir, 'lish-moa', 'train_features.csv'))
    X_train = X_train[X_train['cp_type'] != 'ctl_vehicle']
    y_true = y_true.iloc[X_train.index]
    y_true.reset_index(drop=True, inplace=True)
    del y_true['sig_id']

    y_pred = pd.read_csv(submission)
    del y_pred['sig_id']
    y_true = y_true.values
    y_pred = y_pred.values

    score = 0
    for i in range(y_true.shape[1]):
        score_ = log_loss(y_true[:, i], y_pred[:, i])
        score += score_ / y_true.shape[1]
    return score


def load_val_loss():
    f = open("val_loss002.pkl", 'rb')
    model_val_loss_dict = pickle.load(f)
    print("model_val_loss_dict:", model_val_loss_dict)


def cal_overall_loss():
    overall_val_loss = calculate_overall_loss(root_dir="../input", submission='submission_val.csv')
    print("The overall val loss is {}".format(overall_val_loss))


if __name__ == '__main__':
    load_val_loss()
