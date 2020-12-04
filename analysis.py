import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
import pickle
from moa_pytorch_preprocess_helper_WX import read_data
from tqdm import tqdm
import matplotlib.pylab as plt
import seaborn as sns


def feature_importance(save_path):
    """
    求特征重要性，
    加入了高维特征 square, cubic，来求特征重要性
    :param save_path:
    :return:
    """

    x_train, y_train, y_train_with_non_scored, x_test = read_data(root_dir="../input")
    x_train = pd.get_dummies(x_train, columns=['cp_type', 'cp_dose', 'cp_time'])

    for col in x_train.columns:
        x_train[col + "_square"] = x_train[col].apply(lambda a: a ** 2)
        x_train[col + "_cubic"] = x_train[col].apply(lambda a: a ** 3)
    feature_importance = pd.DataFrame()
    feature_importance['features'] = x_train.columns

    x = x_train
    for col in tqdm(y_train_with_non_scored.columns):
        y = y_train_with_non_scored[col]
        clf = lgb.LGBMClassifier(colsample_bytree=0.3, bagging_seed=123, random_state=1234)
        clf.fit(x, y)
        if 'feature_importances' not in feature_importance.keys():
            feature_importance['feature_importances'] = clf.feature_importances_
        else:
            feature_importance['feature_importances'] += clf.feature_importances_
    # 将求得的特征重要性保存到文件中
    with open(save_path, 'wb') as f:  # open file with write-mode
        pickle.dump(feature_importance, f)  #


def analysis(df_x_train, df_y_train, variance_thresh, scale, encoding, n_gene_comp, n_cell_comp, base_seed):
    x_some = df_x_train[df_x_train.columns[2:32]].apply(lambda x: x ** 2)
    y_some = df_y_train[df_y_train.columns[:30]]
    f, ax = plt.subplots(figsize=(14, 10))

    some = pd.concat([x_some, y_some], axis=1)

    corr = some.corr()
    print(corr)
    # print(corr)
    sns.heatmap(corr, cmap='RdBu', linewidths=0.05, ax=ax)
    # sns.heatmap(df_feature_all[:, :7].corr())
    # 设置Axes的标题
    ax.set_title('Correlation between features')
    plt.show()
    plt.close()
    f.savefig('sns_style_origin.jpg', dpi=100, bbox_inches='tight')


if __name__ == '__main__':
    fn = '../input/pretrained-feature/pretrained_features.pkl'
    # feature_importance(fn)

    with open(fn, 'rb') as f:
        fi = pickle.load(f)
        sorted_fi = fi.sort_values(by='feature_importances', ascending=False)

    # sorted_fi = sorted_fi[sorted_fi['features']]
    # sorted_fi = sorted_fi[sorted_fi['features'].str.contains('_square')]
    num_features = sorted_fi.shape[0]
    print("num features : ", num_features)
    # 最重要的特征
    top_sorted_fi = sorted_fi.iloc[:int(num_features * 0.01)]
    bottome_sorted_fi = sorted_fi.iloc[-int(num_features * 0.02):]

    print("top_sorted_fi:")
    print(top_sorted_fi)
    print("bottome_sorted_fi:")
    print(bottome_sorted_fi)
