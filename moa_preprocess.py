import os
import sys

from sklearn.feature_selection import VarianceThreshold

sys.path.append('../input/rank-gauss')
sys.path.append('../input/')
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from gauss_rank.gauss_rank_scaler import GaussRankScaler


def read_data(root_dir):
    """
    读数据，删除sig_id
    :param root_dir: 数据根目录
    :return:
    """
    x_train = pd.read_csv(os.path.join(root_dir, 'lish-moa', 'train_features.csv'))
    y_train = pd.read_csv(os.path.join(root_dir, 'lish-moa', 'train_targets_scored.csv'))
    x_test = pd.read_csv(os.path.join(root_dir, 'lish-moa', 'test_features.csv'))
    y_train_non_scored = pd.read_csv(os.path.join(root_dir, 'lish-moa', 'train_targets_nonscored.csv'))

    # 删掉不要的列
    del x_train['sig_id']
    del y_train['sig_id']
    del x_test['sig_id']
    del y_train_non_scored['sig_id']
    y_train_with_non_scored = pd.concat([y_train, y_train_non_scored], axis=1)
    print("y_train shape:", y_train.shape)
    print("y_train_non_scored:", y_train_non_scored.shape)
    return x_train, y_train, y_train_with_non_scored, x_test


def drop_ctl_vehicle_samples_and_cols(df_x_train, df_y_train, df_y_train_with_non_scored, df_x_test):
    """
    删除 cp_type = ctl_vehicle的行， 针对 df_x_train，df_x_train
    删除 cp_type所在的列， 针对 df_x_train，df_x_test
    :param df_x_train:
    :param df_y_train:
    :param df_y_train_with_non_scored:
    :param df_x_test:
    :return:
    """
    # delete the row where cp_type!=ctl_vehicle
    df_x_train_copy = df_x_train.copy()
    df_x_train = df_x_train.loc[df_x_train_copy['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    df_y_train = df_y_train.loc[df_x_train_copy['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    df_y_train_with_non_scored = df_y_train_with_non_scored.loc[
        df_x_train_copy['cp_type'] != 'ctl_vehicle'].reset_index(
        drop=True)

    # delete the col cp_type
    del df_x_train['cp_type']
    del df_x_test['cp_type']

    return df_x_train, df_y_train, df_y_train_with_non_scored, df_x_test


def filter_feature_by_variance(df_features_all, cols_feature_category, cols_feature_numeric, variance_thresh):
    """
    滤掉方差太小的特征
    :param df_features_all:
    :param cols_feature_category:
    :param cols_feature_numeric:
    :param variance_thresh:
    :return:
    """
    var_thresh = VarianceThreshold(variance_thresh)  # <-- Update
    tmp = var_thresh.fit_transform(df_features_all[cols_feature_numeric])
    df_features_all = pd.concat([df_features_all[cols_feature_category], pd.DataFrame(tmp)], axis=1)

    cols_feature_numeric = [feat for feat in list(df_features_all.columns) if
                            feat not in cols_feature_category]
    return df_features_all, cols_feature_numeric


def decompose_pca(df_features_all, n_gene_comp, n_cell_comp, cols_feature_category, base_seed):
    """
    PCA提取主成分，并将提取的主成分加入到df_features_all
    :param df_features_all:
    :param n_gene_comp:
    :param n_cell_comp:
    :param cols_feature_category:
    :param base_seed:
    :return: df_features_all： 加入主成分后的特征
            cols_feature_numeric: 加入主成分后的数值列名
    """
    gene_cols = [col for col in df_features_all.columns if col.startswith('g-')]
    cell_cols = [col for col in df_features_all.columns if col.startswith('c-')]

    df_features_gene = pd.DataFrame(df_features_all[gene_cols])
    df_features_cell = pd.DataFrame(df_features_all[cell_cols])

    gene_pca = PCA(n_components=n_gene_comp, random_state=base_seed)
    cell_pca = PCA(n_components=n_cell_comp, random_state=base_seed)
    feature_gene_reduced = gene_pca.fit_transform(df_features_gene[gene_cols])
    feature_cell_reduced = cell_pca.fit_transform(df_features_cell[cell_cols])

    df_feature_gene_reduced = pd.DataFrame(feature_gene_reduced,
                                           columns=['pca_g-{}'.format(col) for col in range(n_gene_comp)])

    df_feature_cell_reduced = pd.DataFrame(feature_cell_reduced,
                                           columns=['pca_c-{}'.format(col) for col in range(n_cell_comp)])

    df_features_all = pd.concat([df_features_all, df_feature_gene_reduced, df_feature_cell_reduced],
                                axis=1)
    cols_feature_numeric = [feat for feat in list(df_features_all.columns) if
                            feat not in cols_feature_category]
    return df_features_all, cols_feature_numeric


def gauss_rank(df_features_all):
    """
    用gauss rank将数据规范化
    :param df_features_all:
    :return:
    """
    gene_cols = [col for col in df_features_all.columns if col.startswith('g-')]
    cell_cols = [col for col in df_features_all.columns if col.startswith('c-')]
    # RankGauss
    for col in (gene_cols + cell_cols):
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        vec_len = len(df_features_all[col].values)
        raw_vec = df_features_all[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        df_features_all[col] = transformer.transform(raw_vec)
    return df_features_all


def normalize_data(df_features_all, cols_feature_numeric, scale):
    def scale_minmax(col):
        return (col - col.min()) / (col.max() - col.min())

    def scale_norm(col):
        return (col - col.mean()) / col.std()

    if scale == 'boxcox':
        # 通过 BoxCox 正态化
        print('boxcox')
        df_features_all[cols_feature_numeric] = df_features_all[cols_feature_numeric].apply(scale_minmax, axis=0)
        trans = []
        for feat in cols_feature_numeric:
            trans_var, lambda_var = stats.boxcox(df_features_all[feat].dropna() + 1)
            trans.append(scale_minmax(trans_var))
        df_features_all[cols_feature_numeric] = np.asarray(trans).T

    elif scale == 'norm':
        # 通过标准化正态化
        print('norm')
        df_features_all[cols_feature_numeric] = df_features_all[cols_feature_numeric].apply(scale_norm, axis=0)

    elif scale == 'minmax':
        # 归一化
        print('minmax')
        df_features_all[cols_feature_numeric] = df_features_all[cols_feature_numeric].apply(scale_minmax, axis=0)

    elif scale == 'rankgauss':
        # RankGauss
        print('rankgauss')
        scaler = GaussRankScaler()
        df_features_all[cols_feature_numeric] = scaler.fit_transform(df_features_all[cols_feature_numeric])
    return df_features_all


def encode_feature(df_features_all, cols_features, encoding):
    """
    编码类别型特征
    :param df_features_all:
    :param cols_features:
    :param encoding: 编码方式['lb','dummy']
    :return:
    """
    # Encoding
    if encoding == 'lb':
        print('Label Encoding')
        for feat in cols_features:
            df_features_all[feat] = LabelEncoder().fit_transform(df_features_all[feat])
    elif encoding == 'dummy':
        print('One-hot')
        df_features_all = pd.get_dummies(df_features_all, columns=cols_features)
    else:
        print('Label Encoding')
        for feat in cols_features:
            df_features_all['lb' + feat] = LabelEncoder().fit_transform(df_features_all[feat])
        df_features_all = pd.get_dummies(df_features_all, columns=cols_features)
    return df_features_all


def add_square_features(df_features_all):
    """
    添加高维度特征，（平方特征）
    :param df_features_all:
    :return:
    """
    # 这四个特征是平方后，并使用lightgbm训练后，显示最重要的四个平方特征
    features_cols = ['c-98', 'g-300', 'g-7', 'g-100']
    for col in features_cols:
        df_features_all[col + "-square"] = df_features_all[col].apply(lambda a: a ** 2)
    return df_features_all


def generate_new_features(df_features_all):
    """
    生成统计特征
    :param df_features_all:
    :return:
    """
    GENES = [col for col in df_features_all.columns if col.startswith('g-')]
    CELLS = [col for col in df_features_all.columns if col.startswith('c-')]

    df_features_all['g_sum'] = df_features_all[GENES].sum(axis=1)
    df_features_all['g_mean'] = df_features_all[GENES].mean(axis=1)
    df_features_all['g_std'] = df_features_all[GENES].std(axis=1)
    df_features_all['g_kurt'] = df_features_all[GENES].kurtosis(axis=1)
    df_features_all['g_skew'] = df_features_all[GENES].skew(axis=1)
    df_features_all['c_sum'] = df_features_all[CELLS].sum(axis=1)
    df_features_all['c_mean'] = df_features_all[CELLS].mean(axis=1)
    df_features_all['c_std'] = df_features_all[CELLS].std(axis=1)
    df_features_all['c_kurt'] = df_features_all[CELLS].kurtosis(axis=1)
    df_features_all['c_skew'] = df_features_all[CELLS].skew(axis=1)
    df_features_all['gc_sum'] = df_features_all[GENES + CELLS].sum(axis=1)
    df_features_all['gc_mean'] = df_features_all[GENES + CELLS].mean(axis=1)
    df_features_all['gc_std'] = df_features_all[GENES + CELLS].std(axis=1)
    df_features_all['gc_kurt'] = df_features_all[GENES + CELLS].kurtosis(axis=1)
    df_features_all['gc_skew'] = df_features_all[GENES + CELLS].skew(axis=1)
    return df_features_all


class MoaPreprocess:
    def __init__(self, root_dir):
        self.df_x_train, self.df_y_train, self.df_y_train_with_non_scored, self.df_x_test = read_data(root_dir)
        self.df_x_train, self.df_y_train, self.df_y_train_with_non_scored, self.df_x_test = drop_ctl_vehicle_samples_and_cols(
            self.df_x_train, self.df_y_train, self.df_y_train_with_non_scored, self.df_x_test)
        # 标签列
        self.cols_label = [col for col in self.df_y_train.columns]
        # 特征列
        self.cols_feature = [col for col in self.df_x_train.columns]
        # 类别型特征
        self.cols_feature_category = ['cp_time', 'cp_dose']
        # 数值型特征
        self.cols_feature_numeric = [feat for feat in self.cols_feature if
                                     feat not in self.cols_feature_category]

        # 标签列的数量
        self.num_cols_label = len(self.cols_label)
        # 特征列的数量
        self.num_features_label = len(self.cols_feature)
        # 训练样本的数量
        self.num_train_samples = self.df_x_train.shape[0]
        # 测试样本的数量
        self.num_test_samples = self.df_x_test.shape[0]

        self.x_train = None
        self.y_train = None
        self.y_train_with_non_scored = None
        self.x_test = None

    def process_for_pytorch2(self, variance_thresh, scale, encoding, n_gene_comp, n_cell_comp, base_seed):
        # 处理
        df_feature_all = pd.concat([self.df_x_train, self.df_x_test], ignore_index=True)
        # 添加高维特征(平方特征)
        df_feature_all = add_square_features(df_feature_all)
        print("after adding square features -- df shape: ", df_feature_all.shape)
        # gauss rank将特征规范化
        df_feature_all = gauss_rank(df_feature_all)
        # 提取主成分，并将主成分加入特征
        df_feature_all, self.cols_feature_numeric = decompose_pca(df_feature_all, n_gene_comp, n_cell_comp,
                                                                  self.cols_feature_category, base_seed)
        # df_feature_all = generate_new_features(df_feature_all)
        # 滤掉方差较小的特征
        df_feature_all, self.cols_feature_numeric = filter_feature_by_variance(df_feature_all,
                                                                               self.cols_feature_category,
                                                                               self.cols_feature_numeric,
                                                                               variance_thresh)
        # 编码类别型特征
        df_feature_all = encode_feature(df_features_all=df_feature_all, cols_features=['cp_time', 'cp_dose'],
                                        encoding=encoding)

        x_train = df_feature_all[:self.num_train_samples]
        x_train.reset_index(drop=True, inplace=True)
        self.x_train = x_train.values

        x_test = df_feature_all[-self.num_test_samples:]
        x_test.reset_index(drop=True, inplace=True)
        self.x_test = x_test.values

        self.y_train = self.df_y_train.values
        self.y_train_with_non_scored = self.df_y_train_with_non_scored.values

        return self.x_train, self.y_train, self.y_train_with_non_scored, self.x_test, self.cols_label, self.num_cols_label

    def process_for_pytorch(self, variance_thresh, scale, encoding, n_gene_comp, n_cell_comp, base_seed):
        # 处理
        df_feature_all = pd.concat([self.df_x_train, self.df_x_test], ignore_index=True)

        df_feature_all = gauss_rank(df_feature_all)
        # df_feature_all = normalize_data(df_feature_all, self.cols_feature_numeric, scale)

        df_feature_all, self.cols_feature_numeric = decompose_pca(df_feature_all, n_gene_comp, n_cell_comp,
                                                                  self.cols_feature_category, base_seed)
        # df_feature_all = generate_new_features(df_feature_all)
        # with new features: 1539, without new features:1524
        df_feature_all, self.cols_feature_numeric = filter_feature_by_variance(df_feature_all,
                                                                               self.cols_feature_category,
                                                                               self.cols_feature_numeric,
                                                                               variance_thresh)
        df_feature_all = encode_feature(df_features_all=df_feature_all, cols_features=['cp_time', 'cp_dose'],
                                        encoding=encoding)

        # with new features : 1045, without new features:1045
        x_train = df_feature_all[:self.num_train_samples]
        x_train.reset_index(drop=True, inplace=True)
        self.x_train = x_train.values

        x_test = df_feature_all[-self.num_test_samples:]
        x_test.reset_index(drop=True, inplace=True)
        self.x_test = x_test.values

        self.y_train = self.df_y_train.values

        return self.x_train, self.y_train, self.x_test, self.cols_label, self.num_cols_label

    def process_for_tabnet(self, variance_thresh, scale, encoding, n_gene_comp, n_cell_comp, base_seed):
        # 处理
        df_feature_all = pd.concat([self.df_x_train, self.df_x_test], ignore_index=True)

        df_feature_all = gauss_rank(df_feature_all)

        df_feature_all, self.cols_feature_numeric = decompose_pca(df_feature_all, n_gene_comp, n_cell_comp,
                                                                  self.cols_feature_category, base_seed)
        df_feature_all, self.cols_feature_numeric = filter_feature_by_variance(df_feature_all,
                                                                               self.cols_feature_category,
                                                                               self.cols_feature_numeric,
                                                                               variance_thresh)
        df_feature_all = encode_feature(df_features_all=df_feature_all, cols_features=['cp_time', 'cp_dose'],
                                        encoding=encoding)

        x_train = df_feature_all[:self.num_train_samples]
        x_train.reset_index(drop=True, inplace=True)
        self.x_train = x_train.values

        x_test = df_feature_all[-self.num_test_samples:]
        x_test.reset_index(drop=True, inplace=True)
        self.x_test = x_test.values

        self.y_train = self.df_y_train.values

        return self.x_train, self.y_train, self.y_train_with_non_scored, self.x_test, self.cols_label, self.num_cols_label

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.cols_label, self.num_cols_label
