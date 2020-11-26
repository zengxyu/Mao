import os
import pickle
import sys

from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm

sys.path.append('../input/rank-gauss')
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from gauss_rank_scaler import GaussRankScaler


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
    # x_test = x_test[:int(0.1 * len(x_test))]
    y_train_with_non_scored = pd.concat([y_train, y_train_non_scored], axis=1)
    print("y_train shape:", y_train.shape)
    print("y_train_with_non_scored:", y_train_with_non_scored.shape)
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


def filter_feature_by_variance(df_features_all, cols_feature_category, variance_thresh, is_train,
                               out_tabnet_data_dir):
    """
    滤掉方差太小的特征
    :param df_features_all:
    :param cols_feature_category:
    :param cols_feature_numeric:
    :param variance_thresh:
    :return:
    """
    cols_numeric = [feat for feat in list(df_features_all.columns) if
                    feat not in cols_feature_category]

    if is_train:
        var_thresh_selector = VarianceThreshold(variance_thresh).fit(df_features_all[cols_numeric])
        pickle.dump(var_thresh_selector, open(os.path.join(out_tabnet_data_dir, 'var_thresh_selector.pkl'), 'wb'))
    else:
        var_thresh_selector = pickle.load(open(os.path.join(out_tabnet_data_dir, 'var_thresh_selector.pkl'), 'rb'))
        print("var_thresh_selector load from file")

    tmp = df_features_all[df_features_all.columns[var_thresh_selector.get_support(indices=True)]]

    df_features_all = pd.concat([df_features_all[cols_feature_category], pd.DataFrame(tmp)], axis=1)

    cols_feature_numeric = [feat for feat in list(df_features_all.columns) if
                            feat not in cols_feature_category]
    return df_features_all, cols_feature_numeric


def decompose_pca(df_features_all, n_gene_comp, n_cell_comp, cols_feature_category, base_seed, is_train,
                  out_tabnet_data_dir):
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
    if is_train:
        gene_pca = PCA(n_components=n_gene_comp, random_state=base_seed).fit(df_features_gene[gene_cols])
        cell_pca = PCA(n_components=n_cell_comp, random_state=base_seed).fit(df_features_cell[cell_cols])
        pickle.dump(gene_pca, open(os.path.join(out_tabnet_data_dir, 'pca_g.pkl'), 'wb'))
        pickle.dump(cell_pca, open(os.path.join(out_tabnet_data_dir, 'pca_c.pkl'), 'wb'))
    else:
        print("gene_pca,cell_pca load from file")
        gene_pca = pickle.load(open(os.path.join(out_tabnet_data_dir, 'pca_g.pkl'), 'rb'))
        cell_pca = pickle.load(open(os.path.join(out_tabnet_data_dir, 'pca_c.pkl'), 'rb'))

    feature_gene_reduced = gene_pca.transform(df_features_gene[gene_cols])
    feature_cell_reduced = cell_pca.transform(df_features_cell[cell_cols])

    df_feature_gene_reduced = pd.DataFrame(feature_gene_reduced,
                                           columns=['pca_g-{}'.format(col) for col in range(n_gene_comp)])

    df_feature_cell_reduced = pd.DataFrame(feature_cell_reduced,
                                           columns=['pca_c-{}'.format(col) for col in range(n_cell_comp)])

    df_features_all = pd.concat([df_features_all, df_feature_gene_reduced, df_feature_cell_reduced],
                                axis=1)
    cols_feature_numeric = [feat for feat in list(df_features_all.columns) if
                            feat not in cols_feature_category]
    return df_features_all, cols_feature_numeric


def decompose_svd(df_features_all, n_gene_comp, n_cell_comp, cols_feature_category, base_seed, is_train,
                  out_tabnet_data_dir):
    """
    SVD分解
    :param df_features_all:
    :param n_gene_comp:
    :param n_cell_comp:
    :param base_seed:
    :return:
    """
    gene_cols = [col for col in df_features_all.columns if col.startswith('g-')]
    cell_cols = [col for col in df_features_all.columns if col.startswith('c-')]

    if is_train:
        svd_g = TruncatedSVD(n_components=n_gene_comp, random_state=base_seed).fit(df_features_all[gene_cols])
        svd_c = TruncatedSVD(n_components=n_cell_comp, random_state=base_seed).fit(df_features_all[cell_cols])
        pickle.dump(svd_g, open(os.path.join(out_tabnet_data_dir, 'svd_g.pkl'), 'wb'))
        pickle.dump(svd_c, open(os.path.join(out_tabnet_data_dir, 'svd_c.pkl'), 'wb'))
    else:
        svd_g = pickle.load(open(os.path.join(out_tabnet_data_dir, 'svd_g.pkl'), 'rb'))
        svd_c = pickle.load(open(os.path.join(out_tabnet_data_dir, 'svd_c.pkl'), 'rb'))
        print("svd_g,svd_g load from file")

    svd_genes = pd.DataFrame(svd_g.transform(df_features_all[gene_cols]),
                             columns=[f'svd_g-{i}' for i in range(n_gene_comp)])
    svd_cells = pd.DataFrame(svd_c.transform(df_features_all[cell_cols]),
                             columns=[f'svd_c-{i}' for i in range(n_cell_comp)])
    df_features_all = pd.concat([df_features_all, svd_genes, svd_cells], axis=1)
    cols_feature_numeric = [feat for feat in list(df_features_all.columns) if
                            feat not in cols_feature_category]
    return df_features_all, cols_feature_numeric


def gauss_rank(df_features_all, cols_feature_category):
    """
    用gauss rank将数据规范化
    :param df_features_all:
    :return:
    """
    # gene_cols = [col for col in df_features_all.columns if col.startswith('g-')]
    # cell_cols = [col for col in df_features_all.columns if col.startswith('c-')]
    cols_numeric = [feat for feat in list(df_features_all.columns) if
                    feat not in cols_feature_category]
    # RankGauss
    for col in (cols_numeric):
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        vec_len = len(df_features_all[col].values)
        raw_vec = df_features_all[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        df_features_all[col] = transformer.transform(raw_vec)

    return df_features_all


def normalize_data(df_features_all, cols_feature_category, scale='rankgauss'):
    cols_numeric = [feat for feat in list(df_features_all.columns) if
                    feat not in cols_feature_category]

    def scale_minmax(col):
        return (col - col.min()) / (col.max() - col.min())

    def scale_norm(col):
        return (col - col.mean()) / col.std()

    if scale == 'boxcox':
        # 通过 BoxCox 正态化
        print('boxcox')
        df_features_all[cols_numeric] = df_features_all[cols_numeric].apply(scale_minmax, axis=0)
        trans = []
        for feat in cols_numeric:
            trans_var, lambda_var = stats.boxcox(df_features_all[feat].dropna() + 1)
            trans.append(scale_minmax(trans_var))
        df_features_all[cols_numeric] = np.asarray(trans).T

    elif scale == 'norm':
        # 通过标准化正态化
        print('norm')
        df_features_all[cols_numeric] = df_features_all[cols_numeric].apply(scale_norm, axis=0)

    elif scale == 'minmax':
        # 归一化
        print('minmax')
        df_features_all[cols_numeric] = df_features_all[cols_numeric].apply(scale_minmax, axis=0)

    elif scale == 'rankgauss':
        # RankGauss
        print('rankgauss')
        scaler = GaussRankScaler()
        df_features_all[cols_numeric] = scaler.fit_transform(df_features_all[cols_numeric])
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
    """
        添加高维度特征，（平方特征）
        :param df_features_all:
        :return:
        """
    # 这四个特征是平方后，并使用lightgbm训练后，显示最重要的四个平方特征
    gene_cols = [col for col in df_features_all.columns if col.startswith('g-')]
    cell_cols = [col for col in df_features_all.columns if col.startswith('c-')]

    print("generate new feature")
    # 特征生成
    for stats in ['sum', 'mean', 'std', 'kurt', 'skew']:
        df_features_all['g_' + stats] = getattr(df_features_all[gene_cols], stats)(axis=1)
        df_features_all['c_' + stats] = getattr(df_features_all[cell_cols], stats)(axis=1)
        df_features_all['gc_' + stats] = getattr(df_features_all[gene_cols + cell_cols], stats)(axis=1)

    df_features_all['c52_c42'] = df_features_all['c-52'] * df_features_all['c-42']
    df_features_all['c13_c73'] = df_features_all['c-13'] * df_features_all['c-73']
    df_features_all['c26_c13'] = df_features_all['c-23'] * df_features_all['c-13']
    df_features_all['c33_c6'] = df_features_all['c-33'] * df_features_all['c-6']
    df_features_all['c11_c55'] = df_features_all['c-11'] * df_features_all['c-55']
    df_features_all['c38_c63'] = df_features_all['c-38'] * df_features_all['c-63']
    df_features_all['c38_c94'] = df_features_all['c-38'] * df_features_all['c-94']
    df_features_all['c13_c94'] = df_features_all['c-13'] * df_features_all['c-94']
    df_features_all['c4_c52'] = df_features_all['c-4'] * df_features_all['c-52']
    df_features_all['c4_c42'] = df_features_all['c-4'] * df_features_all['c-42']
    df_features_all['c13_c38'] = df_features_all['c-13'] * df_features_all['c-38']
    df_features_all['c55_c2'] = df_features_all['c-55'] * df_features_all['c-2']
    df_features_all['c55_c4'] = df_features_all['c-55'] * df_features_all['c-4']
    df_features_all['c4_c13'] = df_features_all['c-4'] * df_features_all['c-13']
    df_features_all['c82_c42'] = df_features_all['c-82'] * df_features_all['c-42']
    df_features_all['c66_c42'] = df_features_all['c-66'] * df_features_all['c-42']
    df_features_all['c6_c38'] = df_features_all['c-6'] * df_features_all['c-38']
    df_features_all['c2_c13'] = df_features_all['c-2'] * df_features_all['c-13']
    df_features_all['c62_c42'] = df_features_all['c-62'] * df_features_all['c-42']
    df_features_all['c90_c55'] = df_features_all['c-90'] * df_features_all['c-55']

    features_cols = ['g-7', 'g-91', 'g-100', 'g-130', 'g-175', 'g-300', 'g-608', 'c-98']
    for col in features_cols:
        df_features_all[col + "_square"] = df_features_all[col].apply(lambda a: a ** 2)

    gsquarecols = ['g-574', 'g-211', 'g-216', 'g-0', 'g-255', 'g-577', 'g-153', 'g-389', 'g-60', 'g-370',
                   'g-248', 'g-167', 'g-203', 'g-177', 'g-301', 'g-332', 'g-517', 'g-6', 'g-744', 'g-224',
                   'g-162', 'g-3', 'g-736', 'g-486', 'g-283', 'g-22', 'g-359', 'g-361', 'g-440', 'g-335',
                   'g-106', 'g-307', 'g-745', 'g-146', 'g-416', 'g-298', 'g-666', 'g-91', 'g-17', 'g-549',
                   'g-145', 'g-157', 'g-768', 'g-568', 'g-396']
    for feature in gsquarecols:
        df_features_all[f'{feature}_squared'] = df_features_all[feature] ** 2

    return df_features_all


def delete_unimportant_features(df_features_all):
    """
    删除不重要的特征
    :param df_features_all:
    :return:
    """
    features_cols = ['g-502', 'g-727', 'c-2', 'c-71']
    for col in features_cols:
        del df_features_all[col]
    return df_features_all


def generate_new_features(df_features_all):
    """
    生成统计特征
    :param df_features_all:
    :return:
    """
    GENES = [col for col in df_features_all.columns if col.startswith("g-")]
    CELLS = [col for col in df_features_all.columns if col.startswith("c-")]

    for stats in tqdm.tqdm(["sum", "mean", "std", "kurt", "skew"]):
        df_features_all["g_" + stats] = getattr(df_features_all[GENES], stats)(axis=1)
        df_features_all["c_" + stats] = getattr(df_features_all[CELLS], stats)(axis=1)
        df_features_all["gc_" + stats] = getattr(df_features_all[GENES + CELLS], stats)(axis=1)
    return df_features_all


class PytorchPreprocessHelper:
    def __init__(self, root_dir, out_data_dir, is_train, read_directly):
        self.out_pytorch_data_dir = out_data_dir
        if not os.path.exists(self.out_pytorch_data_dir):
            os.makedirs(self.out_pytorch_data_dir)

        if read_directly:
            print("read_directly")

        self.is_train = is_train
        self.read_directly = read_directly

        self.df_x_train, self.df_y_train, self.df_y_train_with_non_scored, self.df_x_test = read_data(root_dir)
        self.df_x_train, self.df_y_train, self.df_y_train_with_non_scored, self.df_x_test = \
            drop_ctl_vehicle_samples_and_cols(self.df_x_train, self.df_y_train, self.df_y_train_with_non_scored,
                                              self.df_x_test)
        # 标签列
        self.cols_label = [col for col in self.df_y_train.columns]
        # 特征列
        self.cols_feature = [col for col in self.df_x_train.columns]
        # 类别型特征，类别型特征会被编码
        self.cols_feature_category = ['cp_time', 'cp_dose']
        # 数值型特征， 数值型特征会被修改
        self.cols_feature_numeric = [feat for feat in self.cols_feature if
                                     feat not in self.cols_feature_category]

        # 标签列的数量
        self.num_cols_label = len(self.cols_label)
        # 特征列的数量, 特征列的数量会被修改
        self.num_features_label = len(self.cols_feature)
        # 训练样本的数量
        self.num_train_samples = self.df_x_train.shape[0]
        # 测试样本的数量
        self.num_test_samples = self.df_x_test.shape[0]

        self.x_train = None
        self.y_train = None
        self.y_train_with_non_scored = None
        self.x_test = None

    def process(self, preprocess_param, base_seed):

        df_feature_all = pd.concat([self.df_x_train, self.df_x_test], ignore_index=True)
        # 处理
        if preprocess_param['is_add_square_feature']:
            # 添加高维特征(平方特征)
            df_feature_all = add_square_features(df_feature_all)
            print("add feature processing; after adding square features -- df shape: ", df_feature_all.shape)
        if preprocess_param['is_delete_feature']:
            df_feature_all = delete_unimportant_features(df_feature_all)
            print("delete feature processing; after adding square features -- df shape: ", df_feature_all.shape)
        if preprocess_param['is_gauss_rank']:
            # gauss rank将特征规范化
            df_feature_all = gauss_rank(df_feature_all, self.cols_feature_category)
            print("gauss rank processing ; after gauss rank -- df shape: ", df_feature_all.shape)
        if preprocess_param['is_pca']:
            # 提取主成分，并将主成分加入特征
            df_feature_all, self.cols_feature_numeric = decompose_pca(df_feature_all, preprocess_param['n_gene_comp'],
                                                                      preprocess_param['n_cell_comp'],
                                                                      self.cols_feature_category, base_seed,
                                                                      is_train=self.is_train,
                                                                      out_tabnet_data_dir=self.out_pytorch_data_dir)
            print("pca processing; after pca -- df shape: ", df_feature_all.shape)
        if preprocess_param['is_svd']:
            df_feature_all, self.cols_feature_numeric = decompose_svd(df_feature_all, preprocess_param['n_gene_comp'],
                                                                      preprocess_param['n_cell_comp'],
                                                                      self.cols_feature_category, base_seed,
                                                                      is_train=self.is_train,
                                                                      out_tabnet_data_dir=self.out_pytorch_data_dir)
            print("SVD processing; after SVD -- df shape: ", df_feature_all.shape)

        if preprocess_param['is_filtered_by_var']:
            # 滤掉方差较小的特征
            df_feature_all, self.cols_feature_numeric = filter_feature_by_variance(df_feature_all,
                                                                                   self.cols_feature_category,
                                                                                   preprocess_param['variance_thresh'],
                                                                                   is_train=self.is_train,
                                                                                   out_tabnet_data_dir=self.out_pytorch_data_dir)
            print("filtered by variance processing; after variance processing -- df shape: ", df_feature_all.shape)

        if preprocess_param['is_add_statistic_feature']:
            df_feature_all = generate_new_features(df_feature_all)
            print("generate new features processing ; -- df shape: ", df_feature_all.shape)

        if preprocess_param['is_encoding']:
            # 编码类别型特征
            df_feature_all = encode_feature(df_features_all=df_feature_all, cols_features=self.cols_feature_category,
                                            encoding=preprocess_param['encoding'])
            print("encoding processing; after encoding processing -- df shape: ", df_feature_all.shape)

        x_train = df_feature_all[:self.num_train_samples]
        x_train.reset_index(drop=True, inplace=True)
        self.x_train = x_train.values

        x_test = df_feature_all[-self.num_test_samples:]
        x_test.reset_index(drop=True, inplace=True)
        self.x_test = x_test.values

        self.y_train = self.df_y_train.values
        self.y_train_with_non_scored = self.df_y_train_with_non_scored.values

        return self.x_train, self.y_train, self.y_train_with_non_scored, self.x_test, self.cols_label, self.num_cols_label

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.cols_label, self.num_cols_label
