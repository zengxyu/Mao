import pickle

import numpy as np
import pandas as pd

import os

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer


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
    # del x_train['sig_id']
    # del y_train['sig_id']
    # del x_test['sig_id']
    # del y_train_non_scored['sig_id']
    y_train_with_non_scored = pd.concat([y_train, y_train_non_scored], axis=1)
    print("y_train shape:", y_train.shape)
    print("y_train_with_non_scored:", y_train_with_non_scored.shape)
    return x_train, y_train, y_train_with_non_scored, x_test


def gauss_rank(train_features, test_features):
    """
    用gauss rank将数据规范化
    :return:
    """
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]
    # RankGauss
    qt = QuantileTransformer(n_quantiles=100, random_state=42, output_distribution='normal')
    train_features[GENES + CELLS] = qt.fit_transform(train_features[GENES + CELLS])
    test_features[GENES + CELLS] = qt.transform(test_features[GENES + CELLS])

    return train_features, test_features


def pca(train_features, test_features, is_train, out_tabnet_data_dir):
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    n_comp = 600  # <--Update
    data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

    if is_train:
        pca_g = PCA(n_components=n_comp, random_state=42).fit(data[GENES])
        pickle.dump(pca_g, open(os.path.join(out_tabnet_data_dir, 'pca_g.pkl'), 'wb'))
    else:
        pca_g = pickle.load(open(os.path.join(out_tabnet_data_dir, 'pca_g.pkl'), 'rb'))

    train_g_pca = pd.DataFrame(pca_g.transform(train_features[GENES]), columns=[f'pca_G-{i}' for i in range(n_comp)])
    test_g_pca = pd.DataFrame(pca_g.transform(test_features[GENES]), columns=[f'pca_G-{i}' for i in range(n_comp)])
    # train_features = pd.concat((train_features, train_g_pca), axis=1)
    # test_features = pd.concat((test_features, test_g_pca), axis=1)

    # CELLS
    n_comp = 50  # <--Update
    data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

    if is_train:
        pca_c = PCA(n_components=n_comp, random_state=42).fit(data[CELLS])
        pickle.dump(pca_c, open(os.path.join(out_tabnet_data_dir, 'pca_c.pkl'), 'wb'))
    else:
        pca_c = pickle.load(open(os.path.join(out_tabnet_data_dir, 'pca_c.pkl'), 'rb'))

    train_c_pca = pd.DataFrame(pca_c.transform(train_features[CELLS]), columns=[f'pca_C-{i}' for i in range(n_comp)])
    test_c_pca = pd.DataFrame(pca_c.transform(test_features[CELLS]), columns=[f'pca_C-{i}' for i in range(n_comp)])
    # train_features = pd.concat((train_features, train_c_pca), axis=1)
    # test_features = pd.concat((test_features, test_c_pca), axis=1)

    return train_g_pca, test_g_pca, train_c_pca, test_c_pca


def filter_feature_by_variance(train_features, test_features):
    c_n = [f for f in list(train_features.columns) if f not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
    mask = (train_features[c_n].var() >= 0.85).values
    tmp = train_features[c_n].loc[:, mask]
    train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
    tmp = test_features[c_n].loc[:, mask]
    test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)

    return train_features, test_features


def fe_cluster_genes(train_features, test_features, out_tabnet_data_dir, n_clusters_g=22, SEED=42, is_train=False):
    GENES = [col for col in train_features.columns if col.startswith('g-')]

    # features_c = CELLS

    def create_cluster(train, test, features, kind='g', n_clusters=n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis=0)
        if is_train:
            kmeans_genes = KMeans(n_clusters=n_clusters, random_state=SEED).fit(data)
            pickle.dump(kmeans_genes, open(os.path.join(out_tabnet_data_dir, 'kmeans_genes.pkl'), 'wb'))
        else:
            kmeans_genes = pickle.load(open(os.path.join(out_tabnet_data_dir, 'kmeans_genes.pkl'), 'rb'))

        train[f'clusters_{kind}'] = kmeans_genes.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_genes.predict(test_.values)
        train = pd.get_dummies(train, columns=[f'clusters_{kind}'])
        test = pd.get_dummies(test, columns=[f'clusters_{kind}'])
        return train, test

    train, test = create_cluster(train_features, test_features, GENES, kind='g', n_clusters=n_clusters_g)
    return train, test


def fe_cluster_cells(train_features, test_features, out_tabnet_data_dir, n_clusters_c=4, SEED=42, is_train=False):
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    def create_cluster(train, test, features, kind='c', n_clusters=n_clusters_c):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis=0)
        if is_train:
            kmeans_cells = KMeans(n_clusters=n_clusters, random_state=SEED).fit(data)
            pickle.dump(kmeans_cells, open(os.path.join(out_tabnet_data_dir, 'kmeans_cells.pkl'), 'wb'))
        else:
            kmeans_cells = pickle.load(open(os.path.join(out_tabnet_data_dir, 'kmeans_cells.pkl'), 'rb'))
        train[f'clusters_{kind}'] = kmeans_cells.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_cells.predict(test_.values)
        train = pd.get_dummies(train, columns=[f'clusters_{kind}'])
        test = pd.get_dummies(test, columns=[f'clusters_{kind}'])
        return train, test

    # train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train_features, test_features, CELLS, kind='c', n_clusters=n_clusters_c)
    return train, test


def fe_cluster_pca(train, test, out_tabnet_data_dir, n_clusters=5, SEED=42, is_train=False):
    data = pd.concat([train, test], axis=0)
    if is_train:
        kmeans_pca = KMeans(n_clusters=n_clusters, random_state=SEED).fit(data)
        pickle.dump(kmeans_pca, open(os.path.join(out_tabnet_data_dir, 'kmeans_pca.pkl'), 'wb'))
    else:
        kmeans_pca = pickle.load(open(os.path.join(out_tabnet_data_dir, 'kmeans_pca.pkl'), 'rb'))
    train[f'clusters_pca'] = kmeans_pca.predict(train.values)
    test[f'clusters_pca'] = kmeans_pca.predict(test.values)
    train = pd.get_dummies(train, columns=[f'clusters_pca'])
    test = pd.get_dummies(test, columns=[f'clusters_pca'])
    return train, test


def fe_stats(train_features, test_features):
    gsquarecols = ['g-574', 'g-211', 'g-216', 'g-0', 'g-255', 'g-577', 'g-153', 'g-389', 'g-60', 'g-370', 'g-248',
                   'g-167',
                   'g-203', 'g-177', 'g-301', 'g-332', 'g-517', 'g-6', 'g-744', 'g-224', 'g-162', 'g-3', 'g-736',
                   'g-486',
                   'g-283', 'g-22', 'g-359', 'g-361', 'g-440', 'g-335', 'g-106', 'g-307', 'g-745', 'g-146', 'g-416',
                   'g-298', 'g-666', 'g-91', 'g-17', 'g-549', 'g-145', 'g-157', 'g-768', 'g-568', 'g-396']

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    features_g = GENES
    features_c = CELLS

    for df in train_features, test_features:
        df['g_sum'] = df[features_g].sum(axis=1)
        df['g_mean'] = df[features_g].mean(axis=1)
        df['g_std'] = df[features_g].std(axis=1)
        df['g_kurt'] = df[features_g].kurtosis(axis=1)
        df['g_skew'] = df[features_g].skew(axis=1)
        df['c_sum'] = df[features_c].sum(axis=1)
        df['c_mean'] = df[features_c].mean(axis=1)
        df['c_std'] = df[features_c].std(axis=1)
        df['c_kurt'] = df[features_c].kurtosis(axis=1)
        df['c_skew'] = df[features_c].skew(axis=1)
        df['gc_sum'] = df[features_g + features_c].sum(axis=1)
        df['gc_mean'] = df[features_g + features_c].mean(axis=1)
        df['gc_std'] = df[features_g + features_c].std(axis=1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis=1)
        df['gc_skew'] = df[features_g + features_c].skew(axis=1)

        df['c52_c42'] = df['c-52'] * df['c-42']
        df['c13_c73'] = df['c-13'] * df['c-73']
        df['c26_c13'] = df['c-23'] * df['c-13']
        df['c33_c6'] = df['c-33'] * df['c-6']
        df['c11_c55'] = df['c-11'] * df['c-55']
        df['c38_c63'] = df['c-38'] * df['c-63']
        df['c38_c94'] = df['c-38'] * df['c-94']
        df['c13_c94'] = df['c-13'] * df['c-94']
        df['c4_c52'] = df['c-4'] * df['c-52']
        df['c4_c42'] = df['c-4'] * df['c-42']
        df['c13_c38'] = df['c-13'] * df['c-38']
        df['c55_c2'] = df['c-55'] * df['c-2']
        df['c55_c4'] = df['c-55'] * df['c-4']
        df['c4_c13'] = df['c-4'] * df['c-13']
        df['c82_c42'] = df['c-82'] * df['c-42']
        df['c66_c42'] = df['c-66'] * df['c-42']
        df['c6_c38'] = df['c-6'] * df['c-38']
        df['c2_c13'] = df['c-2'] * df['c-13']
        df['c62_c42'] = df['c-62'] * df['c-42']
        df['c90_c55'] = df['c-90'] * df['c-55']

        for feature in features_c:
            df[f'{feature}_squared'] = df[feature] ** 2

        for feature in gsquarecols:
            df[f'{feature}_squared'] = df[feature] ** 2

    return train_features, test_features


class TabnetPreprocessHelper:
    def __init__(self, root_dir, out_data_dir, is_train, read_directly):
        self.root_dir = root_dir

        self.out_tabnet_data_dir = out_data_dir
        if not os.path.exists(self.out_tabnet_data_dir):
            os.makedirs(self.out_tabnet_data_dir)
        if read_directly:
            print("read_directly")
        self.is_train = is_train
        self.read_directly = read_directly
        self.name = "data_tabnet.pkl"

    def process(self, preprocess_param, base_seed):
        if self.read_directly:
            train, target, df_y_train_with_non_scored, test, feature_cols, target_cols = pickle.load(
                open(os.path.join(self.out_tabnet_data_dir, 'data_tabnet.pkl'), 'rb'))
            print("read_directly")
        else:
            df_x_train, df_y_train, df_y_train_with_non_scored, df_x_test = read_data(self.root_dir)
            train_features2, test_features2 = df_x_train.copy(), df_x_test.copy()

            # gauss rank
            train_features, test_features = gauss_rank(df_x_train, df_x_test)

            # pca
            train_g_pca, test_g_pca, train_c_pca, test_c_pca = pca(train_features, test_features,
                                                                   is_train=self.is_train,
                                                                   out_tabnet_data_dir=self.out_tabnet_data_dir)
            train_features = pd.concat((train_features, train_g_pca, train_c_pca), axis=1)
            test_features = pd.concat((test_features, test_g_pca, test_c_pca), axis=1)

            # filter by variance
            train_features, test_features = filter_feature_by_variance(train_features, test_features)

            # k mean
            #     gene k mean
            train_features2, test_features2 = fe_cluster_genes(train_features2, test_features2,
                                                               out_tabnet_data_dir=self.out_tabnet_data_dir,
                                                               is_train=self.is_train)
            #     cell k mean
            train_features2, test_features2 = fe_cluster_cells(train_features2, test_features2,
                                                               out_tabnet_data_dir=self.out_tabnet_data_dir,
                                                               is_train=self.is_train
                                                               )
            #       pca data k mean
            train_pca = pd.concat((train_g_pca, train_c_pca), axis=1)
            test_pca = pd.concat((test_g_pca, test_c_pca), axis=1)
            train_cluster_pca, test_cluster_pca = fe_cluster_pca(train_pca, test_pca,
                                                                 out_tabnet_data_dir=self.out_tabnet_data_dir,
                                                                 is_train=self.is_train)
            train_cluster_pca = train_cluster_pca.iloc[:, 650:]
            test_cluster_pca = test_cluster_pca.iloc[:, 650:]

            train_features_cluster = train_features2.iloc[:, 876:]
            test_features_cluster = test_features2.iloc[:, 876:]

            # add features
            train_features2, test_features2 = fe_stats(train_features2, test_features2)
            train_features_stats = train_features2.iloc[:, 902:]
            test_features_stats = test_features2.iloc[:, 902:]

            train_features = pd.concat(
                (train_features, train_features_cluster, train_cluster_pca, train_features_stats),
                axis=1)
            test_features = pd.concat((test_features, test_features_cluster, test_cluster_pca, test_features_stats),
                                      axis=1)

            # delete cp_type = ctl_vehicle
            train = train_features.merge(df_y_train, on='sig_id')
            train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
            test = test_features

            target = train[df_y_train.columns]

            train = train.drop('cp_type', axis=1)
            test = test.drop('cp_type', axis=1)

            target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

            target = target[target_cols]

            train = pd.get_dummies(train, columns=['cp_time', 'cp_dose'])
            test_ = pd.get_dummies(test, columns=['cp_time', 'cp_dose'])

            feature_cols = [c for c in train.columns if c not in target_cols]
            feature_cols = [c for c in feature_cols if c not in ['sig_id']]

            train = train[feature_cols]
            test = test_[feature_cols]

            with open(os.path.join(self.out_tabnet_data_dir, self.name), 'wb') as f:
                data_store = (train, target, df_y_train_with_non_scored, test, feature_cols, target_cols)
                pickle.dump(data_store, f)
            print("train shape:", train.shape)
            print("target shape:", target.shape)
            print("train_targets_nonscored shape", df_y_train_with_non_scored.shape)
            print("test shape:", test.shape)

        return train.values, target.values, df_y_train_with_non_scored.values, test.values, feature_cols, target_cols


