from functools import lru_cache

import numpy as np
import pandas as pd
from collections import Counter

FILE_NAMES = {"bag-of-words": "bow", "glove": "glove", "tfidf": "tfidf"}
TRAIN_DATA_FILE = "train.csv"
TEST_DATA_FILE = "test.csv"
EVAL_DATA_FILE = "eval.anon.csv"
LABEL = "label"


def get_file_path(feature_val):
    return "project_data/data/" + feature_val + "/" + FILE_NAMES[feature_val] + "."


def read_data(f, feature):
    return pd.read_csv(get_file_path(feature) + f)


def get_data_labels_counter(data, label):
    data_labels, num_rows = data[label].tolist(), data.shape[0]
    return Counter(data_labels), num_rows


def get_entropy(num_rows, d):
    entropy = 0
    for i in d:
        p = d[i] / num_rows
        entropy += ((-1) * p * np.log2(p))
    return entropy


train_data_glove = read_data(TRAIN_DATA_FILE, "glove")
eval_data_glove = read_data(EVAL_DATA_FILE, "glove")
train_d_glove, train_glove_num_rows = get_data_labels_counter(train_data_glove, LABEL)

total_entropy_glove = get_entropy(train_data_glove.shape[0], train_d_glove)


def get_info_gain(feat, data, label, tot_entropy):
    feat_dic = Counter(data[feat].tolist())
    total_num_rows = data.shape[0]
    exp_entropy = 0
    for i in feat_dic:
        res = data.loc[data[feat] == i, label]
        d = Counter(res.tolist())
        exp_entropy += (feat_dic[i] / total_num_rows) * get_entropy(feat_dic[i], d)

    return tot_entropy - exp_entropy


@lru_cache()
def find_most_informative_feat(data, label, total_entropy):
    data_c = data.copy()
    feat_list = data_c.columns.drop(label)
    max_info_gain = -1
    max_info_feat = ''

    for feat in feat_list:
        feat_info_gain = get_info_gain(feat, data, label, total_entropy)
        if max_info_gain < feat_info_gain:
            max_info_gain = feat_info_gain
            max_info_feat = feat

    return max_info_feat


def gen_subtree(feat_name, data, label, unique_labels):
    feat_value_dic = Counter(data[feat_name].tolist())
    res = {}

    for feat in feat_value_dic:
        feature_value_data = data[data[feat_name] == feat]

        assigned = False
        for c in unique_labels:
            label_count = feature_value_data[feature_value_data[label] == c].shape[0]
            if label_count == feat_value_dic[feat]:
                res[feat] = c
                data = data[data[feat_name] != feat]
                assigned = True

        if not assigned:
            res[feat] = "?"

    return res, data


def gen_tree(root, prev_feat, data, label, unique_label, ent):
    if data.shape[0] != 0:
        max_info_feat = find_most_informative_feat(data, label, ent)
        tr, data = gen_subtree(max_info_feat, data, label, unique_label)
        next_root = None

        if prev_feat is not None:
            root[prev_feat] = dict()
            root[prev_feat][max_info_feat] = tr
            next_root = root[prev_feat][max_info_feat]
        else:
            root[max_info_feat] = tr
            next_root = root[max_info_feat]

        for node, branch in list(next_root.items()):
            if branch == "#":
                feature_value_data = data[data[max_info_feat] == node]
                gen_tree(next_root, node, feature_value_data, label, unique_label, ent)


def id3_algo(data, label, ent):
    data_c = data.copy()
    d = {}
    gen_tree(d, None, data_c, label, data[label].unique(), ent)

    return d


def gen_tree_depth_limit(ent, root, prev_feat, data, label, unique_label, depth=None, curr_depth=0):
    if curr_depth >= depth:
        root[prev_feat] = Counter(data[label].tolist()).most_common(1)[0][0]
        return

    if data.shape[0] != 0:
        max_info_feat = find_most_informative_feat(data, label, ent)
        tr, data = gen_subtree(max_info_feat, data, label, unique_label)
        next_root = None

        if prev_feat is not None:
            root[prev_feat] = dict()
            root[prev_feat][max_info_feat] = tr
            next_root = root[prev_feat][max_info_feat]
        else:
            root[max_info_feat] = tr
            next_root = root[max_info_feat]

        for node, branch in list(next_root.items()):
            if branch == "#":
                feature_value_data = data[data[max_info_feat] == node]
                gen_tree_depth_limit(ent, next_root, node, feature_value_data, label, unique_label, depth, curr_depth + 1)


def id3_algo_depth_limit(ent, data, label, depth=None, curr_depth=0):
    data_c = data.copy()
    d = {}
    gen_tree_depth_limit(ent, d, None, data_c, label, data[label].unique(), depth=depth, curr_depth=curr_depth)
    return d


def prediction_from_tree(t, row):
    if not isinstance(t, dict):
        return t

    root = list(t)[0]
    val = row[root] if root in row else None

    if val in t[root]:
        return prediction_from_tree(t[root][val], row)
    else:
        return None


eval_li = []
tree_glove = id3_algo_depth_limit(total_entropy_glove, train_data_glove, LABEL, depth=10)
for _, row in eval_data_glove.iterrows():
    row_to_dic = row.to_dict()
    result = prediction_from_tree(tree_glove, row_to_dic)
    eval_li.append(result)


print(tree_glove)
# print(eval_li)




