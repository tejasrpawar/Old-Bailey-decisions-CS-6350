import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

np.random.seed(42)

FILE_NAMES = {"bag-of-words": "bow", "glove": "glove", "tfidf": "tfidf"}
TRAIN_DATA_FILE = "train.csv"
TEST_DATA_FILE = "test.csv"
EVAL_DATA_FILE = "eval.anon.csv"
LABEL = "label"


def merge_data_horizontally(df1, df2):
    return pd.concat([df1, df2], axis=0)


def merge_glove_dataset(df1, df2):
    #df1 is glove dataset.
    num_padding_columns = len(df2.columns) - len(df1.columns) - 10

    zero_padding_df = pd.DataFrame(np.zeros((len(df1), num_padding_columns)),
                                   columns=[f'x{309 + i + 1}' for i in range(num_padding_columns)])

    # Concatenate the zero-padding with df1 along axis 1
    df1_padded = pd.concat([df1, zero_padding_df], axis=1)

    # Concatenate df1_padded and df2 along axis 0
    combined_df = pd.concat([df1_padded, df2], axis=0)

    return combined_df


def create_mapping_for_offense_category(train, test, eval, feature):

    unique_categories = train[feature].unique()
    category_mapping = {category: i + 1 for i, category in enumerate(unique_categories)}

    def convert_category_to_numeric(df):
        return df[feature].map(category_mapping)

    train[feature] = convert_category_to_numeric(train)
    test[feature] = convert_category_to_numeric(test)
    eval[feature] = convert_category_to_numeric(eval)

    return train, test, eval


def clean_gender_column(df):
    df['male'] = df['defendant_gender'].apply(lambda x: 1 if x == 'male' else 0)
    df['female'] = df['defendant_gender'].apply(lambda x: 1 if x == 'female' else 0)
    df['indeterminate'] = df['defendant_gender'].apply(lambda x: 1 if x == 'indeterminate' else 0)

    df = df.drop('defendant_gender', axis=1)

    df['victim_male'] = df['victim_genders'].apply(lambda x: 1 if x == 'male' else 0)
    df['victim_female'] = df['victim_genders'].apply(lambda x: 1 if x == 'female' else 0)
    df['victim_indeterminate'] = df['victim_genders'].apply(lambda x: 1 if x == 'indeterminate' else 0)

    df = df.drop('victim_genders', axis=1)
    return df


def clean_defendant_age_column(df):
    df['defendant_age'] = pd.to_numeric(df['defendant_age'], errors='coerce')
    df['defendant_age'].fillna(0, inplace=True)

    return df


def make_mapping_for_misc():
    file_p = "project_data/data/misc/misc-attributes-"
    misc_train, misc_test, misc_eval = pd.read_csv(file_p + "train.csv"), pd.read_csv(file_p + "test.csv"), pd.read_csv(file_p + "eval.csv")
    misc_train = clean_defendant_age_column(clean_gender_column(misc_train))
    misc_test = clean_defendant_age_column(clean_gender_column(misc_test))
    misc_eval = clean_defendant_age_column(clean_gender_column(misc_eval))

    create_mapping_for_offense_category(misc_train, misc_test, misc_eval, 'offence_category')
    create_mapping_for_offense_category(misc_train, misc_test, misc_eval, 'offence_subcategory')

    return misc_train, misc_test, misc_eval


def get_file_path(feature_val):
    return "project_data/data/" + feature_val + "/" + FILE_NAMES[feature_val] + "."


def read_data(f, feature):
    data = pd.read_csv(get_file_path(feature) + f)
    # handle zero values
    data['label'] = data['label'].replace(0, -1)
    return data


def get_most_common_value(data):
    value_c = data[LABEL].value_counts()
    return value_c.idxmax(), value_c.max()


def separate_labels_from_features(data):
    data_copy = data.copy()
    return data_copy.drop(LABEL, axis=1).values, data_copy[LABEL].values


def gen_random_weights_and_bias(features):
    weights = np.random.uniform(low=-0.01, high=0.01, size=features)
    bias = np.random.uniform(low=-0.01, high=0.01)
    return weights, bias


def update_weights_bias(weights, bias, x_values, label_value, learning_rate):
    c = 0
    if label_value * (np.dot(weights, x_values) + bias) < 0:
        weights += learning_rate * label_value * x_values
        bias += learning_rate * label_value
        c = 1

    return weights, bias, c


def calculate_accuracy(x_values, label_value, w, b):
    predictions = np.sign(np.dot(x_values, w) + b)
    accuracy = np.mean(predictions == label_value)
    return accuracy


def train_perceptron(x_values, label_value, learning_rate, epochs):
    update_per_epoch, weight_after_each_epoch, bias_after_each_epoch = [], [], []
    update_count = 0
    weights, bias = gen_random_weights_and_bias(x_values.shape[1])

    for _ in tqdm(range(epochs)):
        updates = 0
        for i in range(x_values.shape[0]):
            weights, bias, c = update_weights_bias(weights, bias, x_values[i], label_value[i], learning_rate)
            updates += c
        update_per_epoch.append(updates)
        update_count += updates
        weight_after_each_epoch.append(np.copy(weights))
        bias_after_each_epoch.append(bias)

    return weights, bias, update_count, update_per_epoch, weight_after_each_epoch, bias_after_each_epoch


def perceptron(x_values, label_values, x_dev_values, label_dev_values, learning_rate, epochs):
    accuracies = []
    _, _, update_count, _, weights_after_epoch, biases_after_epoch = train_perceptron(x_values, label_values, learning_rate, epochs)
    for epoch in range(epochs):
        accuracies.append(calculate_accuracy(x_dev_values, label_dev_values, weights_after_epoch[epoch], biases_after_epoch[epoch]))

    return accuracies, update_count, weights_after_epoch, biases_after_epoch


train_data_tfidf, test_data_tfidf, eval_data_tfidf = read_data(TRAIN_DATA_FILE, "tfidf"), read_data(TEST_DATA_FILE, "tfidf"), read_data(EVAL_DATA_FILE, "tfidf")
train_data_bow, test_data_bow, eval_data_bow = read_data(TRAIN_DATA_FILE, "bag-of-words"), read_data(TEST_DATA_FILE, "bag-of-words"), read_data(EVAL_DATA_FILE, "bag-of-words")
train_data_glove, test_data_glove, eval_data_glove = read_data(TRAIN_DATA_FILE, "glove"), read_data(TEST_DATA_FILE, "glove"), read_data(EVAL_DATA_FILE, "glove")
misc_data_train, misc_data_test, misc_data_eval = make_mapping_for_misc()

# concat the misc files to tfidf
train_data_tfidf, test_data_tfidf, eval_data_tfidf = pd.concat([train_data_tfidf, misc_data_train], axis=1), pd.concat([test_data_tfidf, misc_data_test], axis=1), pd.concat([eval_data_tfidf, misc_data_eval], axis=1)
## concat the misc files to bow
# train_data_bow, test_data_bow, eval_data_bow = pd.concat([train_data_bow, misc_data_train], axis=1), pd.concat([test_data_bow, misc_data_test], axis=1), pd.concat([eval_data_bow, misc_data_eval], axis=1)
## concat the misc files to glove
# train_data_glove, test_data_glove, eval_data_glove = pd.concat([train_data_glove, misc_data_train], axis=1), pd.concat([test_data_glove, misc_data_test], axis=1), pd.concat([eval_data_glove, misc_data_eval], axis=1)


#create a merged training dataset with two datas.
# merged_train_data_tfidf_and_bow = merge_data_horizontally(train_data_tfidf, train_data_bow)

#merging the glove train dataset as well
# merge_all_train = merge_glove_dataset(train_data_glove, merged_train_data_tfidf_and_bow)

x_train_tfidf_features, y_train_tfidf_labels = separate_labels_from_features(train_data_tfidf)
x_test_tfidf, labels_test_tfidf = separate_labels_from_features(test_data_tfidf)
x_eval_tfidf, labels_eval_tfidf = separate_labels_from_features(eval_data_tfidf)


#train accuracy
accuracies_for_train, _, _, _ = perceptron(x_train_tfidf_features, y_train_tfidf_labels, x_train_tfidf_features, y_train_tfidf_labels, 0.1, 100)
print(f"Accuracy on train data: {np.max(accuracies_for_train)}")


#test accuracy
accuracies_for_test, _, weights_test_glove, biases_test_glove = perceptron(x_train_tfidf_features, y_train_tfidf_labels, x_test_tfidf, labels_test_tfidf, 0.1, 100)
index_with_best_acc_for_dev = np.argmax(accuracies_for_test)
best_weights = weights_test_glove[index_with_best_acc_for_dev]
best_bias = biases_test_glove[index_with_best_acc_for_dev]
print(f"Accuracy on test data: {accuracies_for_test[index_with_best_acc_for_dev]}")

accuracy_for_eval = calculate_accuracy(x_eval_tfidf, labels_eval_tfidf, best_weights, best_bias)
preds = np.sign(np.dot(x_eval_tfidf, best_weights) + best_bias)


with open('tfidf_perceptron.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["example_id", "label"]

    writer.writerow(field)
    for i in range(len(preds)):
        x = int(preds[i]) if preds[i] == 1 else 0
        writer.writerow([i, x])












