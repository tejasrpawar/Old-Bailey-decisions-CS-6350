from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import csv

np.random.seed(42)

FILE_NAMES = {"bag-of-words": "bow", "glove": "glove", "tfidf": "tfidf"}
TRAIN_DATA_FILE = "train.csv"
TEST_DATA_FILE = "test.csv"
EVAL_DATA_FILE = "eval.anon.csv"
LABEL = "label"


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


train_data_tfidf, test_data_tfidf, eval_data_tfidf = read_data(TRAIN_DATA_FILE, "tfidf"), read_data(TEST_DATA_FILE, "tfidf"), read_data(EVAL_DATA_FILE, "tfidf")
misc_data_train, misc_data_test, misc_data_eval = make_mapping_for_misc()
train_data_tfidf, test_data_tfidf, eval_data_tfidf = pd.concat([train_data_tfidf, misc_data_train], axis=1), pd.concat([test_data_tfidf, misc_data_test], axis=1), pd.concat([eval_data_tfidf, misc_data_eval], axis=1)

x_train_tfidf_features, y_train_tfidf_labels = separate_labels_from_features(train_data_tfidf)
x_test_tfidf, labels_test_tfidf = separate_labels_from_features(test_data_tfidf)
x_eval_tfidf, labels_eval_tfidf = separate_labels_from_features(eval_data_tfidf)

base_estimator = DecisionTreeClassifier(max_depth=1)
adaboost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)

# Train AdaBoost classifier
adaboost.fit(x_train_tfidf_features, y_train_tfidf_labels)

#Prediction on train set
predictions_train = adaboost.predict(x_train_tfidf_features)
train_accuracy = accuracy_score(y_train_tfidf_labels, predictions_train)
print(f"Accuracy on train data: {train_accuracy}")


if np.isnan(x_test_tfidf).any() or np.isinf(x_test_tfidf).any():
    x_test_tfidf[np.isnan(x_test_tfidf)] = 0  # Replace NaN with 0
    x_test_tfidf[np.isinf(x_test_tfidf)] = np.finfo(x_test_tfidf.dtype).max
x_test_tfidf = x_test_tfidf.astype(np.float64)


# Predictions on test set
predictions_test = adaboost.predict(x_test_tfidf)

#Handle Nan values
if np.isnan(labels_test_tfidf).any() or np.isinf(labels_test_tfidf).any():
    labels_test_tfidf[np.isnan(labels_test_tfidf)] = 0
    labels_test_tfidf[np.isinf(labels_test_tfidf)] = np.finfo(labels_test_tfidf.dtype).max

if np.isnan(predictions_test).any() or np.isinf(predictions_test).any():
    predictions_test[np.isnan(predictions_test)] = 0
    predictions_test[np.isinf(predictions_test)] = np.finfo(predictions_test.dtype).max

labels_test_tfidf = labels_test_tfidf.astype(np.float64)
predictions_test = predictions_test.astype(np.float64)

test_accuracy = accuracy_score(labels_test_tfidf, predictions_test)
print(f"Accuracy on test data: {test_accuracy}")


if np.isnan(x_eval_tfidf).any() or np.isinf(x_eval_tfidf).any():
    # Handle NaN or infinite values (e.g., replace with a specific value)
    x_eval_tfidf[np.isnan(x_eval_tfidf)] = 0  # Replace NaN with 0
    x_eval_tfidf[np.isinf(x_eval_tfidf)] = np.finfo(x_eval_tfidf.dtype).max
x_eval_tfidf = x_eval_tfidf.astype(np.float64)


predictions_eval = adaboost.predict(x_eval_tfidf)

with open('tfidf_adaboost.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["example_id", "label"]

    writer.writerow(field)
    for i in range(len(predictions_eval)):
        x = int(predictions_eval[i]) if predictions_eval[i] == 1 else 0
        writer.writerow([i, x])

