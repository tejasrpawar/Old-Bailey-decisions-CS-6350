from simpleperceptron import perceptron
from svm import svm_sgd, predict, calculate_accuracy
import numpy as np
import csv
import pandas as pd

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


def prediction(perceptron_weights, perceptron_biases, svm_weights, X):
    perceptron_predictions = np.sign(np.dot(X, perceptron_weights.T) + perceptron_biases)

    svm_predictions = np.sign(np.dot(X, svm_weights))

    ensemble_prediction = np.zeros(perceptron_predictions.shape)

    for i in range(len(perceptron_predictions)):
        votes = perceptron_predictions[i] + svm_predictions[i]
        if votes != 0:
            ensemble_prediction[i] = np.sign(votes)  # Majority voting
        else:
            ensemble_prediction[i] = perceptron_predictions[i]  # Example tie-breaking strategy

    return ensemble_prediction


train_data_tfidf, test_data_tfidf, eval_data_tfidf = read_data(TRAIN_DATA_FILE, "tfidf"), read_data(TEST_DATA_FILE, "tfidf"), read_data(EVAL_DATA_FILE, "tfidf")
train_data_bow, test_data_bow, eval_data_bow = read_data(TRAIN_DATA_FILE, "bag-of-words"), read_data(TEST_DATA_FILE, "bag-of-words"), read_data(EVAL_DATA_FILE, "bag-of-words")
train_data_glove, test_data_glove, eval_data_glove = read_data(TRAIN_DATA_FILE, "glove"), read_data(TEST_DATA_FILE, "glove"), read_data(EVAL_DATA_FILE, "glove")
misc_data_train, misc_data_test, misc_data_eval = make_mapping_for_misc()

###################### TFIDF #########################
print("\n###### Working on TFIDF + MISC ######")
# concat the misc files to tfidf
train_data_tfidf, test_data_tfidf, eval_data_tfidf = pd.concat([train_data_tfidf, misc_data_train], axis=1), pd.concat([test_data_tfidf, misc_data_test], axis=1), pd.concat([eval_data_tfidf, misc_data_eval], axis=1)

x_train_tfidf_features, y_train_tfidf_labels = separate_labels_from_features(train_data_tfidf)
x_test_tfidf, labels_test_tfidf = separate_labels_from_features(test_data_tfidf)
x_eval_tfidf, labels_eval_tfidf = separate_labels_from_features(eval_data_tfidf)


# Train the Perceptron
_, _, percep_weights, percep_biases = perceptron(x_train_tfidf_features, y_train_tfidf_labels, x_train_tfidf_features, y_train_tfidf_labels, 0.1, 100)

# Train the SVM
svm_weights = svm_sgd(x_train_tfidf_features, y_train_tfidf_labels, 0.01, 10, 100)
predictions_train = predict(x_train_tfidf_features, svm_weights)
accuracy_train = calculate_accuracy(predictions_train, y_train_tfidf_labels)
print("Ensemble accuracy on train data tfidf:", accuracy_train)

#Test
predictions_test = predict(x_test_tfidf, svm_weights)
accuracy_test = calculate_accuracy(predictions_test, labels_test_tfidf)
print("Ensemble accuracy on test data tfidf:", accuracy_test)


# Get ensemble predictions for evaluation
predictions_eval = prediction(percep_weights[-1], percep_biases[-1], svm_weights, x_eval_tfidf)

with open('tfidf_ensemble.csv', 'w', newline='') as file:
    print("Started writing predictions on eval...")
    writer = csv.writer(file)
    field = ["example_id", "label"]

    writer.writerow(field)
    for i in range(len(predictions_eval)):
        x = int(predictions_eval[i]) if predictions_eval[i] == 1 else 0
        writer.writerow([i, x])


###################### BOW #########################
print("\n###### Working on BOW + MISC ######")
# concat the misc files to bow
train_data_bow, test_data_bow, eval_data_bow = pd.concat([train_data_bow, misc_data_train], axis=1), pd.concat([test_data_bow, misc_data_test], axis=1), pd.concat([eval_data_bow, misc_data_eval], axis=1)

x_train_bow_features, y_train_bow_labels = separate_labels_from_features(train_data_bow)
x_test_bow, labels_test_bow = separate_labels_from_features(test_data_bow)
x_eval_bow, labels_eval_bow = separate_labels_from_features(eval_data_bow)


# Train the Perceptron
_, _, percep_weights, percep_biases = perceptron(x_train_bow_features, y_train_bow_labels, x_train_bow_features, y_train_bow_labels, 0.1, 100)

# Train the SVM
svm_weights = svm_sgd(x_train_bow_features, y_train_bow_labels, 0.01, 10, 100)
predictions_train = predict(x_train_bow_features, svm_weights)
accuracy_train = calculate_accuracy(predictions_train, y_train_bow_labels)
print("Ensemble accuracy on train data bow:", accuracy_train)

#Test
predictions_test = predict(x_test_bow, svm_weights)
accuracy_test = calculate_accuracy(predictions_test, labels_test_bow)
print("Ensemble accuracy on test data bow:", accuracy_test)


# Get ensemble predictions for evaluation
predictions_eval = prediction(percep_weights[-1], percep_biases[-1], svm_weights, x_eval_bow)

with open('bow_ensemble.csv', 'w', newline='') as file:
    print("Started writing predictions on eval...")
    writer = csv.writer(file)
    field = ["example_id", "label"]

    writer.writerow(field)
    for i in range(len(predictions_eval)):
        x = int(predictions_eval[i]) if predictions_eval[i] == 1 else 0
        writer.writerow([i, x])


###################### GLOVE #########################
print("\n###### Working on GLOVE + MISC ######")
# concat the misc files to glove
train_data_glove, test_data_glove, eval_data_glove = pd.concat([train_data_glove, misc_data_train], axis=1), pd.concat([test_data_glove, misc_data_test], axis=1), pd.concat([eval_data_glove, misc_data_eval], axis=1)

x_train_glove_features, y_train_glove_labels = separate_labels_from_features(train_data_glove)
x_test_glove, labels_test_glove = separate_labels_from_features(test_data_glove)
x_eval_glove, labels_eval_glove = separate_labels_from_features(eval_data_glove)


# Train the Perceptron
_, _, percep_weights, percep_biases = perceptron(x_train_glove_features, y_train_glove_labels, x_train_glove_features, y_train_glove_labels, 0.1, 100)

# Train the SVM
svm_weights = svm_sgd(x_train_glove_features, y_train_glove_labels, 0.01, 10, 100)
predictions_train = predict(x_train_glove_features, svm_weights)
accuracy_train = calculate_accuracy(predictions_train, y_train_glove_labels)
print("Ensemble accuracy on train data glove:", accuracy_train)

#Test
predictions_test = predict(x_test_glove, svm_weights)
accuracy_test = calculate_accuracy(predictions_test, labels_test_glove)
print("Ensemble accuracy on test data glove:", accuracy_test)


# Get ensemble predictions for evaluation
predictions_eval = prediction(percep_weights[-1], percep_biases[-1], svm_weights, x_eval_glove)

with open('glove_ensemble.csv', 'w', newline='') as file:
    print("Started writing predictions on eval...")
    writer = csv.writer(file)
    field = ["example_id", "label"]

    writer.writerow(field)
    for i in range(len(predictions_eval)):
        x = int(predictions_eval[i]) if predictions_eval[i] == 1 else 0
        writer.writerow([i, x])
