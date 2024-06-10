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


def svm_sgd(X, y, lr_init, C, epochs):
    w = np.zeros(X.shape[1])
    for epoch in tqdm(range(1, epochs + 1)):
        lr = lr_init / (1 + epoch)

        perm = np.random.permutation(len(y))
        X, y = X[perm], y[perm]

        for i in range(len(y)):
            xi, yi = X[i], y[i]
            common_term = (1 - lr) * w
            if yi * np.dot(w, xi) <= 1:
                w = common_term + (lr * C * yi * xi)
            else:
                w = common_term

    return w


def cross_validation(X, y, lr_candidates, tradeoff_candidates, epochs, folds):
    best_acc = -np.inf
    best_lr = None
    best_tradeoff = None

    #halving the dataset for faster performance
    examples = X.shape[0]
    features = X.shape[1]

    num_samples = len(X)
    fold_size = num_samples // folds
    indices = np.random.permutation(num_samples)

    for lr in lr_candidates:
        for t in tradeoff_candidates:
            fold_accuracy = []

            # Perform cross-validation
            for num in tqdm(range(folds)):
                val_indices = indices[num * fold_size: (num + 1) * fold_size]
                train_indices = np.concatenate([indices[:num * fold_size], indices[(num + 1) * fold_size:]])
                x_train_cv, x_test_cv = X[train_indices], X[val_indices]
                label_train_cv, label_test_cv = y[train_indices], y[val_indices]

                w = svm_sgd(x_train_cv, label_train_cv, lr, t, epochs)

                label_prediction = np.sign(np.dot(x_test_cv, w))

                # Calculate accuracy for the validation predictions
                fold_acc = calculate_accuracy(label_test_cv, label_prediction)
                fold_accuracy.append(fold_acc)

            # Calculate average accuracy across all folds
            avg_acc = np.mean(fold_accuracy)
            print(
                f"lr: {lr} | tradeoff: {t} | avg acc {np.round(avg_acc, 3)} ")
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_lr = lr
                best_tradeoff = t

    return best_lr, best_tradeoff


def calculate_accuracy(predictions, y_test):
    return np.mean(predictions == y_test)


def predict(X, weights):
    prediction = np.sign(np.dot(X, weights))
    return prediction


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

# Find the best hyperparameters
"""
Commenting out this because I ran cross validation just once to get the hyperparamter and 
do not want to run it again and again because it took time.
"""
# best_learning_rate, best_c_value = cross_validation(x_train_tfidf_features, y_train_tfidf_labels, [1, 0.1, 0.01], [10, 1], 10, 5)

# Train SVM
# weights = svm_sgd(x_train_tfidf_features, y_train_tfidf_labels, best_learning_rate, best_c_value, 10)
# After running cross validation and finding the best hyperparameter.
weights = svm_sgd(x_train_tfidf_features, y_train_tfidf_labels, 0.01, 10, 15)

#Train
predictions_train = predict(x_train_tfidf_features, weights)
accuracy_train = calculate_accuracy(predictions_train, y_train_tfidf_labels)
print("SVM accuracy on train data tfidf:", accuracy_train)

# Test
predictions_test = predict(x_test_tfidf, weights)
accuracy_test = calculate_accuracy(predictions_test, labels_test_tfidf)
print("SVM accuracy on test data tfidf:", accuracy_test)

predictions_eval = predict(x_eval_tfidf, weights)

with open('tfidf_svm.csv', 'w', newline='') as file:
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

# Find the best hyperparameters
"""
Commenting out this because I ran cross validation just once to get the hyperparamter and 
do not want to run it again and again because it took time.
"""
# best_learning_rate, best_c_value = cross_validation(x_train_bow_features, y_train_bow_labels, [1, 0.1, 0.01], [10, 1], 10, 5)

# Train SVM
# weights = svm_sgd(x_train_bow_features, y_train_bow_labels, best_learning_rate, best_c_value, 10)
# After running cross validation and finding the best hyperparameter.
weights = svm_sgd(x_train_bow_features, y_train_bow_labels, 0.01, 10, 15)

#Train
predictions_train = predict(x_train_bow_features, weights)
accuracy_train = calculate_accuracy(predictions_train, y_train_bow_labels)
print("SVM accuracy on train data bow:", accuracy_train)

# Test
predictions_test = predict(x_test_bow, weights)
accuracy_test = calculate_accuracy(predictions_test, labels_test_bow)
print("SVM accuracy on test data bow:", accuracy_test)

predictions_eval = predict(x_eval_bow, weights)

with open('bow_svm.csv', 'w', newline='') as file:
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

# Find the best hyperparameters
"""
Commenting out this because I ran cross validation just once to get the hyperparamter and 
do not want to run it again and again because it took time.
"""
# best_learning_rate, best_c_value = cross_validation(x_train_glove_features, y_train_glove_labels, [1, 0.1, 0.01], [10, 1], 10, 5)

# Train SVM
# weights = svm_sgd(x_train_glove_features, y_train_glove_labels, best_learning_rate, best_c_value, 10)
# After running cross validation and finding the best hyperparameter.
weights = svm_sgd(x_train_glove_features, y_train_glove_labels, 0.01, 10, 15)

#Train
predictions_train = predict(x_train_glove_features, weights)
accuracy_train = calculate_accuracy(predictions_train, y_train_glove_labels)
print("SVM accuracy on train data glove:", accuracy_train)

# Test
predictions_test = predict(x_test_glove, weights)
accuracy_test = calculate_accuracy(predictions_test, labels_test_glove)
print("SVM accuracy on test data glove:", accuracy_test)

predictions_eval = predict(x_eval_glove, weights)

with open('glove_svm.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["example_id", "label"]

    writer.writerow(field)
    for i in range(len(predictions_eval)):
        x = int(predictions_eval[i]) if predictions_eval[i] == 1 else 0
        writer.writerow([i, x])


