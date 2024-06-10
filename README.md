# Old Bailey Proceedings Classification Project

## Overview
The Old Bailey is the colloquial name for the Central Criminal Court of England and Wales, which deals with major criminal cases in Greater London, and also sometimes from the rest of England and Wales. This court has existed in some form or another since the 16th century.

The proceedings of this court have been digitized and are available online via the Old Bailey Proceedings Online project. From the project website, it is:

> A fully searchable edition of the largest body of texts detailing the lives of non-elite people ever published, containing 197,752 trials held at London's central criminal court, and 475 Ordinary’s Accounts of the lives of executed convicts.

## Task Definition
Since all the text of the trials from 1674 to 1913 are available, we can ask the following text classification question: Can we predict the decision of the court using the transcribed dialogue during a trial?

The goal of this project is to explore classifiers that predict the outcomes of trials. That is, the instances for classification are the transcripts of trials, and the labels are either guilty (denoted by 0) or not guilty (denoted by 1).

We provide three different feature representations of the data, and you may use some or all of them (and also optionally mix-and-match to develop new ones), and the different learning algorithms we see in class. Through the semester, you will be submitting predictions of various classifiers to Kaggle. The mechanics of the project are described in the last section.


## Evaluation
Since the data is constructed to be balanced, we will use standard accuracy to evaluate classifiers.

The examples are all split randomly among the three splits. So we expect that the cross-validation performance on the training set and the accuracy scores on the test set and the public and private splits of the evaluation set will be similar.

## Dataset Description
Each example for classification is a single trial. In a single trial, there may be more than one defendant, and more than one charge. To simplify things, we have restricted ourselves to trials where there is exactly one defendant and one charge. As a result, each trial has exactly one outcome: either guilty (in our case label 0) or not guilty (in our case label 1).

### Data Splits
We have randomly selected a subset of the data for this project. You will be working with 25,000 examples that have been split into three parts described below:

- **Train:** This is the training split of the data, on which you will be training models after hyper-parameter tuning. We have not split the training data into multiple folds for cross-validation; we expect you to do that on your own. The training split consists of 17,500 examples.
- **Test:** This is the test set that you can use to evaluate your models locally. The test set consists of 2,250 examples.
- **Eval:** This is the "evaluation" set with 5,250 examples. We have hidden the labels for these examples. The idea is that you use your models to make a prediction on these examples, and then upload the predictions to Kaggle, where your model's performance will be ranked on a leaderboard. Kaggle uses a random half these examples for a public leaderboard that will be visible to everyone in class, and the other half for a private leaderboard that is only visible to the instructors.

### Feature Representations of Trials
Instead of having you work with the raw text directly, we have pre-processed the data and a few different feature sets.

- **Bag-of-Words:** The bag of words representation represents the text of the trial as a set of its words, with their counts. That is, each dimension (i.e., feature) corresponds to a word, and the value of the feature is the number of times the word occurs in the trial. To avoid the dimensionality from becoming too large, we have restricted the features to use the 10,000 most frequent words in the data.
- **TF-IDF:** The TF-IDF representation is short for term frequency–inverse document frequency, which is a popular document representation that seeks to improve on the bag of words representation by weighting each feature in the bag-of-words representation such that frequent words are weighted lower. As with the bag-of-words, we use the 10,000 most frequent words in the data.
- **GloVe:** The previous two feature representations are extremely sparse vectors, with only a small fraction of the 10,000 features being non-zero for any vector. The GloVe representation represents a document by the average of its "word embeddings," which are vectors that are trained to capture a word's meaning. The word embeddings are dense, 300-dimensional vectors, and for the purpose of this project, each word is weighted by its TF-IDF score.

The description of the three feature representations here is deliberately brief. If you would like to know more about these representations, feel free to explore the corresponding Wikipedia articles which do a reasonable job of describing them, or use the office hours to discuss them. For the purpose of this class, you can think of these as three different feature representations of the same data.

In addition to these features, we have also provided miscellaneous categorical attributes we have extracted from the trial data. This consists of the following features for each trial:
- Defendant age
- Defendant gender
- Number of victims
- Genders of the victims
- Offence category
- Offence subcategory

You are free to explore using these categorical features to improve, or even build your classifiers. In fact, you are also welcome to try feature space expansions neural networks, and other non-linear methods.

## Data Files
We have three data splits (train, test, eval), and four different kinds of features (bag-of-words, TF-IDF, GloVe, misc). Each of these are available in a different file. All features are provided as CSV files. The CSV files associated with the bag-of-words, TF-IDF, and GloVe features all use the same format: the label is in a column called `label` and the features are in columns called `x0`, `x1`, `x2` and so on. The `misc` files do not contain the labels.

Each row across the four types of features corresponds to the same instance, which means that you can mix-and-match the four feature sets.

All the data is available in the `data` directory, organized as follows:

| Filename                                 | Feature       | Data split |
|------------------------------------------|---------------|------------|
| data/bag-of-words/bow.train.csv          | bag-of-words  | train      |
| data/bag-of-words/bow.test.csv           |               | test       |
| data/bag-of-words/bow.eval.anon.csv      |               | eval       |
| data/tfidf/tfidf.train.csv               | tfidf         | train      |
| data/tfidf/tfidf.test.csv                |               | test       |
| data/tfidf/tfidf.eval.anon.csv           |               | eval       |
| data/glove/glove.train.csv               | glove         | train      |
| data/glove/glove.test.csv                |               | test       |
| data/glove/glove.eval.anon.csv           |               | eval       |
| data/misc/misc-attributes-train.csv      | misc          | train      |
| data/misc/misc-attributes-test.csv       |               | test       |
| data/misc/misc-attributes-eval.csv       |               | eval       |

In addition, the directory also contains a file called `data/eval.ids`. This file has as many rows as the `data.eval.anon` file. Each line consists of an example ID, that uniquely identifies the evaluation example. The IDs from this file will be used to match your uploaded predictions on Kaggle.


