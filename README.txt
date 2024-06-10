Here is the directory structure of the directory:

.
├── README.txt
├── __pycache__
│   ├── simpleperceptron.cpython-36.pyc
│   └── svm.cpython-36.pyc
├── adaboost.py
├── bow_ensemble.csv
├── bow_lr.csv
├── bow_svm.csv
├── decayperceptron.py
├── ensemble_svm_perceptron.py
├── glove_ensemble.csv
├── glove_lr.csv
├── glove_perceptron.csv
├── glove_svm.csv
├── logisticregression.py
├── output.csv
├── perceptron.py
├── simpleperceptron.py
├── svm.py
├── tfidf_adaboost.csv
├── tfidf_ensemble.csv
├── tfidf_lr.csv
├── tfidf_only_perceptron.csv
├── tfidf_perceptron.csv
├── tfidf_perceptron_decay.csv
├── tfidf_svm.csv
└── tree.py

All the .csv files are the submissions I made on kaggle(not all the submission are here since I overwrote most of them which were not performing the best). The final 6 submissions are as follows:
    bow_ensemble.csv(6th)
    tfidf_adaboost.csv(5th)
    tfidf_lr.csv(4th)
    tfidf_svm.csv(3rd)
    tfidf_perceptron.csv(2nd)
    tfidf_only_perceptron.csv(1st)

How to run the code:
- The dataset has not been included in the submission directory.
- In order to run the scripts, please insert the 'project_data' folder at the same level.
- The 'project_data' folder should have the structure as follows:
            ├── project_data
            │   ├── data
            │   │   ├── bag-of-words
            │   │   │   ├── bow.eval.anon.csv
            │   │   │   ├── bow.test.csv
            │   │   │   └── bow.train.csv
            │   │   ├── eval.ids
            │   │   ├── glove
            │   │   │   ├── glove.eval.anon.csv
            │   │   │   ├── glove.test.csv
            │   │   │   └── glove.train.csv
            │   │   ├── info.txt
            │   │   ├── misc
            │   │   │   ├── misc-attributes-eval.csv
            │   │   │   ├── misc-attributes-test.csv
            │   │   │   └── misc-attributes-train.csv
            │   │   └── tfidf
            │   │       ├── tfidf.eval.anon.csv
            │   │       ├── tfidf.test.csv
            │   │       └── tfidf.train.csv
            │   └── sample-submissions
            │       ├── all-negative.csv
            │       └── all-positive.csv

As mentioned in the report every python file corresponds to the algorithm used. In order to run these files there are different shell scripts.

The shell files are as follows:
    First submission :  perceptron.sh
    Second submission:  simpleperceptron.sh
    Third submission :  svm.py
    Fourth submission:  lr.sh
    Fifth submission :  ensemble.sh
    Sixth submission :  adaboost.sh
    Other(optional) :   other.sh



