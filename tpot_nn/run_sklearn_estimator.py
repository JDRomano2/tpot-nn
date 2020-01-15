import numpy as np
import pandas as pd

from tpot import TPOTClassifier

from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


tpot_config = {
    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    }
}


cancer = load_breast_cancer()


X_train, X_test, y_train, y_test = train_test_split(cancer.data.astype(np.float64), cancer.target.astype(np.float64), train_size=0.75, test_size=0.25)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

clf_t = TPOTClassifier(generations=5, population_size=50, verbosity=2, config_dict=tpot_config)

clf_t.fit(X_train, y_train)
print(clf_t.score(X_test, y_test))
clf_t.export('tpot_sklearn_bc_pipeline.py')
