import numpy as np
import pandas as pd

from tpot import TPOTClassifier

from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


tpot_config = {
    'tpot.nn.PytorchLRClassifier': {
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.]
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
clf_t.export('tpot_nn_bc_pipeline.py')
