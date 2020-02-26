import ipdb

from tpot.nn import PytorchLRClassifier

from sklearn.utils.estimator_checks import check_estimator

check_estimator(PytorchLRClassifier)
