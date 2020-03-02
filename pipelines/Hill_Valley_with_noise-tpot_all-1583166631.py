import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9648950376582448
exported_pipeline = make_pipeline(
    Normalizer(norm="max"),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features=0.7000000000000001, min_samples_leaf=20, min_samples_split=9, n_estimators=100, subsample=0.05)),
    GradientBoostingClassifier(learning_rate=0.01, max_depth=4, max_features=0.45, min_samples_leaf=3, min_samples_split=10, n_estimators=100, subsample=0.15000000000000002)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
