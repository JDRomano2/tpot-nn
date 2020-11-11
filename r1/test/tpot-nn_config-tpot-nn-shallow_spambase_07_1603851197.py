import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer, PolynomialFeatures, RobustScaler
from tpot.builtins import PytorchLRClassifier, StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9532608695652174
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=PytorchLRClassifier(batch_size=32, learning_rate=0.001, num_epochs=15, weight_decay=0.0001)),
    RobustScaler(),
    Binarizer(threshold=0.0),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    PytorchLRClassifier(batch_size=4, learning_rate=0.001, num_epochs=10, weight_decay=0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
