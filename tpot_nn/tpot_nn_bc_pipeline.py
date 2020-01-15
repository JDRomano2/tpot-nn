import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.nn import PytorchLRClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8850889192886457
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=PytorchLRClassifier(learning_rate=0.5)),
    StackingEstimator(estimator=PytorchLRClassifier(learning_rate=0.5)),
    PytorchLRClassifier(learning_rate=1.0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)


# Generation 1 - Current best internal CV score: 0.8663201094391246
# Generation 2 - Current best internal CV score: 0.8663201094391246
# Generation 3 - Current best internal CV score: 0.8850889192886457
# Generation 4 - Current best internal CV score: 0.8850889192886457
# Generation 5 - Current best internal CV score: 0.8850889192886457