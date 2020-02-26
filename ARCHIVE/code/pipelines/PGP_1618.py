from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
def opt_pipe(training_features, testing_features):
	exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.05),
    StandardScaler(),
    MLPRegressor(activation="relu", alpha=100.0, learning_rate="adaptive", learning_rate_init=0.1, momentum=0.75, solver="lbfgs")
)
	return({'train_feat': training_features, 'test_feat': testing_features, 'pipe': exported_pipeline})