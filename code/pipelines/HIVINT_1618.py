from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer
def opt_pipe(training_features, testing_features):
	exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=93),
    Binarizer(threshold=0.8500000000000001),
    MLPRegressor(activation="logistic", alpha=10.0, learning_rate="invscaling", learning_rate_init=0.75, momentum=0.1, solver="lbfgs")
)
	return({'train_feat': training_features, 'test_feat': testing_features, 'pipe': exported_pipeline})