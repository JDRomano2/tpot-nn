from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
def opt_pipe(training_features, testing_features):
	exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=85),
    RobustScaler(),
    MLPRegressor(activation="logistic", alpha=100.0, hidden_layer_sizes=(256, 32), learning_rate="adaptive", learning_rate_init=0.01, momentum=0.9, solver="sgd")
)
	return({'train_feat': training_features, 'test_feat': testing_features, 'pipe': exported_pipeline})