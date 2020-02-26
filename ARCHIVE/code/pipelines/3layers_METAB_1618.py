from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
def opt_pipe(training_features, testing_features):
	exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.0005),
    SelectPercentile(score_func=f_regression, percentile=89),
    MinMaxScaler(),
    MLPRegressor(activation="logistic", alpha=100.0, hidden_layer_sizes=(512, 64), learning_rate="constant", learning_rate_init=0.001, momentum=0.75, solver="sgd")
)
	return({'train_feat': training_features, 'test_feat': testing_features, 'pipe': exported_pipeline})