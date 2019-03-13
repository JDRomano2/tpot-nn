from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
def opt_pipe(training_features, testing_features):

	exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=81),
    PCA(iterated_power=3, svd_solver="randomized"),
    MLPRegressor(activation="logistic", alpha=100.0, hidden_layer_sizes=4, learning_rate="invscaling", learning_rate_init=0.001, momentum=0.75, solver="lbfgs")
)
	return({'train_feat': training_features, 'test_feat': testing_features, 'pipe': exported_pipeline})