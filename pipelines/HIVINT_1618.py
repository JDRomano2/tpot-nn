from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
def opt_pipe(training_features, testing_features):

	exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_classif, alpha=0.021),
    FeatureAgglomeration(affinity="euclidean", linkage="ward"),
    MLPRegressor(activation="logistic", alpha=0.001, hidden_layer_sizes=4, learning_rate="constant", learning_rate_init=0.1, momentum=0.5, solver="adam")
)
	return({'train_feat': training_features, 'test_feat': testing_features, 'pipe': exported_pipeline})