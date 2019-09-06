import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.neighbors import NearestNeighbors

class ClassifierDefinition:

    def __init__(self, type, map, map_hyperparams, scaler, scaler_hyperparams, num_models,
                 class_id_mapping, sample_size_reqs, positive_mapped_class_id):
        self.type = type
        self.map = map
        self.map_hyperparams = map_hyperparams
        self.scaler = scaler
        self.scaler_hyperparams = scaler_hyperparams
        self.num_models = num_models
        self.class_id_mapping = class_id_mapping
        self.sample_size_reqs = sample_size_reqs
        self.positive_mapped_class_id = positive_mapped_class_id


def fit(self, X, y, sample_weight=None, sample_size_requirements=None):
    if sample_size_requirements is not None:
        X_sub, y_sub = get_sub_sample(X=X, y=y, sample_size_requirements=sample_size_requirements)
        return self.fit_original(X=X_sub, y=y_sub, sample_weight=sample_weight)
    else:
        return self.fit_original(X=X, y=y, sample_weight=sample_weight)

class MetaEstimator:

    estimator_type = None

    def __init__(self, estimator_class=None, **kwargs):

        if estimator_class is None:
            self.estimator = MetaEstimator.estimator_type(**kwargs)
        else:
            MetaEstimator.estimator_type = estimator_class
            self.estimator = MetaEstimator.estimator_type(**kwargs)

    def fit(self, X, y, sample_weight=None, sample_size_requirements=None):
        if sample_size_requirements is not None:
            X_sub, y_sub = get_sub_sample(X=X, y=y, sample_size_requirements=sample_size_requirements)

        return self.estimator.fit(X=X_sub, y=y_sub, sample_weight=sample_weight)

    def predict(self, X):
        return self.estimator.predict(X=X)

    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        return self.estimator.set_params(**params)


class ConfusionElement:

    def __init__(self, element, pos_label):
        self.__name__ = element
        self.element = element
        self.pos_label = pos_label

    def __call__(self, y_true, y_pred):
        if self.element == 'tp':
            return np.array(y_pred[y_true == self.pos_label] == self.pos_label).sum()
        if self.element == 'tn':
            return np.array(y_pred[y_true != self.pos_label] != self.pos_label).sum()
        if self.element == 'fp':
            return np.array(y_pred[y_true != self.pos_label] == self.pos_label).sum()
        if self.element == 'fn':
            return np.array(y_pred[y_true == self.pos_label] != self.pos_label).sum()

class SubClassScore:

    def __init__(self, score_function):
        self.score_function = score_function
        self.__name__ = score_function.__name__

    def __call__(self, y_true, y_pred, **kwargs):
        y_true = y_true > 0
        y_pred = y_pred > 0
        return self.score_function(y_true, y_pred, **kwargs)


# Utility function to report best scores
def print_results(results, score_names, ranking_score_name, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_' + ranking_score_name] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            for score_name in score_names:
                print("Mean {0!s}: {1:.3f} (std: {2:.3f})".format(
                      score_name,
                      results['mean_test_' + score_name][candidate],
                      results['std_test_' + score_name][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def get_sub_sample(X, y, sample_size_requirements):
    """get_sub_sample

    Reduce the number of samples in a data matrix, whilst trying to include a certain number of each target class.
    Useful for multi-class sampling.

    :param X:                           a numpy array like object containing input variables from at least one class
    :param y:                           a one dimensional numpy array containing target values
    :param sample_size_requirements:    a dict with class labels as keys and number of required observations as values

    :return:    two numpy arrays containing the sub-sampled values from X and y respectively
    """
    temp_X = []
    temp_y = []
    for target_class, required_number in sample_size_requirements.items():
        if required_number == 0:
            continue
        target_positions = y == target_class
        X_target = X[target_positions]
        available_num = (target_positions).sum()
        sample_num = min(required_number, available_num)
        temp_X.append(X_target[np.random.choice(a=X_target.shape[0], size=sample_num, replace=False), :])
        temp_y.append(np.full(shape=(sample_num,), fill_value=target_class))
    X_subsample = np.concatenate(temp_X, axis=0)
    y_subsample = np.concatenate(temp_y, axis=0)

    return X_subsample, y_subsample

    # sampler = RBFSampler(gamma=0.0001618859690178199, n_components=870)
    # rbf_X = sampler.fit_transform(X)
    # positive_positions = y == 1
    # rbf_X_positives = rbf_X[positive_positions]
    # rbf_X_negatives = rbf_X[np.logical_not(positive_positions)]
    # neighbour_finder = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean', n_jobs=1)
    # neighbour_finder.fit(rbf_X_negatives)
    # negative_neighbour_positions = neighbour_finder.kneighbors(X=rbf_X_positives, n_neighbors=50, return_distance=False)
    # negative_neighbour_positions = np.unique(ar=negative_neighbour_positions)
    # return np.concatenate([X[negative_neighbour_positions], X[positive_positions]]), np.concatenate([y[negative_neighbour_positions], y[positive_positions]])