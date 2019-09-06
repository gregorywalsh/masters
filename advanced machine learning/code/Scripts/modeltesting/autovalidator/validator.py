import code.Scripts.modeltesting.autovalidator.utilities as ut

from time import time
from multiprocessing import cpu_count
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer


class RandomisedClassifierValidator():
    """

    Parameters
    ----------
    classifier_map : classifier class
        A classifier which implements the sklearn estimator functions.

    map_type : string ('binary', 'binary_with_subclasses', 'outlier')
        Describes the type of map

    df_X : Pandas DataFrame object
        Contains input variables for all observations

    sr_y : Pandas Series object
        Contains class labels for all observations

    f_beta_value : float >= 0
        Float specifying the value of beta in the f-beta score

    splits : cross-validation generator or an iterable
        An iterable or generator which creates (train_ids, test_ids).

    num_models : int >= 1
        Number of machines to generate using the hyperparam_span

    hyperparams : dict, (hyper_param, {distribution, list})
        Dict of hyperparameter names (string) and values/distributions

    positive_mapped_class_id : int
        The id of the class to be treated as the positive class

    positive_is_subclassed : boolean
        When `True`, subclasses will grouped together for the purpose
        od calculating scores

    sample_size_requirements : dict, (class, n_samples)
        Number of samples for each class

    num_threads : int, optional, default number of cores - 1
        Choose the number of threads to create

    """

    def __init__(self, classifier_definition, df_x, sr_y, splits, num_threads=cpu_count() - 1):

        # ============ CREATE SCORERS ============ #

        # Define scores

        self.classifier_map = classifier_definition.map
        map_type = classifier_definition.type
        map_hyperparams = classifier_definition.map_hyperparams
        scaler = classifier_definition.scaler
        scaler_hyperparams = classifier_definition.scaler_hyperparams
        pos_id = classifier_definition.positive_mapped_class_id
        self.num_models = classifier_definition.num_models
        self.sample_size_requirements = classifier_definition.sample_size_reqs
        self.sr_y = sr_y
        self.df_x = df_x

        tp = {'name': 'True Pos', 'function': ut.ConfusionElement(element='tp', pos_label=pos_id), 'kwargs': {}}
        tn = {'name': 'True Neg', 'function': ut.ConfusionElement(element='tn', pos_label=pos_id), 'kwargs': {}}
        fp = {'name': 'False Pos', 'function': ut.ConfusionElement(element='fp', pos_label=pos_id), 'kwargs': {}}
        fn = {'name': 'False Neg', 'function': ut.ConfusionElement(element='fn', pos_label=pos_id), 'kwargs': {}}
        self.scores = [tp, tn, fp, fn]
        self.ranking_score = tp

        # Create scorers
        self.scorers = {}
        for score in self.scores:
            if map_type == 'binary_with_subclasses':
                score_function = ut.SubClassScore(score['function'])
            else:
                score_function = score['function']
            scorer = make_scorer(score_function, **score['kwargs'])
            self.scorers[score['name']] = scorer


        # ============ CREATE VALIDATOR============ #

        # Create pipeline
        #estimator = ut.MetaEstimator(estimator_class=classifier_map)
        if not hasattr(self.classifier_map, 'fit_original'):
            self.classifier_map.fit_original = self.classifier_map.fit
            self.classifier_map.fit = ut.fit
        estimator = self.classifier_map()

        pipeline_hyperparams = {}
        for map_hyperparam, value in map_hyperparams.items():
            pipeline_hyperparams[type(estimator).__name__.lower() + '__' + map_hyperparam] = value

        for scaler_hyperparam, value in scaler_hyperparams.items():
            pipeline_hyperparams[scaler.__name__.lower() + '__' + scaler_hyperparam] = value

        pipeline = make_pipeline(scaler(), estimator)

        # Create multi-threaded randomised search cross validator
        self.random_search = RandomizedSearchCV(estimator=pipeline,
                                                param_distributions=pipeline_hyperparams,
                                                n_iter=self.num_models,
                                                scoring=self.scorers,
                                                n_jobs=num_threads,
                                                pre_dispatch=num_threads,
                                                cv=splits,
                                                refit=False,
                                                verbose=1,
                                                return_train_score=True)

    def validate(self):

        start = time()
        fit_params = {self.classifier_map.__name__.lower() + '__sample_size_requirements': self.sample_size_requirements}
        self.random_search.fit(X=self.df_x.values, y=self.sr_y.values, **fit_params)

        # Report on performance
        print()
        print("%d machines - running time of %.2fs" % (self.num_models, (time() - start)))
        print()
        ut.print_results(results=self.random_search.cv_results_,
                         score_names=self.scorers.keys(),
                         ranking_score_name=self.ranking_score['name'],
                         n_top=min(self.num_models, 5))
        
        return self.random_search.cv_results_
