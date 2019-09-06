import datetime
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit

import sys

sys.path.insert(0, '../')

from cvsplits.readpickle import get_pickle_cvs
from scoring.evaluation import get_thresholds_table


class XGBModel:

    def __init__(self, param={'objective': 'binary:logistic', 'eval_metric': 'auc', 'booster': 'dart'}, num_round=100):
        self.param = param
        self.num_round = num_round

        self.trained_model = None
        self.y_prob = None
        self.y_class = None
        self.y_true = None

    def random_search(self, n_iter_search=13):
        param_dist = {"max_delta_step": sp_randint(0, 6),
                      "max_depth": sp_randint(0, 6),
                      "min_child_weight": sp_randint(0, 7),
                      "gamma": sp_randint(3, 10),
                      "eta": uniform(0.3, 0.5),
                      "subsample": uniform(0.5, 0.4),
                      "colsample_bytree": uniform(0.65, 0.35),
                      "lambda": uniform(0.5, 8),
                      "rate_drop": uniform(0, 0.4)
                      }

        best_scores = {"F1": 0, "F10": 0, "F50": 0}
        rows = {"F1": 0, "F10": 1, "F50": 2}
        best_params = {}

        for el in range(n_iter_search):
            params = {key: value.rvs(1)[0] for key, value in param_dist.items()}
            print(params)
            results = self.train_cycle(params)

            if el == 0:
                best_results = results

            for key, prev_best in best_scores.items():
                score = results.loc[rows[key], key]
                if score > prev_best:
                    best_scores[key] = score
                    best_results.iloc[rows[key]] = results.iloc[rows[key]]
                    best_params[key] = params

        return best_results, best_params

    def train_cycle(self, params={}):

        for param, value in params.items():
            self.param[param] = value

        best_scores_list = []
        for train, test in get_pickle_cvs():
            self.fit(train)
            y_out = self.predict(test)

            y_true = test.loc[:, 'Class']
            all, bests = get_thresholds_table(y_true.values, y_out.values)
            best_scores_list.append(bests)

        best_scores = pd.concat(best_scores_list)
        avg_best = best_scores.groupby(level=0).mean()
        print(avg_best)
        return avg_best

    def fit(self, train):
        # specify validations set to watch performance
        dtrain, dval = self.split_train_val(train)
        watchlist = [(dval, 'eval'), (dtrain, 'train')]

        # train
        self.trained_model = xgb.train(self.param, dtrain, self.num_round, watchlist)

        # save model
        # self.trained_model.save_model('XGB output/' + self.generate_fname() + '.model')

        # dump model
        # self.trained_model.dump_model('XGB output/dump.raw.txt')
        # dump model with feature map
        # bst.dump_model('XGB output/dump.nice.txt', 'XGB output/featmap.txt')

    def predict(self, test):
        self.y_prob = pd.Series(self.trained_model.predict(self.to_DMatrix(test)))
        return self.y_prob

    @staticmethod
    def to_DMatrix(df):
        return xgb.DMatrix(df.drop('Class', 1), label=df.loc[:, 'Class'])

    def generate_fname(self):
        now = datetime.datetime.now()
        time = now.strftime("%Y%m%dT%H%M")
        return time + "_param-" + str(self.param) + '_num_round-' + str(self.num_round) + ".csv"

    def split_train_val(self, df):
        folds = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        fold_indexes = next(folds.split(df, df.loc[:, ['Class']]))
        df = df.reset_index(drop=True)

        train = df.loc[fold_indexes[0]]
        val = df.loc[fold_indexes[1]]

        return self.to_DMatrix(train), self.to_DMatrix(val)


class uniform_wrapper:

    def __init__(self, a=0, b=1):
        self.a = a
        self.b = b

    def rvs(self):
        uniform.rvs()

if __name__ == "__main__":
    xgb_mod = XGBModel(num_round=50)

    # xgb_mod.train_cycle()
    results, params = xgb_mod.random_search(n_iter_search=25)
