import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import fbeta_score, precision_recall_curve, roc_auc_score

from code.Scripts import get_pickle_cvs
from code.Scripts import precisionRecallScore, fbScore

class XGBModel:

    def __init__(self, param={'objective': 'binary:logistic', 'eval_metric': 'auc'}, num_round=100):
        self.param = param
        self.num_round = num_round

        self.trained_model = None
        self.y_prob = None
        self.y_class = None
        self.y_true = None

    def fit(self, train, test):
        dtrain = self.to_DMatrix(train)

        # specify validations set to watch performance
        watchlist = [(self.to_DMatrix(test), 'eval'), (dtrain, 'train')]
        xgb.XGBClassifier
        # train
        self.trained_model = xgb.train(self.param, dtrain, self.num_round, watchlist)

        # save model
        self.trained_model.save_model('XGB output/0001.model')
        # dump model
        self.trained_model.dump_model('XGB output/dump.raw.txt')
        # dump model with feature map
        # bst.dump_model('XGB output/dump.nice.txt', 'XGB output/featmap.txt')

    def predict_prob(self, test):
        self.y_prob = pd.Series(self.trained_model.predict(self.to_DMatrix(test)))
        return self.y_prob

    def predict_class(self, test, threshold=0.5):
        self.y_class = pd.Series([1 if el > threshold else 0 for el in self.predict_prob(test)])
        return self.y_class

    def score(self, test, beta=1):
        self.y_true = test.loc[:, 'Class']

        # Precision-recall curve
        self.precisions, recalls, thresholds = precision_recall_curve(self.y_true, self.predict_prob(test))

        fig1, ax1 = plt.subplots()
        ax1.plot(self.precisions, recalls)
        ax1.grid()
        ax1.set(xlabel='Precision', ylabel='Recall')
        #fig1.show()

        # 90% Recall point
        self.recalls = pd.Series(recalls)
        self.thresholds = pd.Series(thresholds)
        self.thresh90 = self.thresholds.ix[self.recalls[self.recalls > 0.9].index[-1]]

        self.y_class = self.predict_class(test, threshold=self.thresh90)
        fbeta = fbeta_score(self.y_true, self.y_class, beta)
        print('fbetascore=' + str(fbeta))

        # Histograms of data
        clean = self.y_prob[test['Class'].values == 0]
        fraud = self.y_prob[test['Class'].values == 1]

        fig, ax = plt.subplots()
        clean.plot.hist(logy=True, ax=ax)
        fraud.plot.hist(logy=True, ax=ax)
        ax.grid()
        #fig.show()

        return {'precision': self.precisions, 'recall': self.recalls, 'thresholds': self.thresholds,
                'thresh90': self.thresh90}, [fig, fig1], [ax, ax1]

    @staticmethod
    def to_DMatrix(df):
        return xgb.DMatrix(df.drop('Class', 1), label=df.loc[:, 'Class'])


if __name__ == "__main__":

    avg_tp = np.empty(10)
    avg_fp = np.empty(10)
    threshs = np.empty(10)
    i = 0
    for train, test in get_pickle_cvs():
        xgb_mod = XGBModel(num_round=30)
        xgb_mod.fit(train, test)
        results, figs, axs = xgb_mod.score(test)
        threshs[i] = results['thresh90']

        a = precisionRecallScore(xgb_mod.y_true.values, xgb_mod.y_class.values)
        avg_tp[i] = a[1][0]
        avg_fp[i] = a[1][2]

        i += 1
        break

print(avg_tp.mean())
print(avg_fp.mean())

fig, axs = plt.subplots(1, 3)
axs[0].plot(avg_tp)
axs[1].plot(avg_fp)
axs[2].plot(threshs)
fig.show()
