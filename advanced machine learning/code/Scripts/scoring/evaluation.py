import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_curve
from .fb_score import precisionRecallScore


def get_thresholds_table(y_true, y_out, thresholds=None, betas=[1, 10, 50]):
    """get_thresholds_table

        Generates a dataframe with different scores for each of the thresholds provided.

        Arguments:
        ---------
        y_true: list
            target labels of each point in the set (binary)
        y_out: list
            outputs of the model for each point in the set (same size as y_true)
        thresholds: list
            all the different ways of deciding a classification threshold on the output
        betas: list
            parameters for the different fb_score decisions

        Output:
        ------
        thresh_table: pd.DataFrame
            complete table with performance data of each threshold splitting
        best_results: pd.DataFrame
            best rows of the dataframe according to the corresponding betas

        """

    if thresholds is None:
        thresholds = generate_thresholds(y_true, y_out)

    columns = ['Threshold'] + ['F' + str(beta) for beta in betas] + ['Precision', 'Recall', 'TP', 'FP', 'TN', 'FN']
    thresh_table = pd.DataFrame(index=np.arange(len(thresholds)), columns=columns)

    for i, threshold in enumerate(thresholds):
        y_pred = [1 if el > threshold else 0 for el in y_out]
        thresh_table.iloc[i] = [threshold] + get_single_row(y_true, y_pred, betas)

    best_results = pd.DataFrame(index=np.arange(len(betas)), columns=columns, dtype=float)
    for i, beta in enumerate(betas):
        best_results.iloc[i] = thresh_table.sort_values(by='F' + str(beta), ascending=False).iloc[0]

    return thresh_table, best_results


def get_single_row(y_true, y_pred, betas):
    """get_single_row

            Generates a dataframe with different scores for a specific prediction instance.

            Arguments:
            ---------
            y_true: list
                target labels of each point in the set (binary)
            y_pred: list
                prediction for each point in the set (same size as y_true)
            betas: list
                parameters for the different fb_score decisions

            Output:
            ------
            list of the results with:
                betas: list
                    fbeta scores for the set of betas
                pr: list (size 2)
                    precision and recall
                tfpn: list (size 4)
                    tp, fp, tn, fn

            """
    # Precision-recall curve and fbeta score
    pr, tfpn = precisionRecallScore(y_true, y_pred)
    fbetas = [fbeta_score(y_true, y_pred, beta) for beta in betas]
    return fbetas + pr + tfpn


def generate_thresholds(y_true, y_out):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_out)
    return thresholds


if __name__ == "__main__":
    a, b = get_thresholds_table([0, 1], [0.2, 0.8])
    """
    fig1, ax1 = plt.subplots()
    ax1.plot(self.precisions, recalls)
    ax1.grid()
    ax1.set(xlabel='Precision', ylabel='Recall')
    # fig1.show()

    # 90% Recall point
    self.recalls = pd.Series(recalls)
    self.thresholds = pd.Series(thresholds)
    self.thresh90 = self.thresholds.ix[self.recalls[self.recalls > 0.9].index[-1]]

    # Histograms of data
    clean = self.y_prob[test['Class'].values == 0]
    fraud = self.y_prob[test['Class'].values == 1]

    fig, ax = plt.subplots()
    clean.plot.hist(logy=True, ax=ax)
    fraud.plot.hist(logy=True, ax=ax)
    ax.grid()
    # fig.show()


return {'precision': self.precisions, 'recall': self.recalls, 'thresholds': self.thresholds,
        'thresh90': self.thresh90}, [fig, fig1], [ax, ax1]
"""
