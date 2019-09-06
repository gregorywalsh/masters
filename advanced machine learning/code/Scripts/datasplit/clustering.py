import numpy as np
import pandas as pd
import warnings
import pickle

import code.Scripts.datasplit.preprocessing as pp

from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE
from matplotlib import pyplot as plt, pylab, rcParams
from sklearn.cluster import DBSCAN


class TSNEClusterer():

    def __init__(self, df_x):
        self.df_x = df_x
        self.subclasses = None
        self.multi_tsne_projector = MulticoreTSNE(n_jobs=2,
                                                  n_components=2,
                                                  init='random',
                                                  method='barnes_hut',
                                                  perplexity=50,
                                                  early_exaggeration=100,
                                                  n_iter=6000)
        if 'Class' in self.df_x.columns:
            warnings.warn(message='df_x contains class column. Class column will be ignored when clustering.')
            self.df_x = self.df_x[self.df_x.columns.difference(['Class'])]


    def get_subclasses(self, reproject=False):
        if self.subclasses is None or reproject:
            self.x_projected = self.multi_tsne_projector.fit_transform(self.df_x)
            density_clusterer = DBSCAN(n_jobs=2, eps=5)
            self.subclasses = density_clusterer.fit_predict(X=self.x_projected)
            self.sr_subclasses = pd.Series(self.subclasses + 1, index=self.df_x.index, name='Subclass')
        return self.sr_subclasses


    def plot_clusters(self):

        fig, axes = plt.subplots(nrows=1,
                                 ncols=1,
                                 squeeze=False)

        rcParams['font.sans-serif'] = ['Linux Biolinum',
                                       'Tahoma',
                                       'DejaVu Sans',
                                       'Lucida Grande',
                                       'Verdana']

        self.plot_scatter_plot_subfigure(ax=axes[0][0],
                                         data=self.x_projected,
                                         classes=self.subclasses)

        # Show plots
        fig.tight_layout()
        plt.show()


    def plot_scatter_plot_subfigure(self, ax, data, classes=None, colour=None, title=None, fontsize=9):

        if colour is not None:
            ax.scatter(data[:, 0], data[:, 1], c=colour, s=0.5)
        else:
            ax.scatter(data[:, 0], data[:, 1], c=classes, cmap=pylab.cm.viridis, s=0.5)

        ax.locator_params(nbins=3)
        ax.set_xlabel('First Direction', fontweight='bold', fontsize=fontsize)
        ax.set_ylabel('Second Direction', fontweight='bold', fontsize=fontsize)
        ax.tick_params(labelsize=8)

        if title:
            ax.set_title(title, fontsize=fontsize + 2)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        x0, x1 = ax.get_xlim()
        x_diff = x1 - x0
        y0, y1 = ax.get_ylim()
        y_diff = y1 - y0
        ax.set_aspect(x_diff / y_diff)


if __name__ == '__main__':

    df = pp.get_processed_data(path='../../Dataset/creditcard.csv')
    df_x = df.loc[:, df.columns.difference(['Class'])]
    sr_y = df.loc[:, 'Class']

    df_x_positives = df_x.loc[sr_y == 1]
    clusterer = TSNEClusterer(df_x=df_x_positives)
    sr_subclasses = clusterer.get_subclasses()
    clusterer.plot_clusters()
    # with open('../cvsplits/df_subclasses.pkl', 'wb') as f:
    #     pickle.dump(obj=sr_subclasses, file=f)