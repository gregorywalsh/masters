import numpy as np
import pandas as pd
import code.Scripts.datasplit.preprocessing as pre

from code.Scripts import reduceSize
from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE
from matplotlib import pyplot as plt, pylab, rcParams, transforms
from sklearn.cluster import DBSCAN

# Show plots
rcParams['font.sans-serif'] = ['Linux Biolinum', 'Tahoma', 'DejaVu Sans',
                               'Lucida Grande', 'Verdana']

def plotter(ax, data, classes=None, colour=None, title=None, fontsize=9):
    y = '#F0BE41'
    b = '#5383EC'
    r = '#D85040'
    g = '#58A55C'
    w = '#D8DCD6'
    colours = np.array([b, r, y, g, g, g, w])

    if colour is not None:
        ax.scatter(data[:, 0], data[:, 1], c=colour, s=0.3)
    else:
        ax.scatter(data[:, 0], data[:, 1], c=classes, cmap=pylab.cm.viridis, s=0.3)

    ax.locator_params(nbins=3)
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

    # for j in range(len(data)):
    #     x = data[j][0]
    #     y = data[j][1]
    #     ax.text(x + (0.0175 * x_diff), y + (0.0175 * y_diff), j + 1, fontsize=fontsize)


# Create Projectors
tsne_projector = TSNE(n_components=2,
                      init='random',
                      method='barnes_hut',
                      perplexity=10,
                      early_exaggeration=100,
                      n_iter=3500)

multi_tsne_projector = MulticoreTSNE(n_jobs=2,
                                     n_components=2,
                                     init='random',
                                     method='barnes_hut',
                                     perplexity=50,
                                     early_exaggeration=100,
                                     n_iter=6000)

# Create axes
runs = 1
fig, axes = plt.subplots(nrows=1,
                         ncols=2,
                         dpi=220,
                         squeeze=False,
                         figsize=(3.45, 1.8)
)

# Load data
print('Loading data')
df = pre.get_processed_data('../../Dataset/creditcard.csv')
df = reduceSize(df=df, total_data=1000, proportion=0.5)

# Project positives
print('Projecting positives')
df_positives = df[df['Class'] == 1]
df_project_positives = df_positives[df_positives.columns.difference(['Class'])]
projected_positives = multi_tsne_projector.fit_transform(df_project_positives)

# Cluster positives
print('Clustering positives')
density_cluster = DBSCAN(n_jobs=2, eps=5)
labels = density_cluster.fit_predict(X=projected_positives)

# Plot positives
print('Plotting projection of positives')
plotter(ax=axes[0][0],
        data=projected_positives,
        classes=labels)

# Project all
print('Projecting all data')
df_project_all = df[df.columns.difference(['Class'])]
projected_all = multi_tsne_projector.fit_transform(df_project_all)
df['Class'][df['Class'] == 1] = labels + 2

# Plot all
print('Plotting projection of all data')
projected_positives = projected_all[df['Class'] > 0]
projected_negatives = projected_all[df['Class'] == 0]
plotter(ax=axes[0][1],
        data=projected_negatives,
        colour='#D8DCD6')
plotter(ax=axes[0][1],
        data=projected_positives,
        classes=df['Class'][df['Class'] > 0])

axes[0][0].set_title("A - Fraud Class", fontweight='bold', fontsize=9)
axes[0][1].set_title("B - All Classes", fontweight='bold', fontsize=9)
axes[0][0].set_ylabel('Second Direction', fontsize=9)
axes[0][0].set_xlabel('First Direction', fontsize=9)
axes[0][1].set_xlabel('First Direction', fontsize=9)

axes[0][0].tick_params(labelbottom='off', labelleft='off', axis=u'both', which=u'both',length=0)
axes[0][1].tick_params(labelbottom='off', labelleft='off', axis=u'both', which=u'both',length=0)
axes[0][0].xaxis.labelpad = 7
axes[0][0].yaxis.labelpad = 7
axes[0][1].xaxis.labelpad = 7
axes[0][1].yaxis.labelpad = 7

# plt.tight_layout()
plt.subplots_adjust(left=0.07, right=0.93)

fig.savefig('figures/clustering.pdf',
            bbox_inches=None,
            dpi='figure')

