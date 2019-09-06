import numpy as np
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from matplotlib import pyplot as plt, rcParams

root = Path('doc_matrices')
paths = list(root.glob('**/*.npy'))



mds = MDS(eps=0.0001, max_iter=3000, n_jobs=3, metric=False)
tsne = TSNE(n_components=2, init='random', method='exact', perplexity=7, early_exaggeration=100, n_iter=3500)

projectors = [tsne]
y = '#F0BE41'
b = '#5383EC'
r = '#D85040'
g = '#58A55C'
w = '#D8DCD6'
colours = [g,r,r,r,y,b,g,r,y,y,r,r,b,y,w,y,y,r,r,r,b,y,w,b]

plt.close('all')
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3.33,3.33*0.8), dpi = 220)



rcParams['font.sans-serif'] = ['Linux Biolinum', 'Tahoma', 'DejaVu Sans',
                               'Lucida Grande', 'Verdana']

def plotter(ax, data, title=None, fontsize=9):
    ax.scatter(data[:, 0], data[:, 1], c=colours)
    ax.locator_params(nbins=3)
    ax.set_xlabel('First Direction', fontweight='bold', fontsize=fontsize)
    ax.set_ylabel('Second Direction', fontweight='bold', fontsize=fontsize)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=fontsize+2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    x0, x1 = ax.get_xlim()
    x_diff = x1 - x0
    y0, y1 = ax.get_ylim()
    y_diff = y1 - y0
    ax.set_aspect(x_diff / y_diff)

    for j in range(len(data)):
        x = data[j][0]
        y = data[j][1]
        ax.text(x + (0.0175 * x_diff), y + (0.0175 * y_diff), j + 1, fontsize=fontsize)


sub_figure = 0

pca_machine = PCA(n_components=20)

for i, path in enumerate(paths):

    if path.name not in ['tf_doc_symbol_matrix.npy']:
        continue

    feature_matrix = np.load(path)

    # feature_matrix = pca_machine.fit_transform(feature_matrix)

    for j, projector in enumerate(projectors):

        data = projector.fit_transform(feature_matrix)
        plotter(ax=axes,
                data=data)

        sub_figure += 1


fig.tight_layout(rect=[0.08, 0, 0.86, 1])
fig.savefig('figures/projection.pdf',
            frameon=None)

plt.show()



    #
    # Y = tsne.fit_transform(feature_matrix)
    # ax = fig.add_subplot(2, 5, 10)
    # plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')
    #
    # plt.show()
