import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt, rcParams
from scipy.cluster import hierarchy

def get_leaf_label(id):
    if id < len(doc_names):
        return str("%02d" % (int(doc_names[id].split(' - ')[0]), )) + ' - ' + doc_names[id].split(' - ')[1]
    else:
        return 0

with open("doc_matrices/doc_name_list.txt") as file: # Use file to refer to the file object
    doc_names = [x.strip() for x in file.readlines()]

root = Path('doc_matrices')
paths = list(root.glob('**/*.npy'))


#hierarchy.set_link_color_palette([ '#73B7CE', '#4D83B2', '#2E498D', '#171F5F'])
hierarchy.set_link_color_palette([ '#F0BE41', '#D85040', '#5383EC', '#58A55C' ])

rcParams['font.sans-serif'] = ['Linux Biolinum', 'Tahoma', 'DejaVu Sans',
                               'Lucida Grande', 'Verdana']


for i, path in enumerate(paths):

    if path.name != 'tf_doc_symbol_matrix.npy':
        continue

    feature_matrix = np.load(path)

    methods = ['ward']
    for method in methods:
        # Should consider using L2 distance here for word-to-vec model rather than euclidean
        links = hierarchy.linkage(feature_matrix, method=method, optimal_ordering=True)

        fig = plt.figure(figsize=((3.33 * 2) + 0.33, 3.25), dpi=220)

        hierarchy.dendrogram(
            links,
            leaf_label_func = get_leaf_label,
            orientation='left',
            leaf_rotation=0.,  # rotates the x axis labels

            color_threshold=0.4,
            above_threshold_color='xkcd:light grey',
        )

        # plt.title('Hierarchical Clustering of MSWE Document Representations', x=0.2, y = 1.04)
        plt.xlabel('Euclidean Distance Between Clusters', fontweight='bold', fontsize=9)
        plt.ylabel('Document ID & Title', fontweight='bold', fontsize=9)
        plt.tick_params(labelsize=8)


        fig.tight_layout(rect=[0, 0, 1, 1])

        fig.savefig('figures/hier_clust.pdf',
                    frameon=None)
        plt.show()
