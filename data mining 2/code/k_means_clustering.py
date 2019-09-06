import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt, rcParams

from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

with open("doc_matrices/doc_name_list.txt") as file: # Use file to refer to the file object
    doc_names = [x.strip() for x in file.readlines()]

root = Path('doc_matrices')
paths = list(root.glob('**/*.npy'))

rcParams.update({'figure.autolayout': True})



for i, path in enumerate(paths):

    if path.name != 'tf_doc_symbol_matrix.npy':
        continue

    print("\nPredictions for " + path.name)
    feature_matrix = np.load(path)

    # TODO - Is scaling required for ALL of the feature matrices?
    scale(feature_matrix)

    for k in range(6, 7):

        fig = plt.figure(i+1, figsize=(4, 3))
        kmeans = KMeans(n_clusters=k, n_init=100, max_iter=1000, init='random').fit(feature_matrix)
        labels = kmeans.labels_
        sil_coeff = silhouette_score(feature_matrix, labels, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}, Inertia is {}".format(k, sil_coeff, kmeans.inertia_))
        print(k, kmeans.inertia_)

        predictions = kmeans.predict(feature_matrix)

        named_predictions = []
        for j, prediction in enumerate(predictions):
            named_predictions.append((prediction, doc_names[j]))

        named_predictions.sort(key=lambda x: x[0])

        for cluster_id, doc_name in named_predictions:
            print(str(cluster_id) + " " + doc_name)

plt.show()