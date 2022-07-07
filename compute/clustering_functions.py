import hdbscan
import json
import os
import math

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from imgcat import imgcat

import markov_clustering as mc # python -m pip install markov_clustering
import community as louvain # python -m pip install python-louvain

np.random.seed(0)

def graph_drawer(G, cluster_name):
    # Create positions of all nodes and save them
    G.remove_edges_from(nx.selfloop_edges(G))
    pos = nx.spring_layout(G, k=10, scale=500)

    x = []
    y = []
    for node, xy in pos.items():
        x.append(xy[0])
        y.append(xy[1])

    x_offset = abs(min(x))
    y_offset = abs(min(y))

    for node in G.nodes.data():
        node[1]['x'] = pos[node[0]][0] + x_offset + 50
        node[1]['y'] = pos[node[0]][1] + y_offset + 50

    for edge in G.edges.data():
        edge[2]['x1'] = pos[edge[0]][0] + x_offset + 50
        edge[2]['y1'] = pos[edge[0]][1] + y_offset + 50
        edge[2]['x2'] = pos[edge[1]][0] + x_offset + 50
        edge[2]['y2'] = pos[edge[1]][1] + y_offset + 50

    fig, ax = plt.subplots()
    ax.margins(x=0, y=0)

    nx.draw(G, pos, ax=ax, node_size=15, width=0.6)


    imgcat(fig)
    # plt.show()
    plt.savefig(os.path.join('./data/cluster_htmls/recursive_louvain_cluster_vis/ukr', cluster_name))

    G_json = nx.node_link_data(G)
    with open(os.path.join('./data/cluster_htmls/recursive_louvain_cluster_vis/ukr', f'{cluster_name}.json'), 'w+') as f:
        json.dump(G_json, f)

def spectral_clustering(G, output_file):
    path_to_index = {}
    index_to_path = {}
    index = 0

    tqdm.write('Spectral: IDing images for clustering...')
    for img in list(G.nodes):
        path_to_index[img] = index
        index_to_path[index] = img
        index += 1


    print('Spectral: creating matrix')
    distances = nx.convert_matrix.to_numpy_array(G, dtype=np.double)
    np.fill_diagonal(distances, 250000.)
    distances = np.clip(distances, a_min=0, a_max=None)
    distances = csr_matrix(distances)
    distances.eliminate_zeros()
    n_neigh = min(distances.getnnz(1))
    print(distances.shape)
    print(type(distances))
    print(n_neigh)

    print('Spectral: creating sc')
    sc = SpectralClustering(n_neighbors=n_neigh, n_clusters=150, affinity='precomputed_nearest_neighbors', random_state=0, assign_labels='discretize', n_jobs=-1, verbose=True, eigen_solver='amg')
    print('Spectral: fitting')
    sc.fit(distances)
    # sc.fit_predict(distances)


    # print(len(index_to_path))
    # print(len(list(sc.labels_)))
    # print(len(list(range(len(G)))))
    # print(sc.labels_)
    # exit(69)

    partition = defaultdict(list)
    for n, p in zip(list(range(len(G))), list(sc.labels_)):
        partition[int(p)].append(index_to_path[n])

    with open(output_file, 'w+') as f:
        json.dump(partition, f)


def markov_clustering(G, output_file):
    path_to_index = {}
    index_to_path = {}
    index = 0

    tqdm.write('Markov: IDing images for clustering...')
    for img in list(G.nodes):
        path_to_index[img] = index
        index_to_path[index] = img
        index += 1

    matrix = nx.to_scipy_sparse_matrix(G)
    result = mc.run_mcl(matrix)
    clusters = mc.get_clusters(result)

    # print(clusters[1])
    partition = {}
    c_num = 0
    for cluster in clusters:
        partition[c_num] = [index_to_path[q] for q in cluster]
        c_num += 1

    # print(partition[1])
    # exit(69)
    with open(output_file, 'w+') as f:
        json.dump(partition, f)


def recursive_louvain_clustering(G):
    # distances = np.clip(distances, a_min=0, a_max=None)
    clusters = louvain.best_partition(G)

    cluster_dict = defaultdict(list)
    for image, cluster_id in clusters.items():
        cluster_dict[f'parent_cluster_{cluster_id}'].append(image)

    # print(len(cluster_dict))
    # exit(69)
    # parent_dict = cluster_dict.copy()
    # for cluster_id, member_list in parent_dict.items():
    #     graph_drawer(nx.Graph(G.subgraph(member_list)), cluster_id)
        # clusters = louvain.best_partition(G.subgraph(member_list))

        # for image, cluster_id in clusters.items():
        #     cluster_dict[f'child_cluster_{cluster_id}'].append(image)

    with open('./data/graph_cache/recursive_louvain_clusters.json', 'w+') as f:
        json.dump(cluster_dict, f)

    return clusters


def louvain_clustering(G):
    for n1, n2, d in G.edges.data():
        if d['weight'] < 0:
            d['weight'] = d['weight'] * -1

    # for n in G.nodes():
    #     if G.degree(n, weight='weight') < 0:
    #         print(n)

    # clusters = louvain.best_partition(G, resolution=0.008)
    clusters = louvain.best_partition(G, resolution=0.05)

    cluster_dict = defaultdict(list)
    for image, cluster_id in clusters.items():
        cluster_dict[cluster_id].append(image)

    return cluster_dict

    # print('Clusters:', len(cluster_dict))
    # with open(output_file, 'w+') as f:
    #     json.dump(cluster_dict, f)


def h_dbscan(G, output_file):
    # path_length = dict(nx.all_pairs_shortest_path_length(G))
    # distances = np.zeros((len(G), len(G)))
    # distances = np.full((len(G), len(G)), np.inf)

    path_to_index = {}
    index_to_path = {}
    index = 0

    tqdm.write('IDing images for clustering...')
    for img in list(G.nodes):
        path_to_index[img] = index
        index_to_path[index] = img
        index += 1

    # tqdm.write('Adding distances to matrix...')
    # for u, p in path_length.items():
    #     for v, d in p.items():
    #         distances[path_to_index[u]][path_to_index[v]] = d

    distances = nx.convert_matrix.to_numpy_array(G, nonedge=np.inf, dtype=np.double)

    tqdm.write('HDB scanning...')
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='precomputed')
    cluster_labels = clusterer.fit_predict(distances)

    cluster_dict = defaultdict(list)
    for i in range(0, len(cluster_labels)):
        cluster_dict[int(cluster_labels[i])].append(index_to_path[i])

    with open(output_file, 'w+') as f:
        json.dump(cluster_dict, f)


def hierarchical_clustering(G, output_file):
    # if not i:
    #     i = Index(cache_dir='/nfs/datasets/bill_data/ukr_data/image_index_cache')
    #     i.load_index()

    # path_length = dict(nx.all_pairs_shortest_path_length(G))
    # distances = np.zeros((len(G), len(G)))

    distances = nx.convert_matrix.to_numpy_array(G, dtype=np.double)
    # distances = nx.convert_matrix.to_scipy_sparse_matrix(G, dtype=np.double)
    np.fill_diagonal(distances, 0.0)
    # distances.setdiag(values=0.0)

    np.clip(distances, 0, 1, distances)

    path_to_index = {}
    index_to_path = {}
    index = 0

    tqdm.write('IDing images for clustering...')
    for img in list(G.nodes):
        path_to_index[img] = index
        index_to_path[index] = img
        index += 1

    print(len(index_to_path))

    # tqdm.write('Adding distances to matrix...')
    # for u, p in path_length.items():
    #     for v, d in p.items():
    #         distances[path_to_index[u]][path_to_index[v]] = d

    # Create hierarchical cluster
    print('Doing Clustering')
    Y = distance.squareform(distances)
    Z = hierarchy.complete(Y)  # Creates HC using farthest point linkage
    print('Clustering done')
    # This partition selection is arbitrary, for illustrive purposes

    # membership = list(hierarchy.fcluster(Z, t=1.15))
    membership = list(hierarchy.fcluster(Z, t=1.2))
    # membership = list(hierarchy.fcluster(Z, t=2))
    # Create collection of lists for blockmodel
    partition = defaultdict(list)
    for n, p in zip(list(range(len(G))), membership):
        partition[int(p)].append(index_to_path[n])

    with open(output_file, 'w+') as f:
        json.dump(partition, f)
    # print(list(partition.values()))
    # print(nx.clustering(G))
